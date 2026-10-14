use std::marker::PhantomData;
use std::ptr::NonNull;

use bytemuck::Zeroable;

use crate::allocation::Allocation;

/// A heap buffer starting at a deterministic in-page lane.
///
/// The backing allocation is aligned to both 4096 bytes and `T`; the
/// exposed slice starts at `base + lane`, so `addr() % 4096 == lane`.
/// All exposed elements are initialized and are dropped with the buffer.
pub struct LaneBuffer<T> {
    data: NonNull<T>,
    len: usize,
    allocation: Allocation,
    _marker: PhantomData<T>,
}

impl<T> LaneBuffer<T> {
    /// Allocates `len` zeroed elements starting at in-page byte `lane`.
    ///
    /// `T: Zeroable` guarantees the all-zero representation is valid. This
    /// does not follow from `Default` or `Copy` alone.
    ///
    /// ```compile_fail
    /// use apollo_alloc::LaneBuffer;
    /// use std::num::NonZeroU8;
    /// let _ = LaneBuffer::<NonZeroU8>::zeros(1, 0);
    /// ```
    ///
    /// ```compile_fail
    /// use apollo_alloc::LaneBuffer;
    /// let _ = LaneBuffer::<&u8>::zeros(1, 0);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when `lane` reaches 4096, is misaligned for `T`, the allocation
    /// size exceeds the allocator layout limit, or allocation fails.
    #[must_use]
    pub fn zeros(len: usize, lane: usize) -> Self
    where
        T: Zeroable,
    {
        let (allocation, data) = Allocation::for_elements::<T>(len, lane);
        // SAFETY: the allocation is live and entirely writable; Zeroable
        // guarantees that zeroing it initializes every T, including ZSTs.
        unsafe {
            allocation
                .base
                .as_ptr()
                .write_bytes(0, allocation.layout.size())
        };
        Self {
            data,
            len,
            allocation,
            _marker: PhantomData,
        }
    }

    /// Allocates a lane buffer holding a copy of `src`.
    ///
    /// Values need not have a valid all-zero representation: raw storage is
    /// initialized from `src` before any typed slice is formed.
    ///
    /// # Panics
    ///
    /// Panics under the same lane, size, and allocation conditions as `zeros`.
    #[must_use]
    pub fn from_slice(src: &[T], lane: usize) -> Self
    where
        T: Copy,
    {
        let (allocation, data) = Allocation::for_elements::<T>(src.len(), lane);
        // SAFETY: source contains src.len() initialized T values; destination
        // is aligned raw storage of sufficient size in a distinct allocation.
        // Copy initializes all elements without creating uninitialized &mut T.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), data.as_ptr(), src.len()) };
        Self {
            data,
            len: src.len(),
            allocation,
            _marker: PhantomData,
        }
    }

    /// Returns the lane contents.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: data is non-null/aligned and spans len initialized elements
        // within the retained allocation, whose total size fits isize.
        unsafe { std::slice::from_raw_parts(self.data.as_ptr().cast_const(), self.len) }
    }

    /// Returns the lane contents mutably.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as above, with exclusive access through &mut self.
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }

    /// Returns the slice start address, for aliasing diagnostics.
    #[must_use]
    pub fn addr(&self) -> usize {
        self.data.addr().get()
    }
}

impl<T> Drop for LaneBuffer<T> {
    fn drop(&mut self) {
        debug_assert!(self.len * core::mem::size_of::<T>() < self.allocation.layout.size());
        // SAFETY: data spans exactly len initialized elements exclusively
        // owned here. Drop the elements once; allocation's field destructor
        // frees their storage afterward, including on unwinding.
        unsafe {
            std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(
                self.data.as_ptr(),
                self.len,
            ))
        };
    }
}

// SAFETY: LaneBuffer exclusively owns its allocation and elements, like Vec.
unsafe impl<T: Send> Send for LaneBuffer<T> {}
// SAFETY: shared access exposes shared T only; mutable access is exclusive.
unsafe impl<T: Sync> Sync for LaneBuffer<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocation::PAGE_ALIGNMENT;

    #[test]
    fn lanes_pin_low_address_bits() {
        for lane in [0usize, 512, 1024, 2048, 3072] {
            let buffer = LaneBuffer::<u64>::zeros(32, lane);
            assert_eq!(buffer.addr() % PAGE_ALIGNMENT, lane);
        }
    }

    #[test]
    fn lanes_pin_for_size_ne_align_types() {
        // Complex64 is size 16 / align 8: the lane must still pin exactly.
        #[repr(C)]
        #[derive(Clone, Copy, Default)]
        struct Complex64Like {
            re: f64,
            im: f64,
        }
        // SAFETY: both fields are f64, whose all-zero representation is valid.
        unsafe impl Zeroable for Complex64Like {}
        assert_eq!(core::mem::size_of::<Complex64Like>(), 16);
        assert_eq!(core::mem::align_of::<Complex64Like>(), 8);
        for lane in [0usize, 512, 2048] {
            let buffer = LaneBuffer::<Complex64Like>::zeros(32, lane);
            assert_eq!(buffer.addr() % PAGE_ALIGNMENT, lane);
        }
    }

    #[test]
    fn zeros_are_zero_and_writable() {
        let mut buffer = LaneBuffer::<u64>::zeros(4, 0);
        assert_eq!(buffer.as_slice(), &[0, 0, 0, 0]);
        buffer.as_mut_slice().copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(buffer.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn from_slice_round_trips() {
        let buffer = LaneBuffer::from_slice(&[7u32, 8, 9], 1024);
        assert_eq!(buffer.as_slice(), &[7, 8, 9]);
        assert_eq!(buffer.addr() % PAGE_ALIGNMENT, 1024);
    }

    #[test]
    fn distinct_lanes_do_not_overlap() {
        let mut first = LaneBuffer::<u64>::zeros(64, 0);
        let mut second = LaneBuffer::<u64>::zeros(64, 2048);
        first.as_mut_slice().fill(0xAA);
        second.as_mut_slice().fill(0xBB);
        assert!(first.as_slice().iter().all(|&x| x == 0xAA));
        assert!(second.as_slice().iter().all(|&x| x == 0xBB));
    }

    #[test]
    fn empty_buffer_keeps_its_lane() {
        let buffer = LaneBuffer::<u64>::zeros(0, 512);
        assert!(buffer.as_slice().is_empty());
        assert_eq!(buffer.addr() % PAGE_ALIGNMENT, 512);
    }
}

#[cfg(test)]
mod safety_tests;
