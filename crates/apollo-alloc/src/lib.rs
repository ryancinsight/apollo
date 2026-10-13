#![deny(missing_docs)]

//! Deterministic-lane heap allocation for Apollo measurement.
//!
//! Fresh processes slide heap addresses with ASLR, and the low address bits
//! decide 4 KiB-aliasing stalls and L1-set conflicts for an engine's access
//! pattern. The N=32 comparison cell measured +-70-100% cross-run median
//! swings on one binary that cluster exactly by buffer address lanes
//! (apollo#646), so plain `Vec` working buffers make every ratio a
//! per-process lottery. [`LaneBuffer`] allocates page-aligned and starts at
//! a caller-chosen in-page lane, pinning the low 12 address bits regardless
//! of the ASLR slide and making repeated runs of one binary comparable.
//!
//! This crate owns the `unsafe` allocation core; consumers (notably
//! `apollo-bench`, which forbids `unsafe_code`) use the safe [`LaneBuffer`]
//! API. One fixed lane per role measures one aliasing configuration:
//! absolute medians are lane-specific, but ratios at a fixed lane are stable
//! across runs and commits, which is what the comparison needs. Roles
//! sharing a measurement window must take distinct lanes (a multiple of 512
//! keeps L1 sets disjoint too); roles in disjoint windows may reuse lanes.

use std::alloc::{alloc, dealloc, Layout};
use std::marker::PhantomData;
use std::ptr::NonNull;

/// Page alignment for the backing allocation, pinning the low 12 bits.
const PAGE_ALIGNMENT: usize = 4096;

/// A heap buffer starting at a deterministic in-page lane.
///
/// The backing allocation is page-aligned with room for any lane; the
/// exposed slice starts at `base + lane`, so
/// `as_ptr() as usize % 4096 == lane` on every run of one binary.
pub struct LaneBuffer<T> {
    data: NonNull<T>,
    len: usize,
    alloc: NonNull<u8>,
    layout: Layout,
    _marker: PhantomData<T>,
}

impl<T> LaneBuffer<T> {
    /// Allocates `len` zeroed elements starting at in-page byte `lane`.
    ///
    /// # Panics
    ///
    /// Panics when `lane` reaches the page size, is misaligned for `T`, or
    /// the allocator fails.
    #[must_use]
    pub fn zeros(len: usize, lane: usize) -> Self {
        assert!(
            lane < PAGE_ALIGNMENT,
            "invariant: lane {lane} fits in one page"
        );
        assert_eq!(
            lane % core::mem::align_of::<T>(),
            0,
            "invariant: lane {lane} aligns {}",
            core::mem::align_of::<T>()
        );
        let bytes = len
            .checked_mul(core::mem::size_of::<T>())
            .expect("invariant: lane buffer size fits usize");
        let layout = Layout::from_size_align(
            bytes
                .checked_add(PAGE_ALIGNMENT)
                .expect("invariant: lane overallocation fits usize"),
            PAGE_ALIGNMENT,
        )
        .expect("invariant: page alignment is a power of two");
        // SAFETY: `layout` has non-zero size with a valid power-of-two
        // alignment; on success `base` is a live `layout` allocation.
        let base = unsafe { alloc(layout) };
        let alloc = NonNull::new(base).expect("invariant: lane backing allocation succeeds");
        // SAFETY: `lane < 4096 <= layout.size()`, so `base + lane` stays in
        // bounds and keeps `T` alignment (the base is page-aligned and
        // `lane` is a multiple of `align_of::<T>()`); the full allocation
        // is zeroed before any element is read.
        unsafe {
            base.write_bytes(0, layout.size());
            let data = base.add(lane).cast::<T>();
            Self {
                data: NonNull::new(data).expect("invariant: lane pointer is non-null"),
                len,
                alloc,
                layout,
                _marker: PhantomData,
            }
        }
    }

    /// Allocates a lane buffer holding a copy of `src`.
    #[must_use]
    pub fn from_slice(src: &[T], lane: usize) -> Self
    where
        T: Copy,
    {
        let mut buffer = Self::zeros(src.len(), lane);
        buffer.as_mut_slice().copy_from_slice(src);
        buffer
    }

    /// Returns the lane contents.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `data` points at `len` initialized elements (zeroed at
        // allocation, wholly owned by this buffer) and outlives `&self`.
        unsafe { std::slice::from_raw_parts(self.data.as_ptr().cast_const(), self.len) }
    }

    /// Returns the lane contents mutably.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as above, with exclusive access through `&mut self`.
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
        // SAFETY: `alloc`/`layout` are the pair returned by the allocation
        // call, deallocated exactly once here.
        unsafe { dealloc(self.alloc.as_ptr(), self.layout) };
    }
}

// SAFETY: `LaneBuffer` owns its allocation exclusively; `&LaneBuffer<T>`
// exposes only shared element access, like `Vec<T>`.
unsafe impl<T: Send> Send for LaneBuffer<T> {}
// SAFETY: as above; `&mut` access is exclusive, like `Vec<T>`.
unsafe impl<T: Sync> Sync for LaneBuffer<T> {}

#[cfg(test)]
mod tests {
    use super::*;

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
