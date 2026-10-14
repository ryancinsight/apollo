use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

pub(crate) const PAGE_ALIGNMENT: usize = 4096;

/// Owns backing storage independently of the initialized element owner.
pub(crate) struct Allocation {
    pub(crate) base: NonNull<u8>,
    pub(crate) layout: Layout,
}

impl Allocation {
    pub(crate) fn for_elements<T>(len: usize, lane: usize) -> (Self, NonNull<T>) {
        assert!(
            lane < PAGE_ALIGNMENT,
            "invariant: lane {lane} fits in one page"
        );
        let alignment = core::mem::align_of::<T>();
        assert_eq!(
            lane % alignment,
            0,
            "invariant: lane {lane} aligns {alignment}"
        );
        let bytes = len
            .checked_mul(core::mem::size_of::<T>())
            .expect("invariant: lane buffer size fits usize");
        let layout = Layout::from_size_align(
            bytes
                .checked_add(PAGE_ALIGNMENT)
                .expect("invariant: lane overallocation fits usize"),
            PAGE_ALIGNMENT.max(alignment),
        )
        .expect("invariant: lane allocation size fits isize");
        // SAFETY: `layout` has non-zero size and valid alignment. Successful
        // allocation is exclusively owned here until deallocated by Drop.
        let base = unsafe { alloc(layout) };
        let allocation = Self {
            base: NonNull::new(base).expect("invariant: lane backing allocation succeeds"),
            layout,
        };
        // SAFETY: lane < 4096 and len * size_of::<T>() + 4096 is the layout
        // size, so both the start and the full element span stay in bounds.
        // Rust alignments are powers of two: max(page, T alignment) is a
        // multiple of both, and lane is a multiple of T's alignment.
        let data = unsafe { allocation.base.add(lane).cast() };
        (allocation, data)
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        // SAFETY: this is the live allocator pointer/layout pair, freed once
        // after element destruction (also when an element destructor unwinds).
        unsafe { dealloc(self.base.as_ptr(), self.layout) };
    }
}
