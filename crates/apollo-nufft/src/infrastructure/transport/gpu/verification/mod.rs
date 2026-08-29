//! NUFFT value-semantic verification grouped by operation contract.
//!
//! The direct Type-1/Type-2 pair satisfies the adjoint law
//! `<Type1(c), f> = <c, Type2(f)>` in exact arithmetic. Fast Kaiser--Bessel
//! execution retains the owning plan's derived finite-precision bounds.
//!
//! Direct-operation leaves retain CPU differential, Leto, represented-storage,
//! and rejection contracts. Fast-operation leaves retain independently
//! evaluated gridded CPU comparisons, normalization, and diagnostic-grid
//! contracts. The adjoint statement is a proof sketch; these value-semantic
//! operator tests are empirical finite-precision evidence, not a machine-
//! checked proof.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static ALLOCATION_COUNT: Cell<Option<usize>> = const { Cell::new(None) };
}

fn note_allocation() {
    let _ = ALLOCATION_COUNT.try_with(|counter| {
        if let Some(count) = counter.get() {
            counter.set(Some(count + 1));
        }
    });
}

pub(crate) fn count_allocations<R>(operation: impl FnOnce() -> R) -> (R, usize) {
    ALLOCATION_COUNT.with(|counter| counter.set(Some(0)));
    let result = operation();
    let count = ALLOCATION_COUNT
        .with(|counter| counter.replace(None))
        .unwrap_or(0);
    (result, count)
}

struct CountingAllocator;

// SAFETY: every operation delegates unchanged to the system allocator; the
// thread-local counter observes calls without changing pointer semantics.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

mod direct_type1_1d;
mod direct_type1_3d;
mod direct_type2_1d;
mod direct_type2_3d;
mod fast_type1_1d;
mod fast_type1_3d;
mod fast_type2_1d;
mod fast_type2_3d;
mod metadata;
mod reusable;
mod support;
