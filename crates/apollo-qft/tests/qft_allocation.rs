//! Warm reusable QFT allocation contract.

use apollo_qft::{QftPlan, QuantumStateDimension};
use eunomia::Complex64;
use leto::Array1;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static COUNTS: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}

fn note_allocation(is_reallocation: bool) {
    let _ = COUNTS.try_with(|counts| {
        if let Some((allocations, reallocations)) = counts.get() {
            counts.set(Some(if is_reallocation {
                (allocations, reallocations + 1)
            } else {
                (allocations + 1, reallocations)
            }));
        }
    });
}

fn count_allocations<R>(operation: impl FnOnce() -> R) -> (R, usize, usize) {
    COUNTS.with(|counts| counts.set(Some((0, 0))));
    let result = operation();
    let (allocations, reallocations) = COUNTS.with(|counts| counts.replace(None)).unwrap_or((0, 0));
    (result, allocations, reallocations)
}

struct CountingAllocator;

// SAFETY: every operation delegates unchanged to `System`; the thread-local
// counters only observe calls and never alter pointer or layout semantics.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_allocation(false);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        note_allocation(true);
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn signal(len: usize) -> Array1<Complex64> {
    Array1::from_shape_fn([len], |[index]| {
        let x = index as f64;
        Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
    })
}

#[test]
fn warmed_reusable_qft_execution_allocates_zero_times() {
    for len in [127, 256] {
        let plan = QftPlan::new(QuantumStateDimension::new(len).expect("valid dimension"));
        let input = signal(len);
        let mut spectrum = Array1::zeros([len]);
        let mut recovered = Array1::zeros([len]);

        plan.forward_into(&input, &mut spectrum)
            .expect("warm forward QFT");
        plan.inverse_into(&spectrum, &mut recovered)
            .expect("warm inverse QFT");

        let ((), allocations, reallocations) = count_allocations(|| {
            plan.forward_into(&input, &mut spectrum)
                .expect("measured forward QFT");
            plan.inverse_into(&spectrum, &mut recovered)
                .expect("measured inverse QFT");
        });
        assert_eq!(allocations, 0, "length {len} allocated after warm-up");
        assert_eq!(reallocations, 0, "length {len} reallocated after warm-up");

        let l1 = input.iter().map(|value| value.norm()).sum::<f64>();
        let tolerance = 128.0 * len as f64 * f64::EPSILON * l1.max(1.0);
        for (index, (actual, expected)) in recovered.iter().zip(input.iter()).enumerate() {
            assert!(
                (*actual - *expected).norm() <= tolerance,
                "length {len} index {index}: {actual:?} differs from {expected:?} by {}, bound {tolerance}",
                (*actual - *expected).norm()
            );
        }
    }
}
