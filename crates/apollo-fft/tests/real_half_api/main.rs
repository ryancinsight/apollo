//! The half-spectrum API, in one dimension and in three.
//!
//! A real signal's spectrum satisfies `X[n-k] = conj(X[k])`, so its upper half
//! carries no information, and a real field's 3-D spectrum repeats every bin
//! with `k > nz/2` the same way. The half forms skip that half — its storage
//! and, in three dimensions, the passes over it. These tests pin that skipping
//! it changes nothing about the bins a caller receives or the signal an
//! inverse returns.
//!
//! ## Oracles, in order of authority
//!
//! 1. **Agreement with the full transforms.** The 1-D half forward shares the
//!    full one's code path, so their shared bins must be *identical*; the 3-D
//!    pair runs different passes, so its bins agree within the derived bound.
//! 2. **Conjugate symmetry**, a property of the input rather than of any
//!    implementation: the discarded half must be recoverable, and the 3-D
//!    inverse must be the real part of the full inverse of the Hermitian
//!    completion for any half spectrum.
//! 3. **Differential against RealFFT**, an independently authored real-FFT
//!    implementation whose public contract is this same `n/2 + 1` shape.
//! 4. **Round trips** within the derived bound, for every shipped storage
//!    scalar.
//! 5. **Allocation count**, because halving the output is half the point.

mod forward;
mod inverse;
mod volume;

use apollo_fft::{PlanCacheProvider, RealFftData, F16};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

/// `2c` from the `O(log N · u)` forward-error bound, covering both engines and
/// their differing twiddle generation.
const TOLERANCE_FACTOR: f64 = 16.0;

/// Lengths the split admits: `n >= 4` and a multiple of four.
const SPLIT_SIZES: [usize; 7] = [4, 8, 16, 64, 256, 1024, 4096];

/// Lengths it does not, which must still honour the same contract through the
/// full-transform fallback.
const FALLBACK_SIZES: [usize; 6] = [2, 6, 7, 9, 10, 14];

thread_local! {
    /// `None` while this thread is not measuring, otherwise the running count.
    ///
    /// Thread-local rather than a process-global counter, and that is not a
    /// detail: the harness runs these tests in parallel threads, so a global
    /// counts every sibling's allocations too. The first version of this test
    /// did exactly that and failed against its own siblings while passing in
    /// isolation — a test that depends on execution order is a flake authored
    /// in, not a result.
    ///
    /// `const`-initialized so that arming the counter cannot itself allocate.
    static COUNTER: Cell<Option<usize>> = const { Cell::new(None) };
}

fn note_allocation() {
    // `try_with` because a thread-local is unavailable during TLS teardown, and
    // an allocation there must not panic the allocator.
    let _ = COUNTER.try_with(|c| {
        if let Some(seen) = c.get() {
            c.set(Some(seen + 1));
        }
    });
}

/// Runs `f` and reports how many allocations it made **on this thread**.
fn count_allocations<R>(f: impl FnOnce() -> R) -> (R, usize) {
    COUNTER.with(|c| c.set(Some(0)));
    let value = f();
    let seen = COUNTER.with(|c| c.replace(None)).unwrap_or(0);
    (value, seen)
}

struct Counting;

// SAFETY: every method forwards to `System` unchanged; the counter observes and
// never affects the returned pointer.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        note_allocation();
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            (0.017 * x).sin() + 0.4 * (0.083 * x).cos()
        })
        .collect()
}

/// `‖x‖₁`, the scale the forward-error bound grows with.
fn l1(values: &[f64]) -> f64 {
    values.iter().map(|v| v.abs()).sum()
}

/// Per-bin error bound of a length-`n` forward transform of a signal with
/// 1-norm `l1`, in arithmetic of unit roundoff `unit`.
fn tolerance(n: usize, l1: f64, unit: f64) -> f64 {
    // Mixed-radix lengths have fewer, wider stages than log2(n) implies, so
    // using log2 keeps the bound conservative rather than tight; a length-2
    // transform still rounds once.
    TOLERANCE_FACTOR * (n as f64).log2().max(1.0) * unit * l1
}

/// A storage scalar the half-spectrum API ships for, with the roundings its
/// round trips are bounded by.
trait Sample: RealFftData + PlanCacheProvider {
    /// Unit roundoff `u = ε/2` of the plan arithmetic.
    const PLAN_UNIT: f64;
    /// Relative rounding of the narrowing back to storage; zero where storage
    /// is the plan precision.
    const STORAGE_UNIT: f64;
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl Sample for f64 {
    const PLAN_UNIT: f64 = f64::EPSILON / 2.0;
    const STORAGE_UNIT: f64 = 0.0;
    fn from_f64(value: f64) -> Self {
        value
    }
    fn to_f64(self) -> f64 {
        self
    }
}

impl Sample for f32 {
    const PLAN_UNIT: f64 = f32::EPSILON as f64 / 2.0;
    const STORAGE_UNIT: f64 = 0.0;
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl Sample for F16 {
    const PLAN_UNIT: f64 = f32::EPSILON as f64 / 2.0;
    /// Eleven significand bits.
    const STORAGE_UNIT: f64 = 1.0 / 2048.0;
    fn from_f64(value: f64) -> Self {
        F16::from_f32(value as f32)
    }
    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }
}
