//! The `n/2 + 1` half-spectrum forward API.
//!
//! A real signal's spectrum satisfies `X[n-k] = conj(X[k])`, so the upper half
//! carries no information. [`apollo_fft::fft_1d_slice`] materializes it anyway,
//! which costs a mirror pass and twice the output storage; the half API skips
//! both. These tests pin that skipping them changes nothing about the bins the
//! caller actually receives.
//!
//! ## Oracles, in order of authority
//!
//! 1. **Bitwise agreement with the full spectrum.** Both forms run the same
//!    pack-transform-untangle, so the shared bins must be *identical*, not
//!    merely close. An epsilon here would hide exactly the defect worth
//!    catching: a half path that quietly computes something else.
//! 2. **Conjugate symmetry**, a property of the input rather than of any
//!    implementation — the discarded half must genuinely be recoverable.
//! 3. **Differential against RealFFT**, an independently authored real-FFT
//!    implementation whose public contract is this same `n/2 + 1` shape.
//! 4. **Allocation count**, because halving the output is half the point.

use eunomia::Complex64;
use realfft::RealFftPlanner;
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

fn tolerance(n: usize, input: &[f64]) -> f64 {
    let l1: f64 = input.iter().map(|v| v.abs()).sum();
    // Mixed-radix lengths have fewer, wider stages than log2(n) implies, so
    // using log2 keeps the bound conservative rather than tight.
    TOLERANCE_FACTOR * (n as f64).log2() * (f64::EPSILON / 2.0) * l1
}

#[test]
fn half_spectrum_is_bitwise_the_full_spectrum_truncated() {
    for n in SPLIT_SIZES.into_iter().chain(FALLBACK_SIZES) {
        let src = signal(n);
        let full = apollo_fft::fft_1d_slice::<f64>(&src);
        let half = apollo_fft::fft_1d_slice_half::<f64>(&src);

        assert_eq!(half.len(), n / 2 + 1, "N={n}: half spectrum length");
        assert_eq!(full.len(), n, "N={n}: full spectrum length");

        for (bin, (h, f)) in half.iter().zip(full.iter()).enumerate() {
            // Bitwise: the two share a code path, so any difference is a
            // defect rather than rounding.
            assert!(
                h.re.to_bits() == f.re.to_bits() && h.im.to_bits() == f.im.to_bits(),
                "N={n} bin {bin}: half {h:?} differs from full {f:?}"
            );
        }
    }
}

#[test]
fn the_discarded_half_is_recoverable_by_conjugate_symmetry() {
    for n in SPLIT_SIZES {
        let src = signal(n);
        let half = apollo_fft::fft_1d_slice_half::<f64>(&src);
        let full = apollo_fft::fft_1d_slice::<f64>(&src);

        // X[n-k] = conj(X[k]) is a property of a real input, so reconstructing
        // the upper half from the retained bins must reproduce the full
        // spectrum exactly.
        for k in 1..n / 2 {
            let mirrored = full[n - k];
            let from_half = half[k];
            assert!(
                (mirrored.re - from_half.re).abs() <= f64::EPSILON * from_half.re.abs().max(1.0)
                    && (mirrored.im + from_half.im).abs()
                        <= f64::EPSILON * from_half.im.abs().max(1.0),
                "N={n} bin {k}: X[n-k] is not conj(X[k]) — {mirrored:?} against {from_half:?}"
            );
        }
    }
}

#[test]
fn matches_realfft_bin_for_bin() {
    let mut planner = RealFftPlanner::<f64>::new();
    for n in SPLIT_SIZES {
        let src = signal(n);
        let ours = apollo_fft::fft_1d_slice_half::<f64>(&src);

        let r2c = planner.plan_fft_forward(n);
        let mut input = r2c.make_input_vec();
        input.copy_from_slice(&src);
        let mut theirs = r2c.make_output_vec();
        r2c.process(&mut input, &mut theirs)
            .expect("realfft length agrees with the plan");

        assert_eq!(ours.len(), theirs.len(), "N={n}: both engines return n/2+1");
        let bound = tolerance(n, &src);
        for (bin, (a, b)) in ours.iter().zip(theirs.iter()).enumerate() {
            let err = (a.re - b.re).hypot(a.im - b.im);
            assert!(
                err <= bound,
                "N={n} bin {bin}: {err:.3e} exceeds {bound:.3e} against RealFFT"
            );
        }
    }
}

#[test]
fn the_into_form_allocates_nothing() {
    for n in SPLIT_SIZES {
        let src = signal(n);
        let mut out = vec![Complex64::default(); n / 2 + 1];

        // One warm call first: plan, twiddles and scratch are cached on the
        // first use of a length, and those are not per-call costs.
        apollo_fft::fft_1d_slice_half_into::<f64>(&src, &mut out);

        let ((), observed) =
            count_allocations(|| apollo_fft::fft_1d_slice_half_into::<f64>(&src, &mut out));
        assert_eq!(
            observed, 0,
            "N={n}: the _into form allocated {observed} times; the caller owns the output"
        );
    }
}

#[test]
#[should_panic(expected = "exactly n/2 + 1 bins")]
fn a_wrong_output_length_is_rejected() {
    let src = signal(64);
    let mut out = vec![Complex64::default(); 64];
    apollo_fft::fft_1d_slice_half_into::<f64>(&src, &mut out);
}
