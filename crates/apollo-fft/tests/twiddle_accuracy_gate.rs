//! Error-growth gate on twiddle construction.
//!
//! ## What this catches that the rest of the suite does not
//!
//! A twiddle table advanced by a multiplicative recurrence — `w[j+1] = w[j]*w`
//! — accumulates phase drift, degrading the transform's forward error from
//! `O(log N · u)` to `O(N · u)` (Higham, *Accuracy and Stability of Numerical
//! Algorithms*, 2nd ed., section 24.1, whose `log N` result requires accurately
//! computed twiddles). That defect lived in three builders in this tree while
//! 403 tests passed, because every accuracy assertion in the suite is at a size,
//! or on a signal, where the drift is still under its tolerance.
//!
//! An absolute per-size error threshold cannot catch it: at `N = 256` the
//! recurrence's error is genuinely small. **Growth is the signal.** So the
//! assertion here is on the dimensionless ratio
//!
//! ```text
//! ratio(N) = max_k |X_k - X_k^exact| / (log2(N) · u · l1_norm(x))
//! ```
//!
//! which is `O(1)` for accurate twiddles and `O(N / log N)` for a recurrence —
//! about 341 at `N = 4096` against 32 at `N = 256`. Bounding that ratio by a
//! constant across a ladder spanning `2^3` to `2^18` *is* the non-growth
//! assertion.
//!
//! The `l1` norm is the scale in the cited bound (via `|X_k| <= l1_norm(x)`) and
//! is used in preference to `max|X_k|`, which for a spread spectrum can sit far
//! below the bound's scale and would inflate the ratio for reasons unrelated to
//! accuracy.
//!
//! ## Oracles
//!
//! 1. **Unit delta**, for the growth ladder. For `x[t] = [t == 1]` the spectrum
//!    is `X[k] = exp(-2πi k/N)`: the transform's output *is* the twiddle table,
//!    read out bin by bin, which makes this the most direct probe of the
//!    property being gated. The input is exactly representable, every `|X_k|` is
//!    1, and `l1_norm(x)` is 1, so the ratio needs no scale bookkeeping.
//!
//!    An earlier version of this test used a sum of tones at bins
//!    `0, N/4, N/2, 3N/4`, whose twiddles are `1, i, -1, -i`, to get an oracle
//!    with *zero* error on both sides. It measured a ratio of exactly zero at
//!    every size — and that is a defect, not a pass: the alignment making the
//!    construction exact also means every butterfly is either exact or applied
//!    to a zero, so the inexact twiddles never touch live data. It would have
//!    passed against any twiddle table whatsoever. Recorded because the trap is
//!    inviting: the property that makes such an oracle attractive is the same
//!    one that makes it blind.
//! 2. **Compensated direct DFT**, for path coverage at sizes with no closed-form
//!    oracle. Accumulated in double-double arithmetic (Knuth two-sum, Dekker
//!    two-product via `mul_add`).
//!
//! Both oracles evaluate `sin_cos` on an exactly reduced integer angle, so each
//! carries a half-ulp floor of about `u · l1_norm(x)` — a factor `log2 N` below
//! the quantity being measured. Stated rather than assumed, and it biases the
//! ratio slightly *up* at the small end of the ladder, which is the conservative
//! direction for a growth test.
//!
//! ## Sizes
//!
//! Read off the dispatch in `mixed_radix/dispatch.rs` and
//! `plan/fft/dimension_1d/dynamic_impl.rs` rather than assumed — the throwaway
//! probe that motivated this test initially mislabelled which sizes reach
//! `four_step_fft`. Sized ZST codelets claim `log2` 4 through 10 *before* the
//! four-step gate is consulted, so `N = 1024` is a codelet path, not four-step.

use apollo_fft::{FftPlan1D, Shape1D};
use eunomia::Complex64;
use std::f64::consts::TAU;

/// Unit roundoff.
const U: f64 = f64::EPSILON / 2.0;

/// Ceiling on `ratio(N)`. The textbook constant in the `O(log N · u)` bound is a
/// small single digit; 8 leaves margin for the differing stage counts of the
/// codelet, Stockham, and four-step routes while sitting two orders of magnitude
/// below the ratio a recurrence produces at the top of the ladder.
const RATIO_CEILING: f64 = 8.0;

/// Permitted drift in the ratio between the small and large ends of the ladder.
/// Accurate twiddles give a flat ratio; a recurrence grows it by `N / log N`,
/// four orders of magnitude across this range.
const GROWTH_ALLOWANCE: f64 = 4.0;

/// Floor for the small-size baseline in the growth comparison. The ratio's
/// expected magnitude is order one, so without a floor a fortuitously small
/// measurement at the bottom of the ladder would make the comparison spurious
/// rather than sensitive.
const BASELINE_FLOOR: f64 = 0.5;

/// Power-of-two ladder covering every route the power-of-two dispatch takes.
///
/// | sizes | route |
/// | --- | --- |
/// | 16, 64, 256, 1024 | sized ZST Stockham codelets (`log2` 4..=10) |
/// | 8, 2048, 8192, 32768 | Stockham autosort against a twiddle table |
/// | 4096, 16384 | four-step on the batched layout |
/// | 65536, 262144 | four-step with threaded row transforms |
const POT_LADDER: [usize; 12] = [
    8, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 262144,
];

/// Mixed-radix ladder: powers of three, which are 3-smooth and so reach the
/// composite builder in `radix_composite/cache.rs`, spanning a factor of 729 in
/// length so growth has room to show.
///
/// This ladder exists because the power-of-two one does not cover that builder,
/// and neither did the compensated-reference test: with the composite recurrence
/// deliberately reintroduced, both passed. A broad-spectrum signal lets twiddle
/// errors cancel in a random walk, which is how the original defect survived 403
/// tests. The impulse is the worst case rather than the average one, and that is
/// the point of it.
const COMPOSITE_LADDER: [usize; 6] = [81, 243, 729, 2187, 6561, 19683];

/// Primes whose Rader/Bluestein convolution buffer is a power of two with an
/// even `log2`, which is the only route by which a *one-dimensional* transform
/// reaches `four_step_fft` and, through it, the batched layout.
///
/// The 1-D power-of-two plan does not: it runs sized codelets to `N = 64` and
/// Stockham autosort above that, and the four-step gate lives in
/// `try_power_of_two_fast_path`, reached only from `dispatch_inplace` — 2-D and
/// 3-D lane transforms and these convolution buffers. Verified by instrumenting
/// `four_step_fft` and observing zero calls across the whole power-of-two
/// ladder, after reading the dispatch had twice suggested otherwise.
///
/// Buffers here are 4096, 16384, 65536 and 262144: the first two reach the
/// batched layout, the second two the four-step twiddle matrix, so between them
/// the two builders the power-of-two ladder cannot see are covered.
const CONVOLUTION_LADDER: [usize; 4] = [2039, 8191, 32749, 131071];

/// Position of the unit impulse. Bin `k` of the result is then `W_N^k`, so the
/// probe walks the twiddle table in unit steps — the index sequence along which
/// a recurrence's drift is largest.
const IMPULSE_AT: usize = 1;

/// `x[t] = [t == IMPULSE_AT]`, whose exact spectrum is `exp(-2πi k / N)`.
fn impulse_signal(n: usize) -> (Vec<Complex64>, Vec<Complex64>) {
    let mut signal = vec![Complex64::new(0.0, 0.0); n];
    signal[IMPULSE_AT] = Complex64::new(1.0, 0.0);

    let expected = (0..n)
        .map(|k| {
            // The index product is reduced modulo n first, so the angle is
            // formed from a small integer and no cancellation precedes sin_cos.
            let (sin, cos) = (-TAU * ((k * IMPULSE_AT) % n) as f64 / n as f64).sin_cos();
            Complex64::new(cos, sin)
        })
        .collect();
    (signal, expected)
}

fn l1_norm(x: &[Complex64]) -> f64 {
    x.iter().map(|v| v.re.hypot(v.im)).sum()
}

fn max_deviation(computed: &[Complex64], expected: &[Complex64]) -> f64 {
    computed
        .iter()
        .zip(expected)
        .map(|(a, b)| (a.re - b.re).hypot(a.im - b.im))
        .fold(0.0f64, f64::max)
}

/// `ratio(N)` for one size, using the impulse oracle.
///
/// `log2(N)` stands in for the stage count at every length: it is exact for the
/// power-of-two routes and an upper bound for mixed-radix factorizations, whose
/// stages are fewer and wider. Overstating the stage count understates the
/// ratio, which is the conservative direction here.
fn normalized_error(n: usize) -> f64 {
    let (signal, expected) = impulse_signal(n);
    let mut data = signal.clone();
    FftPlan1D::<f64>::new(Shape1D { n }).forward_complex_slice_inplace(&mut data);

    let stages = (n as f64).log2();
    max_deviation(&data, &expected) / (stages * U * l1_norm(&signal))
}

/// Asserts the ratio is bounded and does not grow across `sizes`, which must be
/// ordered ascending and span enough range for growth to be visible.
fn assert_ratio_is_flat(route: &str, sizes: &[usize]) {
    let ratios: Vec<(usize, f64)> = sizes.iter().map(|&n| (n, normalized_error(n))).collect();

    for &(n, ratio) in &ratios {
        println!("{route}: N={n:<8} ratio = {ratio:.3}");
    }

    for &(n, ratio) in &ratios {
        assert!(
            ratio <= RATIO_CEILING,
            "{route}, N={n}: normalized error {ratio:.3} exceeds {RATIO_CEILING}. This \
             is the O(N·u) signature of a twiddle recurrence, not a tolerance to \
             widen — the twiddles for this size are being advanced rather than \
             evaluated."
        );
    }

    // Growth is the discriminating signal, so it is asserted directly rather
    // than left implied by the ceiling.
    let split = sizes[sizes.len() / 2];
    let extreme = |keep: fn(usize, usize) -> bool| {
        ratios
            .iter()
            .filter(|&&(n, _)| keep(n, split))
            .map(|&(_, r)| r)
            .fold(0.0f64, f64::max)
    };
    let small = extreme(|n, split| n < split);
    let large = extreme(|n, split| n >= split);
    let baseline = small.max(BASELINE_FLOOR);
    assert!(
        large <= GROWTH_ALLOWANCE * baseline,
        "{route}: normalized error grows with size — {large:.3} at N >= {split} \
         against {small:.3} below it. Accurate twiddles give a flat ratio; growth \
         means error is accumulating per element rather than per stage."
    );
}

#[test]
fn normalized_error_does_not_grow_across_the_power_of_two_ladder() {
    assert_ratio_is_flat("power-of-two", &POT_LADDER);
}

#[test]
fn normalized_error_does_not_grow_across_the_mixed_radix_ladder() {
    assert_ratio_is_flat("mixed-radix", &COMPOSITE_LADDER);
}

#[test]
fn normalized_error_does_not_grow_across_the_convolution_ladder() {
    assert_ratio_is_flat("convolution", &CONVOLUTION_LADDER);
}

// ---------------------------------------------------------------------------
// Compensated reference, for the sizes with no exact oracle.
// ---------------------------------------------------------------------------

/// Knuth's two-sum: exact for any inputs, with no ordering assumption.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// Dekker's two-product, expressed with a fused multiply-add so the error term
/// comes out in one rounding rather than by splitting.
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

/// Double-double accumulator: the running value is `hi + lo`, with `lo` holding
/// the rounding residue, giving roughly twice working precision.
#[derive(Clone, Copy, Default)]
struct DoubleDouble {
    hi: f64,
    lo: f64,
}

impl DoubleDouble {
    fn add(&mut self, value: f64) {
        let (s, e) = two_sum(self.hi, value);
        self.lo += e;
        let (hi, lo) = two_sum(s, self.lo);
        self.hi = hi;
        self.lo = lo;
    }

    /// Accumulates `a * b` without first rounding the product away.
    fn add_product(&mut self, a: f64, b: f64) {
        let (p, e) = two_prod(a, b);
        self.add(p);
        self.add(e);
    }

    fn value(self) -> f64 {
        self.hi + self.lo
    }
}

/// Direct DFT accumulated in double-double arithmetic.
///
/// The index product is reduced modulo `n` before scaling, so the angle is
/// formed from a small integer and `sin_cos` receives an argument free of
/// cancellation. `sin_cos`'s own half-ulp is the reference's error floor.
fn compensated_dft(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    (0..n)
        .map(|k| {
            let (mut re, mut im) = (DoubleDouble::default(), DoubleDouble::default());
            for (t, v) in input.iter().enumerate() {
                let (sin, cos) = (-TAU * ((k * t) % n) as f64 / n as f64).sin_cos();
                re.add_product(v.re, cos);
                re.add_product(-v.im, sin);
                im.add_product(v.re, sin);
                im.add_product(v.im, cos);
            }
            Complex64::new(re.value(), im.value())
        })
        .collect()
}

/// Deterministic dyadic samples: exactly representable, so the input carries no
/// error of its own, and broad-spectrum, so no bin is accidentally spared.
fn dyadic_signal(n: usize) -> Vec<Complex64> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // 24 significant bits over a power of two: exact in f64.
        f64::from(((state >> 40) as u32) as i32 - (1 << 23)) / f64::from(1u32 << 23)
    };
    (0..n).map(|_| Complex64::new(next(), next())).collect()
}

/// Sizes reaching the non-power-of-two routes, plus the small power-of-two
/// codelets, all within the `O(N^2)` reference's reach.
///
/// | sizes | route |
/// | --- | --- |
/// | 36, 48, 90, 144, 180, 198, 200, 385 | composite mixed radix |
/// | 72, 511 | Good-Thomas |
/// | 127, 251 | Rader |
/// | 512, 1024, 2048 | power-of-two codelet and Stockham |
const PATH_SIZES: [usize; 15] = [
    36, 48, 72, 90, 127, 144, 180, 198, 200, 251, 385, 511, 512, 1024, 2048,
];

#[test]
fn every_dispatch_path_matches_a_compensated_reference() {
    for n in PATH_SIZES {
        let signal = dyadic_signal(n);
        let expected = compensated_dft(&signal);
        let mut data = signal.clone();
        FftPlan1D::<f64>::new(Shape1D { n }).forward_complex_slice_inplace(&mut data);

        // A mixed-radix length's stage count is bounded by log2(N); using that
        // rather than the actual radix sequence keeps the bound conservative for
        // the paths whose factorization is chosen at plan time.
        let stages = (n as f64).log2();
        let bound = RATIO_CEILING * stages * U * l1_norm(&signal);
        let err = max_deviation(&data, &expected);
        assert!(
            err <= bound,
            "N={n}: error {err:.3e} exceeds {bound:.3e} against a double-double \
             reference. Derive the cause; do not widen the bound."
        );
    }
}
