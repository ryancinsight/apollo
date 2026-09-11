//! Twiddle-table construction and real-FFT half-complex split routines.
//!
//! ## Current role
//!
//! This module no longer contains a DIT execution kernel. The radix-2 iterative
//! DIT butterfly engine was retired in favour of the Stockham autosort kernel
//! (`mixed_radix.rs`) which requires no bit-reversal permutation pass and
//! delivers better throughput via cache-friendly ping-pong buffering.
//!
//! This module owns two related scalar-FFT concerns:
//!
//! - `RealFft` delegates contiguous per-stage twiddle-table construction to
//!   the single implementation in `twiddle_table.rs` for Stockham and the
//!   multidimensional plan axes.
//! - The real-input split helpers pack an even real signal into a half-length
//!   complex transform, untangle its independent bins in place, and mirror the
//!   conjugate half only when a full spectrum is requested.
//!
//! ## Twiddle-table mathematical contract
//!
//! Theorem (Unified Twiddle Table): A single (N-1)-entry contiguous table
//! with per-stage layout suffices for all log2(N) Stockham stages.
//!
//! Layout invariant: for stage s with sub-transform length L = 2^s,
//! table[base..base+L/2] holds W_L^j = exp(-2*pi*i*j/L) for j = 0..L/2-1,
//! where base = L/2 - 1 (sum of all shorter stage lengths). This lets
//! the Stockham kernel read twiddles sequentially with no stride. QED.
//!
//! ## Failure modes
//!
//! - Empty slice: returns immediately (N=0).
//! - N=1: returns immediately (trivial transform).
//! - N not a power of 2: triggers `debug_assert!` in debug builds.

/// Kernel-level twiddle-table trait consumed by the active twiddle caches.
pub(crate) trait RealFft:
    crate::application::execution::kernel::mixed_radix::MixedRadixScalar
{
    fn build_forward_twiddle_table(n: usize) -> Vec<Self::Complex>;
    fn build_inverse_twiddle_table(n: usize) -> Vec<Self::Complex>;
}

impl RealFft for f64 {
    #[inline]
    fn build_forward_twiddle_table(n: usize) -> Vec<eunomia::Complex64> {
        super::twiddle_table::build_twiddle_table(n, -1.0)
    }

    #[inline]
    fn build_inverse_twiddle_table(n: usize) -> Vec<eunomia::Complex64> {
        super::twiddle_table::build_twiddle_table(n, 1.0)
    }
}

impl RealFft for f32 {
    #[inline]
    fn build_forward_twiddle_table(n: usize) -> Vec<eunomia::Complex32> {
        super::twiddle_table::build_twiddle_table(n, -1.0)
    }

    #[inline]
    fn build_inverse_twiddle_table(n: usize) -> Vec<eunomia::Complex32> {
        super::twiddle_table::build_twiddle_table(n, 1.0)
    }
}

// ── Real-to-complex half-complex split ────────────────────────────────────────

/// Untangles a packed half-length transform into the `N/2 + 1` real-input bins.
///
/// ## Why this exists
///
/// A real signal of length `N` has a conjugate-symmetric spectrum: `X[N-k] =
/// conj(X[k])`, so only `N/2 + 1` bins carry information. Running a full
/// size-`N` complex transform on real input widened with a zero imaginary part
/// computes the redundant half as well, at roughly twice the arithmetic.
///
/// The caller packs the `N` reals as `M = N/2` complex samples
/// `z[k] = x[2k] + i·x[2k+1]`, transforms those with a size-`M` complex FFT
/// into `out[..M]`, and calls this to untangle in place. Writing `a = Z[k]` and
/// `b = Z[M-k]`:
///
/// ```text
/// Fe[k] = (a + conj(b)) / 2          spectrum of the even-indexed samples
/// Fo[k] = (a - conj(b)) / (2i)       spectrum of the odd-indexed samples
/// X[k]  = Fe[k] + W_N^k · Fo[k],     W_N^k = exp(-2πi k / N)
/// ```
///
/// The paired bin follows from `Fe[M-k] = conj(Fe[k])`, `Fo[M-k] =
/// conj(Fo[k])`, and `W_N^{M-k} = -conj(W_N^k)`, which give
/// `X[M-k] = conj(Fe[k] - W_N^k · Fo[k])`. Both bins come from one twiddle
/// multiply, so the loop runs to `M/2` rather than `M`.
///
/// The purely real bins are the special cases: `X[0] = Z[0].re + Z[0].im` and
/// `X[M] = Z[0].re - Z[0].im`. When `M` is even the midpoint is
/// `X[M/2] = conj(Z[M/2])`, since there `a = b` and `W_N^{M/2} = -i`.
///
/// ## Allocation
///
/// None, and the untangle is in place: `out[..M]` arrives holding `Z` and
/// leaves holding `X[..M]`, with `X[M]` written to the spare slot.
///
/// ## Panics
///
/// Panics if `out.len() < n / 2 + 1` or if `n` is not even.
pub(crate) fn untangle_real_half<T>(
    out: &mut [eunomia::Complex<T>],
    n: usize,
    twiddles: impl IntoIterator<Item = eunomia::Complex<T>>,
) where
    T: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<T>,
    >,
{
    assert!(
        n % 2 == 0,
        "real transform requires an even length, got {n}"
    );
    let m = n / 2;
    assert!(
        out.len() > m,
        "real spectrum needs n/2 + 1 slots, got {}",
        out.len()
    );
    if m == 0 {
        return;
    }

    let two = T::from_precise(2.0);
    let zero = T::from_precise(0.0);
    let z0 = out[0];
    out[0] = eunomia::Complex::new(z0.re + z0.im, zero);
    out[m] = eunomia::Complex::new(z0.re - z0.im, zero);

    let mut twiddles = twiddles.into_iter();
    for k in 1..m.div_ceil(2) {
        let w = twiddles
            .next()
            .expect("invariant: the split twiddles cover every bin pair below N/4");
        let (wr, wi) = (w.re, w.im);
        let a = out[k];
        let b = out[m - k];

        let fe_re = (a.re + b.re) / two;
        let fe_im = (a.im - b.im) / two;
        let fo_re = (a.im + b.im) / two;
        let fo_im = (b.re - a.re) / two;

        let t_re = fo_re * wr - fo_im * wi;
        let t_im = fo_re * wi + fo_im * wr;

        out[k] = eunomia::Complex::new(fe_re + t_re, fe_im + t_im);
        out[m - k] = eunomia::Complex::new(fe_re - t_re, t_im - fe_im);
    }

    if m % 2 == 0 && m >= 2 {
        let mid = out[m / 2];
        out[m / 2] = eunomia::Complex::new(mid.re, -mid.im);
    }
}

/// Retangles the `N/2 + 1` real-input bins into the packed half-length
/// spectrum, in place: the inverse of [`untangle_real_half`].
///
/// With `M = N/2` and `W = W_N`, the untangle wrote `X[k] = Fe[k] + W^k·Fo[k]`
/// and `X[M-k] = conj(Fe[k] - W^k·Fo[k])`. Adding and subtracting the conjugate
/// of the paired bin recovers both halves, since `|W^k| = 1`:
///
/// ```text
/// Fe[k] = (X[k] + conj(X[M-k])) / 2
/// Fo[k] = (X[k] - conj(X[M-k])) · conj(W^k) / 2
/// Z[k]  = Fe[k] + i·Fo[k]
/// ```
///
/// and a size-`M` inverse of `Z`, normalized by `1/M`, yields
/// `z[j] = x[2j] + i·x[2j+1]`. The paired bin needs no second twiddle:
/// `Fe[M-k] = conj(Fe[k])` and `Fo[M-k] = conj(Fo[k])`, so
/// `Z[M-k] = conj(Fe[k]) + i·conj(Fo[k])`.
///
/// The ends mirror the untangle's: `X[0]` and `X[M]` are taken as real and give
/// `Z[0] = (X[0] + X[M])/2 + i·(X[0] - X[M])/2`, and when `M` is even the
/// midpoint is `Z[M/2] = conj(X[M/2])`. Their imaginary parts, which the
/// spectrum of a real signal does not have, are ignored.
///
/// ## Allocation
///
/// None: `bins[..=M]` arrives holding `X` and `bins[..M]` leaves holding `Z`.
/// The twiddle advances by the untangle's block-restarted recurrence, so the
/// two directions carry the same rounding.
///
/// ## Panics
///
/// Panics if `bins.len() < n / 2 + 1` or if `n` is odd.
pub(crate) fn retangle_real_half<T>(
    bins: &mut [eunomia::Complex<T>],
    n: usize,
    twiddles: impl IntoIterator<Item = eunomia::Complex<T>>,
) where
    T: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<T>,
    >,
{
    assert!(
        n % 2 == 0,
        "real transform requires an even length, got {n}"
    );
    let m = n / 2;
    assert!(
        bins.len() > m,
        "real spectrum needs n/2 + 1 slots, got {}",
        bins.len()
    );
    if m == 0 {
        return;
    }

    let two = T::from_precise(2.0);
    let (first, last) = (bins[0].re, bins[m].re);
    bins[0] = eunomia::Complex::new((first + last) / two, (first - last) / two);

    let mut twiddles = twiddles.into_iter();
    for k in 1..m.div_ceil(2) {
        let w = twiddles
            .next()
            .expect("invariant: the split twiddles cover every bin pair below N/4");
        let (wr, wi) = (w.re, w.im);
        let a = bins[k];
        let b = bins[m - k];

        let fe_re = (a.re + b.re) / two;
        let fe_im = (a.im - b.im) / two;
        // (a - conj(b)) · conj(w) / 2
        let d_re = a.re - b.re;
        let d_im = a.im + b.im;
        let fo_re = (d_re * wr + d_im * wi) / two;
        let fo_im = (d_im * wr - d_re * wi) / two;

        bins[k] = eunomia::Complex::new(fe_re - fo_im, fe_im + fo_re);
        bins[m - k] = eunomia::Complex::new(fe_re + fo_im, fo_re - fe_im);
    }

    if m % 2 == 0 && m >= 2 {
        let mid = bins[m / 2];
        bins[m / 2] = eunomia::Complex::new(mid.re, -mid.im);
    }
}

/// The split twiddles `W_N^k` for `k = 1..⌈M/2⌉` (`M = N/2`), by the
/// block-restarted recurrence [`untangle_real_half`] and
/// [`retangle_real_half`] consume.
///
/// Each eight-bin block is seeded directly and advanced in native precision, so
/// the recurrence error is bounded by eight steps rather than `N`, and every
/// transcendental evaluation is amortized over its block. That suits one split
/// of a length; many splits of one length — the z lanes of a 3-D half-spectrum
/// transform — take [`split_twiddle_table`], which evaluates each twiddle once
/// for all of them.
pub(crate) fn split_twiddles<T>(n: usize) -> impl Iterator<Item = eunomia::Complex<T>>
where
    T: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<T>,
    >,
{
    const TWIDDLE_RESTART_INTERVAL: usize = 8;
    let limit = (n / 2).div_ceil(2);
    let scale = -core::f64::consts::TAU / n as f64;
    let (step_sin, step_cos) = scale.sin_cos();
    let (step_re, step_im) = (T::from_precise(step_cos), T::from_precise(step_sin));
    let mut w = eunomia::Complex::new(T::from_precise(1.0), T::from_precise(0.0));
    (1..limit).map(move |k| {
        if (k - 1) % TWIDDLE_RESTART_INTERVAL == 0 {
            let (sin, cos) = (scale * k as f64).sin_cos();
            w = eunomia::Complex::new(T::from_precise(cos), T::from_precise(sin));
        }
        let current = w;
        w = eunomia::Complex::new(
            w.re * step_re - w.im * step_im,
            w.re * step_im + w.im * step_re,
        );
        current
    })
}

/// The split twiddles for length `n`, each evaluated directly: `⌈N/4⌉ - 1`
/// entries, the same `W_N^k` [`split_twiddles`] yields.
///
/// A plan that runs many real splits of one length keeps this, so the
/// transcendental evaluations happen once per plan rather than once per split.
pub(crate) fn split_twiddle_table<T>(n: usize) -> Box<[eunomia::Complex<T>]>
where
    T: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<T>,
    >,
{
    let limit = (n / 2).div_ceil(2);
    let scale = -core::f64::consts::TAU / n as f64;
    (1..limit)
        .map(|k| {
            let (sin, cos) = (scale * k as f64).sin_cos();
            eunomia::Complex::new(T::from_precise(cos), T::from_precise(sin))
        })
        .collect()
}

/// Mirrors the `N/2 + 1` independent bins over the upper half, in place.
///
/// `X[N-k] = conj(X[k])` for a real input signal, so the upper half is a
/// reflection and costs a copy rather than a transform. `full[..=N/2]` must
/// already hold the independent bins.
///
/// ## Panics
///
/// Panics if `full.len()` is odd.
pub(crate) fn mirror_half_spectrum_in_place<T>(full: &mut [eunomia::Complex<T>])
where
    T: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<T>,
    >,
{
    let n = full.len();
    assert!(n % 2 == 0, "full spectrum length must be even, got {n}");
    for k in 1..n / 2 {
        let v = full[k];
        full[n - k] = eunomia::Complex::new(v.re, -v.im);
    }
}

#[cfg(test)]
mod tests;
