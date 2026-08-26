//! Twiddle-table construction and real-FFT half-complex split routines.
//!
//! ## Current role
//!
//! This module no longer contains a DIT execution kernel. The radix-2 iterative
//! DIT butterfly engine was retired in favour of the Stockham autosort kernel
//! (`mixed_radix.rs`) which requires no bit-reversal permutation pass and
//! delivers better throughput via cache-friendly ping-pong buffering.
//!
//! The functions remaining here are twiddle-table builders
//! (`build_forward_twiddle_table_{32,64}`, `build_inverse_twiddle_table_{32,64}`).
//! They construct contiguous per-stage twiddle tables used by the Stockham
//! kernel and by the 2-D / 3-D plan axes. All four delegate to the SSOT in
//! `twiddle_table.rs`.
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
pub(crate) fn untangle_real_half<T>(out: &mut [eunomia::Complex<T>], n: usize)
where
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

    let scale = -core::f64::consts::TAU / n as f64;
    for k in 1..m.div_ceil(2) {
        let a = out[k];
        let b = out[m - k];

        let fe_re = (a.re + b.re) / two;
        let fe_im = (a.im - b.im) / two;
        let fo_re = (a.im + b.im) / two;
        let fo_im = (b.re - a.re) / two;

        // Direct evaluation per entry, matching the twiddle-accuracy contract
        // in this module: a recurrence here would reintroduce O(N·u) error.
        let (sin, cos) = (scale * k as f64).sin_cos();
        let (wr, wi) = (T::from_precise(cos), T::from_precise(sin));
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
