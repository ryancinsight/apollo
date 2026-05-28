//! Bluestein chirp-Z convolution for large Rader prime lengths.
//!
//! # Algorithm
//!
//! For a convolution length `M = N - 1` (N prime), the circular convolution
//! `Y = X ⊛ K` is computed via the standard convolution theorem with
//! zero-padding:
//!
//! 1. **Zero-pad** `X` and `K` to length `P = next_pow2(2M - 1)`.
//! 2. **Forward FFT** of padded `X` and `K` (via Apollo Stockham, unnormalized).
//! 3. **Pointwise multiply** `FFT(X) · FFT(K)`.
//! 4. **Inverse FFT** (via Apollo Stockham with inverse twiddles, unnormalized).
//! 5. **Fold**: `Y[j] = d[j] + d[j+M]` for `j = 0..M-2`, `Y[M-1] = d[M-1]`.
//! 6. **Scale by `1/P`** matching the existing Rader convolution convention.
//!
//! The precomputed cache stores `FFT(K_padded)` so each convolve call only
//! pays for one forward FFT, one pointwise multiply, and one inverse FFT.
//!
//! # Cache ownership
//!
//! `BluesteinStore for f32` and `BluesteinStore for f64` impls live in
//! `mixed_radix/scalar/impls.rs` alongside `MixedRadixScalar` impls.
//! All global statics and thread-locals reside there; this file holds no
//! precision-specific cache state.
//!
//! # Performance rationale
//!
//! For large primes (N ≥ 2051, M ≥ 2050), the Bluestein path replaces a
//! deep recursive Rader chain (e.g. N=10007 → HalfCyclic 5003 → Rader 5003
//! → HalfCyclic 2501 → PFA 41×61) with two Stockham power-of-two FFTs.
//! Apollo's Stockham kernel at N=32768 is heavily optimized (pair+3×triple+quad
//! fused stages), making this a large net win.

use crate::application::execution::kernel::mixed_radix::scalar::{
    BluesteinEntry, BluesteinKey, BluesteinStore,
};
use crate::application::execution::kernel::mixed_radix::{dispatch_inplace, MixedRadixScalar};
use num_complex::Complex;
use std::sync::Arc;

/// Minimum Rader convolution length `M = N - 1` that triggers the Bluestein
/// strategy instead of the half-cyclic CRT split.
///
/// For N=10007, M=10006 → uses Bluestein with P=32768.
/// For N=521,   M=520   → uses existing FullCyclic path.
// pub(crate) const BLUESTEIN_RADER_THRESHOLD: usize = 2048;

// ── Cache fetch ────────────────────────────────────────────────────────────

/// Fetch or build the Bluestein cache entry (kernel FFT).
///
/// The two-level cache (thread-local first, then global `RwLock`) follows the
/// same pattern as all other Apollo FFT caches. The global write lock is
/// acquired only on a miss after confirming the miss under the read lock,
/// preventing self-deadlock.
fn cached_bluestein_entry<F>(
    n: usize,
    inverse: bool,
    generator_inverse: usize,
) -> BluesteinEntry<F::Complex>
where
    F: MixedRadixScalar<Complex = Complex<F>> + BluesteinStore<Cpx = Complex<F>>,
{
    let m = n - 1;
    let key: BluesteinKey = (m, inverse, generator_inverse);

    if let Some(v) = F::tl_get(key) {
        return v;
    }

    let cached = F::global().read().get(&key).cloned();
    if let Some(v) = cached {
        F::tl_insert(key, v.clone());
        return v;
    }

    let entry = build_bluestein_entry::<F>(n, m, inverse, generator_inverse);
    let v = {
        let mut write_guard = F::global().write();
        write_guard.entry(key).or_insert(entry).clone()
    };
    F::tl_insert(key, v.clone());
    v
}

// ── Build cache entry ──────────────────────────────────────────────────────

/// Build a fresh Bluestein cache entry: the forward FFT of the zero-padded
/// Rader kernel.
///
/// Steps:
/// 1. Rader time-domain kernel: `K[j] = exp(sign · 2πi · g_inv^j / N)`.
/// 2. Zero-pad to `P = next_pow2(2M - 1)`.
/// 3. Forward Stockham FFT (unnormalized) → `kernel_fft`.
fn build_bluestein_entry<F>(
    n: usize,
    m: usize,
    inverse: bool,
    generator_inverse: usize,
) -> BluesteinEntry<F::Complex>
where
    F: MixedRadixScalar<Complex = Complex<F>>,
{
    let p = next_7_smooth(2 * m - 1);
    debug_assert!(is_7_smooth(p));

    // Build Rader time-domain kernel, zero-padded to P.
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut cur = 1usize;
    let mut kernel_padded = vec![F::complex(0.0, 0.0); p];
    for j in 0..m {
        let a = sign * std::f64::consts::TAU * (cur as f64) / (n as f64);
        kernel_padded[j] = F::complex(a.cos(), a.sin());
        cur = (cur * generator_inverse) % n;
    }

    // Forward FFT (unnormalized) of the zero-padded kernel.
    // P is a 7-smooth composite number, so routing via dispatch_inplace
    // avoids Rader re-entry.
    dispatch_inplace::<F, false, false>(&mut kernel_padded, None);

    Arc::from(kernel_padded)
}

// ── Utilities ──────────────────────────────────────────────────────────────

#[inline]
fn is_7_smooth(mut n: usize) -> bool {
    if n == 0 {
        return false;
    }
    while n % 2 == 0 {
        n /= 2;
    }
    while n % 3 == 0 {
        n /= 3;
    }
    while n % 5 == 0 {
        n /= 5;
    }
    while n % 7 == 0 {
        n /= 7;
    }
    n == 1
}

#[inline]
fn next_7_smooth(min_val: usize) -> usize {
    let mut candidate = min_val;
    while !is_7_smooth(candidate) {
        candidate += 1;
    }
    candidate
}

// ── Bluestein convolution ──────────────────────────────────────────────────

/// In-place Bluestein circular convolution of length `padded.len() = N - 1`.
///
/// Replaces the deep recursive Rader ↔ HalfCyclic chain for large primes
/// with the standard convolution theorem: zero-pad → FFT → pointwise → IFFT
/// → fold → scale.
///
/// # Contract
///
/// `padded` holds the permuted Rader input sequence on entry and the
/// convolution result on exit. The Rader kernel FFT is accessed from the
/// precomputed cache entry keyed by `(m, inverse, generator_inverse)`.
///
/// # Normalization
///
/// The output is scaled by `1/P` where `P = next_pow2(2M - 1)`, matching
/// the existing convention: forward FFT (unnormalized) → pointwise →
/// inverse FFT (unnormalized) → divide by P. This is compatible with the
/// existing Rader scatter path which adds unscaled `x0` to each result.
///
/// Marked `#[inline(never)]` to keep the Stockham FFT frame independent
/// of the Rader dispatch.
#[inline(never)]
pub(super) fn rader_bluestein_convolve_inplace<
    F: MixedRadixScalar<Complex = Complex<F>> + BluesteinStore<Cpx = Complex<F>>,
>(
    padded: &mut [F::Complex],
    n: usize,
    inverse: bool,
    generator_inverse: usize,
) {
    let m = padded.len();
    debug_assert_eq!(m, n - 1);

    let kernel_fft = cached_bluestein_entry::<F>(n, inverse, generator_inverse);
    let p = kernel_fft.len();
    debug_assert!(is_7_smooth(p));
    debug_assert!(p >= 2 * m - 1);

    // ── 1-4. Zero-pad → Forward FFT → pointwise → Inverse FFT ────────────
    F::with_bluestein_scratch(p, |data_buf| {
        data_buf[..m].copy_from_slice(&padded[..m]);
        data_buf[m..p].fill(F::complex(0.0, 0.0));

        // Forward FFT (unnormalized)
        dispatch_inplace::<F, false, false>(data_buf, None);

        // Pointwise multiply with precomputed kernel FFT
        F::pointwise_mul(data_buf, kernel_fft.as_ref());

        // Inverse FFT (unnormalized)
        dispatch_inplace::<F, true, false>(data_buf, None);

        // ── 5. Fold tail into head for circular convolution ─────────────
        // After two unnormalized FFTs and pointwise multiply:
        //   data_buf = P · (X_padded ⊛_lin K_padded)
        // The circular convolution of length M is obtained by folding the tail:
        //   Y[j] = d[j] + d[j+M]   for j = 0..M-2
        //   Y[M-1] = d[M-1]
        // Raw-field writes avoid Complex::new() overhead in the hot fold loop.
        let inv_p = 1.0 / (p as f64);
        // Scale factor in the correct scalar precision: F::complex(re, im).re gives F from f64.
        let scale = F::complex(inv_p, 0.0).re;
        let last = m - 1;
        let m4 = (last / 4) * 4;
        let mut j = 0usize;
        while j < m4 {
            unsafe {
                let d0 = data_buf.get_unchecked(j);
                let d1 = data_buf.get_unchecked(j + 1);
                let d2 = data_buf.get_unchecked(j + 2);
                let d3 = data_buf.get_unchecked(j + 3);
                let t0 = data_buf.get_unchecked(j + m);
                let t1 = data_buf.get_unchecked(j + m + 1);
                let t2 = data_buf.get_unchecked(j + m + 2);
                let t3 = data_buf.get_unchecked(j + m + 3);
                padded.get_unchecked_mut(j).re = (d0.re + t0.re) * scale;
                padded.get_unchecked_mut(j).im = (d0.im + t0.im) * scale;
                padded.get_unchecked_mut(j + 1).re = (d1.re + t1.re) * scale;
                padded.get_unchecked_mut(j + 1).im = (d1.im + t1.im) * scale;
                padded.get_unchecked_mut(j + 2).re = (d2.re + t2.re) * scale;
                padded.get_unchecked_mut(j + 2).im = (d2.im + t2.im) * scale;
                padded.get_unchecked_mut(j + 3).re = (d3.re + t3.re) * scale;
                padded.get_unchecked_mut(j + 3).im = (d3.im + t3.im) * scale;
            }
            j += 4;
        }
        while j < last {
            unsafe {
                let d = data_buf.get_unchecked(j);
                let t = data_buf.get_unchecked(j + m);
                padded.get_unchecked_mut(j).re = (d.re + t.re) * scale;
                padded.get_unchecked_mut(j).im = (d.im + t.im) * scale;
            }
            j += 1;
        }
        unsafe {
            let d_last = data_buf.get_unchecked(last);
            padded.get_unchecked_mut(last).re = d_last.re * scale;
            padded.get_unchecked_mut(last).im = d_last.im * scale;
        }
    });
}

#[cfg(test)]
mod tests {
    use crate::application::execution::kernel::direct::dft_forward;
    use num_complex::Complex64;

    fn signal(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|k| {
                let t = k as f64;
                Complex64::new((0.27 * t).sin(), (0.35 * t).cos())
            })
            .collect()
    }

    fn max_abs_err(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max)
    }

    /// Verify that forced Bluestein convolution matches the direct transform on
    /// a bounded prime. The production threshold is a routing decision, not a
    /// mathematical precondition of the convolution theorem.
    #[test]
    fn bluestein_matches_direct_for_n17() {
        let n = 17usize;
        let input = signal(n);
        let expected = dft_forward(&input);
        let mut got = input;

        super::super::rader_fft_with_convolution_strategy::<f64, false>(
            &mut got,
            super::super::RaderConvolutionStrategy::Bluestein,
        );

        let err = max_abs_err(&got, &expected);
        assert!(err < 1e-8, "Bluestein N=17 forward max_err={err:.2e}");
    }

    /// Verify unnormalized inverse composition for forced Bluestein on the same
    /// bounded prime.
    #[test]
    fn bluestein_roundtrip_n17() {
        let n = 17usize;
        let input = signal(n);
        let mut data = input.clone();

        super::super::rader_fft_with_convolution_strategy::<f64, false>(
            &mut data,
            super::super::RaderConvolutionStrategy::Bluestein,
        );
        super::super::rader_fft_with_convolution_strategy::<f64, true>(
            &mut data,
            super::super::RaderConvolutionStrategy::Bluestein,
        );
        for x in &mut data {
            *x /= n as f64;
        }

        let err = max_abs_err(&data, &input);
        assert!(err < 1e-8, "Bluestein N=17 roundtrip max_err={err:.2e}");
    }

    #[test]
    fn test_is_7_smooth() {
        use super::{is_7_smooth, next_7_smooth};
        assert!(is_7_smooth(1));
        assert!(is_7_smooth(2));
        assert!(is_7_smooth(3));
        assert!(is_7_smooth(4));
        assert!(is_7_smooth(5));
        assert!(is_7_smooth(6));
        assert!(is_7_smooth(7));
        assert!(is_7_smooth(8));
        assert!(is_7_smooth(9));
        assert!(is_7_smooth(10));
        assert!(!is_7_smooth(11));
        assert!(!is_7_smooth(13));
        assert!(is_7_smooth(20160));
        assert_eq!(next_7_smooth(20013), 20160);
    }
}
