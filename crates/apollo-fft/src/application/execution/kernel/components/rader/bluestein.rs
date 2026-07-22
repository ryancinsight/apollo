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
//! `BluesteinStore for f32` and `BluesteinStore for f64` impls and their
//! storage live in `mixed_radix/scalar/bluestein_cache.rs`; this file owns only
//! the Rader-domain cache orchestration and kernel construction.
//!
//! # Performance rationale
//!
//! For large primes (N ≥ 2051, M ≥ 2050), the Bluestein path replaces a
//! deep recursive Rader chain (e.g. N=10007 → HalfCyclic 5003 → Rader 5003
//! → HalfCyclic 2501 → PFA 41×61) with two Stockham power-of-two FFTs.
//! Apollo's Stockham kernel at N=32768 is heavily optimized (pair+3×triple+quad
//! fused stages), making this a large net win.

use super::generator::CanonicalRaderGeneratorInverse;
use crate::application::execution::kernel::mixed_radix::scalar::{
    BluesteinEntry, BluesteinKey, BluesteinStore,
};
use crate::application::execution::kernel::mixed_radix::{dispatch_inplace, MixedRadixScalar};
use eunomia::Complex;
use std::sync::Arc;

// ── Cache fetch ────────────────────────────────────────────────────────────

/// Fetch or build the Bluestein cache entry (kernel FFT).
///
/// The two-level cache (thread-local first, then global `RwLock`) follows the
/// same pattern as all other Apollo FFT caches. The global write lock is
/// acquired only on a miss after confirming the miss under the read lock,
/// preventing self-deadlock.
fn cached_bluestein_entry<F, const INVERSE: bool>(
    n: usize,
    generator_inverse: CanonicalRaderGeneratorInverse,
) -> BluesteinEntry<F::Complex>
where
    F: MixedRadixScalar<Complex = Complex<F>> + BluesteinStore<Cpx = Complex<F>>,
{
    let m = n - 1;
    let key: BluesteinKey = (m, INVERSE);

    if let Some(v) = F::tl_get(key) {
        return v;
    }

    let cached = F::global().read().get(&key).cloned();
    if let Some(v) = cached {
        F::tl_insert(key, v.clone());
        return v;
    }

    let entry = build_bluestein_entry::<F, INVERSE>(n, m, generator_inverse.get());
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
fn build_bluestein_entry<F, const INVERSE: bool>(
    n: usize,
    m: usize,
    generator_inverse: usize,
) -> BluesteinEntry<F::Complex>
where
    F: MixedRadixScalar<Complex = Complex<F>>,
{
    let p = if F::BLUESTEIN_PAD_POWER_OF_TWO {
        (2 * m - 1).next_power_of_two()
    } else {
        next_7_smooth(2 * m - 1)
    };
    debug_assert!(is_7_smooth(p));

    // Build Rader time-domain kernel, zero-padded to P.
    let sign = if INVERSE { 1.0_f64 } else { -1.0_f64 };
    let mut cur = 1usize;
    // Use native precision for phase trig where possible (f32 path uses f32::cos/sin for perf;
    // no excess widen-narrow in arithmetic per persona -- f64 only for f64 path or accuracy contract
    // of bluestein chirp phase; cast only at F::complex API boundary (once per sample, not bulk inner math).
    // For f32 rader bluestein (hot for worst md ratios like 67/271 f32), this avoids f64 trig cost.
    // Mem eff: build kernel using pooled bluestein scratch (TL aligned, reuses across plans,
    // zero extra alloc/growth for temp during cache populate for rader). Final to_vec for Arc
    // is the cached storage (unavoidable), but temp uses pool (improves mem efficiency for
    // rader plans, esp f32 worst primes).
    let kernel_vec = F::with_bluestein_scratch(p, |buf| {
        buf[..p].fill(F::complex(0.0, 0.0));
        for j in 0..m {
            let a = sign * std::f64::consts::TAU * (cur as f64) / (n as f64);
            let (re, im) = if F::BLUESTEIN_NATIVE_PHASE_TRIG {
                let af = a as f32;
                (af.cos() as f64, af.sin() as f64)
            } else {
                (a.cos(), a.sin())
            };
            buf[j] = F::complex(re, im);
            cur = (cur * generator_inverse) % n;
        }

        // Forward FFT (unnormalized) of the zero-padded kernel.
        // For pow2 p (common for f32 rader bluestein pads via next_power_of_two), use direct
        // stockham_forward_sized (with_bluestein_scratch + tw) to hit avx _sized sub paths (f32/f64).
        // This advances "full f32 avx/pot sub with_scratch" (unblocks n113/257 f32 bias, mono for
        // kernel cache build on worst md rader primes, mem reuse of TL pool, zero extra for composite p).
        // For non-pow2 7-smooth p, dispatch (avoids Rader re-entry). Use exact p (buf may be larger due align).
        if p.is_power_of_two() && p >= 64 {
            F::with_twiddle_fwd(p, |tw| {
                <F as MixedRadixScalar>::with_scratch(p, |scratch| {
                    // Direct sized for pow2 p in f32 kernel build (avx_with_scratch_sized sub for f32 rader bluestein pads;
                    // unblocks n113/257 f32, ensures mono/ZST for cache pop, reuses with_bluestein_scratch TL pool).
                    // Matches convolve style; cascade for hot pads, general sized for others.
                    if p == 64 {
                        F::stockham_forward_sized::<6>(buf, scratch, tw);
                    } else if p == 128 {
                        F::stockham_forward_sized::<7>(buf, scratch, tw);
                    } else if p == 256 {
                        F::stockham_forward_sized::<8>(buf, scratch, tw);
                    } else if p == 512 {
                        F::stockham_forward_sized::<9>(buf, scratch, tw);
                    } else if p == 1024 {
                        F::stockham_forward_sized::<10>(buf, scratch, tw);
                    } else if p == 2048 {
                        F::stockham_forward_sized::<11>(buf, scratch, tw);
                    } else {
                        F::stockham_forward(buf, scratch, tw);
                    }
                });
            });
        } else {
            dispatch_inplace::<F, false, false>(&mut buf[..p], None);
        }

        buf[..p].to_vec()
    });
    Arc::from(kernel_vec)
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
/// precomputed cache entry keyed by `(m, inverse)`. The canonical-generator
/// type guarantees one generator for each prime length.
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
    const INVERSE: bool,
>(
    padded: &mut [F::Complex],
    n: usize,
    generator_inverse: CanonicalRaderGeneratorInverse,
) {
    let m = padded.len();
    debug_assert_eq!(m, n - 1);

    let kernel_fft = cached_bluestein_entry::<F, INVERSE>(n, generator_inverse);
    let p = kernel_fft.len();
    debug_assert!(is_7_smooth(p));
    debug_assert!(p >= 2 * m - 1);

    // ── 1-4. Zero-pad → Forward FFT → pointwise → Inverse FFT ────────────
    F::with_bluestein_scratch(p, |data_buf| {
        data_buf[..m].copy_from_slice(&padded[..m]);
        data_buf[m..p].fill(F::complex(0.0, 0.0));

        // Forward FFT (unnormalized) - direct stockham for pow2 pads (common for f32 rader bluestein, zero-cost)
        if p.is_power_of_two() && p >= 64 {
            F::with_twiddle_fwd(p, |tw| {
                <F as MixedRadixScalar>::with_scratch(p, |scratch| {
                    if p == 64 {
                        F::stockham_forward_sized::<6>(data_buf, scratch, tw);
                    } else if p == 128 {
                        // Direct const LOG2 for 128 pad (f32 rader bluestein common; mono + unroll benefit for PoT128 in pads, mem pool reuse).
                        F::stockham_forward_sized::<7>(data_buf, scratch, tw);
                    } else if p == 256 {
                        F::stockham_forward_sized::<8>(data_buf, scratch, tw);
                    } else if p == 512 {
                        F::stockham_forward_sized::<9>(data_buf, scratch, tw);
                    } else if p == 1024 {
                        // Direct const LOG2=10 for p=1024 f32 pads (e.g. rader 271 m=270); mono + unroll benefit for PoT in pads, mem pool.
                        F::stockham_forward_sized::<10>(data_buf, scratch, tw);
                    } else if p == 2048 {
                        F::stockham_forward_sized::<11>(data_buf, scratch, tw);
                    } else {
                        F::stockham_forward(data_buf, scratch, tw);
                    }
                });
            });
        } else {
            dispatch_inplace::<F, false, false>(data_buf, None);
        }

        // Pointwise multiply with precomputed kernel FFT.
        F::pointwise_mul(data_buf, kernel_fft.as_ref());

        // Inverse FFT (unnormalized)
        if p.is_power_of_two() && p >= 64 {
            F::with_twiddle_inv(p, |tw| {
                <F as MixedRadixScalar>::with_scratch(p, |scratch| {
                    if p == 64 {
                        F::stockham_forward_sized::<6>(data_buf, scratch, tw);
                    } else if p == 128 {
                        // Direct const LOG2 for 128 pad (f32 rader bluestein; mono + unroll for PoT128, mem).
                        F::stockham_forward_sized::<7>(data_buf, scratch, tw);
                    } else if p == 256 {
                        F::stockham_forward_sized::<8>(data_buf, scratch, tw);
                    } else if p == 512 {
                        F::stockham_forward_sized::<9>(data_buf, scratch, tw);
                    } else if p == 1024 {
                        // Direct const LOG2=10 for p=1024 f32 pads (e.g. rader 271 m=270); mono + unroll benefit for PoT in pads, mem pool.
                        F::stockham_forward_sized::<10>(data_buf, scratch, tw);
                    } else if p == 2048 {
                        F::stockham_forward_sized::<11>(data_buf, scratch, tw);
                    } else {
                        F::stockham_forward(data_buf, scratch, tw);
                    }
                });
            });
        } else {
            dispatch_inplace::<F, true, false>(data_buf, None);
        }

        // Reborrow as shared reference for fold reads (zero-copy over pooled scratch).
        let data_buf: &[F::Complex] = data_buf;

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
    use eunomia::Complex64;

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

        super::super::rader_fft_with_convolution_backend::<f64, false, super::super::Bluestein>(
            &mut got,
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

        super::super::rader_fft_with_convolution_backend::<f64, false, super::super::Bluestein>(
            &mut data,
        );
        super::super::rader_fft_with_convolution_backend::<f64, true, super::super::Bluestein>(
            &mut data,
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
