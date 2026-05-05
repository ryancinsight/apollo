//! Mixed-radix Cooley-Tukey DIT FFT for 5-smooth composite lengths.
//!
//! ## Mathematical foundation
//!
//! Theorem (Cooley-Tukey, 1965): For N = r·m, the N-point DFT can be
//! computed from `r` DFTs of length `m` plus N twiddle multiplications and
//! one `r`-point DFT per group, yielding O(N log N) total operations when
//! the factorization is applied recursively.
//!
//! ## Algorithm — iterative DIT mixed-radix
//!
//! Given factorization N = r₀ · r₁ · … · r_{L-1} (innermost radix r₀ first):
//!
//! 1. **Digit-reversal permutation**: reinterpret each index i as a mixed-radix
//!    number in base [r₀, r₁, …, r_{L-1}] and reverse the digit order.  This
//!    reorders the input so that after all butterfly stages the output is in
//!    natural (in-order) frequency-bin layout.
//!
//! 2. **Butterfly stages** (innermost to outermost, s = 0 … L-1):
//!    - Current sub-transform length: `stage_len = r₀ · … · rₛ`
//!    - Previous sub-transform length: `prev_len = stage_len / rₛ`
//!    - For each chunk of `stage_len` contiguous elements and each offset
//!      `j = 0 … prev_len-1`:
//!      a. Gather r_s elements spaced `prev_len` apart:
//!         `x[k] = chunk[j + k · prev_len]` for k = 0…rₛ-1
//!      b. Apply inter-stage twiddles `W_{stage_len}^{j·k}` for k = 1…rₛ-1
//!         (k = 0 is always W⁰ = 1, skipped).
//!      c. Apply DFT-rₛ butterfly (in-place on the gathered array).
//!      d. Scatter results back: `chunk[j + k · prev_len] = x[k]`.
//!
//! ## Twiddle computation
//!
//! For each (stage_len, j) pair, the base twiddle W = exp(±2πi·j/stage_len)
//! is computed once.  Powers W^{j·k} = W^j · (W^j)^{k-1} are then computed
//! by successive multiplication (one complex multiply per additional twiddle
//! order), requiring only one trigonometric evaluation per (stage_len, j) pair.
//!
//! ## Supported radix set
//!
//! {2, 3, 4, 5, 8}.  N must have no prime factor outside {2, 3, 5}; sizes
//! with other prime factors fall back to Bluestein chirp-Z.
//!
//! ## Performance
//!
//! The first correct implementation intentionally avoids precomputed twiddle
//! tables to keep the code auditable.  One `sin`/`cos` evaluation per
//! (stage, j) replaces the O(N log N) trig evaluations in Bluestein; the
//! resulting algorithm is typically 10–50× faster than Bluestein for
//! N = 2^a · 3^b · 5^c (e.g., 100, 300, 1000, 10000).  A stage-twiddle
//! cache can be layered on top without changing the correctness invariants.
//!
//! ## References
//!
//! - Cooley, J.W. & Tukey, J.W. (1965). An algorithm for the machine
//!   calculation of complex Fourier series. *Math. Comp.* 19, 297–301.
//! - Blahut, R.E. (2010). *Fast Algorithms for Signal Processing*. Cambridge.

use num_complex::{Complex32, Complex64};

use super::radix_permute::digit_reverse_permute_mixed;
use super::radix_shape::factorize_composite;
use super::radix_stage::normalize_inplace;
use super::winograd::{
    apply_twiddle_32, apply_twiddle_64, dft2_32, dft2_64, dft3_32, dft3_64, dft4_32, dft4_64,
    dft5_32, dft5_64, dft5_32_simd, dft5_64_simd, dft8_32, dft8_64,
};

// ── inner butterfly dispatchers ───────────────────────────────────────────────

/// Apply DFT-r butterfly in-place on `data[..r]` (f64).
///
/// # Preconditions
/// `r` must be in {2, 3, 4, 5, 8} and `data.len() >= r`.
#[inline(always)]
fn apply_dft_r_64(data: &mut [Complex64], r: usize, inverse: bool) {
    match r {
        2 => {
            let (lo, hi) = data.split_at_mut(1);
            dft2_64(&mut lo[0], &mut hi[0]);
        }
        3 => {
            let mut b: [Complex64; 3] = data[..3].try_into().unwrap();
            dft3_64(&mut b, inverse);
            data[..3].copy_from_slice(&b);
        }
        4 => {
            let mut b: [Complex64; 4] = data[..4].try_into().unwrap();
            dft4_64(&mut b, inverse);
            data[..4].copy_from_slice(&b);
        }
        5 => {
            let mut b: [Complex64; 5] = data[..5].try_into().unwrap();
            dft5_64_simd(&mut b, inverse);
            data[..5].copy_from_slice(&b);
        }
        8 => {
            let mut b: [Complex64; 8] = data[..8].try_into().unwrap();
            dft8_64(&mut b, inverse);
            data[..8].copy_from_slice(&b);
        }
        _ => unreachable!("unsupported radix {r}"),
    }
}

/// Apply DFT-r butterfly in-place on `data[..r]` (f32).
#[inline(always)]
fn apply_dft_r_32(data: &mut [Complex32], r: usize, inverse: bool) {
    match r {
        2 => {
            let (lo, hi) = data.split_at_mut(1);
            dft2_32(&mut lo[0], &mut hi[0]);
        }
        3 => {
            let mut b: [Complex32; 3] = data[..3].try_into().unwrap();
            dft3_32(&mut b, inverse);
            data[..3].copy_from_slice(&b);
        }
        4 => {
            let mut b: [Complex32; 4] = data[..4].try_into().unwrap();
            dft4_32(&mut b, inverse);
            data[..4].copy_from_slice(&b);
        }
        5 => {
            let mut b: [Complex32; 5] = data[..5].try_into().unwrap();
            dft5_32_simd(&mut b, inverse);
            data[..5].copy_from_slice(&b);
        }
        8 => {
            let mut b: [Complex32; 8] = data[..8].try_into().unwrap();
            dft8_32(&mut b, inverse);
            data[..8].copy_from_slice(&b);
        }
        _ => unreachable!("unsupported radix {r}"),
    }
}

// ── public inplace kernels ────────────────────────────────────────────────────

/// In-place forward FFT (unnormalized) for 5-smooth composite N (f64).
///
/// # Panics
/// Debug-panics if `N` is not 5-smooth.  In release, undefined output.
pub fn forward_inplace_64(data: &mut [Complex64]) {
    composite_core_64(data, false);
}

/// In-place forward FFT (unnormalized) for 5-smooth composite N (f64)
/// with precomputed radices.
///
/// `radices` must be ordered innermost-first and equal to `N` when multiplied.
#[inline]
pub fn forward_inplace_64_with_radices(data: &mut [Complex64], radices: &[usize]) {
    composite_core_64_with_radices(data, false, radices);
}

/// In-place inverse FFT (unnormalized) for 5-smooth composite N (f64).
pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    composite_core_64(data, true);
}

/// In-place inverse FFT (unnormalized) for 5-smooth composite N (f64)
/// with precomputed radices.
///
/// `radices` must be ordered innermost-first and equal to `N` when multiplied.
#[inline]
pub fn inverse_inplace_unnorm_64_with_radices(data: &mut [Complex64], radices: &[usize]) {
    composite_core_64_with_radices(data, true, radices);
}

/// In-place inverse FFT normalized by 1/N for 5-smooth composite N (f64).
pub fn inverse_inplace_64(data: &mut [Complex64]) {
    composite_core_64(data, true);
    normalize_inplace(data, 1.0 / data.len() as f64);
}

/// In-place inverse FFT normalized by 1/N for 5-smooth composite N (f64)
/// with precomputed radices.
///
/// `radices` must be ordered innermost-first and equal to `N` when multiplied.
#[inline]
pub fn inverse_inplace_64_with_radices(data: &mut [Complex64], radices: &[usize]) {
    composite_core_64_with_radices(data, true, radices);
    normalize_inplace(data, 1.0 / data.len() as f64);
}

/// In-place forward FFT (unnormalized) for 5-smooth composite N (f32).
pub fn forward_inplace_32(data: &mut [Complex32]) {
    composite_core_32(data, false);
}

/// In-place forward FFT (unnormalized) for 5-smooth composite N (f32)
/// with precomputed radices.
///
/// `radices` must be ordered innermost-first and equal to `N` when multiplied.
#[inline]
pub fn forward_inplace_32_with_radices(data: &mut [Complex32], radices: &[usize]) {
    composite_core_32_with_radices(data, false, radices);
}

/// In-place inverse FFT (unnormalized) for 5-smooth composite N (f32).
pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
    composite_core_32(data, true);
}

/// In-place inverse FFT (unnormalized) for 5-smooth composite N (f32)
/// with precomputed radices.
///
/// `radices` must be ordered innermost-first and equal to `N` when multiplied.
#[inline]
pub fn inverse_inplace_unnorm_32_with_radices(data: &mut [Complex32], radices: &[usize]) {
    composite_core_32_with_radices(data, true, radices);
}

/// In-place inverse FFT normalized by 1/N for 5-smooth composite N (f32).
pub fn inverse_inplace_32(data: &mut [Complex32]) {
    composite_core_32(data, true);
}

/// In-place inverse FFT normalized by 1/N for 5-smooth composite N (f32)
/// with precomputed radices.
///
/// `radices` must be ordered innermost-first and equal to `N` when multiplied.
#[inline]
pub fn inverse_inplace_32_with_radices(data: &mut [Complex32], radices: &[usize]) {
    composite_core_32_with_radices(data, true, radices);
    normalize_inplace(data, 1.0 / data.len() as f32);
}

// ── iterative DIT core (f64) ──────────────────────────────────────────────────

fn composite_core_64(data: &mut [Complex64], inverse: bool) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let radices = factorize_composite(n)
        .expect("composite_core_64: N has prime factors outside {2,3,5}");
    composite_core_64_with_radices(data, inverse, &radices);
}

fn composite_core_64_with_radices(data: &mut [Complex64], inverse: bool, radices: &[usize]) {
    let n = data.len();
    if n <= 1 || radices.is_empty() {
        return;
    }
    debug_assert_eq!(radices.iter().product::<usize>(), n);
    debug_assert!(radices.iter().all(|r| [2usize, 3, 4, 5, 8].contains(r)));

    // Step 1: mixed-radix digit-reversal permutation.
    digit_reverse_permute_mixed(data, &radices);

    // twiddle exponent sign: −1 for forward (exp(−2πi·…)), +1 for inverse.
    let sign: f64 = if inverse { 1.0 } else { -1.0 };

    // Step 2: precompute twiddles for ALL stages upfront.
    // For stage s, twiddle[j] = exp(sign·2πi·j / stage_len).
    // Flatten into a single buffer with stage offsets for cache efficiency.
    let mut all_twiddles = Vec::new();
    let mut stage_offsets = Vec::new();
    let mut prev_len = 1usize;
    for &r in radices {
        let stage_len = prev_len * r;
        stage_offsets.push(all_twiddles.len());
        for j in 0..prev_len {
            if j == 0 {
                all_twiddles.push(Complex64::new(1.0, 0.0));
            } else {
                let angle = sign * std::f64::consts::TAU * j as f64 / stage_len as f64;
                all_twiddles.push(Complex64::new(angle.cos(), angle.sin()));
            }
        }
        prev_len = stage_len;
    }

    // Step 3: iterative butterfly stages (innermost first), using precomputed twiddles.
    let mut prev_len = 1usize;
    let mut stage_idx = 0usize;
    let mut buf = [Complex64::default(); 8];

    for &r in radices {
        let stage_len = prev_len * r;
        let stage_twiddles = &all_twiddles[stage_offsets[stage_idx]..stage_offsets[stage_idx] + prev_len];

        for chunk in data.chunks_mut(stage_len) {
            // j = 0: twiddle is always 1, just apply DFT-r.
            for k in 0..r {
                buf[k] = chunk[k * prev_len];
            }
            apply_dft_r_64(&mut buf[..r], r, inverse);
            for k in 0..r {
                chunk[k * prev_len] = buf[k];
            }

            // j = 1..prev_len: use precomputed base twiddle, build powers by multiplication.
            for j in 1..prev_len {
                let base_tw = stage_twiddles[j]; // W_{stage_len}^j

                buf[0] = chunk[j];
                let mut tw_k = base_tw;
                for k in 1..r {
                    buf[k] = apply_twiddle_64(chunk[j + k * prev_len], tw_k);
                    if k + 1 < r {
                        tw_k = apply_twiddle_64(tw_k, base_tw);
                    }
                }
                apply_dft_r_64(&mut buf[..r], r, inverse);
                for k in 0..r {
                    chunk[j + k * prev_len] = buf[k];
                }
            }
        }

        prev_len = stage_len;
        stage_idx += 1;
    }
}

// ── iterative DIT core (f32) ──────────────────────────────────────────────────

fn composite_core_32(data: &mut [Complex32], inverse: bool) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let radices = factorize_composite(n)
        .expect("composite_core_32: N has prime factors outside {2,3,5}");
    composite_core_32_with_radices(data, inverse, &radices);
}

fn composite_core_32_with_radices(data: &mut [Complex32], inverse: bool, radices: &[usize]) {
    let n = data.len();
    if n <= 1 || radices.is_empty() {
        return;
    }
    debug_assert_eq!(radices.iter().product::<usize>(), n);
    debug_assert!(radices.iter().all(|r| [2usize, 3, 4, 5, 8].contains(r)));

    digit_reverse_permute_mixed(data, &radices);

    let sign: f32 = if inverse { 1.0 } else { -1.0 };

    // Precompute ALL stage twiddles upfront.
    let mut all_twiddles = Vec::new();
    let mut stage_offsets = Vec::new();
    let mut prev_len = 1usize;
    for &r in radices {
        let stage_len = prev_len * r;
        stage_offsets.push(all_twiddles.len());
        for j in 0..prev_len {
            if j == 0 {
                all_twiddles.push(Complex32::new(1.0, 0.0));
            } else {
                let angle = sign * std::f32::consts::TAU * j as f32 / stage_len as f32;
                all_twiddles.push(Complex32::new(angle.cos(), angle.sin()));
            }
        }
        prev_len = stage_len;
    }

    // Iterative butterfly stages using precomputed twiddles.
    let mut prev_len = 1usize;
    let mut stage_idx = 0usize;
    let mut buf = [Complex32::default(); 8];

    for &r in radices {
        let stage_len = prev_len * r;
        let stage_twiddles = &all_twiddles[stage_offsets[stage_idx]..stage_offsets[stage_idx] + prev_len];

        for chunk in data.chunks_mut(stage_len) {
            // j = 0: trivial twiddles.
            for k in 0..r {
                buf[k] = chunk[k * prev_len];
            }
            apply_dft_r_32(&mut buf[..r], r, inverse);
            for k in 0..r {
                chunk[k * prev_len] = buf[k];
            }

            for j in 1..prev_len {
                let base_tw = stage_twiddles[j];

                buf[0] = chunk[j];
                let mut tw_k = base_tw;
                for k in 1..r {
                    buf[k] = apply_twiddle_32(chunk[j + k * prev_len], tw_k);
                    if k + 1 < r {
                        tw_k = apply_twiddle_32(tw_k, base_tw);
                    }
                }
                apply_dft_r_32(&mut buf[..r], r, inverse);
                for k in 0..r {
                    chunk[j + k * prev_len] = buf[k];
                }
            }
        }

        prev_len = stage_len;
        stage_idx += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward_64, dft_inverse_64};

    fn max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0f64, f64::max)
    }

    // ── factorize_composite ───────────────────────────────────────────────────

    #[test]
    fn factorize_5smooth_sizes() {
        // N = 2^a * 3^b * 5^c (not pure-PoT) → Some(radices) with product = N.
        for &n in &[3usize, 5, 6, 9, 10, 12, 15, 18, 24, 25, 48, 50, 75, 100, 120,
                    125, 150, 192, 240, 250, 300, 375, 384, 500, 600, 768, 1000,
                    1200, 1500, 2000, 3000, 4500, 5000, 6000, 7500, 10000] {
            let result = factorize_composite(n);
            assert!(result.is_some(), "factorize_composite({n}) returned None");
            let radices = result.unwrap();
            assert_eq!(
                radices.iter().product::<usize>(),
                n,
                "factorize_composite({n}): product mismatch"
            );
            for &r in &radices {
                assert!(
                    [2, 3, 4, 5, 8].contains(&r),
                    "factorize_composite({n}): unsupported radix {r}"
                );
            }
        }
    }

    #[test]
    fn factorize_pow2_returns_none() {
        // Pure powers of two must return None (handled by pow2_dispatch!).
        for exp in 1..=20u32 {
            let n = 1usize << exp;
            assert!(
                factorize_composite(n).is_none(),
                "factorize_composite({n}) should be None for pure-PoT"
            );
        }
    }

    #[test]
    fn factorize_non_smooth_returns_none() {
        for &n in &[7usize, 11, 13, 14, 21, 22, 35, 49, 77, 91, 100 * 7] {
            assert!(
                factorize_composite(n).is_none(),
                "factorize_composite({n}) should be None (has prime > 5)"
            );
        }
    }

    // ── forward + roundtrip correctness ──────────────────────────────────────

    /// Test forward FFT against direct O(N²) reference for a given N.
    fn check_forward(n: usize, tol: f64) {
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.37).sin(), (k as f64 * 0.19).cos()))
            .collect();
        let expected = dft_forward_64(&input);
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let err = max_err(&got, &expected);
        assert!(err < tol, "forward N={n}: max_err={err:.2e} (tol={tol:.2e})");
    }

    /// Test inverse roundtrip (fwd → inv / N = id) for a given N.
    fn check_roundtrip(n: usize, tol: f64) {
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.53).cos(), (k as f64 * 0.27).sin()))
            .collect();
        let mut buf = input.clone();
        forward_inplace_64(&mut buf);
        inverse_inplace_unnorm_64(&mut buf);
        let recovered: Vec<Complex64> = buf.iter().map(|x| x / n as f64).collect();
        let err = max_err(&recovered, &input);
        assert!(err < tol, "roundtrip N={n}: max_err={err:.2e} (tol={tol:.2e})");
    }

    /// Test inverse against direct reference.
    fn check_inverse(n: usize, tol: f64) {
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.61).cos(), (k as f64 * 0.43).sin()))
            .collect();
        let expected_unnorm: Vec<Complex64> = dft_inverse_64(&input)
            .into_iter()
            .map(|x| x * n as f64)
            .collect();
        let mut got = input.clone();
        inverse_inplace_unnorm_64(&mut got);
        let err = max_err(&got, &expected_unnorm);
        assert!(err < tol, "inverse N={n}: max_err={err:.2e} (tol={tol:.2e})");
    }

    // Pure-prime base cases.
    #[test]
    fn forward_n3()   { check_forward(3,   1e-13); }
    #[test]
    fn forward_n5()   { check_forward(5,   1e-13); }
    #[test]
    fn forward_n9()   { check_forward(9,   1e-12); }
    #[test]
    fn forward_n15()  { check_forward(15,  1e-12); }
    #[test]
    fn forward_n25()  { check_forward(25,  1e-12); }
    #[test]
    fn forward_n6()   { check_forward(6,   1e-13); }
    #[test]
    fn forward_n10()  { check_forward(10,  1e-12); }

    // Mixed 2^a × 5^b sizes (benchmark targets).
    #[test]
    fn forward_n100()  { check_forward(100,  1e-11); }
    #[test]
    fn forward_n1000() { check_forward(1000, 1e-9);  }
    #[test]
    fn forward_n10000(){ check_forward(10000,1e-8);  }

    // 3-smooth non-PoT sizes.
    #[test]
    fn forward_n12()  { check_forward(12,  1e-13); }
    #[test]
    fn forward_n24()  { check_forward(24,  1e-12); }
    #[test]
    fn forward_n48()  { check_forward(48,  1e-12); }
    #[test]
    fn forward_n192() { check_forward(192, 1e-11); }
    #[test]
    fn forward_n384() { check_forward(384, 1e-10); }

    // Roundtrip tests.
    #[test]
    fn roundtrip_n100()  { check_roundtrip(100,  1e-12); }
    #[test]
    fn roundtrip_n1000() { check_roundtrip(1000, 1e-11); }
    #[test]
    fn roundtrip_n10000(){ check_roundtrip(10000,1e-10); }

    // Inverse against reference.
    #[test]
    fn inverse_n100()    { check_inverse(100,   1e-11); }
    #[test]
    fn inverse_n1000()   { check_inverse(1000,  1e-10); }

    // DC input: all ones → only bin 0 nonzero.
    #[test]
    fn forward_dc_n100() {
        let mut buf = vec![Complex64::new(1.0, 0.0); 100];
        forward_inplace_64(&mut buf);
        assert!((buf[0] - Complex64::new(100.0, 0.0)).norm() < 1e-10);
        for x in &buf[1..] {
            assert!(x.norm() < 1e-10, "non-zero bin: {:?}", x);
        }
    }

    #[test]
    fn forward_dc_n1000() {
        let mut buf = vec![Complex64::new(1.0, 0.0); 1000];
        forward_inplace_64(&mut buf);
        assert!((buf[0] - Complex64::new(1000.0, 0.0)).norm() < 1e-9);
        for x in &buf[1..] {
            assert!(x.norm() < 1e-9, "non-zero bin: {:?}", x);
        }
    }

    // f32 forward correctness.
    #[test]
    fn forward_f32_n100_matches_f64_reference() {
        let input: Vec<Complex64> = (0..100usize)
            .map(|k| Complex64::new((k as f64 * 0.29).sin(), (k as f64 * 0.47).cos()))
            .collect();
        let expected = dft_forward_64(&input);
        let mut buf: Vec<Complex32> = input
            .iter()
            .map(|x| Complex32::new(x.re as f32, x.im as f32))
            .collect();
        forward_inplace_32(&mut buf);
        let got: Vec<Complex64> = buf
            .iter()
            .map(|x| Complex64::new(x.re as f64, x.im as f64))
            .collect();
        let err = max_err(&got, &expected);
        assert!(err < 1e-4, "f32 forward N=100 max_err={err:.2e}");
    }

    #[test]
    fn forward_f32_n1000_matches_f64_reference() {
        let input: Vec<Complex64> = (0..1000usize)
            .map(|k| Complex64::new((k as f64 * 0.13).sin(), (k as f64 * 0.31).cos()))
            .collect();
        let expected = dft_forward_64(&input);
        let mut buf: Vec<Complex32> = input
            .iter()
            .map(|x| Complex32::new(x.re as f32, x.im as f32))
            .collect();
        forward_inplace_32(&mut buf);
        let got: Vec<Complex64> = buf
            .iter()
            .map(|x| Complex64::new(x.re as f64, x.im as f64))
            .collect();
        let err = max_err(&got, &expected);
        assert!(err < 2e-3, "f32 forward N=1000 max_err={err:.2e}");
    }
}
