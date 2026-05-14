//! Bluestein chirp-Z FFT for arbitrary-length DFT.
//!
//! ## Mathematical derivation
//!
//! Using the Bluestein identity `kn = [-(k-n)² + k² + n²] / 2`:
//! ```text
//! X[k] = chirp[k] * (a_M ⊛ b_M)[k]
//! ```
//! where `chirp[k] = exp(-πi k²/N)`, `a_M[n] = x[n] * chirp[n]` (zero-padded to M),
//! `b_M` is the filter `exp(+πi m²/N)` arranged for circular convolution,
//! and `M = next_pow2(2N-1)`.
//!
//! ## Module layout
//!
//! ```text
//! bluestein/
//!   mod.rs       — public API: forward/inverse in-place + re-exports
//!   scalar.rs    — BluesteinScalar sealed trait + Complex64/32 impls
//!   pointwise.rs — generic zero_fill, fill_and_mul, mul_pointwise kernels
//!   avx_f64.rs   — x86_64 AVX/FMA intrinsics for Complex64
//!   avx_f32.rs   — x86_64 AVX/FMA intrinsics for Complex32
//!   plan.rs      — BluesteinPlan64/32, two-level caching, scratch accessors
//! ```

#![allow(clippy::uninit_vec)]
#![allow(clippy::empty_line_after_doc_comments)]

pub(super) mod avx_f32;
pub(super) mod avx_f64;
pub(crate) mod plan;
pub(crate) mod pointwise;
pub(crate) mod scalar;

pub(crate) use plan::BluesteinPlan64;
pub(crate) use pointwise::zero_fill;

use super::mixed_radix;
use super::radix_stage::{normalize_inplace_c32, normalize_inplace_c64};
use num_complex::{Complex32, Complex64};
use plan::{cached_plan32, cached_plan64, with_scratch32, with_scratch64};
use std::sync::OnceLock;

// ── AVX capability probe ──────────────────────────────────────────────────────

#[inline]
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_avx_fma() -> bool {
    static HAS_AVX_FMA: OnceLock<bool> = OnceLock::new();
    *HAS_AVX_FMA.get_or_init(|| {
        std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma")
    })
}

#[inline]
#[cfg(not(target_arch = "x86_64"))]
pub(crate) const fn has_avx_fma() -> bool {
    false
}

// ── Public transform API ──────────────────────────────────────────────────────

/// In-place forward Bluestein chirp-Z transform for `Complex64`.
///
/// Computes the exact discrete Fourier transform in O(N log N) time for arbitrary
/// and non-power-of-two lengths.  Power-of-two lengths delegate to the radix-2 kernel.
pub fn forward_inplace_64(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        mixed_radix::forward_inplace_64(data);
        return;
    }
    let plan = cached_plan64(n);
    with_scratch64(plan.m(), |a_m| plan.forward_with_scratch(data, a_m));
}

/// In-place unnormalized inverse Bluestein chirp-Z transform for `Complex64`.
///
/// Computes the adjoint DFT without the `1/N` scaling factor.
pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        mixed_radix::inverse_inplace_unnorm_64(data);
        return;
    }
    let plan = cached_plan64(n);
    with_scratch64(plan.m(), |a_m| plan.inverse_unnorm_with_scratch(data, a_m));
}

/// In-place normalized inverse Bluestein chirp-Z transform for `Complex64`.
pub fn inverse_inplace_64(data: &mut [Complex64]) {
    inverse_inplace_unnorm_64(data);
    normalize_inplace_c64(data, 1.0 / data.len() as f64);
}

/// In-place forward Bluestein chirp-Z transform for `Complex32`.
///
/// Evaluates arbitrary-length DFT in-place on native `Complex32` with a cached
/// plan and reusable scratch.
pub fn forward_inplace_32(data: &mut [Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        mixed_radix::forward_inplace_32(data);
        return;
    }
    let plan = cached_plan32(n);
    with_scratch32(plan.m(), |a_m| plan.forward_with_scratch(data, a_m));
}

/// In-place unnormalized inverse Bluestein chirp-Z transform for `Complex32`.
pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        mixed_radix::inverse_inplace_unnorm_32(data);
        return;
    }
    let plan = cached_plan32(n);
    with_scratch32(plan.m(), |a_m| plan.inverse_unnorm_with_scratch(data, a_m));
}

/// In-place normalized inverse Bluestein chirp-Z transform for `Complex32`.
pub fn inverse_inplace_32(data: &mut [Complex32]) {
    inverse_inplace_unnorm_32(data);
    normalize_inplace_c32(data, 1.0f32 / data.len() as f32);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::test_utils::max_abs_err_64 as max_abs_err;
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};

    fn sig(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|k| {
                let t = k as f64 / n as f64;
                Complex64::new(
                    (std::f64::consts::TAU * 3.0 * t).sin()
                        + 0.5 * (std::f64::consts::TAU * 7.0 * t).cos(),
                    0.25 * (std::f64::consts::TAU * 2.0 * t).sin(),
                )
            })
            .collect()
    }

    #[test]
    fn forward_matches_direct_for_n3() {
        let input = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let expected = dft_forward(&input);
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        assert!(max_abs_err(&got, &expected) < 1e-12);
    }

    #[test]
    fn forward_matches_direct_for_n5() {
        let input = sig(5);
        let expected = dft_forward(&input);
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        assert!(
            max_abs_err(&got, &expected) < 1e-12,
            "n=5 err={}",
            max_abs_err(&got, &expected)
        );
    }

    #[test]
    fn forward_matches_direct_for_n6() {
        let input = sig(6);
        let expected = dft_forward(&input);
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        assert!(
            max_abs_err(&got, &expected) < 1e-12,
            "n=6 err={}",
            max_abs_err(&got, &expected)
        );
    }

    #[test]
    fn forward_matches_direct_for_n7() {
        let input = sig(7);
        let expected = dft_forward(&input);
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        assert!(
            max_abs_err(&got, &expected) < 1e-12,
            "n=7 err={}",
            max_abs_err(&got, &expected)
        );
    }

    #[test]
    fn forward_matches_direct_for_n11() {
        let input = sig(11);
        let expected = dft_forward(&input);
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        assert!(
            max_abs_err(&got, &expected) < 1e-11,
            "n=11 err={}",
            max_abs_err(&got, &expected)
        );
    }

    #[test]
    fn roundtrip_for_non_power_of_two() {
        for &n in &[3usize, 5, 6, 7, 9, 11] {
            let input = sig(n);
            let mut data = input.clone();
            forward_inplace_64(&mut data);
            inverse_inplace_64(&mut data);
            let err = max_abs_err(&data, &input);
            assert!(err < 1e-11, "roundtrip n={n} err={err:.2e}");
        }
    }

    #[test]
    fn repeated_forward_same_input_same_output() {
        let input = sig(45);
        let mut a = input.clone();
        let mut b = input;
        forward_inplace_64(&mut a);
        forward_inplace_64(&mut b);
        let err = max_abs_err(&a, &b);
        assert!(err < 1e-12, "repeat determinism err={err:.2e}");
    }

    #[test]
    fn unnorm_inverse_differs_from_norm_by_n() {
        let n = 7usize;
        let input = sig(n);
        let mut spec = input.clone();
        forward_inplace_64(&mut spec);
        let mut unnorm = spec.clone();
        inverse_inplace_unnorm_64(&mut unnorm);
        let mut norm = spec.clone();
        inverse_inplace_64(&mut norm);
        let err = max_abs_err(
            &unnorm,
            &norm.iter().map(|x| x * n as f64).collect::<Vec<_>>(),
        );
        assert!(err < 1e-11, "unnorm/norm ratio failed n={n}: err={err:.2e}");
    }

    #[test]
    fn inverse_matches_direct_for_n5() {
        let input = sig(5);
        let expected = dft_inverse(&input);
        let mut got = input.clone();
        inverse_inplace_64(&mut got);
        assert!(
            max_abs_err(&got, &expected) < 1e-12,
            "inverse n=5 err={}",
            max_abs_err(&got, &expected)
        );
    }

    #[test]
    fn power_of_two_uses_stockham() {
        let n = 8usize;
        let input = sig(n);
        let mut bl = input.clone();
        forward_inplace_64(&mut bl);
        let mut st = input.clone();
        mixed_radix::forward_inplace_64(&mut st);
        assert!(max_abs_err(&bl, &st) < 1e-14, "bluestein vs stockham n=8");
    }
}
