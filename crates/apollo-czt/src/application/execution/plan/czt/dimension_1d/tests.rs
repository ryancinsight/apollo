//! Unit tests for 1D Chirp Z-Transform.

use super::helpers::{forward_workspace_capacity, typed_scratch_capacities};
use super::plan::CztPlan;
use crate::domain::contracts::error::CztError;
use apollo_fft::{f16, PrecisionProfile};
use eunomia::{assert_abs_diff_eq, assert_relative_eq};
use eunomia::{Complex32, Complex64};
use leto::Array1;

fn contiguous_array<T>(values: Vec<T>) -> Array1<T> {
    Array1::from_shape_vec([values.len()], values).expect("contiguous test input")
}

mod spiral_collapse_tests {
    use super::*;

    /// CZT with A=1, W=exp(-2pi*i/N), M=N equals the N-point DFT (spiral-collapse theorem).
    #[test]
    fn czt_with_dft_parameters_equals_dft() {
        let n = 8usize;
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64);
        let input = Array1::from_shape_fn([n], |[i]| {
            Complex64::new((i as f64 * 0.7).sin(), (i as f64 * 0.3).cos())
        });
        let plan = CztPlan::new(n, n, a, w).expect("valid DFT plan");
        let fast = plan.forward(&input).expect("fast");
        let direct = plan.forward_direct(&input).expect("direct");
        for (k, (cv, dv)) in fast.iter().zip(direct.iter()).enumerate() {
            let err = (cv - dv).norm();
            assert!(err < 1e-10, "CZT != DFT at k={k}: err={err}");
        }
    }

    /// Bluestein fast path matches direct for multiple non-trivial (N, M) pairs.
    #[test]
    fn czt_bluestein_equals_direct_for_fixed_inputs() {
        for (n, m) in [(3usize, 3usize), (5, 7), (11, 11), (13, 8)] {
            let a = Complex64::from_polar(0.9, 0.3);
            let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64 * 0.8);
            let inp = Array1::from_shape_fn([n], |[i]| {
                Complex64::new((i as f64 * 0.31).sin(), -(i as f64 * 0.19).cos())
            });
            let plan = CztPlan::new(n, m, a, w).expect("valid");
            let fast = plan.forward(&inp).expect("fast");
            let direct = plan.forward_direct(&inp).expect("direct");
            for (k, (f, d)) in fast.iter().zip(direct.iter()).enumerate() {
                let err = (f - d).norm();
                assert!(err < 1e-9, "n={n} m={m} k={k} err={err}");
            }
        }
    }

    /// Theorem (Spiral-Collapse, independent cross-check): CZT(x, N, exp(-2πi/N), 1)
    /// equals the N-point DFT, verified here against `apollo_fft::fft_1d_complex`
    /// which is entirely independent of the CZT implementation path.
    #[test]
    fn czt_dft_parameters_match_independent_fft_implementation() {
        let n = 8usize;
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64);
        let input = Array1::from_shape_fn([n], |[i]| {
            Complex64::new((i as f64 * 0.7).sin(), (i as f64 * 0.3).cos())
        });
        let plan = CztPlan::new(n, n, a, w).expect("valid DFT plan");
        let czt_output = plan.forward(&input).expect("CZT forward");
        let fft_output = apollo_fft::fft_1d_complex(&input);
        for (k, (cv, fv)) in czt_output.iter().zip(fft_output.iter()).enumerate() {
            let err = (cv - fv).norm();
            assert!(
                err < 1e-9,
                "CZT does not match independent FFT at k={k}: czt={cv:?}, fft={fv:?}, err={err:.3e}"
            );
        }
    }

    #[test]
    fn rejects_zero_length() {
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::new(0.9, 0.1);
        assert!(matches!(
            CztPlan::new(0, 5, a, w),
            Err(CztError::EmptyLength)
        ));
        assert!(matches!(
            CztPlan::new(5, 0, a, w),
            Err(CztError::EmptyLength)
        ));
    }

    #[test]
    fn rejects_zero_a() {
        let a = Complex64::new(0.0, 0.0);
        let w = Complex64::new(0.9, 0.1);
        assert!(matches!(
            CztPlan::new(4, 4, a, w),
            Err(CztError::InvalidParameters)
        ));
    }
}

mod general_czt_tests {
    use super::*;

    #[test]
    fn direct_matches_reference_for_small_sequence() {
        let input = contiguous_array(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.5, 0.25),
            Complex64::new(-0.75, 0.5),
        ]);
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / 8.0);
        let plan = CztPlan::new(input.size(), 4, a, w).expect("valid plan");
        let direct = plan.forward_direct(&input).expect("direct");
        let fast = plan.forward(&input).expect("fast");
        for (lhs, rhs) in direct.iter().zip(fast.iter()) {
            assert_relative_eq!(lhs.re, rhs.re, epsilon = 1.0e-9);
            assert_relative_eq!(lhs.im, rhs.im, epsilon = 1.0e-9);
        }
    }

    #[test]
    fn rejects_invalid_lengths() {
        assert!(matches!(
            CztPlan::new(0, 4, Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)),
            Err(CztError::EmptyLength)
        ));
        assert!(matches!(
            CztPlan::new(4, 0, Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)),
            Err(CztError::EmptyLength)
        ));
    }

    #[test]
    fn length_mismatch_is_rejected() {
        let plan = CztPlan::new(
            4,
            4,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, -std::f64::consts::TAU / 8.0),
        )
        .expect("valid plan");
        let bad = contiguous_array(vec![Complex64::new(0.0, 0.0); 3]);
        assert!(matches!(plan.forward(&bad), Err(CztError::LengthMismatch)));
        let good = contiguous_array(vec![Complex64::new(0.0, 0.0); 4]);
        let mut bad_output = contiguous_array(vec![Complex64::new(0.0, 0.0); 3]);
        assert!(matches!(
            plan.forward_into(&good, &mut bad_output),
            Err(CztError::LengthMismatch)
        ));
    }

    #[test]
    fn forward_into_matches_allocating_fast_path() {
        let input = contiguous_array(vec![
            Complex64::new(0.25, 0.5),
            Complex64::new(-0.75, 1.0),
            Complex64::new(1.25, -0.25),
            Complex64::new(0.5, 0.125),
            Complex64::new(-0.375, -0.75),
        ]);
        let plan = CztPlan::new(
            input.size(),
            7,
            Complex64::from_polar(1.0, 0.125),
            Complex64::from_polar(1.0, -std::f64::consts::TAU / 11.0),
        )
        .expect("valid plan");

        let expected = plan.forward(&input).expect("allocating fast path");
        let mut actual = Array1::<Complex64>::zeros([plan.output_len()]);
        plan.forward_into(&input, &mut actual)
            .expect("caller-owned fast path");

        for (lhs, rhs) in expected.iter().zip(actual.iter()) {
            assert_relative_eq!(lhs.re, rhs.re, epsilon = 1.0e-12);
            assert_relative_eq!(lhs.im, rhs.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn forward_into_reuses_plan_convolution_workspace() {
        let input = contiguous_array(vec![
            Complex64::new(0.25, -0.5),
            Complex64::new(-0.75, 1.0),
            Complex64::new(1.25, 0.25),
            Complex64::new(0.5, -0.125),
            Complex64::new(-0.375, 0.75),
        ]);
        let plan = CztPlan::new(
            input.size(),
            7,
            Complex64::from_polar(1.0, 0.125),
            Complex64::from_polar(1.0, -std::f64::consts::TAU / 11.0),
        )
        .expect("valid plan");
        let mut first = Array1::<Complex64>::zeros([plan.output_len()]);
        let mut second = Array1::<Complex64>::zeros([plan.output_len()]);

        plan.forward_into(&input, &mut first)
            .expect("first caller-owned fast path");
        let capacity_after_first = forward_workspace_capacity();
        assert!(capacity_after_first >= plan.convolution_len());

        plan.forward_into(&input, &mut second)
            .expect("second caller-owned fast path");
        assert_eq!(forward_workspace_capacity(), capacity_after_first);
        for (actual, expected) in second.iter().zip(first.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn typed_paths_support_complex64_complex32_and_mixed_f16_storage() {
        let input64 = contiguous_array(vec![
            Complex64::new(0.25, 0.5),
            Complex64::new(-0.75, 1.0),
            Complex64::new(1.25, -0.25),
            Complex64::new(0.5, 0.125),
            Complex64::new(-0.375, -0.75),
        ]);
        let plan = CztPlan::new(
            input64.size(),
            7,
            Complex64::from_polar(1.0, 0.125),
            Complex64::from_polar(1.0, -std::f64::consts::TAU / 11.0),
        )
        .expect("valid plan");
        let expected = plan.forward(&input64).expect("reference");

        let mut out64 = Array1::<Complex64>::zeros([plan.output_len()]);
        plan.forward_typed_into(&input64, &mut out64, PrecisionProfile::HIGH_ACCURACY_F64)
            .expect("complex64 typed");
        for (actual, expected) in out64.iter().zip(expected.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }

        let input32 = input64.mapv(|value| Complex32::new(value.re as f32, value.im as f32));
        let mut out32 = Array1::<Complex32>::zeros([plan.output_len()]);
        plan.forward_typed_into(&input32, &mut out32, PrecisionProfile::LOW_PRECISION_F32)
            .expect("complex32 typed");
        for (actual, expected) in out32.iter().zip(expected.iter()) {
            assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
            assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
        }

        let input16 = input64.mapv(|value| {
            [
                f16::from_f32(value.re as f32),
                f16::from_f32(value.im as f32),
            ]
        });
        let mut out16 = Array1::from_elem([plan.output_len()], [f16::from_f32(0.0); 2]);
        plan.forward_typed_into(
            &input16,
            &mut out16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("mixed f16 typed");
        for (actual, expected) in out16.iter().zip(expected.iter()) {
            let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!((f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound);
            assert!((f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound);
        }
    }

    #[test]
    fn typed_path_rejects_profile_storage_mismatch() {
        let plan = CztPlan::new(
            4,
            4,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, -std::f64::consts::TAU / 8.0),
        )
        .expect("valid plan");
        let input = contiguous_array(vec![Complex32::new(1.0, 0.0); 4]);
        let mut output = Array1::<Complex32>::zeros([4]);
        assert!(matches!(
            plan.forward_typed_into(&input, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
            Err(CztError::PrecisionMismatch)
        ));
    }

    #[test]
    fn typed_complex32_forward_and_inverse_reuse_complex64_workspaces() {
        let n = 5usize;
        let input64 = Array1::from_shape_fn([n], |[i]| {
            Complex64::new((i as f64 * 0.23).sin(), (i as f64 * 0.41).cos())
        });
        let input32 = input64.mapv(|value| Complex32::new(value.re as f32, value.im as f32));
        let plan = CztPlan::new(
            n,
            n,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64),
        )
        .expect("valid DFT-equivalent CZT plan");
        let mut first_spectrum = Array1::<Complex32>::zeros([n]);
        let mut second_spectrum = Array1::<Complex32>::zeros([n]);
        let mut first_recovered = Array1::<Complex32>::zeros([n]);
        let mut second_recovered = Array1::<Complex32>::zeros([n]);

        plan.forward_typed_into(
            &input32,
            &mut first_spectrum,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("first typed forward");
        let forward_caps = typed_scratch_capacities();
        plan.forward_typed_into(
            &input32,
            &mut second_spectrum,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("second typed forward");
        assert_eq!(typed_scratch_capacities(), forward_caps);

        plan.inverse_typed_into(
            &first_spectrum,
            &mut first_recovered,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("first typed inverse");
        let inverse_caps = typed_scratch_capacities();
        plan.inverse_typed_into(
            &first_spectrum,
            &mut second_recovered,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("second typed inverse");
        assert_eq!(typed_scratch_capacities(), inverse_caps);
        assert!(inverse_caps.0 >= n);
        assert!(inverse_caps.1 >= n);

        for ((first, second), original) in first_recovered
            .iter()
            .zip(second_recovered.iter())
            .zip(input32.iter())
        {
            assert!((first.re - second.re).abs() < 1.0e-7);
            assert!((first.im - second.im).abs() < 1.0e-7);
            assert!((first.re - original.re).abs() < 1.0e-4);
            assert!((first.im - original.im).abs() < 1.0e-4);
        }
    }
}

mod inverse_tests {
    use super::*;

    /// Theorem (exact ICZT roundtrip, DFT case):
    /// For A=1, W=exp(-2πi/N), the CZT is the N-point DFT.
    /// `inverse(forward(x)) == x` must hold to machine precision.
    #[test]
    fn inverse_roundtrip_dft_parameters() {
        for n in [2usize, 3, 4, 5, 7, 8, 11, 13, 16] {
            let a = Complex64::new(1.0, 0.0);
            let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64);
            let input: Array1<Complex64> = Array1::from_shape_fn([n], |[i]| {
                Complex64::new((i as f64 * 0.31 + 0.7).sin(), (i as f64 * 0.17).cos())
            });
            let plan = CztPlan::new(n, n, a, w).expect("DFT plan");
            let spectrum = plan.forward(&input).expect("forward");
            let recovered = plan.inverse(&spectrum).expect("inverse");
            for i in 0..n {
                let err = (recovered[[i]] - input[[i]]).norm();
                assert!(err < 1e-10, "n={n} i={i} err={err:.3e}");
            }
        }
    }

    /// Theorem (exact ICZT roundtrip, general A≠1):
    /// For |A|=1 and W=exp(-2πi*θ) with θ irrational, the Vandermonde nodes
    /// z_k = W^k are distinct and Björck-Pereyra gives the exact inverse.
    #[test]
    fn inverse_roundtrip_general_a_parameter() {
        let n = 6usize;
        let a = Complex64::from_polar(1.0, 0.5);
        let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64);
        let input: Array1<Complex64> = Array1::from_shape_fn([n], |[i]| {
            Complex64::new((i as f64 * 0.53 + 1.1).sin(), (i as f64 * 0.37 - 0.5).cos())
        });
        let plan = CztPlan::new(n, n, a, w).expect("plan");
        let spectrum = plan.forward(&input).expect("forward");
        let recovered = plan.inverse(&spectrum).expect("inverse");
        for i in 0..n {
            assert_abs_diff_eq!(recovered[[i]].re, input[[i]].re, epsilon = 1e-9);
            assert_abs_diff_eq!(recovered[[i]].im, input[[i]].im, epsilon = 1e-9);
        }
    }

    /// Theorem (exact ICZT roundtrip, |W|≠1):
    /// When |W| > 1 the spiral points z_k = W^k diverge; they are always
    /// distinct since |z_j - z_k| = |W|^k * |W^{j-k} - 1| > 0 for j≠k.
    #[test]
    fn inverse_roundtrip_non_unit_w() {
        let n = 5usize;
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::from_polar(1.1, -0.9);
        let input: Array1<Complex64> = Array1::from_shape_fn([n], |[i]| {
            Complex64::new((i as f64 * 0.7).cos(), (i as f64 * 0.4 + 0.3).sin())
        });
        let plan = CztPlan::new(n, n, a, w).expect("plan");
        let spectrum = plan.forward(&input).expect("forward");
        let recovered = plan.inverse(&spectrum).expect("inverse");
        for i in 0..n {
            assert_abs_diff_eq!(recovered[[i]].re, input[[i]].re, epsilon = 1e-8);
            assert_abs_diff_eq!(recovered[[i]].im, input[[i]].im, epsilon = 1e-8);
        }
    }

    /// Rectangular CZT (M ≠ N) must return NotInvertible.
    #[test]
    fn rectangular_czt_is_not_invertible() {
        let plan = CztPlan::new(
            4,
            6,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, -0.7),
        )
        .expect("plan");
        let spectrum = Array1::zeros([6]);
        assert!(matches!(
            plan.inverse(&spectrum),
            Err(CztError::NotInvertible { .. })
        ));
    }

    /// Spectrum length mismatch must return LengthMismatch.
    #[test]
    fn inverse_rejects_wrong_spectrum_length() {
        let n = 4usize;
        let plan = CztPlan::new(
            n,
            n,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64),
        )
        .expect("plan");
        let bad_spectrum = Array1::zeros([n + 1]);
        assert!(matches!(
            plan.inverse(&bad_spectrum),
            Err(CztError::LengthMismatch)
        ));
    }
}
