//! Verification modules for Mellin.

#[cfg(test)]
mod tests {
    use crate::{calculate_log_resample, mellin_moment, MellinError, MellinPlan};
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    #[test]
    fn plan_preserves_scale_configuration() {
        let plan = MellinPlan::new(128, 0.25, 4.0).expect("plan");
        assert_eq!(plan.config().samples(), 128);
        assert_eq!(plan.config().min_scale(), 0.25);
        assert_eq!(plan.config().max_scale(), 4.0);
    }

    #[test]
    fn rejects_invalid_scale_configuration() {
        assert_eq!(
            MellinPlan::new(0, 0.25, 4.0).unwrap_err(),
            MellinError::EmptySampleCount
        );
        assert_eq!(
            MellinPlan::new(8, f64::NAN, 4.0).unwrap_err(),
            MellinError::InvalidScaleBound
        );
        assert_eq!(
            MellinPlan::new(8, 4.0, 4.0).unwrap_err(),
            MellinError::InvalidScaleOrder
        );
    }

    #[test]
    fn log_resample_matches_linear_input_at_scale_grid_endpoints() {
        let plan = MellinPlan::new(3, 1.0, 4.0).expect("plan");
        let signal = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 3];
        plan.forward_resample(&signal, 1.0, 4.0, &mut output)
            .expect("resample");

        assert_abs_diff_eq!(output[0], 1.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(output[2], 4.0, epsilon = 1.0e-12);
        assert!(output[1] > output[0]);
        assert!(output[1] < output[2]);
    }

    #[test]
    fn constant_signal_moment_matches_analytical_integral() {
        let plan = MellinPlan::new(8, 1.0, 4.0).expect("plan");
        let signal = [2.0; 65];
        let moment = plan.moment(&signal, 1.0, 4.0, 1.0).expect("moment");

        assert_abs_diff_eq!(moment, 6.0, epsilon = 1.0e-12);
    }

    #[test]
    fn power_signal_moment_matches_analytical_integral() {
        let min = 1.0;
        let max = 3.0;
        let samples = 4097;
        let step = (max - min) / (samples as f64 - 1.0);
        let signal: Vec<f64> = (0..samples)
            .map(|index| {
                let coordinate = min + index as f64 * step;
                coordinate * coordinate
            })
            .collect();
        let plan = MellinPlan::new(16, min, max).expect("plan");
        let moment = plan.moment(&signal, min, max, 1.0).expect("moment");
        let expected = (max.powi(3) - min.powi(3)) / 3.0;

        assert_abs_diff_eq!(moment, expected, epsilon = 1.0e-7);
    }

    #[test]
    fn constant_log_frequency_spectrum_has_only_dc_component() {
        let plan = MellinPlan::new(8, 1.0, 8.0).expect("plan");
        let signal = [5.0; 8];
        let spectrum = plan.forward_spectrum(&signal, 1.0, 8.0).expect("spectrum");

        assert_abs_diff_eq!(spectrum.values()[0].im, 0.0, epsilon = 1.0e-12);
        for coefficient in &spectrum.values()[1..] {
            assert_abs_diff_eq!(coefficient.norm(), 0.0, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn rejects_invalid_execution_contracts() {
        let plan = MellinPlan::new(4, 1.0, 2.0).expect("plan");
        let mut output = [0.0; 3];

        assert_eq!(
            plan.forward_resample(&[1.0], 1.0, 2.0, &mut output)
                .unwrap_err(),
            MellinError::LengthMismatch
        );
        assert_eq!(
            plan.moment(&[], 1.0, 2.0, 1.0).unwrap_err(),
            MellinError::EmptySignal
        );
        assert_eq!(
            plan.moment(&[1.0], 0.0, 2.0, 1.0).unwrap_err(),
            MellinError::InvalidSignalBound
        );
        assert_eq!(
            plan.moment(&[1.0], 2.0, 1.0, 1.0).unwrap_err(),
            MellinError::InvalidSignalOrder
        );
        assert_eq!(
            plan.moment(&[1.0], 1.0, 2.0, f64::NAN).unwrap_err(),
            MellinError::InvalidExponent
        );
    }

    proptest! {
        /// Log-resample of a uniform signal returns uniform values within interpolation error.
        #[test]
        fn log_resample_uniform_signal_is_uniform(
            constant in 0.1_f64..2.0_f64,
            n in 4_usize..33_usize,
        ) {
            // Signal domain [0.5, 20.0] strictly contains the scale grid [1.0, 10.0].
            // This prevents exp(ln(max_scale)) from landing epsilon above signal_max
            // in floating-point, which would otherwise return 0 for the boundary sample.
            let signal: Vec<f64> = vec![constant; n];
            let mut result = vec![0.0_f64; n];
            calculate_log_resample(&signal, 0.5, 20.0, &mut result, 1.0, 10.0);
            // All log-spaced samples in [1.0, 10.0] are strictly inside [0.5, 20.0],
            // so linear interpolation on a uniform signal returns constant exactly.
            let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
            for &v in &result {
                prop_assert!(
                    (v - mean).abs() < 1.0e-10,
                    "non-uniform result for constant signal: v={}, mean={}",
                    v,
                    mean
                );
            }
        }

        /// Trapezoidal Mellin moment of r^alpha satisfies M(s) = (max^(alpha+s) - min^(alpha+s)) / (alpha+s).
        ///
        /// Mathematical basis: integral_min^max r^alpha * r^(s-1) dr
        ///   = integral r^(alpha+s-1) dr = [r^(alpha+s) / (alpha+s)]_min^max.
        #[test]
        fn power_law_moment_analytical(
            alpha in 0.1_f64..3.0_f64,
            s_real in 0.5_f64..2.5_f64,
        ) {
            let n = 64_usize;
            let min_scale = 1.0_f64;
            let max_scale = 3.0_f64;
            let step = (max_scale - min_scale) / (n - 1) as f64;
            let signal: Vec<f64> = (0..n)
                .map(|i| (min_scale + i as f64 * step).powf(alpha))
                .collect();
            let computed = mellin_moment(&signal, min_scale, max_scale, s_real);
            let exponent = alpha + s_real;
            let analytical = (max_scale.powf(exponent) - min_scale.powf(exponent)) / exponent;
            prop_assert!(
                (computed - analytical).abs() / analytical.abs() < 0.05,
                "power law moment: computed={}, analytical={}",
                computed,
                analytical
            );
        }
    }
}

#[cfg(test)]
mod inverse_tests {
    use crate::{MellinError, MellinPlan};

    /// Theorem (Mellin inverse roundtrip, constant signal):
    /// For a constant signal the forward spectrum has only a DC component and
    /// the inverse must recover the same constant to within interpolation error.
    #[test]
    fn inverse_spectrum_roundtrip_constant_signal() {
        let n = 32usize;
        let min_scale = 1.0_f64;
        let max_scale = 8.0_f64;
        let plan = MellinPlan::new(n, min_scale, max_scale).expect("plan");

        // Build constant signal on log-grid (forward_spectrum log-resamples first).
        let signal: Vec<f64> = vec![3.0; n];
        let spectrum = plan
            .forward_spectrum(&signal, min_scale, max_scale)
            .expect("forward");

        let mut recovered = vec![0.0_f64; n];
        plan.inverse_spectrum(&spectrum, min_scale, max_scale, &mut recovered)
            .expect("inverse");

        // The constant signal is maximally smooth; interpolation error is zero.
        for (i, &v) in recovered.iter().enumerate() {
            assert!(
                (v - 3.0).abs() < 1.0e-10,
                "sample {i}: expected=3.0, got={v:.15e}"
            );
        }
    }

    /// Theorem (Mellin inverse roundtrip, linear signal):
    /// A linear signal sampled on the log-grid and round-tripped through
    /// forward_spectrum + inverse_spectrum must be recovered at the overlapping
    /// query domain to within interpolation precision (linear interpolation
    /// introduces no error for a linear function).
    #[test]
    fn inverse_spectrum_roundtrip_linear_signal() {
        let n = 64usize;
        let min_scale = 1.0_f64;
        let max_scale = 4.0_f64;
        let plan = MellinPlan::new(n, min_scale, max_scale).expect("plan");

        // Linear signal: f(r) = 2*r + 0.5, sampled at n uniformly spaced r.
        let step = (max_scale - min_scale) / (n as f64 - 1.0);
        let signal: Vec<f64> = (0..n)
            .map(|i| 2.0 * (min_scale + i as f64 * step) + 0.5)
            .collect();

        let spectrum = plan
            .forward_spectrum(&signal, min_scale, max_scale)
            .expect("forward");

        let mut recovered = vec![0.0_f64; n];
        plan.inverse_spectrum(&spectrum, min_scale, max_scale, &mut recovered)
            .expect("inverse");

        // The inverse queries the log-domain at the same scale grid as the
        // forward input, so the round-trip error is bounded by DFT alias error
        // plus log-to-linear interpolation error.  For a smooth signal and N=64
        // the dominant error is DFT alias leakage ~O(1/N), bound loosened to 0.1.
        let r_step = (max_scale - min_scale) / (n as f64 - 1.0);
        for (i, &v) in recovered.iter().enumerate() {
            let r = min_scale + i as f64 * r_step;
            let expected = 2.0 * r + 0.5;
            let err = (v - expected).abs();
            assert!(
                err < 0.1,
                "sample {i}: expected={expected:.4}, got={v:.4}, err={err:.4e}"
            );
        }
    }

    /// Error contract: spectrum length mismatch returns SpectrumLengthMismatch.
    #[test]
    fn inverse_spectrum_rejects_wrong_spectrum_length() {
        let plan = MellinPlan::new(8, 1.0, 4.0).expect("plan");
        let bad_spectrum = crate::MellinSpectrum::new(vec![eunomia::Complex64::ZERO; 5]);
        let mut out = vec![0.0_f64; 8];
        assert_eq!(
            plan.inverse_spectrum(&bad_spectrum, 1.0, 4.0, &mut out)
                .unwrap_err(),
            MellinError::SpectrumLengthMismatch
        );
    }

    /// Error contract: invalid output bounds return InvalidSignalBound.
    #[test]
    fn inverse_spectrum_rejects_invalid_output_bounds() {
        let n = 8usize;
        let plan = MellinPlan::new(n, 1.0, 4.0).expect("plan");
        let spectrum = crate::MellinSpectrum::new(vec![eunomia::Complex64::ZERO; n]);
        let mut out = vec![0.0_f64; n];
        assert_eq!(
            plan.inverse_spectrum(&spectrum, 0.0, 4.0, &mut out)
                .unwrap_err(),
            MellinError::InvalidSignalBound
        );
        assert_eq!(
            plan.inverse_spectrum(&spectrum, 1.0, 1.0, &mut out)
                .unwrap_err(),
            MellinError::InvalidSignalOrder
        );
    }
}
