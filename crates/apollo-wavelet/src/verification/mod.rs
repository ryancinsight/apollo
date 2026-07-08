//! Verification modules for wavelet transforms.

#[cfg(test)]
mod tests {
    use crate::{ContinuousWavelet, CwtPlan, DiscreteWavelet, DwtPlan, WaveletError};
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;
    use proptest::proptest;

    #[test]
    fn haar_single_level_matches_analytical_coefficients() {
        let plan = DwtPlan::new(4, 1, DiscreteWavelet::Haar).expect("plan");
        let coefficients = plan.forward(&[1.0, 3.0, 2.0, 0.0]).expect("forward");
        let scale = std::f64::consts::FRAC_1_SQRT_2;

        assert_abs_diff_eq!(
            coefficients.approximation()[0],
            4.0 * scale,
            epsilon = 1.0e-12
        );
        assert_abs_diff_eq!(
            coefficients.approximation()[1],
            2.0 * scale,
            epsilon = 1.0e-12
        );
        assert_abs_diff_eq!(
            coefficients.details()[0][0],
            -2.0 * scale,
            epsilon = 1.0e-12
        );
        assert_abs_diff_eq!(coefficients.details()[0][1], 2.0 * scale, epsilon = 1.0e-12);
    }

    #[test]
    fn dwt_inverse_recovers_signal_for_supported_wavelets() {
        let signal = [1.0, -2.0, 0.5, 4.0, -1.25, 0.75, 3.0, -0.5];
        for wavelet in [DiscreteWavelet::Haar, DiscreteWavelet::Daubechies4] {
            let plan = DwtPlan::new(signal.len(), 3, wavelet).expect("plan");
            let coefficients = plan.forward(&signal).expect("forward");
            let recovered = plan.inverse(&coefficients).expect("inverse");
            for (actual, expected) in recovered.iter().zip(signal.iter()) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-10);
            }
        }
    }

    #[test]
    fn dwt_rejects_invalid_contracts() {
        assert_eq!(
            DwtPlan::new(0, 1, DiscreteWavelet::Haar).unwrap_err(),
            WaveletError::EmptySignal
        );
        assert_eq!(
            DwtPlan::new(6, 1, DiscreteWavelet::Haar).unwrap_err(),
            WaveletError::NonPowerOfTwoLength
        );
        assert_eq!(
            DwtPlan::new(8, 0, DiscreteWavelet::Haar).unwrap_err(),
            WaveletError::EmptyLevelCount
        );
        assert_eq!(
            DwtPlan::new(8, 4, DiscreteWavelet::Haar).unwrap_err(),
            WaveletError::LevelExceedsLength
        );
        let plan = DwtPlan::new(8, 1, DiscreteWavelet::Haar).expect("plan");
        assert_eq!(
            plan.forward(&[1.0, 2.0]).unwrap_err(),
            WaveletError::LengthMismatch
        );
    }

    #[test]
    fn cwt_impulse_response_peaks_at_impulse_shift() {
        let signal = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let plan = CwtPlan::new(7, vec![1.0], ContinuousWavelet::Ricker).expect("plan");
        let coefficients = plan.transform(&signal).expect("cwt");
        let values = coefficients.values();
        let cols = values.shape()[1];
        let row = &values.as_slice().expect("contiguous coefficients")[0..cols];
        let (max_index, _) = row
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.abs().partial_cmp(&rhs.abs()).unwrap())
            .expect("non-empty row");
        assert_eq!(max_index, 3);
    }

    #[test]
    fn cwt_rejects_invalid_contracts() {
        assert_eq!(
            CwtPlan::new(0, vec![1.0], ContinuousWavelet::Ricker).unwrap_err(),
            WaveletError::EmptySignal
        );
        assert_eq!(
            CwtPlan::new(8, vec![], ContinuousWavelet::Ricker).unwrap_err(),
            WaveletError::EmptyScales
        );
        assert_eq!(
            CwtPlan::new(8, vec![0.0], ContinuousWavelet::Ricker).unwrap_err(),
            WaveletError::InvalidScale
        );
        let plan = CwtPlan::new(8, vec![1.0], ContinuousWavelet::Ricker).expect("plan");
        assert_eq!(
            plan.transform(&[1.0, 2.0]).unwrap_err(),
            WaveletError::LengthMismatch
        );
    }

    proptest! {
        #[test]
        fn haar_roundtrip_holds_for_random_power_of_two_signals(
            data in prop::collection::vec(-10.0f64..10.0f64, 8)
        ) {
            let plan = DwtPlan::new(8, 3, DiscreteWavelet::Haar).expect("plan");
            let coefficients = plan.forward(&data).expect("forward");
            let recovered = plan.inverse(&coefficients).expect("inverse");
            for (actual, expected) in recovered.iter().zip(data.iter()) {
                prop_assert!((actual - expected).abs() < 1.0e-10);
            }
        }
    }

    /// Theorem: Morlet CWT resonates at the scale matching the input signal frequency.
    ///
    /// For the DC-corrected real Morlet wavelet with angular carrier ω₀, the CWT
    /// coefficient W(a, b) is maximized when the scale a = ω₀ / ω_signal, where
    /// ω_signal is the signal's dominant angular frequency.
    ///
    /// **Derivation:** The Morlet mother wavelet ψ(t) = π^(-1/4)(cos(ω₀t) - e^{-ω₀²/2})e^{-t²/2}
    /// acts as a bandpass filter with center frequency ω₀/a at scale a. For a signal
    /// x[n] = cos(ω_signal·n), the CWT at resonant scale a₀ = ω₀/ω_signal evaluates
    /// the inner product of x with a scaled copy of ψ, giving amplitude ≈ π^(1/4) / √a₀
    /// (Gaussian envelope integral × cosine resonance). At a mismatched scale a_far = 16,
    /// the Gaussian window width is ~4× the signal period, and the incoherent product
    /// integrates to near zero.
    ///
    /// **Verified:** max|W(a=2, b)| > 2 · max|W(a=16, b)| and max|W(a=2, b)| > 0.5.
    #[test]
    fn morlet_cwt_resonates_at_matching_scale() {
        let omega0 = 5.0_f64;
        let s_resonant = 2.0_f64;
        // Signal frequency matching resonant scale: ω_signal = ω₀ / s_resonant = 2.5 rad/sample.
        let omega_signal = omega0 / s_resonant;
        let n = 32usize;
        let signal: Vec<f64> = (0..n).map(|i| (omega_signal * i as f64).cos()).collect();
        let scales = vec![s_resonant, 16.0_f64];
        let plan = CwtPlan::new(n, scales, ContinuousWavelet::Morlet { omega0 }).expect("plan");
        let coeffs = plan.transform(&signal).expect("transform");
        let values = coeffs.values();
        let cols = values.shape()[1];
        let flat = values.as_slice().expect("contiguous coefficients");
        // Max coefficient magnitude at resonant scale (row 0)
        let resonant_row = &flat[0..cols];
        let max_resonant = resonant_row.iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
        // Max coefficient magnitude at far scale (row 1)
        let far_row = &flat[cols..2 * cols];
        let max_far = far_row.iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
        // At the resonant scale, the Morlet CWT coefficient is O(π^(1/4)) ≈ 1.33.
        // At scale 16 (mismatched frequency), the broad Gaussian and incoherent oscillation give ~0.
        assert!(
            max_resonant > 2.0 * max_far.max(1e-10),
            "Resonant scale s={s_resonant} should dominate far scale s=16: \
             max_resonant={max_resonant:.4}, max_far={max_far:.4}"
        );
        assert!(
            max_resonant > 0.5,
            "Morlet CWT at resonant scale should have nontrivial amplitude: max_resonant={max_resonant:.4}"
        );
    }

    #[test]
    fn morlet_mother_wavelet_has_bounded_dc_response() {
        let omega0 = 5.0;
        let step = 0.01;
        let support = 12.0;
        let sample_count = (2.0 * support / step) as usize + 1;
        let integral: f64 = (0..sample_count)
            .map(|i| {
                let t = -support + i as f64 * step;
                crate::infrastructure::kernel::continuous::mother_wavelet(
                    ContinuousWavelet::Morlet { omega0 },
                    t,
                ) * step
            })
            .sum();
        assert_abs_diff_eq!(integral, 0.0, epsilon = 1.0e-10);
    }

    #[test]
    fn ricker_cwt_zero_signal_gives_zero_coefficients() {
        let signal = vec![0.0_f64; 16];
        let scales = vec![1.0, 2.0];
        let plan = CwtPlan::new(signal.len(), scales, ContinuousWavelet::Ricker).unwrap();
        let coeffs = plan.transform(&signal).unwrap();
        for &c in coeffs.values().iter() {
            assert!(
                c.abs() < 1e-14,
                "Zero signal should give zero coefficients, got {c}"
            );
        }
    }
}
