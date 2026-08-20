//! Continuous wavelet transform analysis kernels.

use crate::domain::metadata::wavelet::ContinuousWavelet;

const HERMES_CWT_LEN_THRESHOLD: usize = 8_192;

thread_local! {
    static CWT_WEIGHT_SCRATCH: mnemosyne::scratch::ScratchPool<f64> =
        const { mnemosyne::scratch::ScratchPool::new() };
}

/// Evaluate the selected mother wavelet at normalized time `t`.
#[must_use]
pub fn mother_wavelet(wavelet: ContinuousWavelet, t: f64) -> f64 {
    match wavelet {
        ContinuousWavelet::Ricker => {
            let normalization = 2.0 / (3.0_f64.sqrt() * std::f64::consts::PI.powf(0.25));
            normalization * (1.0 - t * t) * (-0.5 * t * t).exp()
        }
        ContinuousWavelet::Morlet { omega0 } => {
            let correction = (-0.5 * omega0 * omega0).exp();
            std::f64::consts::PI.powf(-0.25)
                * ((omega0 * t).cos() - correction)
                * (-0.5 * t * t).exp()
        }
    }
}

/// Compute one real-valued CWT coefficient.
#[must_use]
pub fn coefficient(signal: &[f64], wavelet: ContinuousWavelet, scale: f64, shift: usize) -> f64 {
    if signal.len() >= HERMES_CWT_LEN_THRESHOLD {
        coefficient_hermes(signal, wavelet, scale, shift)
    } else {
        coefficient_scalar(signal, wavelet, scale, shift)
    }
}

fn coefficient_hermes(signal: &[f64], wavelet: ContinuousWavelet, scale: f64, shift: usize) -> f64 {
    CWT_WEIGHT_SCRATCH.with(|pool| {
        pool.with_scratch(signal.len(), |weights| {
            fill_cwt_weights(weights, wavelet, scale, shift);
            hermes_simd::dot::<f64>(signal, weights)
                .expect("CWT Hermes dot uses equal-length signal and weight slices")
        })
    })
}

fn fill_cwt_weights(weights: &mut [f64], wavelet: ContinuousWavelet, scale: f64, shift: usize) {
    let inv_sqrt_scale = 1.0 / scale.sqrt();
    for (index, weight) in weights.iter_mut().enumerate() {
        let normalized_time = (index as f64 - shift as f64) / scale;
        *weight = inv_sqrt_scale * mother_wavelet(wavelet, normalized_time);
    }
}

fn coefficient_scalar(signal: &[f64], wavelet: ContinuousWavelet, scale: f64, shift: usize) -> f64 {
    let inv_sqrt_scale = 1.0 / scale.sqrt();
    signal
        .iter()
        .enumerate()
        .map(|(index, &sample)| {
            let normalized_time = (index as f64 - shift as f64) / scale;
            sample * inv_sqrt_scale * mother_wavelet(wavelet, normalized_time)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::assert_abs_diff_eq;

    #[test]
    fn hermes_cwt_coefficient_matches_scalar_formula_at_threshold() {
        let signal = (0..HERMES_CWT_LEN_THRESHOLD)
            .map(|index| (index as f64 * 0.01).sin() + (index % 7) as f64 * 0.125)
            .collect::<Vec<_>>();
        let wavelet = ContinuousWavelet::Morlet { omega0: 5.0 };
        let scale = 3.5;
        let shift = HERMES_CWT_LEN_THRESHOLD / 3;

        let expected = coefficient_scalar(&signal, wavelet, scale, shift);
        let actual = coefficient_hermes(&signal, wavelet, scale, shift);

        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-9);
    }

    #[test]
    fn cwt_weights_match_mother_wavelet_formula() {
        let wavelet = ContinuousWavelet::Ricker;
        let scale = 2.0;
        let shift = 3;
        let mut weights = [0.0; 8];

        fill_cwt_weights(&mut weights, wavelet, scale, shift);

        for (index, actual) in weights.iter().enumerate() {
            let normalized_time = (index as f64 - shift as f64) / scale;
            let expected = scale.sqrt().recip() * mother_wavelet(wavelet, normalized_time);
            assert_abs_diff_eq!(*actual, expected, epsilon = 1.0e-12);
        }
    }
}
