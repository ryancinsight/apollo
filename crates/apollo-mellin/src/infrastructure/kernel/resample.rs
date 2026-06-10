//! Mellin transform kernels: log-resampling, trapezoidal-rule moments, and direct DFT spectrum.
//!
//! ## Mathematical foundation
//!
//! The Mellin transform of f(r) on (0, inf) is M(s) = integral_0^inf f(r) r^(s-1) dr.
//!
//! **Theorem (Mellin-Fourier substitution)**: Under the substitution r = e^u,
//! the Mellin transform becomes M(s) = integral f(e^u) e^{su} du where g(u) = f(e^u).
//! Along the imaginary axis s = i*xi this is the Fourier transform of g:
//! M(i*xi) = integral g(u) e^{i*xi*u} du.
//!
//! **Consequence**: The Mellin spectrum on the imaginary axis equals the Fourier
//! transform of the log-resampled signal. calculate_log_resample performs the
//! r to e^u substitution; log_frequency_spectrum then applies the discrete Fourier sum.

use moirai::ParallelSliceMut;
use num_complex::Complex64;

const PAR_THRESHOLD: usize = 256;
const HERMES_MOMENT_LEN_THRESHOLD: usize = 8_192;
const HERMES_SPECTRUM_OP_THRESHOLD: usize = 16_384;

thread_local! {
    static MOMENT_WEIGHT_SCRATCH: mnemosyne::scratch::ScratchPool<f64> = const { mnemosyne::scratch::ScratchPool::new() };
    static LOG_FREQUENCY_WEIGHT_LANE_SCRATCH: mnemosyne::scratch::ScratchPool<f64> = const { mnemosyne::scratch::ScratchPool::new() };
    static REAL_LANES_SCRATCH: mnemosyne::scratch::ScratchPool<f64> = const { mnemosyne::scratch::ScratchPool::new() };
}

/// Interpolate a positive-domain signal onto logarithmically spaced samples.
///
/// The output sample `i` evaluates the input at
/// `exp(log(min_scale) + i * du)` using linear interpolation in the original
/// positive coordinate. Values outside `[signal_min, signal_max]` map to zero.
pub fn calculate_log_resample(
    signal: &[f64],
    signal_min: f64,
    signal_max: f64,
    output: &mut [f64],
    min_scale: f64,
    max_scale: f64,
) {
    let samples = output.len();
    if samples == 0 {
        return;
    }

    let log_min = min_scale.ln();
    let log_max = max_scale.ln();
    let step = if samples > 1 {
        (log_max - log_min) / (samples as f64 - 1.0)
    } else {
        0.0
    };

    let signal_len = signal.len();
    if signal_len == 0 {
        output.fill(0.0);
        return;
    }

    let domain_width = signal_max - signal_min;

    let eval = |i: usize| -> f64 {
        let current_scale = (log_min + i as f64 * step).exp();

        if current_scale < signal_min || current_scale > signal_max || domain_width <= 0.0 {
            return 0.0;
        }

        let fraction = (current_scale - signal_min) / domain_width;
        let exact_idx = fraction * (signal_len as f64 - 1.0);

        let lower_idx = exact_idx.floor() as usize;
        let upper_idx = (lower_idx + 1).min(signal_len - 1);

        let weight = exact_idx - lower_idx as f64;

        signal[lower_idx] * (1.0 - weight) + signal[upper_idx] * weight
    };

    if samples >= PAR_THRESHOLD {
        output.par_mut().enumerate(|i, val| {
            *val = eval(i);
        });
    } else {
        output.iter_mut().enumerate().for_each(|(i, val)| {
            *val = eval(i);
        });
    }
}

/// Evaluate the real Mellin moment `int f(r) r^(exponent - 1) dr`.
#[must_use]
pub fn mellin_moment(signal: &[f64], signal_min: f64, signal_max: f64, exponent: f64) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    if signal.len() == 1 {
        return signal[0] * moment_antiderivative(signal_min, signal_max, exponent);
    }

    let step = (signal_max - signal_min) / (signal.len() as f64 - 1.0);
    if signal.len() >= HERMES_MOMENT_LEN_THRESHOLD {
        return mellin_moment_hermes(signal, signal_min, exponent, step) * step;
    }
    mellin_moment_scalar(signal, signal_min, exponent, step) * step
}

fn mellin_moment_hermes(signal: &[f64], signal_min: f64, exponent: f64, step: f64) -> f64 {
    MOMENT_WEIGHT_SCRATCH.with(|pool| {
        pool.with_scratch(signal.len(), |weights| {
            fill_moment_weights(weights, signal_min, exponent, step);
            hermes_simd::dot::<f64>(signal, weights)
                .expect("Mellin moment Hermes dot uses equal-length signal and weight slices")
        })
    })
}

fn fill_moment_weights(weights: &mut [f64], signal_min: f64, exponent: f64, step: f64) {
    let len = weights.len();
    for (index, value) in weights.iter_mut().enumerate() {
        let coordinate = signal_min + index as f64 * step;
        let trapezoid = if index == 0 || index + 1 == len {
            0.5
        } else {
            1.0
        };
        *value = trapezoid * coordinate.powf(exponent - 1.0);
    }
}

fn mellin_moment_scalar(signal: &[f64], signal_min: f64, exponent: f64, step: f64) -> f64 {
    signal
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            let coordinate = signal_min + index as f64 * step;
            let weight = if index == 0 || index + 1 == signal.len() {
                0.5
            } else {
                1.0
            };
            weight * sample * coordinate.powf(exponent - 1.0)
        })
        .sum::<f64>()
}

/// Compute the direct log-frequency Mellin spectrum from log-domain samples.
///
/// The returned coefficient `k` is `du * sum_j g[j] exp(-2 pi i k j / N)`,
/// where `g[j] = f(exp(u_j))`. This is the DFT form of the imaginary-axis
/// Mellin transform after the substitution `r = exp(u)`.
#[must_use]
pub fn log_frequency_spectrum(log_samples: &[f64], log_min: f64, log_max: f64) -> Vec<Complex64> {
    let len = log_samples.len();
    if len == 0 {
        return Vec::new();
    }
    let du = if len > 1 {
        (log_max - log_min) / (len as f64 - 1.0)
    } else {
        1.0
    };
    let factor = -std::f64::consts::TAU / len as f64;
    let work_items = len.saturating_mul(len);
    if work_items >= HERMES_SPECTRUM_OP_THRESHOLD {
        return REAL_LANES_SCRATCH.with(|pool| {
            pool.with_scratch(len * 2, |input_lanes| {
                for (i, &val) in log_samples.iter().enumerate() {
                    input_lanes[2 * i] = val;
                    input_lanes[2 * i + 1] = 0.0;
                }
                moirai::map_collect_index_with::<moirai::Adaptive, _, _>(len, |k| {
                    log_frequency_coeff_hermes(input_lanes, factor, du, k)
                })
            })
        });
    }

    let dft_coeff = |k: usize| -> Complex64 {
        du * log_samples
            .iter()
            .enumerate()
            .map(|(n, sample)| Complex64::from_polar(*sample, factor * k as f64 * n as f64))
            .sum::<Complex64>()
    };
    if len >= PAR_THRESHOLD {
        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(len, dft_coeff)
    } else {
        (0..len).map(dft_coeff).collect()
    }
}

fn log_frequency_coeff_hermes(input_lanes: &[f64], factor: f64, scale: f64, k: usize) -> Complex64 {
    LOG_FREQUENCY_WEIGHT_LANE_SCRATCH.with(|pool| {
        pool.with_scratch(input_lanes.len(), |weight_lanes| {
            fill_log_frequency_weight_lanes(weight_lanes, factor, k);
            let (re, im) = hermes_simd::interleaved_complex_dot_runtime::<f64, false>(
                input_lanes,
                weight_lanes,
            )
            .expect("Mellin forward spectrum Hermes dot uses equal-length interleaved lanes");
            Complex64::new(re * scale, im * scale)
        })
    })
}

fn moment_antiderivative(min: f64, max: f64, exponent: f64) -> f64 {
    if exponent.abs() < f64::EPSILON {
        max.ln() - min.ln()
    } else {
        (max.powf(exponent) - min.powf(exponent)) / exponent
    }
}

/// Compute the inverse Mellin spectrum: recover log-domain samples from a
/// log-frequency spectrum via IDFT.
///
/// # Mathematical basis
///
/// The forward spectrum is `F[k] = du · Σ_n g[n] exp(−2πi·kn/N)`.
/// Dividing by `du` and applying the standard IDFT recovers
/// `g[n] = (1/N) · Σ_k (F[k]/du) · exp(2πi·kn/N)`.
///
/// The result `g[n]` represents `f(exp(u_n))` where
/// `u_n = log_min + n·du` and `du = (log_max − log_min) / (N−1)`.
#[must_use]
pub fn inverse_log_frequency_spectrum(
    spectrum: &[Complex64],
    log_min: f64,
    log_max: f64,
) -> Vec<f64> {
    let len = spectrum.len();
    if len == 0 {
        return Vec::new();
    }
    let du = if len > 1 {
        (log_max - log_min) / (len as f64 - 1.0)
    } else {
        1.0
    };

    // Divide spectrum by du to undo the du scaling from the forward DFT.
    let inv_du = if du.abs() > f64::EPSILON {
        1.0 / du
    } else {
        1.0
    };
    let factor = std::f64::consts::TAU / len as f64;
    let work_items = len.saturating_mul(len);
    if work_items >= HERMES_SPECTRUM_OP_THRESHOLD {
        let spectrum_lanes = complex_interleaved_lanes(spectrum);
        return moirai::map_collect_index_with::<moirai::Adaptive, _, _>(len, |n| {
            inverse_log_frequency_coeff_hermes(spectrum_lanes, factor, inv_du / len as f64, n)
        });
    }

    let idft_coeff = |n: usize| -> f64 {
        let re_sum: f64 = spectrum
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let angle = factor * k as f64 * n as f64;
                s.re * angle.cos() - s.im * angle.sin()
            })
            .sum();
        re_sum * inv_du / len as f64
    };

    if len >= PAR_THRESHOLD {
        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(len, idft_coeff)
    } else {
        (0..len).map(idft_coeff).collect()
    }
}

fn inverse_log_frequency_coeff_hermes(
    spectrum_lanes: &[f64],
    factor: f64,
    scale: f64,
    n: usize,
) -> f64 {
    LOG_FREQUENCY_WEIGHT_LANE_SCRATCH.with(|pool| {
        pool.with_scratch(spectrum_lanes.len(), |weight_lanes| {
            fill_log_frequency_weight_lanes(weight_lanes, factor, n);
            let (re, _) = hermes_simd::interleaved_complex_dot_runtime::<f64, false>(
                spectrum_lanes,
                weight_lanes,
            )
            .expect("Mellin inverse spectrum Hermes dot uses equal-length interleaved lanes");
            re * scale
        })
    })
}

#[cfg(test)]
fn real_interleaved_lanes(values: &[f64]) -> Vec<f64> {
    let mut lanes = Vec::with_capacity(values.len() * 2);
    for value in values {
        lanes.push(*value);
        lanes.push(0.0);
    }
    lanes
}

#[inline]
fn complex_interleaved_lanes(values: &[Complex64]) -> &[f64] {
    // SAFETY: Complex64 is #[repr(C)] and has the same layout and alignment as [f64; 2].
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast::<f64>(), values.len() * 2) }
}

fn fill_log_frequency_weight_lanes(lanes: &mut [f64], factor: f64, row: usize) {
    for (index, lane_pair) in lanes.chunks_exact_mut(2).enumerate() {
        let angle = factor * row as f64 * index as f64;
        lane_pair[0] = angle.cos();
        lane_pair[1] = angle.sin();
    }
}

/// Interpolate a log-domain signal back onto a linear-scale output grid.
///
/// Given `log_samples[n]` representing `f(exp(u_n))` where
/// `u_n = log_min + n·du`, this function evaluates `f` at linearly spaced
/// points in `[output_min, output_max]` via linear interpolation in the
/// log-domain.
pub fn exp_resample(
    log_samples: &[f64],
    log_min: f64,
    log_max: f64,
    output: &mut [f64],
    output_min: f64,
    output_max: f64,
) {
    let n = log_samples.len();
    let out_len = output.len();
    if n == 0 || out_len == 0 {
        output.fill(0.0);
        return;
    }

    let du = if n > 1 {
        (log_max - log_min) / (n as f64 - 1.0)
    } else {
        0.0
    };
    let out_step = if out_len > 1 {
        (output_max - output_min) / (out_len as f64 - 1.0)
    } else {
        0.0
    };

    let eval = |i: usize| -> f64 {
        let r = output_min + i as f64 * out_step;
        if r <= 0.0 {
            return 0.0;
        }
        let u = r.ln();
        if u < log_min || u > log_max || du.abs() < f64::EPSILON {
            return 0.0;
        }
        let exact_idx = (u - log_min) / du;
        let lower = exact_idx.floor() as usize;
        let upper = (lower + 1).min(n - 1);
        let frac = exact_idx - lower as f64;
        log_samples[lower] * (1.0 - frac) + log_samples[upper] * frac
    };

    if out_len >= PAR_THRESHOLD {
        output.par_mut().enumerate(|i, v| {
            *v = eval(i);
        });
    } else {
        output.iter_mut().enumerate().for_each(|(i, v)| {
            *v = eval(i);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn hermes_mellin_moment_matches_scalar_formula_at_threshold() {
        let len = HERMES_MOMENT_LEN_THRESHOLD;
        let signal_min = 1.0_f64;
        let signal_max = 5.0_f64;
        let exponent = 1.75_f64;
        let step = (signal_max - signal_min) / (len as f64 - 1.0);
        let signal = (0..len)
            .map(|index| {
                let coordinate = signal_min + index as f64 * step;
                coordinate.ln().sin() + 2.0
            })
            .collect::<Vec<_>>();

        let actual = mellin_moment_hermes(&signal, signal_min, exponent, step);
        let expected = mellin_moment_scalar(&signal, signal_min, exponent, step);

        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-9);
    }

    #[test]
    fn moment_weights_match_trapezoid_formula() {
        let len = HERMES_MOMENT_LEN_THRESHOLD;
        let signal_min = 0.5_f64;
        let step = 0.001_f64;
        let exponent = 2.25_f64;
        let mut weights = vec![0.0; len];

        fill_moment_weights(&mut weights, signal_min, exponent, step);

        for &index in &[0usize, 1, 257, len - 2, len - 1] {
            let coordinate = signal_min + index as f64 * step;
            let trapezoid = if index == 0 || index + 1 == len {
                0.5
            } else {
                1.0
            };
            let expected = trapezoid * coordinate.powf(exponent - 1.0);
            assert_eq!(weights[index].to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn hermes_log_frequency_rows_match_scalar_formulas_at_threshold() {
        let len = PAR_THRESHOLD;
        let log_min = -0.25_f64;
        let log_max = 1.75_f64;
        let du = (log_max - log_min) / (len as f64 - 1.0);
        let factor = -std::f64::consts::TAU / len as f64;
        let samples = (0..len)
            .map(|index| {
                let x = log_min + index as f64 * du;
                x.sin() + (2.0 * x).cos()
            })
            .collect::<Vec<_>>();
        let input_lanes = real_interleaved_lanes(&samples);

        for k in [0usize, 1, 17, 64, 127, 255] {
            let actual = log_frequency_coeff_hermes(&input_lanes, factor, du, k);
            let expected = du
                * samples
                    .iter()
                    .enumerate()
                    .map(|(n, sample)| Complex64::from_polar(*sample, factor * k as f64 * n as f64))
                    .sum::<Complex64>();

            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-10);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-10);
        }
    }

    #[test]
    fn hermes_inverse_log_frequency_rows_match_scalar_formulas_at_threshold() {
        let len = PAR_THRESHOLD;
        let log_min = -0.25_f64;
        let log_max = 1.75_f64;
        let du = (log_max - log_min) / (len as f64 - 1.0);
        let inv_du = 1.0 / du;
        let factor = std::f64::consts::TAU / len as f64;
        let spectrum = (0..len)
            .map(|index| {
                Complex64::new(
                    (index as f64 * 0.03125).sin(),
                    (index as f64 * 0.015625).cos(),
                )
            })
            .collect::<Vec<_>>();
        let spectrum_lanes = complex_interleaved_lanes(&spectrum);

        for n in [0usize, 1, 17, 64, 127, 255] {
            let actual =
                inverse_log_frequency_coeff_hermes(spectrum_lanes, factor, inv_du / len as f64, n);
            let expected = spectrum
                .iter()
                .enumerate()
                .map(|(k, s)| {
                    let angle = factor * k as f64 * n as f64;
                    s.re * angle.cos() - s.im * angle.sin()
                })
                .sum::<f64>()
                * inv_du
                / len as f64;

            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-10);
        }
    }

    #[test]
    fn log_frequency_weight_lanes_match_trigonometric_formula() {
        let len = PAR_THRESHOLD * 2;
        let factor = -std::f64::consts::TAU / PAR_THRESHOLD as f64;
        let row = 17usize;
        let mut lanes = vec![0.0; len];

        fill_log_frequency_weight_lanes(&mut lanes, factor, row);

        for &index in &[0usize, 1, 17, 64, 127, 255] {
            let angle = factor * row as f64 * index as f64;
            assert_eq!(lanes[index * 2].to_bits(), angle.cos().to_bits());
            assert_eq!(lanes[index * 2 + 1].to_bits(), angle.sin().to_bits());
        }
    }
}
