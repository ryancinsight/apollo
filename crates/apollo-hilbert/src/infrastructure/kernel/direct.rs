//! Direct frequency-domain Hilbert kernel.
//!
//! The Hilbert transform is computed via the analytic signal mask applied in the
//! frequency domain. The forward and inverse DFT steps delegate to
//! `apollo_fft::FftPlan1D` slice execution, which uses the O(N log N)
//! radix-2/Bluestein strategy, replacing the former private O(N²)
//! direct-summation kernels and eliminating the SSOT violation documented in
//! `apollo-radon::infrastructure::kernel::filter`.

use crate::domain::contracts::error::{HilbertError, HilbertResult};
use apollo_fft::{FftPlan1D, Shape1D};
use moirai::ParallelSliceMut;
use num_complex::Complex64;

thread_local! {
    static QUADRATURE_ANALYTIC_SCRATCH: mnemosyne::scratch::ScratchPool<Complex64> = const { mnemosyne::scratch::ScratchPool::new() };
}

/// Below this length, serial loops avoid parallel scheduling overhead.
const HILBERT_PAR_LEN_THRESHOLD: usize = 16_384;

/// Compute the Hilbert quadrature component of a real signal.
pub fn hilbert_transform(signal: &[f64]) -> HilbertResult<Vec<f64>> {
    let mut output = vec![0.0; signal.len()];
    hilbert_transform_into(signal, &mut output)?;
    Ok(output)
}

/// Compute the Hilbert quadrature component into caller-owned storage.
pub fn hilbert_transform_into(signal: &[f64], output: &mut [f64]) -> HilbertResult<()> {
    if output.len() != signal.len() {
        return Err(HilbertError::LengthMismatch);
    }

    with_quadrature_analytic_workspace(signal.len(), |analytic| {
        analytic_signal_into(signal, analytic)?;
        write_quadrature(signal.len(), analytic, output);
        Ok(())
    })
}

/// Compute the analytic signal `x[n] + i H{x}[n]`.
///
/// # Theorem: Analytic Signal via Frequency-Domain Mask
///
/// For a real signal x ∈ ℝᴺ, the analytic signal z ∈ ℂᴺ is defined by
/// doubling the positive-frequency components of the DFT spectrum and zeroing
/// the negative-frequency components, then applying the IDFT:
///
/// ```text
/// Z[k] = { X[0]        k = 0
///         { 2 X[k]     1 ≤ k < N/2
///         { X[N/2]     k = N/2 (even N only)
///         { 0          N/2 < k < N
/// z[n] = IDFT(Z)[n];  re(z[n]) ← x[n]  (Hartley–Zygmund constraint)
/// ```
///
/// The Hilbert quadrature is `H{x}[n] = im(z[n])`.
///
/// # Proof sketch
///
/// The analytic signal z is the unique complex extension of x whose
/// negative-frequency content vanishes. Doubling positive frequencies preserves
/// the convolution-with-signum interpretation of the Hilbert transform
/// (`H{x} = IDFT(−i·sgn(k)·X[k])`). The DC and Nyquist bins are unscaled so
/// that `re(IDFT(Z)) = x` exactly, and the real-part is then forced from the
/// original input to eliminate rounding accumulation across the FFT/IFFT pair.
///
/// Reference: *The Analytic Signal*, Gabor D., J. IEE, 93(3):429–441, 1946.
///
/// # Complexity
///
/// O(N log N) — one O(N log N) FFT, O(N) mask application, one O(N log N) IFFT.
/// The previous O(N²) direct-summation implementation has been replaced.
pub fn analytic_signal(signal: &[f64]) -> HilbertResult<Vec<Complex64>> {
    if signal.is_empty() {
        return Err(HilbertError::EmptySignal);
    }

    let mut analytic = vec![Complex64::new(0.0, 0.0); signal.len()];
    analytic_signal_into(signal, &mut analytic)?;
    Ok(analytic)
}

/// Compute the analytic signal `x[n] + i H{x}[n]` into caller-owned storage.
pub fn analytic_signal_into(signal: &[f64], output: &mut [Complex64]) -> HilbertResult<()> {
    if output.len() != signal.len() {
        return Err(HilbertError::LengthMismatch);
    }
    if signal.is_empty() {
        return Err(HilbertError::EmptySignal);
    }

    let shape = Shape1D::new(signal.len()).expect("non-empty Hilbert signal length");
    let plan = FftPlan1D::<f64>::new(shape);
    write_real_input(signal, output);
    plan.forward_complex_slice_inplace(output);
    apply_analytic_mask(output);
    plan.inverse_complex_slice_inplace(output);

    // Force the real part to equal the original input to eliminate IFFT rounding.
    restore_original_real(signal, output);
    Ok(())
}

fn with_quadrature_analytic_workspace<R>(
    len: usize,
    f: impl FnOnce(&mut [Complex64]) -> HilbertResult<R>,
) -> HilbertResult<R> {
    QUADRATURE_ANALYTIC_SCRATCH.with(|pool| pool.with_scratch(len, f))
}

#[cfg(test)]
fn quadrature_analytic_workspace_capacity() -> usize {
    QUADRATURE_ANALYTIC_SCRATCH.with(|pool| pool.capacity())
}

fn apply_analytic_mask(spectrum: &mut [Complex64]) {
    let len = spectrum.len();
    let positive_end = (len + 1) / 2;
    if len >= HILBERT_PAR_LEN_THRESHOLD {
        apply_analytic_mask_hermes(spectrum, positive_end);
    } else {
        spectrum.iter_mut().enumerate().for_each(|(k, value)| {
            *value *= analytic_mask_scale(len, positive_end, k);
        });
    }
}

fn apply_analytic_mask_hermes(spectrum: &mut [Complex64], positive_end: usize) {
    let len = spectrum.len();
    if positive_end > 1 {
        hermes_simd::scale(complex_lanes_mut(&mut spectrum[1..positive_end]), 2.0);
    }

    let negative_start = if len.is_multiple_of(2) {
        len / 2 + 1
    } else {
        positive_end
    };
    if negative_start < len {
        hermes_simd::scale(complex_lanes_mut(&mut spectrum[negative_start..]), 0.0);
    }
}

fn complex_lanes_mut(values: &mut [Complex64]) -> &mut [f64] {
    let scalar_len = values.len() * 2;
    // SAFETY: num_complex::Complex64 stores adjacent `re, im` f64 lanes; this
    // helper preserves the slice lifetime and does not change alignment.
    unsafe { core::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<f64>(), scalar_len) }
}

fn analytic_mask_scale(len: usize, positive_end: usize, k: usize) -> f64 {
    if k == 0 || (len % 2 == 0 && k == len / 2) {
        1.0
    } else if k < positive_end {
        2.0
    } else {
        0.0
    }
}

fn write_real_input(signal: &[f64], output: &mut [Complex64]) {
    if signal.len() >= HILBERT_PAR_LEN_THRESHOLD {
        output.par_mut().enumerate(|index, value| {
            *value = Complex64::new(signal[index], 0.0);
        });
    } else {
        output.iter_mut().zip(signal.iter()).for_each(|(dst, src)| {
            *dst = Complex64::new(*src, 0.0);
        });
    }
}

fn restore_original_real(signal: &[f64], output: &mut [Complex64]) {
    if signal.len() >= HILBERT_PAR_LEN_THRESHOLD {
        output.par_mut().enumerate(|index, value| {
            value.re = signal[index];
        });
    } else {
        output
            .iter_mut()
            .zip(signal.iter())
            .for_each(|(sample, original)| {
                sample.re = *original;
            });
    }
}

fn write_quadrature(len: usize, analytic: &[Complex64], output: &mut [f64]) {
    if len >= HILBERT_PAR_LEN_THRESHOLD {
        output.par_mut().enumerate(|index, value| {
            *value = analytic[index].im;
        });
    } else {
        output
            .iter_mut()
            .zip(analytic.iter())
            .for_each(|(slot, value)| {
                *slot = value.im;
            });
    }
}

#[cfg(test)]
fn analytic_mask_for_test(spectrum: &mut [Complex64]) {
    apply_analytic_mask(spectrum);
}

#[cfg(test)]
fn write_real_input_for_test(signal: &[f64], output: &mut [Complex64]) {
    write_real_input(signal, output);
}

#[cfg(test)]
fn restore_original_real_for_test(signal: &[f64], output: &mut [Complex64]) {
    restore_original_real(signal, output);
}

#[cfg(test)]
fn write_quadrature_for_test(analytic: &[Complex64], output: &mut [f64]) {
    write_quadrature(analytic.len(), analytic, output);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn hilbert_transform_into_matches_owned_quadrature() {
        let len = 16;
        let signal: Vec<f64> = (0..len)
            .map(|n| (std::f64::consts::TAU * n as f64 / len as f64).cos())
            .collect();
        let expected = hilbert_transform(&signal).expect("owned quadrature");
        let mut output = vec![f64::NAN; len];

        hilbert_transform_into(&signal, &mut output).expect("caller-owned quadrature");

        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn analytic_signal_into_matches_owned_analytic_signal() {
        let len = 16;
        let signal: Vec<f64> = (0..len)
            .map(|n| (std::f64::consts::TAU * n as f64 / len as f64).cos())
            .collect();
        let expected = analytic_signal(&signal).expect("owned analytic");
        let mut output = vec![Complex64::new(f64::NAN, f64::NAN); len];

        analytic_signal_into(&signal, &mut output).expect("caller-owned analytic");

        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn hilbert_transform_into_reuses_analytic_workspace_capacity() {
        let len = 16;
        let signal: Vec<f64> = (0..len)
            .map(|n| (std::f64::consts::TAU * n as f64 / len as f64).cos())
            .collect();
        let mut first = vec![0.0; len];
        let mut second = vec![0.0; len];

        hilbert_transform_into(&signal, &mut first).expect("first caller-owned quadrature");
        let after_first = quadrature_analytic_workspace_capacity();
        assert!(after_first >= len);

        hilbert_transform_into(&signal, &mut second).expect("second caller-owned quadrature");
        assert_eq!(quadrature_analytic_workspace_capacity(), after_first);

        for (actual, expected) in second.iter().zip(first.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn moirai_parallel_helpers_match_serial_formulas_at_threshold() {
        let len = HILBERT_PAR_LEN_THRESHOLD;
        let signal: Vec<f64> = (0..len)
            .map(|n| (std::f64::consts::TAU * n as f64 / len as f64).sin())
            .collect();
        let mut complex = vec![Complex64::new(f64::NAN, f64::NAN); len];

        write_real_input_for_test(&signal, &mut complex);
        for (index, value) in complex.iter().enumerate() {
            assert_eq!(value.re.to_bits(), signal[index].to_bits());
            assert_eq!(value.im.to_bits(), 0.0_f64.to_bits());
        }

        analytic_mask_for_test(&mut complex);
        for (index, value) in complex.iter().enumerate() {
            let scale = analytic_mask_scale(len, (len + 1) / 2, index);
            assert_eq!(value.re.to_bits(), (signal[index] * scale).to_bits());
            assert_eq!(value.im.to_bits(), 0.0_f64.to_bits());
        }

        let replacement: Vec<f64> = (0..len)
            .map(|n| (std::f64::consts::TAU * n as f64 / len as f64).cos())
            .collect();
        restore_original_real_for_test(&replacement, &mut complex);
        for (index, value) in complex.iter().enumerate() {
            assert_eq!(value.re.to_bits(), replacement[index].to_bits());
        }

        let mut quadrature = vec![f64::NAN; len];
        write_quadrature_for_test(&complex, &mut quadrature);
        for (actual, expected) in quadrature.iter().zip(complex.iter()) {
            assert_eq!(actual.to_bits(), expected.im.to_bits());
        }
    }

    #[test]
    fn hermes_mask_matches_scalar_formula_for_odd_threshold_length() {
        let len = HILBERT_PAR_LEN_THRESHOLD + 1;
        let mut spectrum: Vec<Complex64> = (0..len)
            .map(|index| Complex64::new(index as f64 * 0.25, -(index as f64) * 0.125))
            .collect();
        let original = spectrum.clone();

        analytic_mask_for_test(&mut spectrum);

        let positive_end = (len + 1) / 2;
        for (index, actual) in spectrum.iter().enumerate() {
            let scale = analytic_mask_scale(len, positive_end, index);
            assert_eq!(actual.re.to_bits(), (original[index].re * scale).to_bits());
            assert_eq!(actual.im.to_bits(), (original[index].im * scale).to_bits());
        }
    }

    #[test]
    fn analytic_signal_into_rejects_output_length_mismatch() {
        let signal = [1.0, 0.0, -1.0, 0.0];
        let mut output = [Complex64::new(0.0, 0.0); 3];

        assert!(matches!(
            analytic_signal_into(&signal, &mut output),
            Err(HilbertError::LengthMismatch)
        ));
    }

    #[test]
    fn hilbert_transform_into_rejects_output_length_mismatch() {
        let signal = [1.0, 0.0, -1.0, 0.0];
        let mut output = [0.0; 3];

        assert!(matches!(
            hilbert_transform_into(&signal, &mut output),
            Err(HilbertError::LengthMismatch)
        ));
    }
}
