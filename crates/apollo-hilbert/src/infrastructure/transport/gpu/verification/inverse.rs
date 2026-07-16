//! Value-semantic Hilbert GPU inverse-projection contracts.

use crate::HilbertPlan;

use super::support::backend;

#[test]
fn inverse_roundtrip_recovers_dc_nyquist_projection_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let original = vec![1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
    let plan = backend.plan(original.len());
    let quadrature = backend
        .execute_forward(&plan, &original)
        .expect("forward Hilbert");
    let recovered = backend
        .execute_inverse(&plan, &quadrature)
        .expect("inverse Hilbert");
    assert_eq!(recovered.len(), original.len());

    // The Hilbert transform loses DC and, for even lengths, Nyquist. The
    // inverse recovers the original signal after both contributions are removed.
    let len = original.len();
    let dc = original.iter().map(|&value| f64::from(value)).sum::<f64>() / len as f64;
    let tau = std::f64::consts::TAU;
    let mut nyquist_real = 0.0_f64;
    for index in 0..len {
        let angle = -tau * (len as f64) / 2.0 * (index as f64) / (len as f64);
        nyquist_real += f64::from(original[index]) * angle.cos();
    }
    let nyquist_component = |index: usize| -> f64 {
        if len % 2 == 0 {
            nyquist_real * if index % 2 == 0 { 1.0 } else { -1.0 } / len as f64
        } else {
            0.0
        }
    };
    let expected: Vec<f64> = original
        .iter()
        .enumerate()
        .map(|(index, &value)| f64::from(value) - dc - nyquist_component(index))
        .collect();
    const TOLERANCE: f64 = 5.0e-3;
    for (index, (actual, expected)) in recovered.iter().zip(expected.iter()).enumerate() {
        let error = (f64::from(*actual) - *expected).abs();
        assert!(
            error < TOLERANCE,
            "roundtrip mismatch at index {index}: actual={actual}, expected={expected}, error={error}"
        );
    }
}

#[test]
fn inverse_matches_cpu_frequency_domain_reference_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let original = [1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
    let original_reference: Vec<f64> = original.iter().map(|&value| f64::from(value)).collect();
    let plan = backend.plan(original.len());

    let cpu_plan = HilbertPlan::new(original.len()).expect("CPU plan");
    let cpu_quadrature = cpu_plan
        .transform(&original_reference)
        .expect("CPU forward Hilbert");
    let accelerated_quadrature: Vec<f32> =
        cpu_quadrature.iter().map(|&value| value as f32).collect();
    let actual = backend
        .execute_inverse(&plan, &accelerated_quadrature)
        .expect("inverse Hilbert");

    let len = cpu_quadrature.len();
    let positive_end = (len + 1) / 2;
    let tau = std::f64::consts::TAU;
    let mut quadrature_spectrum = vec![eunomia::Complex64::new(0.0, 0.0); len];
    for frequency in 0..len {
        let mut real = 0.0_f64;
        let mut imaginary = 0.0_f64;
        for sample in 0..len {
            let angle = -tau * (frequency as f64) * (sample as f64) / (len as f64);
            real += cpu_quadrature[sample] * angle.cos();
            imaginary += cpu_quadrature[sample] * angle.sin();
        }
        quadrature_spectrum[frequency] = eunomia::Complex64::new(real, imaginary);
    }
    let mut signal_spectrum = vec![eunomia::Complex64::new(0.0, 0.0); len];
    for frequency in 0..len {
        signal_spectrum[frequency] = if frequency == 0 || (len % 2 == 0 && frequency == len / 2) {
            eunomia::Complex64::new(0.0, 0.0)
        } else if frequency < positive_end {
            eunomia::Complex64::new(
                -quadrature_spectrum[frequency].im,
                quadrature_spectrum[frequency].re,
            )
        } else {
            eunomia::Complex64::new(
                quadrature_spectrum[frequency].im,
                -quadrature_spectrum[frequency].re,
            )
        };
    }
    let mut expected = vec![0.0_f64; len];
    for sample in 0..len {
        let mut real = 0.0_f64;
        for frequency in 0..len {
            let angle = tau * (frequency as f64) * (sample as f64) / (len as f64);
            real += signal_spectrum[frequency].re * angle.cos()
                - signal_spectrum[frequency].im * angle.sin();
        }
        expected[sample] = real / len as f64;
    }

    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let error = (f64::from(*actual) - *expected).abs();
        assert!(
            error < 5.0e-3,
            "inverse mismatch at index {index}: GPU={actual}, CPU={expected}, error={error}"
        );
    }
}
