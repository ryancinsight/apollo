//! Shared real-device acquisition and numerical assertions for SFT verification.

use crate::infrastructure::transport::gpu::SftWgpuBackend;
use eunomia::{Complex32, Complex64};

// Four-term `f32` inverse evaluation is bounded by twice gamma_12 times the
// largest retained magnitude (1/3); gamma_12 = 12u / (1 - 12u).
pub(super) const INVERSE_N0_ERROR_BOUND: f32 = {
    let unit_roundoff = f32::EPSILON / 2.0;
    let gamma_12 = (12.0 * unit_roundoff) / (1.0 - 12.0 * unit_roundoff);
    2.0 * gamma_12 / 3.0
};

pub(super) fn backend() -> Option<SftWgpuBackend> {
    SftWgpuBackend::try_default().ok()
}

pub(super) fn two_tone_signal(len: usize, tones: &[(usize, f64)]) -> Vec<Complex64> {
    (0..len)
        .map(|n| {
            tones
                .iter()
                .map(|(frequency, amplitude)| {
                    let angle = 2.0 * std::f64::consts::PI * (*frequency as f64) * (n as f64)
                        / (len as f64);
                    Complex64::new(amplitude * angle.cos(), amplitude * angle.sin())
                })
                .sum()
        })
        .collect()
}

pub(super) fn represented_signal(signal: &[Complex64]) -> Vec<Complex32> {
    signal
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect()
}

pub(super) fn assert_reference_complex_close(
    actual: Complex64,
    expected: Complex64,
    tolerance: f64,
) {
    assert!(
        (actual.re - expected.re).abs() <= tolerance,
        "real mismatch: actual={actual:?}, expected={expected:?}"
    );
    assert!(
        (actual.im - expected.im).abs() <= tolerance,
        "imag mismatch: actual={actual:?}, expected={expected:?}"
    );
}

pub(super) fn assert_accelerated_complex_close(
    actual: Complex32,
    expected: Complex32,
    tolerance: f32,
) {
    assert!(
        (actual.re - expected.re).abs() <= tolerance,
        "real mismatch: actual={actual:?}, expected={expected:?}"
    );
    assert!(
        (actual.im - expected.im).abs() <= tolerance,
        "imag mismatch: actual={actual:?}, expected={expected:?}"
    );
}
