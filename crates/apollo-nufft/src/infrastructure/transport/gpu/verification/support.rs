//! Shared device availability and assertion contracts for NUFFT verification.

use eunomia::Complex64;

use crate::infrastructure::transport::gpu::{NufftWgpuBackend, NufftWgpuError};

pub(super) fn backend() -> Option<NufftWgpuBackend> {
    NufftWgpuBackend::try_default().ok()
}

pub(super) fn assert_complex64_close(actual: Complex64, expected: Complex64, tolerance: f64) {
    assert!(
        (actual.re - expected.re).abs() <= tolerance,
        "real mismatch: actual={actual:?}, expected={expected:?}"
    );
    assert!(
        (actual.im - expected.im).abs() <= tolerance,
        "imag mismatch: actual={actual:?}, expected={expected:?}"
    );
}

pub(super) fn assert_input_length_mismatch(error: NufftWgpuError, expected: usize, actual: usize) {
    match error {
        NufftWgpuError::InputLengthMismatch {
            expected: actual_expected,
            actual: actual_actual,
        } => {
            assert_eq!(actual_expected, expected);
            assert_eq!(actual_actual, actual);
        }
        other => panic!("expected input-length mismatch, received {other:?}"),
    }
}

pub(super) fn assert_invalid_plan(error: NufftWgpuError, expected_message: &'static str) {
    match error {
        NufftWgpuError::InvalidPlan { message } => assert_eq!(message, expected_message),
        other => panic!("expected invalid plan, received {other:?}"),
    }
}
