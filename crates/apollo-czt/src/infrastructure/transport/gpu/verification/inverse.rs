//! Value-semantic CZT GPU inverse contracts.

use crate::infrastructure::transport::gpu::WgpuError;
use eunomia::Complex32;

use super::support::{backend, dft_input, dft_parameters, DFT_ROUNDTRIP_BOUND};

#[test]
fn inverse_roundtrip_recovers_dft_specialization_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let len = 8;
    let (a, w) = dft_parameters(len);
    let input = dft_input(len);
    let plan = backend.plan(len, len, a, w);
    let spectrum = backend.execute_forward(&plan, &input).expect("GPU forward");
    let recovered = backend
        .execute_inverse(&plan, &spectrum)
        .expect("GPU inverse");
    assert_eq!(recovered.len(), len);
    for (index, (actual, expected)) in recovered.iter().zip(input.iter()).enumerate() {
        let real_error = (actual.re - expected.re).abs();
        let imaginary_error = (actual.im - expected.im).abs();
        assert!(
            real_error < DFT_ROUNDTRIP_BOUND,
            "sample {index} real error: {real_error:.3e}"
        );
        assert!(
            imaginary_error < DFT_ROUNDTRIP_BOUND,
            "sample {index} imaginary error: {imaginary_error:.3e}"
        );
    }
}

#[test]
fn inverse_rejects_non_square_plan_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = backend.plan(4, 6, Complex32::new(1.0, 0.0), Complex32::new(1.0, 0.0));
    let spectrum = vec![Complex32::new(0.0, 0.0); 6];
    assert!(matches!(
        backend.execute_inverse(&plan, &spectrum),
        Err(WgpuError::LengthMismatch { .. })
    ));
}
