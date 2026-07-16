use crate::{QftPlan, QuantumStateDimension};
use eunomia::Complex64;
use leto::Array1;

use super::support::{
    assert_matches_cpu, backend, inverse_input, roundtrip_input, ROUNDTRIP_TOLERANCE,
};

#[test]
fn inverse_matches_cpu_reference_when_device_exists() {
    let Ok(backend) = backend() else {
        return;
    };
    let input = inverse_input();
    let plan = backend.plan(input.len());
    let gpu = backend
        .execute_inverse(&plan, &input)
        .expect("wgpu inverse execution");

    let cpu_plan = QftPlan::new(QuantumStateDimension::new(input.len()).expect("dimension"));
    let cpu_input = Array1::from(
        input
            .iter()
            .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
            .collect::<Vec<_>>(),
    );
    let cpu = cpu_plan.inverse(&cpu_input).expect("cpu inverse");

    assert_matches_cpu(&gpu, &cpu, "inverse");
}

#[test]
fn inverse_recovers_forward_input_when_device_exists() {
    let Ok(backend) = backend() else {
        return;
    };
    let input = roundtrip_input();
    let plan = backend.plan(input.len());
    let transformed = backend
        .execute_forward(&plan, &input)
        .expect("wgpu forward execution");
    let recovered = backend
        .execute_inverse(&plan, &transformed)
        .expect("wgpu inverse execution");

    assert_eq!(recovered.len(), input.len());
    for (index, (actual, expected)) in recovered.iter().zip(input.iter()).enumerate() {
        let real_error = (actual.re - expected.re).abs();
        let imag_error = (actual.im - expected.im).abs();
        assert!(
            real_error < ROUNDTRIP_TOLERANCE && imag_error < ROUNDTRIP_TOLERANCE,
            "roundtrip mismatch at index {index}: actual=({},{}) expected=({},{}) real_error={} imag_error={}",
            actual.re,
            actual.im,
            expected.re,
            expected.im,
            real_error,
            imag_error
        );
    }
}
