use crate::{QftPlan, QuantumStateDimension};

use super::support::{
    assert_matches_cpu, backend, cpu_input, inverse_input, roundtrip_input, ROUNDTRIP_TOLERANCE,
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
    let cpu_input = cpu_input(&input);
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
