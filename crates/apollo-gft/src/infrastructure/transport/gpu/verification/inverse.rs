//! Value-semantic GFT GPU inverse and reconstruction contracts.

use super::support::{backend, path4_plan_and_basis, PATH4_F32_DOT_ABS_TOLERANCE};

#[test]
fn inverse_matches_cpu_reference_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let (cpu_plan, basis, signal) = path4_plan_and_basis();
    let signal = leto::Array1::from(
        signal
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>(),
    );
    let cpu_spectrum = cpu_plan.forward(&signal).expect("CPU spectrum");
    let spectrum: Vec<f32> = cpu_spectrum.iter().map(|&value| value as f32).collect();
    let plan = crate::infrastructure::transport::gpu::GftWgpuPlan::new(4);
    let actual = backend
        .execute_inverse(&plan, &spectrum, &basis)
        .expect("GFT inverse");
    let expected = cpu_plan.inverse(&cpu_spectrum).expect("CPU inverse");

    assert_eq!(actual.len(), 4);
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (f64::from(*actual) - expected).abs() < PATH4_F32_DOT_ABS_TOLERANCE,
            "inverse {index}: GPU={actual}, CPU={expected}"
        );
    }
}

#[test]
fn roundtrip_recovers_path_signal_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let (_cpu_plan, basis, signal) = path4_plan_and_basis();
    let plan = crate::infrastructure::transport::gpu::GftWgpuPlan::new(4);
    let spectrum = backend
        .execute_forward(&plan, &signal, &basis)
        .expect("roundtrip forward");
    let recovered = backend
        .execute_inverse(&plan, &spectrum, &basis)
        .expect("roundtrip inverse");
    assert_eq!(recovered.len(), 4);
    for (index, (actual, expected)) in recovered.iter().zip(signal.iter()).enumerate() {
        assert!(
            f64::from((actual - expected).abs()) < PATH4_F32_DOT_ABS_TOLERANCE,
            "roundtrip {index}: got={actual}, expected={expected}"
        );
    }
}

#[test]
fn caller_owned_forward_and_inverse_match_allocating_api_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let (_cpu_plan, basis, signal) = path4_plan_and_basis();
    let plan = crate::infrastructure::transport::gpu::GftWgpuPlan::new(4);
    let expected_forward = backend
        .execute_forward(&plan, &signal, &basis)
        .expect("allocating forward");
    let mut actual_forward = vec![0.0_f32; signal.len()];
    backend
        .execute_forward_into(&plan, &signal, &basis, &mut actual_forward)
        .expect("caller-owned forward");
    assert_eq!(actual_forward, expected_forward);

    let expected_inverse = backend
        .execute_inverse(&plan, &expected_forward, &basis)
        .expect("allocating inverse");
    let mut actual_inverse = vec![0.0_f32; signal.len()];
    backend
        .execute_inverse_into(&plan, &actual_forward, &basis, &mut actual_inverse)
        .expect("caller-owned inverse");
    assert_eq!(actual_inverse, expected_inverse);
}
