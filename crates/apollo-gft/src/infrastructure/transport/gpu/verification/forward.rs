//! Value-semantic GFT GPU forward-execution contracts.

use super::support::{backend, path4_plan_and_basis, PATH4_F32_DOT_ABS_TOLERANCE};

#[test]
fn forward_matches_cpu_reference_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let (cpu_plan, basis, signal) = path4_plan_and_basis();
    let plan = crate::infrastructure::transport::gpu::GftWgpuPlan::new(4);
    let actual = backend
        .execute_forward(&plan, &signal, &basis)
        .expect("GFT forward");
    let signal = leto::Array1::from(
        signal
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>(),
    );
    let expected = cpu_plan.forward(&signal).expect("CPU GFT forward");

    assert_eq!(actual.len(), 4);
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (f64::from(*actual) - expected).abs() < PATH4_F32_DOT_ABS_TOLERANCE,
            "forward {index}: GPU={actual}, CPU={expected}"
        );
    }
}
