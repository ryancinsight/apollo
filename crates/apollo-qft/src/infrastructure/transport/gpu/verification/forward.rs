use crate::{QftPlan, QuantumStateDimension};
use eunomia::Complex64;
use leto::Array1;

use super::support::{assert_matches_cpu, backend, forward_input};

#[test]
fn forward_matches_cpu_reference_when_device_exists() {
    let Ok(backend) = backend() else {
        return;
    };
    let input = forward_input();
    let plan = backend.plan(input.len());
    let gpu = backend
        .execute_forward(&plan, &input)
        .expect("wgpu forward execution");

    let cpu_plan = QftPlan::new(QuantumStateDimension::new(input.len()).expect("dimension"));
    let cpu_input = Array1::from(
        input
            .iter()
            .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
            .collect::<Vec<_>>(),
    );
    let cpu = cpu_plan.forward(&cpu_input).expect("cpu forward");

    assert_matches_cpu(&gpu, &cpu, "forward");
}
