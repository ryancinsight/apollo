use crate::{QftPlan, QuantumStateDimension};

use super::support::{assert_matches_cpu, backend, cpu_input, forward_input};

#[test]
fn forward_matches_cpu_reference_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let input = forward_input();
    let plan = backend.plan(input.len());
    let gpu = backend
        .execute_forward(&plan, &input)
        .expect("wgpu forward execution");

    let cpu_plan = QftPlan::new(QuantumStateDimension::new(input.len()).expect("dimension"));
    let cpu_input = cpu_input(&input);
    let cpu = cpu_plan.forward(&cpu_input).expect("cpu forward");

    assert_matches_cpu(&gpu, &cpu, "forward");
}
