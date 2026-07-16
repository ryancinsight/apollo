//! Value-semantic Hilbert GPU analytic and quadrature contracts.

use crate::HilbertPlan;

use super::support::backend;

#[test]
fn analytic_signal_matches_cpu_reference_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let input = vec![1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
    let plan = backend.plan(input.len());
    let gpu = backend
        .execute_analytic_signal(&plan, &input)
        .expect("WGPU analytic execution");

    let cpu_plan = HilbertPlan::new(input.len()).expect("CPU plan");
    let cpu = cpu_plan
        .analytic_signal(
            &input
                .iter()
                .map(|&value| f64::from(value))
                .collect::<Vec<_>>(),
        )
        .expect("CPU analytic");

    assert_eq!(gpu.len(), cpu.values().len());
    for (actual, expected) in gpu.iter().zip(cpu.values().iter()) {
        assert!((f64::from(actual.re) - expected.re).abs() < 5.0e-4);
        assert!((f64::from(actual.im) - expected.im).abs() < 5.0e-4);
    }
}

#[test]
fn quadrature_matches_cpu_reference_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let input = vec![0.5_f32, -1.25, 2.75, 4.0, -3.5, 1.0];
    let plan = backend.plan(input.len());
    let gpu = backend
        .execute_forward(&plan, &input)
        .expect("WGPU transform execution");

    let cpu_plan = HilbertPlan::new(input.len()).expect("CPU plan");
    let cpu = cpu_plan
        .transform(
            &input
                .iter()
                .map(|&value| f64::from(value))
                .collect::<Vec<_>>(),
        )
        .expect("CPU transform");

    assert_eq!(gpu.len(), cpu.len());
    for (actual, expected) in gpu.iter().zip(cpu.iter()) {
        assert!((f64::from(*actual) - *expected).abs() < 5.0e-4);
    }
}
