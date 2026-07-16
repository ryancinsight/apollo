//! Value-semantic CZT GPU forward-execution contracts.

use crate::CztPlan;
use eunomia::Complex64;
use leto::Array1;

use super::support::{backend, reference_input, reference_parameters, DIRECT_DIFFERENTIAL_BOUND};

#[test]
fn forward_matches_cpu_direct_reference_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let (a, w) = reference_parameters();
    let input = reference_input();
    let gpu_plan = backend.plan(input.len(), 6, a, w);
    let gpu = backend
        .execute_forward(&gpu_plan, &input)
        .expect("WGPU forward execution");

    let cpu_plan = CztPlan::new(
        input.len(),
        6,
        Complex64::new(f64::from(a.re), f64::from(a.im)),
        Complex64::new(f64::from(w.re), f64::from(w.im)),
    )
    .expect("CPU plan");
    let cpu_values = input
        .iter()
        .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
        .collect::<Vec<_>>();
    let cpu_input = Array1::from_shape_vec([cpu_values.len()], cpu_values).expect("CPU input");
    let cpu = cpu_plan.forward_direct(&cpu_input).expect("CPU direct");

    assert_eq!(gpu.len(), cpu.size());
    for (actual, expected) in gpu.iter().zip(cpu.iter()) {
        assert!((f64::from(actual.re) - expected.re).abs() < DIRECT_DIFFERENTIAL_BOUND);
        assert!((f64::from(actual.im) - expected.im).abs() < DIRECT_DIFFERENTIAL_BOUND);
    }
}

#[test]
fn unit_impulse_is_constant_on_the_spiral_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let (a, w) = reference_parameters();
    let input = [
        eunomia::Complex32::new(1.0, 0.0),
        eunomia::Complex32::new(0.0, 0.0),
        eunomia::Complex32::new(0.0, 0.0),
        eunomia::Complex32::new(0.0, 0.0),
    ];
    let plan = backend.plan(input.len(), 7, a, w);
    let actual = backend
        .execute_forward(&plan, &input)
        .expect("impulse forward");
    assert_eq!(actual, vec![eunomia::Complex32::new(1.0, 0.0); 7]);
}
