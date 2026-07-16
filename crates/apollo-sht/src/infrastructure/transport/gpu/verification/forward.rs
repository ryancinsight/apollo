use crate::ShtPlan;
use eunomia::Complex64;

use super::support::{assert_complex_close, backend, complex_samples, represented_samples};

#[test]
fn forward_matches_cpu_complex_coefficients_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = crate::infrastructure::transport::gpu::ShtWgpuPlan::new(4, 5, 1);
    let cpu_plan = ShtPlan::new(plan.latitudes(), plan.longitudes(), plan.max_degree())
        .expect("valid CPU SHT plan");
    let samples = complex_samples(&plan);
    let accelerator_samples = represented_samples(&samples);
    let represented =
        accelerator_samples.mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im)));
    let expected = cpu_plan
        .forward_complex(&represented)
        .expect("CPU forward represented samples");
    let actual = backend
        .execute_forward(&plan, &accelerator_samples)
        .expect("GPU forward");
    for degree in 0..=plan.max_degree() {
        for order in -(degree as isize)..=(degree as isize) {
            assert_complex_close(actual.get(degree, order), expected.get(degree, order));
        }
    }
}
