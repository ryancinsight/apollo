use crate::ShtPlan;

use super::support::{assert_complex_close, backend, complex_samples, represented_samples};

#[test]
fn inverse_matches_cpu_complex_samples_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = crate::infrastructure::transport::gpu::ShtWgpuPlan::new(4, 5, 1);
    let cpu_plan = ShtPlan::new(plan.latitudes(), plan.longitudes(), plan.max_degree())
        .expect("valid CPU SHT plan");
    let represented = represented_samples(&complex_samples(&plan));
    let coefficients = backend
        .execute_forward(&plan, &represented)
        .expect("GPU forward coefficients");
    let expected = cpu_plan
        .inverse_complex(&coefficients)
        .expect("CPU inverse GPU coefficients");
    let actual = backend
        .execute_inverse(&plan, &coefficients)
        .expect("GPU inverse");
    assert_eq!(actual.shape(), expected.shape());
    for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {
        assert_complex_close(*actual_value, *expected_value);
    }
}
