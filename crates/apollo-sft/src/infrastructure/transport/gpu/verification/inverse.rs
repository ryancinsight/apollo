//! Value-semantic SFT GPU inverse-reconstruction contracts.

use crate::{infrastructure::transport::gpu::SftWgpuPlan, SparseFftPlan};
use eunomia::Complex32;

use super::support::{
    assert_accelerated_complex_close, backend, represented_signal, two_tone_signal,
};
use crate::infrastructure::transport::gpu::{SparseExecution, SparsityPlan};

#[test]
fn inverse_matches_cpu_sparse_reconstruction_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = SftWgpuPlan::new(SparsityPlan::new(8, 2));
    let signal = two_tone_signal(8, &[(1, 3.0), (3, 1.25)]);
    let cpu_plan =
        SparseFftPlan::new(plan.len(), plan.payload().sparsity()).expect("valid CPU plan");
    let spectrum = backend
        .sparse_forward(&plan, &represented_signal(&signal))
        .expect("GPU forward SFT");
    let expected = cpu_plan.inverse(&spectrum).expect("CPU inverse");

    let actual = backend
        .sparse_inverse(&plan, &spectrum)
        .expect("GPU inverse");

    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_accelerated_complex_close(
            *actual,
            Complex32::new(expected.re as f32, expected.im as f32),
            2.0e-4,
        );
    }
}
