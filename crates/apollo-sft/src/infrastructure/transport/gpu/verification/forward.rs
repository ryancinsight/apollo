//! Value-semantic SFT GPU forward contracts.

use crate::{infrastructure::transport::gpu::SftWgpuPlan, SparseFftPlan};

use super::support::{
    assert_reference_complex_close, backend, represented_signal, two_tone_signal,
};
use crate::infrastructure::transport::gpu::{SparseExecution, SparsityPlan};

#[test]
fn forward_matches_cpu_sparse_support_and_coefficients_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = SftWgpuPlan::new(SparsityPlan::new(8, 2));
    let signal = two_tone_signal(8, &[(1, 3.0), (3, 1.25)]);
    let represented_signal = represented_signal(&signal);

    let cpu = SparseFftPlan::new(plan.len(), plan.payload().sparsity())
        .expect("valid CPU plan")
        .forward(&signal)
        .expect("CPU SFT");
    let gpu = backend
        .sparse_forward(&plan, &represented_signal)
        .expect("GPU SFT");

    assert_eq!(gpu.frequencies, cpu.frequencies);
    assert_eq!(gpu.values.len(), cpu.values.len());
    for (actual, expected) in gpu.values.iter().zip(cpu.values.iter()) {
        assert_reference_complex_close(*actual, *expected, 2.0e-4);
    }
}
