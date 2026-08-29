//! Rejection-before-mutation and device-provenance contracts.

use apollo_fft::WgpuError;

use crate::infrastructure::transport::gpu::{
    forward_output_len, FramePlan, FramedExecution, StftWgpuPlan,
};

use super::oracle::{frame_count, spectrum};
use super::support::backend;

#[test]
fn geometry_overflow_returns_typed_error_without_panicking() {
    for (frame_len, hop_len, expected_message) in [
        (
            1,
            1,
            "1 + ceil(signal_len / hop_len) overflows host address space",
        ),
        (2, 2, "frame_count * frame_len overflows host address space"),
    ] {
        let plan = StftWgpuPlan::new(FramePlan::new(frame_len, hop_len));
        let error = forward_output_len(&plan, usize::MAX).expect_err("geometry must overflow");
        let WgpuError::InvalidPlan { message } = error else {
            panic!("geometry overflow returned the wrong error: {error:?}");
        };
        assert_eq!(message, expected_message);
    }
}

#[test]
fn geometry_mismatch_rejects_before_host_output_mutation() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = StftWgpuPlan::new(FramePlan::new(8, 4));
    let signal = (0..16)
        .map(|index| index as f32 * 0.125)
        .collect::<Vec<_>>();
    let mut buffers = backend.make_buffers(&plan, signal.len()).expect("buffers");
    backend
        .execute_forward_with_buffers(&plan, &signal, &mut buffers)
        .expect("forward warm-up");
    let forward_before = buffers.fwd_output().to_vec();
    let inverse_before = buffers.inv_output().to_vec();

    let wrong_plan = StftWgpuPlan::new(FramePlan::new(6, 3));
    let error = backend
        .execute_forward_with_buffers(&wrong_plan, &signal, &mut buffers)
        .expect_err("foreign geometry must fail");
    assert!(matches!(error, WgpuError::InvalidPlan { .. }));
    assert_eq!(buffers.fwd_output(), forward_before);
    assert_eq!(buffers.inv_output(), inverse_before);

    let short_spectrum =
        vec![eunomia::Complex32::new(1.0, 0.0); frame_count(signal.len(), 4) * 8 - 1];
    let error = backend
        .execute_inverse_with_buffers(&plan, &short_spectrum, signal.len(), &mut buffers)
        .expect_err("spectrum geometry must fail");
    assert!(matches!(error, WgpuError::InvalidPlan { .. }));
    assert_eq!(buffers.fwd_output(), forward_before);
    assert_eq!(buffers.inv_output(), inverse_before);
}

#[test]
fn foreign_device_rejects_before_host_output_mutation() {
    let Some(owner) = backend() else {
        return;
    };
    let Some(foreign) = backend() else {
        return;
    };
    let plan = StftWgpuPlan::new(FramePlan::new(8, 4));
    let signal = (0..16)
        .map(|index| (index as f32 * 0.31).sin())
        .collect::<Vec<_>>();
    let mut buffers = owner.make_buffers(&plan, signal.len()).expect("buffers");
    let forward_before = buffers.fwd_output().to_vec();
    let inverse_before = buffers.inv_output().to_vec();

    let error = foreign
        .execute_forward_with_buffers(&plan, &signal, &mut buffers)
        .expect_err("foreign forward device must fail");
    assert_eq!(
        error.to_string(),
        "accelerator provider: kernel dispatch failed: prepared WGPU FFT belongs to a different device"
    );
    assert_eq!(buffers.fwd_output(), forward_before);
    assert_eq!(buffers.inv_output(), inverse_before);

    let input = spectrum(frame_count(signal.len(), 4), 8, 1);
    let error = foreign
        .execute_inverse_with_buffers(&plan, &input, signal.len(), &mut buffers)
        .expect_err("foreign inverse device must fail");
    assert_eq!(
        error.to_string(),
        "accelerator provider: kernel dispatch failed: prepared WGPU FFT belongs to a different device"
    );
    assert_eq!(buffers.fwd_output(), forward_before);
    assert_eq!(buffers.inv_output(), inverse_before);
}
