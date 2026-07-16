//! Value-semantic Wavelet GPU Leto host-boundary contracts.

use leto::{SliceArg, Storage};

use crate::infrastructure::transport::gpu::WaveletWgpuPlan;

use super::support::backend;

#[test]
fn leto_forward_and_inverse_match_slice_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let signal = vec![1.0_f32, -0.5, 2.0, 0.25, -1.25, 0.75, 3.0, -2.0];
    let plan = WaveletWgpuPlan::new(signal.len(), 3);
    let expected_forward = backend
        .execute_forward(&plan, &signal)
        .expect("slice forward");
    let signal_leto = leto::Array1::from_shape_vec([signal.len()], signal).expect("leto signal");
    let actual_forward = backend
        .execute_forward_leto(&plan, signal_leto.view())
        .expect("leto forward");
    assert_eq!(
        actual_forward.storage().as_slice(),
        expected_forward.as_slice()
    );

    let expected_inverse = backend
        .execute_inverse(&plan, &expected_forward)
        .expect("slice inverse");
    let coeffs_leto = leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
        .expect("leto coeffs");
    let actual_inverse = backend
        .execute_inverse_leto(&plan, coeffs_leto.view())
        .expect("leto inverse");
    assert_eq!(
        actual_inverse.storage().as_slice(),
        expected_inverse.as_slice()
    );
}

#[test]
fn leto_strided_forward_matches_logical_slice_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let signal = vec![1.0_f32, -0.5, 2.0, 0.25, -1.25, 0.75, 3.0, -2.0];
    let backing = signal
        .iter()
        .flat_map(|value| [*value, 99.0_f32])
        .collect::<Vec<_>>();
    let signal_leto =
        leto::Array1::from_shape_vec([backing.len()], backing).expect("interleaved signal");
    let strided = signal_leto
        .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
        .expect("strided signal");
    let plan = WaveletWgpuPlan::new(signal.len(), 3);
    let expected = backend
        .execute_forward(&plan, &signal)
        .expect("slice forward");
    let actual = backend
        .execute_forward_leto(&plan, strided)
        .expect("strided leto forward");
    assert_eq!(actual.storage().as_slice(), expected.as_slice());
}
