//! Device-present verification for reusable Leto host staging.

use hephaestus_core::HephaestusError;
use hephaestus_wgpu::WgpuDevice;

use crate::{f16, ApolloError};

use super::{GpuFft3d, GpuFft3dBuffers};

fn device_or_skip(application_name: &str) -> Option<WgpuDevice> {
    match WgpuDevice::try_default(application_name) {
        Ok(device) => Some(device),
        Err(HephaestusError::AdapterUnavailable { .. }) => None,
        Err(error) => panic!("reusable Leto FFT staging requires a working provider: {error}"),
    }
}

fn assert_buffer_shape_mismatch(error: ApolloError) {
    match error {
        ApolloError::ShapeMismatch { expected, actual } => {
            assert_eq!(expected, "FFT reusable buffer shape [2, 3, 2]");
            assert_eq!(actual, "FFT reusable buffer shape [2, 2, 2]");
        }
        other => panic!("expected reusable-buffer shape mismatch, got {other}"),
    }
}

fn assert_staging_sentinels(buffers: &GpuFft3dBuffers) {
    assert!(buffers.real_host.iter().all(|value| *value == 3.0));
    assert!(buffers.imaginary_host.iter().all(|value| *value == -2.0));
}

#[test]
fn retained_staging_matches_allocating_leto_roundtrip_when_device_exists() {
    let Some(device) = device_or_skip("apollo-fft-leto-staging-test") else {
        return;
    };
    let plan = GpuFft3d::new(device, 2, 2, 2)
        .expect("2x2x2 Leto staging plan must fit the acquired device");
    let input = leto::Array::<f64, leto::MnemosyneStorage<f64>, 3>::from_mnemosyne_slice(
        [2, 2, 2],
        &[1.0, -0.5, 0.25, 2.0, -1.0, 0.75, 0.125, -0.25],
    )
    .expect("input shape matches its values");
    let mut buffers = GpuFft3dBuffers::new(&plan).expect("reusable staging allocation");
    let staging_addresses = (buffers.real_host.as_ptr(), buffers.imaginary_host.as_ptr());

    let expected_spectrum = plan.forward_leto(input.view()).expect("allocating forward");
    let actual_spectrum = plan
        .forward_leto_with_buffers(input.view(), &mut buffers)
        .expect("retained-staging forward");
    assert_eq!(actual_spectrum.as_slice(), expected_spectrum.as_slice());

    let expected_field = plan
        .inverse_leto(expected_spectrum.view())
        .expect("allocating inverse");
    let actual_field = plan
        .inverse_leto_with_buffers(actual_spectrum.view(), &mut buffers)
        .expect("retained-staging inverse");
    assert_eq!(actual_field.as_slice(), expected_field.as_slice());
    assert_eq!(
        (buffers.real_host.as_ptr(), buffers.imaginary_host.as_ptr(),),
        staging_addresses
    );
}

#[test]
fn retained_staging_matches_half_leto_roundtrip_when_device_exists() {
    let Some(device) = device_or_skip("apollo-fft-half-leto-staging-test") else {
        return;
    };
    let plan = GpuFft3d::new(device, 2, 2, 2)
        .expect("2x2x2 half Leto staging plan must fit the acquired device");
    let values = [1.0_f32, -0.5, 0.25, 2.0, -1.0, 0.75, 0.125, -0.25].map(f16::from_f32);
    let input = leto::Array::<f16, leto::MnemosyneStorage<f16>, 3>::from_mnemosyne_slice(
        [2, 2, 2],
        &values,
    )
    .expect("half input shape matches its values");
    let mut buffers = GpuFft3dBuffers::new(&plan).expect("reusable staging allocation");

    let expected_spectrum = plan
        .forward_f16_leto(input.view())
        .expect("allocating half forward");
    let actual_spectrum = plan
        .forward_half_leto_with_buffers(input.view(), &mut buffers)
        .expect("retained-staging half forward");
    assert_eq!(actual_spectrum.as_slice(), expected_spectrum.as_slice());

    let expected_field = plan
        .inverse_f16_leto(expected_spectrum.view())
        .expect("allocating half inverse");
    let actual_field = plan
        .inverse_half_leto_with_buffers(actual_spectrum.view(), &mut buffers)
        .expect("retained-staging half inverse");
    assert_eq!(actual_field.as_slice(), expected_field.as_slice());
}

#[test]
fn retained_staging_rejects_a_different_plan_shape_when_device_exists() {
    let Some(device) = device_or_skip("apollo-fft-leto-staging-shape-test") else {
        return;
    };
    let original = GpuFft3d::new(device.clone(), 2, 2, 2)
        .expect("2x2x2 staging plan must fit the acquired device");
    let other =
        GpuFft3d::new(device, 2, 3, 2).expect("2x3x2 staging plan must fit the acquired device");
    let mut buffers = GpuFft3dBuffers::new(&original).expect("reusable staging allocation");
    buffers.real_host.fill(3.0);
    buffers.imaginary_host.fill(-2.0);
    let input = leto::Array::<f64, leto::MnemosyneStorage<f64>, 3>::from_mnemosyne_slice(
        [2, 3, 2],
        &[0.0; 12],
    )
    .expect("input shape matches its values");

    let Err(error) = other.forward_leto_with_buffers(input.view(), &mut buffers) else {
        panic!("staging from a different plan shape must be rejected");
    };
    assert_buffer_shape_mismatch(error);
    assert_staging_sentinels(&buffers);

    let spectrum =
        leto::Array::<f32, leto::MnemosyneStorage<f32>, 1>::from_mnemosyne_slice([24], &[0.0; 24])
            .expect("spectrum shape matches its values");
    let Err(error) = other.inverse_leto_with_buffers(spectrum.view(), &mut buffers) else {
        panic!("inverse staging from a different plan shape must be rejected");
    };
    assert_buffer_shape_mismatch(error);
    assert_staging_sentinels(&buffers);
}
