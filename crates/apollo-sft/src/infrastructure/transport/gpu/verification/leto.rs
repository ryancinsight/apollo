//! Value-semantic SFT GPU Leto host-boundary contracts.

use leto::{SliceArg, Storage};

use crate::infrastructure::transport::gpu::SftWgpuPlan;

use super::support::{backend, represented_signal, two_tone_signal};

#[test]
fn leto_forward_matches_slice_forward_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = SftWgpuPlan::new(8, 2);
    let signal = two_tone_signal(8, &[(1, 3.0), (3, 1.25)]);
    let input = represented_signal(&signal);
    let leto_input = leto::Array1::from_shape_vec([input.len()], input.clone()).expect("input");

    let expected = backend
        .execute_forward(&plan, &input)
        .expect("slice forward");
    let actual = backend
        .execute_forward_leto(&plan, leto_input.view())
        .expect("leto forward");
    assert_eq!(actual.frequencies, expected.frequencies);
    assert_eq!(actual.values, expected.values);
}

#[test]
fn leto_strided_forward_matches_logical_slice_forward_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = SftWgpuPlan::new(8, 2);
    let signal = two_tone_signal(8, &[(1, 3.0), (3, 1.25)]);
    let logical = represented_signal(&signal);
    let mut backing = Vec::with_capacity(logical.len() * 2);
    for value in &logical {
        backing.push(*value);
        backing.push(eunomia::Complex32::new(99.0, -99.0));
    }
    let leto_input = leto::Array1::from_shape_vec([backing.len()], backing).expect("input");
    let strided = leto_input
        .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
        .expect("strided view");

    let expected = backend
        .execute_forward(&plan, &logical)
        .expect("slice forward");
    let actual = backend
        .execute_forward_leto(&plan, strided)
        .expect("strided Leto forward");
    assert_eq!(actual.frequencies, expected.frequencies);
    assert_eq!(actual.values, expected.values);
}

#[test]
fn leto_inverse_matches_slice_inverse_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = SftWgpuPlan::new(8, 2);
    let signal = two_tone_signal(8, &[(1, 3.0), (3, 1.25)]);
    let spectrum = backend
        .execute_forward(&plan, &represented_signal(&signal))
        .expect("GPU SFT");
    let expected = backend
        .execute_inverse(&plan, &spectrum)
        .expect("slice inverse");
    let actual = backend
        .execute_inverse_leto(&plan, &spectrum)
        .expect("Leto inverse");
    assert_eq!(actual.storage().as_slice(), expected.as_slice());
}
