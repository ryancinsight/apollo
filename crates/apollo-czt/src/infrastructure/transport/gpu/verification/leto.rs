//! Value-semantic CZT GPU Leto host-boundary contracts.

use eunomia::Complex32;
use leto::{SliceArg, Storage};

use super::support::{backend, dft_input, dft_parameters, reference_input, reference_parameters};

#[test]
fn leto_forward_matches_slice_forward_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let (a, w) = reference_parameters();
    let input = reference_input().to_vec();
    let plan = backend.plan(input.len(), 6, a, w);
    let expected = backend
        .execute_forward(&plan, &input)
        .expect("slice forward");
    let leto_input = leto::Array1::from_shape_vec([input.len()], input).expect("Leto input");
    let actual = backend
        .execute_forward_leto(&plan, leto_input.view())
        .expect("Leto forward");
    assert_eq!(actual.storage().as_slice(), expected.as_slice());
}

#[test]
fn leto_strided_forward_matches_logical_slice_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let (a, w) = reference_parameters();
    let logical = reference_input().to_vec();
    let sentinel = Complex32::new(99.0, -99.0);
    let mut backing = Vec::with_capacity(logical.len() * 2);
    for value in logical.iter().copied() {
        backing.push(value);
        backing.push(sentinel);
    }
    let plan = backend.plan(logical.len(), 6, a, w);
    let expected = backend
        .execute_forward(&plan, &logical)
        .expect("slice forward");
    let leto_input = leto::Array1::from_shape_vec([backing.len()], backing).expect("input");
    let strided = leto_input
        .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
        .expect("strided view");
    let actual = backend
        .execute_forward_leto(&plan, strided)
        .expect("strided Leto forward");
    assert_eq!(actual.storage().as_slice(), expected.as_slice());
}

#[test]
fn leto_inverse_matches_slice_inverse_for_dft_parameters_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let len = 8;
    let (a, w) = dft_parameters(len);
    let input = dft_input(len);
    let plan = backend.plan(len, len, a, w);
    let spectrum = backend
        .execute_forward(&plan, &input)
        .expect("slice forward");
    let expected = backend
        .execute_inverse(&plan, &spectrum)
        .expect("slice inverse");
    let leto_spectrum =
        leto::Array1::from_shape_vec([spectrum.len()], spectrum).expect("Leto spectrum");
    let actual = backend
        .execute_inverse_leto(&plan, leto_spectrum.view())
        .expect("Leto inverse");
    assert_eq!(actual.storage().as_slice(), expected.as_slice());
}
