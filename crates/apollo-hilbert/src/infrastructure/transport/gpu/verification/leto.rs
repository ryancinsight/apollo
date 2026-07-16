//! Value-semantic Hilbert GPU Leto host-boundary contracts.

use leto::{SliceArg, Storage};

use super::support::backend;

#[test]
fn leto_analytic_signal_matches_slice_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let input = vec![1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
    let plan = backend.plan(input.len());
    let expected = backend
        .execute_analytic_signal(&plan, &input)
        .expect("analytic signal");
    let leto_input = leto::Array1::from_shape_vec([input.len()], input).expect("Leto input");
    let actual = backend
        .execute_analytic_signal_leto(&plan, leto_input.view())
        .expect("Leto analytic signal");
    assert_eq!(actual.storage().as_slice(), expected.as_slice());
}

#[test]
fn leto_strided_forward_matches_logical_slice_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let logical = vec![0.5_f32, -1.25, 2.75, 4.0, -3.5, 1.0];
    let mut backing = Vec::with_capacity(logical.len() * 2);
    for value in logical.iter().copied() {
        backing.push(value);
        backing.push(99.0);
    }
    let plan = backend.plan(logical.len());
    let expected = backend.execute_forward(&plan, &logical).expect("forward");
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
fn leto_inverse_matches_slice_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let input = vec![1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
    let plan = backend.plan(input.len());
    let quadrature = backend.execute_forward(&plan, &input).expect("forward");
    let expected = backend
        .execute_inverse(&plan, &quadrature)
        .expect("inverse");
    let leto_quadrature =
        leto::Array1::from_shape_vec([quadrature.len()], quadrature).expect("Leto input");
    let actual = backend
        .execute_inverse_leto(&plan, leto_quadrature.view())
        .expect("Leto inverse");
    assert_eq!(actual.storage().as_slice(), expected.as_slice());
}
