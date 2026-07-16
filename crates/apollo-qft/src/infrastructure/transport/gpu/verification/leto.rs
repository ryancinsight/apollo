use leto::{Array1, SliceArg, Storage};

use super::support::{backend, inverse_input, roundtrip_input};

#[test]
fn leto_forward_matches_slice_forward_when_device_exists() {
    let Ok(backend) = backend() else {
        return;
    };
    let input = roundtrip_input();
    let leto_input = Array1::from_shape_vec([input.len()], input.clone()).expect("input");
    let plan = backend.plan(input.len());
    let expected = backend
        .execute_forward(&plan, &input)
        .expect("slice forward");
    let actual = backend
        .execute_forward_leto(&plan, leto_input.view())
        .expect("leto forward");
    assert_eq!(actual.storage().as_slice(), expected.as_slice());
}

#[test]
fn leto_strided_forward_matches_logical_slice_forward_when_device_exists() {
    let Ok(backend) = backend() else {
        return;
    };
    let logical = roundtrip_input();
    let mut backing = Vec::with_capacity(logical.len() * 2);
    for value in &logical {
        backing.push(*value);
        backing.push(eunomia::Complex32::new(99.0, -99.0));
    }
    let leto_input = Array1::from_shape_vec([backing.len()], backing).expect("input");
    let strided = leto_input
        .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
        .expect("strided view");
    let plan = backend.plan(logical.len());
    let expected = backend
        .execute_forward(&plan, &logical)
        .expect("slice forward");
    let actual = backend
        .execute_forward_leto(&plan, strided)
        .expect("strided leto forward");
    assert_eq!(actual.storage().as_slice(), expected.as_slice());
}

#[test]
fn leto_inverse_matches_slice_inverse_when_device_exists() {
    let Ok(backend) = backend() else {
        return;
    };
    let input = inverse_input();
    let leto_input = Array1::from_shape_vec([input.len()], input.clone()).expect("input");
    let plan = backend.plan(input.len());
    let expected = backend
        .execute_inverse(&plan, &input)
        .expect("slice inverse");
    let actual = backend
        .execute_inverse_leto(&plan, leto_input.view())
        .expect("leto inverse");
    assert_eq!(actual.storage().as_slice(), expected.as_slice());
}
