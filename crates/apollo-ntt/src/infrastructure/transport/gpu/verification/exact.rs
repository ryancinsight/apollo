use leto::{Array1, SliceArg, Storage};

use crate::NttPlan;

use super::support::backend;

#[test]
fn forward_impulse_matches_exact_cpu_reference() {
    let Some(backend) = backend() else {
        return;
    };
    let input = vec![1_u64, 0, 0, 0, 0, 0, 0, 0];
    let plan = backend.plan(input.len());
    let actual = backend
        .execute_forward(&plan, &input)
        .expect("GPU forward execution");
    assert_eq!(actual, vec![1_u64; input.len()]);

    let cpu_plan = NttPlan::new(input.len()).expect("CPU plan");
    let expected = cpu_plan
        .forward(&Array1::from(input))
        .expect("CPU forward")
        .into_vec();
    assert_eq!(actual, expected);
}

#[test]
fn forward_fibonacci_matches_exact_cpu_reference() {
    let Some(backend) = backend() else {
        return;
    };
    let input = vec![1_u64, 1, 2, 3, 5, 8, 13, 21];
    let plan = backend.plan(input.len());
    let actual = backend
        .execute_forward(&plan, &input)
        .expect("GPU forward execution");
    let cpu_plan = NttPlan::new(input.len()).expect("CPU plan");
    let expected = cpu_plan
        .forward(&Array1::from(input))
        .expect("CPU forward")
        .into_vec();
    assert_eq!(actual, expected);
}

#[test]
fn inverse_recovers_fibonacci_residues_exactly() {
    let Some(backend) = backend() else {
        return;
    };
    let input = vec![1_u64, 1, 2, 3, 5, 8, 13, 21];
    let plan = backend.plan(input.len());
    let spectrum = backend
        .execute_forward(&plan, &input)
        .expect("GPU forward execution");
    let recovered = backend
        .execute_inverse(&plan, &spectrum)
        .expect("GPU inverse execution");
    assert_eq!(recovered, input);
}

#[test]
fn leto_forward_and_inverse_match_allocating_slice() {
    let Some(backend) = backend() else {
        return;
    };
    let input = vec![1_u64, 1, 2, 3, 5, 8, 13, 21];
    let plan = backend.plan(input.len());
    let expected_forward = backend
        .execute_forward(&plan, &input)
        .expect("slice forward");
    let leto_input = leto::Array1::from_shape_vec([input.len()], input).expect("Leto input");
    let actual_forward = backend
        .execute_forward_leto(&plan, leto_input.view())
        .expect("Leto forward");
    assert_eq!(
        actual_forward.storage().as_slice(),
        expected_forward.as_slice()
    );

    let expected_inverse = backend
        .execute_inverse(&plan, &expected_forward)
        .expect("slice inverse");
    let leto_spectrum = leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
        .expect("Leto spectrum");
    let actual_inverse = backend
        .execute_inverse_leto(&plan, leto_spectrum.view())
        .expect("Leto inverse");
    assert_eq!(
        actual_inverse.storage().as_slice(),
        expected_inverse.as_slice()
    );
}

#[test]
fn strided_leto_forward_matches_logical_slice() {
    let Some(backend) = backend() else {
        return;
    };
    let logical = vec![1_u64, 4, 9, 16, 25, 36, 49, 64];
    let backing = logical
        .iter()
        .copied()
        .flat_map(|value| [value, 99])
        .collect::<Vec<_>>();
    let plan = backend.plan(logical.len());
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
