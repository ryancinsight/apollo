use leto::{Array2, Storage};

use crate::{infrastructure::transport::gpu::WgpuError, DctDstPlan, RealTransformKind};

use super::support::{assert_cpu_differential, assert_roundtrip, backend, DIMENSIONAL_TOLERANCE};

fn input() -> Array2<f32> {
    Array2::from_shape_vec(
        [3, 3],
        vec![1.0, -2.0, 0.5, 0.25, 3.0, -1.5, -0.75, 0.5, 2.0],
    )
    .expect("square Leto input")
}

fn cpu_reference(input: &Array2<f32>) -> Vec<f64> {
    let plan = DctDstPlan::new(3, RealTransformKind::DctII).expect("CPU plan");
    let mut rows = [[0.0_f64; 3]; 3];
    for row in 0..3 {
        let values = (0..3)
            .map(|column| f64::from(input[[row, column]]))
            .collect::<Vec<_>>();
        rows[row].copy_from_slice(&plan.forward(&values).expect("CPU row forward"));
    }
    let mut output = vec![0.0_f64; 9];
    for column in 0..3 {
        let values = (0..3).map(|row| rows[row][column]).collect::<Vec<_>>();
        let transformed = plan.forward(&values).expect("CPU column forward");
        for row in 0..3 {
            output[row * 3 + column] = transformed[row];
        }
    }
    output
}

#[test]
fn dct_two_forward_matches_independent_cpu_separable_reference() {
    let Some(backend) = backend() else {
        return;
    };
    let input = input();
    let plan = backend.plan(3, RealTransformKind::DctII);
    let actual = backend
        .execute_forward_2d(&plan, input.as_slice().expect("contiguous input"))
        .expect("GPU 2D forward");
    assert_cpu_differential(&actual, &cpu_reference(&input), DIMENSIONAL_TOLERANCE);
}

#[test]
fn dct_two_inverse_recovers_input() {
    let Some(backend) = backend() else {
        return;
    };
    let input = input();
    let plan = backend.plan(3, RealTransformKind::DctII);
    let spectrum = backend
        .execute_forward_2d(&plan, input.as_slice().expect("contiguous input"))
        .expect("GPU 2D forward");
    let actual = backend
        .execute_inverse_2d(&plan, &spectrum)
        .expect("GPU 2D inverse");
    assert_roundtrip(
        &actual,
        input.as_slice().expect("contiguous input"),
        DIMENSIONAL_TOLERANCE as f32,
    );
}

#[test]
fn leto_two_dimensional_boundary_matches_slice_execution() {
    let Some(backend) = backend() else {
        return;
    };
    let input = input();
    let plan = backend.plan(3, RealTransformKind::DctII);
    let expected_forward = backend
        .execute_forward_2d(&plan, input.as_slice().expect("contiguous input"))
        .expect("2D forward");
    let leto_input =
        Array2::from_shape_vec([3, 3], input.iter().copied().collect()).expect("Leto input");
    let actual_forward = backend
        .execute_forward_2d_leto(&plan, leto_input.view())
        .expect("Leto 2D forward");
    assert_eq!(
        actual_forward.storage().as_slice(),
        expected_forward.as_slice()
    );

    let expected_inverse = backend
        .execute_inverse_2d(&plan, &expected_forward)
        .expect("2D inverse");
    let leto_spectrum = Array2::from_shape_vec([3, 3], expected_forward).expect("Leto spectrum");
    let actual_inverse = backend
        .execute_inverse_2d_leto(&plan, leto_spectrum.view())
        .expect("Leto 2D inverse");
    assert_eq!(
        actual_inverse.storage().as_slice(),
        expected_inverse.as_slice()
    );
}

#[test]
fn two_dimensional_execution_rejects_non_square_input() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = backend.plan(3, RealTransformKind::DctII);
    let input = Array2::<f32>::zeros([2, 3]);
    let error = backend
        .execute_forward_2d(&plan, input.as_slice().expect("contiguous input"))
        .expect_err("non-square 2D input must fail");
    assert!(matches!(error, WgpuError::ShapeMismatch { .. }));
}
