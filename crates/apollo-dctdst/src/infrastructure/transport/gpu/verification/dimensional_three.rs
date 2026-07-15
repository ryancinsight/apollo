use leto::{Array3, Storage};

use crate::{infrastructure::transport::gpu::WgpuError, DctDstPlan, RealTransformKind};

use super::support::{assert_cpu_differential, assert_roundtrip, backend, DIMENSIONAL_TOLERANCE};

const INPUT: [f32; 27] = [
    1.0, -2.0, 0.5, 0.25, 3.0, -1.5, -0.75, 0.5, 2.0, 0.1, -0.3, 1.2, -0.5, 2.1, -1.1, 0.7, -0.9,
    0.3, 1.5, -0.2, 0.8, -1.4, 0.6, -0.1, 0.9, -2.5, 1.3,
];

fn input() -> Array3<f32> {
    Array3::from_shape_vec([3, 3, 3], INPUT.to_vec()).expect("cubic Leto input")
}

fn cpu_reference(input: &Array3<f32>) -> Vec<f64> {
    let plan = DctDstPlan::new(3, RealTransformKind::DctII).expect("CPU plan");
    let mut axis_zero = [[[0.0_f64; 3]; 3]; 3];
    for column in 0..3 {
        for depth in 0..3 {
            let values = (0..3)
                .map(|row| f64::from(input[[row, column, depth]]))
                .collect::<Vec<_>>();
            let transformed = plan.forward(&values).expect("CPU axis-zero forward");
            for row in 0..3 {
                axis_zero[row][column][depth] = transformed[row];
            }
        }
    }
    let mut axis_one = [[[0.0_f64; 3]; 3]; 3];
    for row in 0..3 {
        for depth in 0..3 {
            let values = (0..3)
                .map(|column| axis_zero[row][column][depth])
                .collect::<Vec<_>>();
            let transformed = plan.forward(&values).expect("CPU axis-one forward");
            for column in 0..3 {
                axis_one[row][column][depth] = transformed[column];
            }
        }
    }
    let mut output = vec![0.0_f64; 27];
    for row in 0..3 {
        for column in 0..3 {
            let values = (0..3)
                .map(|depth| axis_one[row][column][depth])
                .collect::<Vec<_>>();
            let transformed = plan.forward(&values).expect("CPU axis-two forward");
            for depth in 0..3 {
                output[(row * 3 + column) * 3 + depth] = transformed[depth];
            }
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
        .execute_forward_3d(&plan, input.as_slice().expect("contiguous input"))
        .expect("GPU 3D forward");
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
        .execute_forward_3d(&plan, input.as_slice().expect("contiguous input"))
        .expect("GPU 3D forward");
    let actual = backend
        .execute_inverse_3d(&plan, &spectrum)
        .expect("GPU 3D inverse");
    assert_roundtrip(
        &actual,
        input.as_slice().expect("contiguous input"),
        DIMENSIONAL_TOLERANCE as f32,
    );
}

#[test]
fn leto_three_dimensional_boundary_matches_slice_execution() {
    let Some(backend) = backend() else {
        return;
    };
    let input = input();
    let plan = backend.plan(3, RealTransformKind::DctII);
    let expected_forward = backend
        .execute_forward_3d(&plan, input.as_slice().expect("contiguous input"))
        .expect("3D forward");
    let leto_input = Array3::from_shape_vec([3, 3, 3], INPUT.to_vec()).expect("Leto input");
    let actual_forward = backend
        .execute_forward_3d_leto(&plan, leto_input.view())
        .expect("Leto 3D forward");
    assert_eq!(
        actual_forward.storage().as_slice(),
        expected_forward.as_slice()
    );

    let expected_inverse = backend
        .execute_inverse_3d(&plan, &expected_forward)
        .expect("3D inverse");
    let leto_spectrum = Array3::from_shape_vec([3, 3, 3], expected_forward).expect("Leto spectrum");
    let actual_inverse = backend
        .execute_inverse_3d_leto(&plan, leto_spectrum.view())
        .expect("Leto 3D inverse");
    assert_eq!(
        actual_inverse.storage().as_slice(),
        expected_inverse.as_slice()
    );
}

#[test]
fn three_dimensional_execution_rejects_non_cubic_input() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = backend.plan(3, RealTransformKind::DctII);
    let input = Array3::<f32>::zeros([2, 3, 3]);
    let error = backend
        .execute_forward_3d(&plan, input.as_slice().expect("contiguous input"))
        .expect_err("non-cubic 3D input must fail");
    assert!(matches!(error, WgpuError::ShapeMismatch { .. }));
}
