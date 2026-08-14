use crate::{DctDstError, DctDstPlan, RealTransformKind};
use eunomia::assert_abs_diff_eq;
use leto::{Array2, Array3};

#[test]
fn forward_2d_matches_separable_manual_application() {
    let plan = DctDstPlan::new(4, RealTransformKind::DctII).unwrap();
    let input = Array2::from_shape_vec(
        [4, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, //
            0.5, -1.0, 2.0, 1.5, //
            -2.0, 0.0, 1.0, 3.0, //
            4.0, -3.0, 2.5, -0.5,
        ],
    )
    .unwrap();

    let actual = plan.forward_2d(&input).unwrap();

    let mut row_stage = Array2::<f64>::zeros([4, 4]);
    let mut expected = Array2::<f64>::zeros([4, 4]);
    for i in 0..4 {
        let row: Vec<f64> = (0..4).map(|j| input[[i, j]]).collect();
        let transformed = plan.forward(&row).unwrap();
        for j in 0..4 {
            row_stage[[i, j]] = transformed[j];
        }
    }
    for j in 0..4 {
        let col: Vec<f64> = (0..4).map(|i| row_stage[[i, j]]).collect();
        let transformed = plan.forward(&col).unwrap();
        for i in 0..4 {
            expected[[i, j]] = transformed[i];
        }
    }

    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(lhs, rhs, epsilon = 1.0e-12);
    }
}

#[test]
fn inverse_2d_roundtrip_recovers_signal() {
    let plan = DctDstPlan::new(4, RealTransformKind::DstII).unwrap();
    let input = Array2::from_shape_vec(
        [4, 4],
        vec![
            0.25, -1.0, 3.0, 2.0, //
            1.0, 0.0, -2.5, 4.0, //
            -3.0, 2.0, 1.5, -0.5, //
            2.25, -4.0, 0.75, 1.0,
        ],
    )
    .unwrap();
    let coeff = plan.forward_2d(&input).unwrap();
    let recovered = plan.inverse_2d(&coeff).unwrap();
    for (lhs, rhs) in recovered.iter().zip(input.iter()) {
        assert_abs_diff_eq!(lhs, rhs, epsilon = 1.0e-10);
    }
}

#[test]
fn inverse_3d_roundtrip_recovers_signal() {
    let plan = DctDstPlan::new(3, RealTransformKind::DctIII).unwrap();
    let input = Array3::from_shape_fn([3, 3, 3], |[i, j, k]| {
        (0.7 * i as f64).sin() + 0.5 * (1.3 * j as f64).cos() - 0.25 * k as f64
    });
    let coeff = plan.forward_3d(&input).unwrap();
    let recovered = plan.inverse_3d(&coeff).unwrap();
    for (lhs, rhs) in recovered.iter().zip(input.iter()) {
        assert_abs_diff_eq!(lhs, rhs, epsilon = 1.0e-10);
    }
}

#[test]
fn rejects_non_square_or_non_cubic_shapes() {
    let plan2 = DctDstPlan::new(4, RealTransformKind::DctII).unwrap();
    let nonsquare = Array2::<f64>::zeros([4, 3]);
    assert!(matches!(
        plan2.forward_2d(&nonsquare),
        Err(DctDstError::LengthMismatch)
    ));

    let plan3 = DctDstPlan::new(3, RealTransformKind::DstIII).unwrap();
    let noncubic = Array3::<f64>::zeros([3, 3, 2]);
    assert!(matches!(
        plan3.inverse_3d(&noncubic),
        Err(DctDstError::LengthMismatch)
    ));
}
