//! Verification modules for the Radon transform.

#[cfg(test)]
mod tests {
    use crate::{ramp_filter_projection, ramp_filter_projection_into, RadonError, RadonPlan};
    use approx::assert_abs_diff_eq;
    use leto::Array2;
    use proptest::prelude::*;
    use proptest::proptest;

    #[test]
    fn zero_angle_projection_equals_column_sums() {
        let image =
            leto::Array2::from_shape_vec([3, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();
        let plan = RadonPlan::new(3, 3, vec![0.0], 3, 1.0).expect("plan");
        let sinogram = plan.forward(&image).expect("forward");
        let row = sinogram.values().as_slice().expect("contiguous sinogram");

        assert_abs_diff_eq!(row[0], 12.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(row[1], 15.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(row[2], 18.0, epsilon = 1.0e-12);
    }

    #[test]
    fn right_angle_projection_equals_row_sums() {
        let image =
            leto::Array2::from_shape_vec([3, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();
        let plan = RadonPlan::new(3, 3, vec![std::f64::consts::FRAC_PI_2], 3, 1.0).expect("plan");
        let sinogram = plan.forward(&image).expect("forward");
        let row = sinogram.values().as_slice().expect("contiguous sinogram");

        assert_abs_diff_eq!(row[0], 6.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(row[1], 15.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(row[2], 24.0, epsilon = 1.0e-12);
    }

    #[test]
    fn forward_and_backproject_satisfy_adjoint_identity() {
        let image = leto::Array2::from_shape_vec([2, 2], vec![1.0, -2.0, 0.5, 3.0]).unwrap();
        let detector_values =
            leto::Array2::from_shape_vec([2, 3], vec![2.0, -1.0, 0.25, 1.5, 0.0, -0.75]).unwrap();
        let plan =
            RadonPlan::new(2, 2, vec![0.0, std::f64::consts::FRAC_PI_4], 3, 1.0).expect("plan");
        let forward = plan.forward(&image).expect("forward");
        let probe = crate::Sinogram::new(detector_values);
        let backprojected = plan.backproject(&probe).expect("backproject");

        let projection_inner: f64 = forward
            .values()
            .iter()
            .zip(probe.values().iter())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum();
        let image_inner: f64 = image
            .iter()
            .zip(backprojected.iter())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum();

        assert_abs_diff_eq!(projection_inner, image_inner, epsilon = 1.0e-12);
    }

    #[test]
    fn ramp_filter_removes_projection_dc_component() {
        let filtered = ramp_filter_projection(&[3.0; 8], 1.0);
        for value in filtered {
            assert_abs_diff_eq!(value, 0.0, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn ramp_filter_into_matches_allocating_path() {
        let projection = [1.0, -2.0, 0.5, 4.0, -1.0, 0.25, 3.0, -0.75];
        let allocating = ramp_filter_projection(&projection, 0.5);
        let mut caller_owned = [0.0; 8];

        ramp_filter_projection_into(&projection, 0.5, &mut caller_owned);

        for (actual, expected) in caller_owned.iter().zip(allocating.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn filtered_backprojection_rejects_shape_mismatch() {
        let plan = RadonPlan::new(2, 2, vec![0.0], 2, 1.0).expect("plan");
        let sinogram = crate::Sinogram::new(Array2::zeros([2, 2]));
        assert_eq!(
            plan.filtered_backprojection(&sinogram).unwrap_err(),
            RadonError::SinogramShapeMismatch
        );
    }

    #[test]
    fn rejects_invalid_contracts() {
        assert_eq!(
            RadonPlan::new(0, 2, vec![0.0], 2, 1.0).unwrap_err(),
            RadonError::EmptyRows
        );
        assert_eq!(
            RadonPlan::new(2, 0, vec![0.0], 2, 1.0).unwrap_err(),
            RadonError::EmptyCols
        );
        assert_eq!(
            RadonPlan::new(2, 2, vec![], 2, 1.0).unwrap_err(),
            RadonError::EmptyAngles
        );
        assert_eq!(
            RadonPlan::new(2, 2, vec![f64::NAN], 2, 1.0).unwrap_err(),
            RadonError::InvalidAngle
        );
        assert_eq!(
            RadonPlan::new(2, 2, vec![0.0], 0, 1.0).unwrap_err(),
            RadonError::EmptyDetectors
        );
        assert_eq!(
            RadonPlan::new(2, 2, vec![0.0], 2, 0.0).unwrap_err(),
            RadonError::InvalidDetectorSpacing
        );
    }

    proptest! {
        #[test]
        fn zero_angle_mass_is_conserved_for_square_images(
            values in prop::collection::vec(-10.0f64..10.0f64, 9)
        ) {
            let image = Array2::from_shape_vec([3, 3], values).expect("shape");
            let plan = RadonPlan::new(3, 3, vec![0.0], 3, 1.0).expect("plan");
            let sinogram = plan.forward(&image).expect("forward");
            let image_sum: f64 = image.iter().sum();
            let projection_sum: f64 = sinogram.values().iter().sum();
            prop_assert!((image_sum - projection_sum).abs() < 1.0e-10);
        }
    }
}
