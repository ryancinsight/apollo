//! Value-semantic Radon GPU Leto host-boundary contracts.

use leto::{SliceArg, Storage};

use super::support::backend;

#[test]
fn leto_forward_inverse_and_fbp_match_leto() {
    let Some(backend) = backend() else {
        return;
    };
    let image = leto::Array2::from_shape_vec(
        [3, 3],
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let angles = vec![
        0.0_f32,
        std::f32::consts::FRAC_PI_4,
        std::f32::consts::FRAC_PI_2,
    ];
    let plan = backend.plan(3, 3, angles.len(), 7, 1.0);
    let image_leto =
        leto::Array::from_mnemosyne_slice([3, 3], &image.iter().copied().collect::<Vec<_>>())
            .expect("leto image");
    let angles_leto =
        leto::Array::from_mnemosyne_slice([angles.len()], &angles).expect("leto angles");

    let expected_forward = backend
        .execute_forward(&plan, &image, &angles)
        .expect("leto forward");
    let actual_forward = backend
        .execute_forward_leto(&plan, image_leto.view(), angles_leto.view())
        .expect("leto forward");
    assert_eq!(
        actual_forward.storage().as_slice(),
        expected_forward.as_slice().expect("contiguous forward")
    );

    let expected_inverse = backend
        .execute_inverse(&plan, &expected_forward, &angles)
        .expect("leto inverse");
    let actual_inverse = backend
        .execute_inverse_leto(&plan, actual_forward.view(), angles_leto.view())
        .expect("leto inverse");
    assert_eq!(
        actual_inverse.storage().as_slice(),
        expected_inverse.as_slice().expect("contiguous inverse")
    );

    let expected_fbp = backend
        .execute_filtered_backproject(&plan, &expected_forward, &angles)
        .expect("leto fbp");
    let actual_fbp = backend
        .execute_filtered_backproject_leto(&plan, actual_forward.view(), angles_leto.view())
        .expect("leto fbp");
    assert_eq!(
        actual_fbp.storage().as_slice(),
        expected_fbp.as_slice().expect("contiguous fbp")
    );
}

#[test]
fn leto_strided_forward_matches_logical_leto() {
    let Some(backend) = backend() else {
        return;
    };
    let image = leto::Array2::from_shape_vec(
        [3, 3],
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let mut interleaved = Vec::with_capacity(3 * 6);
    for row in image.rows().expect("row axis view") {
        for value in row.iter().copied() {
            interleaved.push(value);
            interleaved.push(99.0);
        }
    }
    let image_leto = leto::Array::from_mnemosyne_slice([3, 6], &interleaved).expect("leto image");
    let strided = image_leto
        .slice_with::<2>(&[
            SliceArg::range(Some(0), None, 1),
            SliceArg::range(Some(0), None, 2),
        ])
        .expect("strided image");
    let angles = vec![0.0_f32, std::f32::consts::FRAC_PI_2];
    let angles_leto =
        leto::Array::from_mnemosyne_slice([angles.len()], &angles).expect("leto angles");
    let plan = backend.plan(3, 3, angles.len(), 5, 1.0);
    let expected = backend
        .execute_forward(&plan, &image, &angles)
        .expect("leto forward");
    let actual = backend
        .execute_forward_leto(&plan, strided, angles_leto.view())
        .expect("strided leto forward");
    assert_eq!(
        actual.storage().as_slice(),
        expected.as_slice().expect("contiguous forward")
    );
}
