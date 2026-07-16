use crate::infrastructure::transport::gpu::ShtWgpuPlan;
use eunomia::Complex32;
use leto::{Array2, SliceArg, Storage};

use super::support::backend;

#[test]
fn leto_forward_and_inverse_match_slice_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = ShtWgpuPlan::new(4, 5, 1);
    let samples = Array2::from_shape_fn([plan.latitudes(), plan.longitudes()], |[lat, lon]| {
        Complex32::new(
            0.25 + lat as f32 * 0.5 - lon as f32 * 0.125,
            0.1 * (lat as f32 + 1.0) * (lon as f32 + 1.0),
        )
    });
    let expected_forward = backend
        .execute_forward(&plan, &samples)
        .expect("slice forward");
    let samples_leto = leto::Array::from_mnemosyne_slice(
        [plan.latitudes(), plan.longitudes()],
        &samples.iter().copied().collect::<Vec<_>>(),
    )
    .expect("leto samples");
    let actual_forward = backend
        .execute_forward_leto(&plan, samples_leto.view())
        .expect("leto forward");
    assert_eq!(
        actual_forward.storage().as_slice(),
        expected_forward
            .values()
            .as_slice()
            .expect("contiguous coeffs")
    );
    let expected_inverse = backend
        .execute_inverse(&plan, &expected_forward)
        .expect("slice inverse");
    let actual_inverse = backend
        .execute_inverse_leto(&plan, actual_forward.view())
        .expect("leto inverse");
    assert_eq!(
        actual_inverse.storage().as_slice(),
        expected_inverse.as_slice().expect("contiguous inverse")
    );
}

#[test]
fn leto_strided_forward_matches_logical_slice_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = ShtWgpuPlan::new(4, 5, 1);
    let samples = Array2::from_shape_fn([plan.latitudes(), plan.longitudes()], |[lat, lon]| {
        Complex32::new(lat as f32 + lon as f32 * 0.25, 0.5 + lon as f32 * 0.1)
    });
    let mut interleaved = Vec::with_capacity(plan.sample_count() * 2);
    for value in samples.iter().copied() {
        interleaved.push(value);
        interleaved.push(Complex32::new(99.0, -99.0));
    }
    let interleaved_leto =
        leto::Array::from_mnemosyne_slice([plan.latitudes(), plan.longitudes() * 2], &interleaved)
            .expect("interleaved samples");
    let strided = interleaved_leto
        .slice_with::<2>(&[
            SliceArg::range(Some(0), None, 1),
            SliceArg::range(Some(0), None, 2),
        ])
        .expect("strided samples");
    let expected = backend
        .execute_forward(&plan, &samples)
        .expect("slice forward");
    let actual = backend
        .execute_forward_leto(&plan, strided)
        .expect("strided leto forward");
    assert_eq!(
        actual.storage().as_slice(),
        expected.values().as_slice().expect("contiguous coeffs")
    );
}
