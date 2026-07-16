use apollo_fft::{f16, PrecisionProfile};
use eunomia::Complex32;
use leto::{Array2, Storage};

use crate::{
    infrastructure::transport::gpu::{ShtWgpuPlan, WgpuError},
    SphericalHarmonicCoefficients,
};

use super::support::{backend, REPRESENTED_STORAGE_TOLERANCE};

#[test]
fn typed_flat_mixed_storage_matches_represented_forward_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = ShtWgpuPlan::new(3, 5, 2);
    let signal: Vec<Complex32> = (0..plan.sample_count())
        .map(|index| Complex32::new(0.5 + index as f32 * 0.1, 0.1 * (index as f32 + 1.0)))
        .collect();
    let reduced_input: Vec<[f16; 2]> = signal
        .iter()
        .map(|value| [f16::from_f32(value.re), f16::from_f32(value.im)])
        .collect();
    let represented: Vec<Complex32> = reduced_input
        .iter()
        .map(|value| Complex32::new(value[0].to_f32(), value[1].to_f32()))
        .collect();
    let samples = Array2::from_shape_vec([plan.latitudes(), plan.longitudes()], represented)
        .expect("reshape represented samples");

    let expected = backend
        .execute_forward(&plan, &samples)
        .expect("represented f32 forward");
    let actual = backend
        .execute_forward_flat_typed(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &reduced_input,
        )
        .expect("typed flat mixed forward");

    assert_eq!(actual.max_degree(), plan.max_degree());
    assert_eq!(actual.max_degree(), expected.max_degree());
    for degree in 0..=plan.max_degree() {
        for order in -(degree as isize)..=(degree as isize) {
            let actual_value = actual.get(degree, order);
            let expected_value = expected.get(degree, order);
            assert!(
                (actual_value.re - expected_value.re).abs() < REPRESENTED_STORAGE_TOLERANCE,
                "real mismatch degree={degree} order={order}: actual={actual_value:?} expected={expected_value:?}"
            );
            assert!(
                (actual_value.im - expected_value.im).abs() < REPRESENTED_STORAGE_TOLERANCE,
                "imaginary mismatch degree={degree} order={order}: actual={actual_value:?} expected={expected_value:?}"
            );
        }
    }
}

#[test]
fn typed_flat_leto_forward_and_inverse_match_slice_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = ShtWgpuPlan::new(3, 5, 2);
    let input: Vec<[f16; 2]> = (0..plan.sample_count())
        .map(|index| {
            [
                f16::from_f32(0.5 + index as f32 * 0.1),
                f16::from_f32(0.1 * (index as f32 + 1.0)),
            ]
        })
        .collect();

    let expected_forward = backend
        .execute_forward_flat_typed(&plan, PrecisionProfile::MIXED_PRECISION_F16_F32, &input)
        .expect("typed slice forward");
    let input_leto =
        leto::Array::from_mnemosyne_slice([input.len()], &input).expect("typed leto input");
    let actual_forward = backend
        .execute_forward_flat_leto_typed(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            input_leto.view(),
        )
        .expect("typed leto forward");
    assert_eq!(
        actual_forward.storage().as_slice(),
        expected_forward
            .values()
            .as_slice()
            .expect("contiguous coefficients")
    );

    let mut expected_inverse = vec![[f16::from_f32(0.0); 2]; plan.sample_count()];
    backend
        .execute_inverse_flat_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &expected_forward,
            &mut expected_inverse,
        )
        .expect("typed slice inverse");
    let actual_inverse = backend
        .execute_inverse_flat_leto_typed::<[f16; 2]>(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            actual_forward.view(),
        )
        .expect("typed leto inverse");
    let actual_bits: Vec<[u16; 2]> = actual_inverse
        .storage()
        .as_slice()
        .iter()
        .map(|value| [value[0].to_bits(), value[1].to_bits()])
        .collect();
    let expected_bits: Vec<[u16; 2]> = expected_inverse
        .iter()
        .map(|value| [value[0].to_bits(), value[1].to_bits()])
        .collect();
    assert_eq!(actual_bits, expected_bits);
}

#[test]
fn typed_flat_path_rejects_profile_mismatch_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = ShtWgpuPlan::new(3, 5, 2);
    let flat_input = vec![[f16::from_f32(0.0); 2]; plan.sample_count()];

    let forward_error = backend
        .execute_forward_flat_typed::<[f16; 2]>(
            &plan,
            PrecisionProfile::LOW_PRECISION_F32,
            &flat_input,
        )
        .expect_err("profile mismatch must fail");
    assert!(matches!(forward_error, WgpuError::InvalidPrecisionProfile));

    let coefficients = SphericalHarmonicCoefficients::zeros(plan.max_degree());
    let mut output = vec![[f16::from_f32(0.0); 2]; plan.sample_count()];
    let inverse_error = backend
        .execute_inverse_flat_typed_into::<[f16; 2]>(
            &plan,
            PrecisionProfile::LOW_PRECISION_F32,
            &coefficients,
            &mut output,
        )
        .expect_err("profile mismatch must fail");
    assert!(matches!(inverse_error, WgpuError::InvalidPrecisionProfile));
}
