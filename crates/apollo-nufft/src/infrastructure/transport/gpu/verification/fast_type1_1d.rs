use apollo_fft::{f16, PrecisionProfile};
use eunomia::{Complex32, Complex64};
use leto::Storage;

use crate::infrastructure::transport::gpu::NufftWgpuPlan1D;
use crate::{nufft_type1_1d_fast, UniformDomain1D};

use super::support::{assert_complex64_close, backend};

#[test]
fn typed_leto_fast_type1_1d_matches_typed_slice_path() {
    let Some(backend) = backend() else {
        return;
    };

    let domain = UniformDomain1D::new(8, 0.25).expect("domain");
    let plan = NufftWgpuPlan1D::new(domain, 2, 6);
    let positions = [0.0_f32, 0.25, 0.7, 1.15];
    let values16 = [
        [f16::from_f32(1.0), f16::from_f32(0.0)],
        [f16::from_f32(0.5), f16::from_f32(-0.25)],
        [f16::from_f32(-0.75), f16::from_f32(0.5)],
        [f16::from_f32(0.25), f16::from_f32(0.75)],
    ];
    let mut expected = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; domain.n];
    backend
        .execute_fast_type1_1d_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &positions,
            &values16,
            &mut expected,
        )
        .expect("typed fast slice");
    let leto_positions =
        leto::Array1::from_shape_vec([positions.len()], positions.to_vec()).expect("positions");
    let leto_values =
        leto::Array1::from_shape_vec([values16.len()], values16.to_vec()).expect("values");

    let actual = backend
        .execute_fast_type1_1d_leto_typed::<[f16; 2]>(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            leto_positions.view(),
            leto_values.view(),
        )
        .expect("typed fast leto");

    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_eq!(actual[0].to_bits(), expected[0].to_bits());
        assert_eq!(actual[1].to_bits(), expected[1].to_bits());
    }
}

#[test]
fn fast_type1_1d_matches_cpu_gridded_reference() {
    let Some(backend) = backend() else {
        return;
    };

    let domain = UniformDomain1D::new(8, 0.25).expect("domain");
    let plan = NufftWgpuPlan1D::new(domain, 2, 6);
    let positions = [0.0_f32, 0.25, 0.7, 1.15];
    let values = [
        Complex32::new(1.0, 0.0),
        Complex32::new(0.5, -0.25),
        Complex32::new(-0.75, 0.5),
        Complex32::new(0.25, 0.75),
    ];
    let expected_positions: Vec<f64> = positions.iter().map(|value| *value as f64).collect();
    let expected_values: Vec<Complex64> = values
        .iter()
        .map(|value| Complex64::new(value.re as f64, value.im as f64))
        .collect();
    let expected = nufft_type1_1d_fast(&expected_positions, &expected_values, domain, 6);

    let actual = backend
        .execute_fast_type1_1d(&plan, &positions, &values)
        .expect("GPU fast type1 1D");

    assert_eq!(actual.size(), expected.size());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_complex64_close(*actual, *expected, 1.5e-3);
    }
}

#[test]
fn fast_type1_1d_typed_mixed_storage_matches_represented_input() {
    let Some(backend) = backend() else {
        return;
    };

    let domain = UniformDomain1D::new(8, 0.25).expect("domain");
    let plan = NufftWgpuPlan1D::new(domain, 2, 6);
    let positions = [0.0_f32, 0.25, 0.7, 1.15];
    let values16 = [
        [f16::from_f32(1.0), f16::from_f32(0.0)],
        [f16::from_f32(0.5), f16::from_f32(-0.25)],
        [f16::from_f32(-0.75), f16::from_f32(0.5)],
        [f16::from_f32(0.25), f16::from_f32(0.75)],
    ];
    let represented: Vec<Complex32> = values16
        .iter()
        .map(|value| Complex32::new(value[0].to_f32(), value[1].to_f32()))
        .collect();
    let expected = backend
        .execute_fast_type1_1d(&plan, &positions, &represented)
        .expect("represented fast type1 1D");
    let mut actual = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; domain.n];

    backend
        .execute_fast_type1_1d_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &positions,
            &values16,
            &mut actual,
        )
        .expect("mixed fast type1 1D");

    assert_eq!(actual.len(), expected.size());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        let expected_re = f16::from_f32(expected.re as f32);
        let expected_im = f16::from_f32(expected.im as f32);
        assert_eq!(actual[0].to_bits(), expected_re.to_bits());
        assert_eq!(actual[1].to_bits(), expected_im.to_bits());
    }
}
