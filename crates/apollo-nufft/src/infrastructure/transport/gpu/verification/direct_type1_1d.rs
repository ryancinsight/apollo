//! Direct one-dimensional Type-1 CPU, Leto, and typed-storage contracts.

use apollo_fft::{f16, PrecisionProfile};
use eunomia::{Complex32, Complex64};
use leto::Storage;

use crate::{infrastructure::transport::gpu::NufftWgpuPlan1D, nufft_type1_1d, UniformDomain1D};

use super::support::{assert_complex64_close, backend};

#[test]
fn type1_matches_cpu_exact_reference() {
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
    let expected_positions = positions
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    let expected_values = values
        .iter()
        .map(|value| Complex64::new(value.re as f64, value.im as f64))
        .collect::<Vec<_>>();
    let expected = nufft_type1_1d(&expected_positions, &expected_values, domain);
    let actual = backend
        .execute_type1_1d(&plan, &positions, &values)
        .expect("GPU type1 1D");

    assert_eq!(actual.size(), expected.size());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_complex64_close(*actual, *expected, 4.0e-5);
    }
}

#[test]
fn type1_leto_matches_slice_path() {
    let Some(backend) = backend() else {
        return;
    };
    let domain = UniformDomain1D::new(8, 0.25).expect("domain");
    let plan = NufftWgpuPlan1D::new(domain, 2, 6);
    let positions = vec![0.0_f32, 0.25, 0.7, 1.15];
    let values = vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(0.5, -0.25),
        Complex32::new(-0.75, 0.5),
        Complex32::new(0.25, 0.75),
    ];
    let expected = backend
        .execute_type1_1d(&plan, &positions, &values)
        .expect("slice type1");
    let leto_positions =
        leto::Array1::from_shape_vec([positions.len()], positions).expect("positions");
    let leto_values = leto::Array1::from_shape_vec([values.len()], values).expect("values");
    let actual = backend
        .execute_type1_1d_leto(&plan, leto_positions.view(), leto_values.view())
        .expect("leto type1");

    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_complex64_close(*actual, *expected, 1.0e-6);
    }
}

#[test]
fn type1_typed_storage_matches_represented_input() {
    let Some(backend) = backend() else {
        return;
    };
    let domain = UniformDomain1D::new(8, 0.25).expect("domain");
    let plan = NufftWgpuPlan1D::new(domain, 2, 6);
    let positions = [0.0_f32, 0.25, 0.7, 1.15];
    let values = [
        [f16::from_f32(1.0), f16::from_f32(0.0)],
        [f16::from_f32(0.5), f16::from_f32(-0.25)],
        [f16::from_f32(-0.75), f16::from_f32(0.5)],
        [f16::from_f32(0.25), f16::from_f32(0.75)],
    ];
    let represented = values
        .iter()
        .map(|value| Complex32::new(value[0].to_f32(), value[1].to_f32()))
        .collect::<Vec<_>>();
    let expected = backend
        .execute_type1_1d(&plan, &positions, &represented)
        .expect("represented type1 1D");
    let mut actual = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; domain.n];
    backend
        .execute_type1_1d_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &positions,
            &values,
            &mut actual,
        )
        .expect("mixed type1 1D");

    assert_eq!(actual.len(), expected.size());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_eq!(
            actual[0].to_bits(),
            f16::from_f32(expected.re as f32).to_bits()
        );
        assert_eq!(
            actual[1].to_bits(),
            f16::from_f32(expected.im as f32).to_bits()
        );
    }
}
