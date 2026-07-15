//! Direct one-dimensional Type-2 rejection, CPU, Leto, and typed-storage contracts.

use apollo_fft::{f16, PrecisionProfile};
use eunomia::{Complex32, Complex64};
use leto::{Array1, SliceArg, Storage};

use crate::{infrastructure::transport::gpu::NufftWgpuPlan1D, nufft_type2_1d, UniformDomain1D};

use super::support::{assert_complex64_close, assert_input_length_mismatch, backend};

#[test]
fn type2_rejects_coefficient_length_mismatch() {
    let Some(backend) = backend() else {
        return;
    };
    let domain = UniformDomain1D::new(8, 0.25).expect("domain");
    let plan = NufftWgpuPlan1D::new(domain, 2, 6);
    let error = backend
        .execute_type2_1d(&plan, &[Complex32::new(1.0, 0.0); 4], &[0.0_f32, 0.25, 0.7])
        .expect_err("coefficient length mismatch must fail");
    assert_input_length_mismatch(error, 8, 4);
}

#[test]
fn type2_matches_cpu_exact_reference() {
    let Some(backend) = backend() else {
        return;
    };
    let domain = UniformDomain1D::new(8, 0.25).expect("domain");
    let plan = NufftWgpuPlan1D::new(domain, 2, 6);
    let positions = [0.0_f32, 0.25, 0.7, 1.15, 1.8];
    let coefficients = [
        Complex32::new(1.0, 0.0),
        Complex32::new(0.5, -0.25),
        Complex32::new(-0.75, 0.5),
        Complex32::new(0.25, 0.75),
        Complex32::new(-0.5, -0.1),
        Complex32::new(0.125, 0.25),
        Complex32::new(0.8, -0.6),
        Complex32::new(-0.3, 0.4),
    ];
    let expected_positions = positions
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    let expected_coefficients = coefficients
        .iter()
        .map(|value| Complex64::new(value.re as f64, value.im as f64))
        .collect::<Vec<_>>();
    let expected = nufft_type2_1d(
        &Array1::from(expected_coefficients),
        &expected_positions,
        domain,
    );
    let actual = backend
        .execute_type2_1d(&plan, &coefficients, &positions)
        .expect("GPU type2 1D");
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_complex64_close(*actual, *expected, 8.0e-5);
    }
}

#[test]
fn type2_strided_leto_matches_slice_path() {
    let Some(backend) = backend() else {
        return;
    };
    let domain = UniformDomain1D::new(4, 0.25).expect("domain");
    let plan = NufftWgpuPlan1D::new(domain, 2, 6);
    let coefficients = vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(0.25, -0.5),
        Complex32::new(-0.75, 0.25),
        Complex32::new(0.5, 0.75),
    ];
    let logical_positions = vec![0.0_f32, 0.25, 0.7, 1.15];
    let expected = backend
        .execute_type2_1d(&plan, &coefficients, &logical_positions)
        .expect("slice type2");
    let positions =
        leto::Array1::from_shape_vec([8], vec![0.0_f32, 99.0, 0.25, 99.0, 0.7, 99.0, 1.15, 99.0])
            .expect("positions");
    let strided_positions = positions
        .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
        .expect("strided positions");
    let leto_coefficients =
        leto::Array1::from_shape_vec([coefficients.len()], coefficients).expect("coeffs");
    let actual = backend
        .execute_type2_1d_leto(&plan, leto_coefficients.view(), strided_positions)
        .expect("leto type2");
    for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
        assert_complex64_close(*actual, *expected, 1.0e-6);
    }
}

#[test]
fn type2_typed_storage_matches_represented_input() {
    let Some(backend) = backend() else {
        return;
    };
    let domain = UniformDomain1D::new(8, 0.25).expect("domain");
    let plan = NufftWgpuPlan1D::new(domain, 2, 6);
    let positions = [0.0_f32, 0.25, 0.7, 1.15, 1.8];
    let coefficients = [
        [f16::from_f32(1.0), f16::from_f32(0.0)],
        [f16::from_f32(0.5), f16::from_f32(-0.25)],
        [f16::from_f32(-0.75), f16::from_f32(0.5)],
        [f16::from_f32(0.25), f16::from_f32(0.75)],
        [f16::from_f32(-0.5), f16::from_f32(-0.1)],
        [f16::from_f32(0.125), f16::from_f32(0.25)],
        [f16::from_f32(0.8), f16::from_f32(-0.6)],
        [f16::from_f32(-0.3), f16::from_f32(0.4)],
    ];
    let represented = coefficients
        .iter()
        .map(|value| Complex32::new(value[0].to_f32(), value[1].to_f32()))
        .collect::<Vec<_>>();
    let expected = backend
        .execute_type2_1d(&plan, &represented, &positions)
        .expect("represented type2 1D");
    let mut actual = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; positions.len()];
    backend
        .execute_type2_1d_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &coefficients,
            &positions,
            &mut actual,
        )
        .expect("mixed type2 1D");
    assert_eq!(actual.len(), expected.len());
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
