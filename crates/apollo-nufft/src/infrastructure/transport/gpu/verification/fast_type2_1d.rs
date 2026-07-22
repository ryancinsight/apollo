use apollo_fft::{f16, PrecisionProfile};
use eunomia::{Complex32, Complex64};
use leto::Array1;

use crate::infrastructure::transport::gpu::NufftWgpuPlan1D;
use crate::{
    nufft_type2_1d_fast, UniformDomain1D, DEFAULT_NUFFT_KERNEL_WIDTH, DEFAULT_NUFFT_OVERSAMPLING,
};

use super::support::{assert_complex64_close, backend};

#[test]
fn fast_type2_1d_matches_cpu_gridded_reference() {
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
    let expected_positions: Vec<f64> = positions.iter().map(|value| *value as f64).collect();
    let expected_coefficients: Vec<Complex64> = coefficients
        .iter()
        .map(|value| Complex64::new(value.re as f64, value.im as f64))
        .collect();
    let expected = nufft_type2_1d_fast(
        &Array1::from(expected_coefficients),
        &expected_positions,
        domain,
        6,
    );

    let actual = backend
        .execute_fast_type2_1d(&plan, &coefficients, &positions)
        .expect("GPU fast type2 1D");

    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_complex64_close(*actual, *expected, 1.5e-3);
    }
}

#[test]
fn fast_type2_1d_typed_mixed_storage_matches_represented_input() {
    let Some(backend) = backend() else {
        return;
    };

    let domain = UniformDomain1D::new(8, 0.25).expect("domain");
    let plan = NufftWgpuPlan1D::new(domain, 2, 6);
    let positions = [0.0_f32, 0.25, 0.7, 1.15, 1.8];
    let coefficients16 = [
        [f16::from_f32(1.0), f16::from_f32(0.0)],
        [f16::from_f32(0.5), f16::from_f32(-0.25)],
        [f16::from_f32(-0.75), f16::from_f32(0.5)],
        [f16::from_f32(0.25), f16::from_f32(0.75)],
        [f16::from_f32(-0.5), f16::from_f32(-0.1)],
        [f16::from_f32(0.125), f16::from_f32(0.25)],
        [f16::from_f32(0.8), f16::from_f32(-0.6)],
        [f16::from_f32(-0.3), f16::from_f32(0.4)],
    ];
    let represented: Vec<Complex32> = coefficients16
        .iter()
        .map(|value| Complex32::new(value[0].to_f32(), value[1].to_f32()))
        .collect();
    let expected = backend
        .execute_fast_type2_1d(&plan, &represented, &positions)
        .expect("represented fast type2 1D");
    let mut actual = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; positions.len()];

    backend
        .execute_fast_type2_1d_typed_into(
            &plan,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
            &coefficients16,
            &positions,
            &mut actual,
        )
        .expect("mixed fast type2 1D");

    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        let expected_re = f16::from_f32(expected.re as f32);
        let expected_im = f16::from_f32(expected.im as f32);
        assert_eq!(actual[0].to_bits(), expected_re.to_bits());
        assert_eq!(actual[1].to_bits(), expected_im.to_bits());
    }
}

#[test]
fn fast_type2_1d_normalization_invariance() {
    let Some(backend) = backend() else {
        return;
    };

    let n = 16;
    let domain = UniformDomain1D::new(n, 0.25).expect("domain");
    let plan = NufftWgpuPlan1D::new(
        domain,
        DEFAULT_NUFFT_OVERSAMPLING,
        DEFAULT_NUFFT_KERNEL_WIDTH,
    );
    let mut coefficients = [Complex32::new(0.0, 0.0); 16];
    coefficients[0] = Complex32::new(1.0, 0.0);
    let positions: [f32; 5] = [0.0, 0.5, 1.25, 2.75, 3.875];
    let expected_coefficients: Vec<Complex64> = coefficients
        .iter()
        .map(|value| Complex64::new(value.re as f64, value.im as f64))
        .collect();
    let expected_positions: Vec<f64> = positions.iter().map(|&x| x as f64).collect();
    let expected = nufft_type2_1d_fast(
        &Array1::from(expected_coefficients),
        &expected_positions,
        domain,
        DEFAULT_NUFFT_KERNEL_WIDTH,
    );
    let actual = backend
        .execute_fast_type2_1d(&plan, &coefficients, &positions)
        .expect("GPU fast type2 1D with single nonzero coefficient");

    assert_eq!(
        actual.len(),
        expected.len(),
        "output length must match CPU reference"
    );
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            eunomia::abs_diff_eq!(actual.re, expected.re, epsilon = 1e-4),
            "real mismatch at position index {index}: actual={actual:?}, expected={expected:?}"
        );
        assert!(
            eunomia::abs_diff_eq!(actual.im, expected.im, epsilon = 1e-4),
            "imag mismatch at position index {index}: actual={actual:?}, expected={expected:?}"
        );
    }

    let reference = actual[0];
    for (index, value) in actual.iter().enumerate() {
        assert!(
            eunomia::abs_diff_eq!(value.re, reference.re, epsilon = 1e-4),
            "constancy regression at position index {index}: {value:?} vs reference {reference:?}"
        );
        assert!(
            eunomia::abs_diff_eq!(value.im, reference.im, epsilon = 1e-4),
            "constancy regression at position index {index}: {value:?} vs reference {reference:?}"
        );
    }
}

#[test]
fn fast_type2_1d_diagnostics_capture_load_and_ifft_grids() {
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
    let expected = backend
        .execute_fast_type2_1d(&plan, &coefficients, &positions)
        .expect("standard fast type2 1D");
    let (actual, diagnostics) = backend
        .execute_fast_type2_1d_with_diagnostics(&plan, &coefficients, &positions)
        .expect("diagnostic fast type2 1D");

    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_complex64_close(*actual, *expected, 1.0e-6);
    }

    let oversampled_len = plan.oversampling() * plan.domain().n;
    assert_eq!(diagnostics.after_load.re.len(), oversampled_len);
    assert_eq!(diagnostics.after_load.im.len(), oversampled_len);
    assert_eq!(diagnostics.after_ifft.re.len(), oversampled_len);
    assert_eq!(diagnostics.after_ifft.im.len(), oversampled_len);
    assert!(diagnostics
        .after_load
        .re
        .iter()
        .chain(diagnostics.after_load.im.iter())
        .all(|value| value.is_finite()));
    assert!(diagnostics
        .after_ifft
        .re
        .iter()
        .chain(diagnostics.after_ifft.im.iter())
        .all(|value| value.is_finite()));
    assert!(
        diagnostics
            .after_load
            .re
            .iter()
            .chain(diagnostics.after_load.im.iter())
            .any(|value| value.abs() > 0.0),
        "loaded diagnostic grid must contain deconvolved Fourier coefficients"
    );
    assert!(
        diagnostics
            .after_ifft
            .re
            .iter()
            .chain(diagnostics.after_ifft.im.iter())
            .any(|value| value.abs() > 0.0),
        "IFFT diagnostic grid must contain interpolable spatial samples"
    );
}
