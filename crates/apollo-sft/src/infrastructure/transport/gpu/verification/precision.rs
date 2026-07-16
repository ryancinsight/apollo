//! Value-semantic SFT GPU explicit-precision contracts.

use apollo_fft::{f16, PrecisionProfile};
use eunomia::Complex64;

use crate::{
    infrastructure::transport::gpu::{SftWgpuPlan, WgpuError},
    SparseSpectrum,
};

use super::support::{assert_accelerated_complex_close, backend, INVERSE_N0_ERROR_BOUND};

#[test]
fn typed_path_rejects_profile_mismatch_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = SftWgpuPlan::new(4, 2);
    let mixed_input: Vec<[f16; 2]> = vec![[f16::from_f32(0.0); 2]; 4];

    let fwd_err = backend
        .execute_forward_typed::<[f16; 2]>(&plan, PrecisionProfile::LOW_PRECISION_F32, &mixed_input)
        .expect_err("profile mismatch must fail");
    assert!(matches!(fwd_err, WgpuError::InvalidPrecisionProfile));

    let spectrum = SparseSpectrum::new(4);
    let mut mixed_output: Vec<[f16; 2]> = vec![[f16::from_f32(0.0); 2]; 4];
    let inv_err = backend
        .execute_inverse_typed_into::<[f16; 2]>(
            &plan,
            PrecisionProfile::LOW_PRECISION_F32,
            &spectrum,
            &mut mixed_output,
        )
        .expect_err("profile mismatch must fail");
    assert!(matches!(inv_err, WgpuError::InvalidPrecisionProfile));
}

#[test]
fn high_accuracy_sparse_coefficients_are_not_silently_narrowed_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = SftWgpuPlan::new(4, 1);
    let mut spectrum = SparseSpectrum::new(4);
    spectrum
        .insert(1, Complex64::new(1.0 / 3.0, 0.0))
        .expect("in-range sparse coefficient");
    let error = backend
        .execute_inverse(&plan, &spectrum)
        .expect_err("non-f32 sparse coefficient must be rejected");
    assert!(matches!(
        error,
        WgpuError::PrecisionLoss {
            component: "real",
            value
        } if value == 1.0 / 3.0
    ));
}

#[test]
fn explicit_quantization_is_required_and_value_visible_when_device_exists() {
    let Some(backend) = backend() else {
        return;
    };
    let plan = SftWgpuPlan::new(4, 1);
    let mut spectrum = SparseSpectrum::new(4);
    spectrum
        .insert(1, Complex64::new(1.0 / 3.0, -1.0 / 7.0))
        .expect("in-range sparse coefficient");
    let quantized = backend
        .quantize_spectrum(&plan, &spectrum)
        .expect("explicit accelerator quantization");
    assert_eq!(quantized.frequencies, spectrum.frequencies);
    assert_eq!(
        quantized.values,
        vec![Complex64::new(
            f64::from((1.0_f64 / 3.0) as f32),
            f64::from((-1.0_f64 / 7.0) as f32),
        )]
    );
    let actual = backend
        .execute_inverse(&plan, &quantized)
        .expect("quantized spectrum executes");
    assert_eq!(actual.len(), plan.len());
    assert_accelerated_complex_close(
        actual[0],
        eunomia::Complex32::new(1.0 / 12.0, -1.0 / 28.0),
        INVERSE_N0_ERROR_BOUND,
    );
}
