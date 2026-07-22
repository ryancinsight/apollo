use crate::{
    NufftComplexStorage, NufftPlan1D, UniformDomain1D, DEFAULT_NUFFT_KERNEL_WIDTH,
    DEFAULT_NUFFT_OVERSAMPLING,
};
use apollo_fft::{f16, ApolloError, Complex32, PrecisionProfile};
use eunomia::Complex64;
use leto::Array1;

/// Invariant: `NufftPlan1D::type1_typed_into` produces value-identical output
/// for `Complex64`, within-f32-precision for `Complex32`, and within-f16-
/// quantization for `[f16; 2]`, and rejects profile mismatches.
///
/// **Verification data** (DC-mode pattern):
/// - positions: `[0.1, 0.5, 1.3, 1.8]` (non-uniform, off-grid)
/// - values: `[(1,0), (0.5,-0.3), (-0.25,0.8), (0.1,0.1)]`
/// - f64 path matches allocating `type1` to machine precision
/// - f32 path matches f64 reference to within 1e-5 relative tolerance
/// - f16 path matches f64 reference to within `|v|·2⁻¹⁰ + 2⁻¹⁴`
/// - f32 storage with f64 profile returns `ApolloError::Validation { field: "precision_profile" }`
#[test]
fn typed_type1_1d_supports_complex64_complex32_and_f16_storage() {
    let domain = UniformDomain1D::new(8, 0.25).expect("domain");
    let sigma = DEFAULT_NUFFT_OVERSAMPLING;
    let kernel_width = DEFAULT_NUFFT_KERNEL_WIDTH;
    let plan = NufftPlan1D::new(domain, sigma, kernel_width);
    // Buffer sizes derived from plan construction: m = sigma * n, n_out = n
    let m = sigma * domain.n;
    let n_out = domain.n;
    let positions = vec![0.1_f64, 0.5, 1.3, 1.8];
    let values64 = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.5, -0.3),
        Complex64::new(-0.25, 0.8),
        Complex64::new(0.1, 0.1),
    ];

    // Allocating reference (Complex64 owner path)
    let expected64 = plan.type1(&positions, &values64);

    // ── f64 path ──────────────────────────────────────────────────────
    let mut scratch = vec![Complex64::new(0.0, 0.0); m];
    let mut output64 = vec![Complex64::new(0.0, 0.0); n_out];
    plan.type1_typed_into(
        &positions,
        &values64,
        &mut scratch,
        &mut output64,
        PrecisionProfile::HIGH_ACCURACY_F64,
    )
    .expect("typed complex64 type1");
    for (actual, expected) in output64.iter().zip(expected64.iter()) {
        eunomia::assert_abs_diff_eq!(actual.re, expected.re);
        eunomia::assert_abs_diff_eq!(actual.im, expected.im);
    }

    // ── f32 path ──────────────────────────────────────────────────────
    let values32: Vec<Complex32> = values64
        .iter()
        .map(|v| Complex32::new(v.re as f32, v.im as f32))
        .collect();
    let represented32: Vec<Complex64> = values32
        .iter()
        .copied()
        .map(Complex32::to_complex64)
        .collect();
    let expected32 = plan.type1(&positions, &represented32);
    let mut output32 = vec![Complex32::new(0.0, 0.0); n_out];
    plan.type1_typed_into(
        &positions,
        &values32,
        &mut scratch,
        &mut output32,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed complex32 type1");
    let mut max_rel_err_f32 = 0.0_f64;
    for (actual, expected) in output32.iter().zip(expected32.iter()) {
        let denom_re = expected.re.abs().max(1e-30);
        let denom_im = expected.im.abs().max(1e-30);
        max_rel_err_f32 = max_rel_err_f32
            .max((f64::from(actual.re) - expected.re).abs() / denom_re)
            .max((f64::from(actual.im) - expected.im).abs() / denom_im);
    }
    assert!(
        max_rel_err_f32 < 1e-5,
        "f32 type1 max relative error {max_rel_err_f32:.3e} exceeds 1e-5"
    );

    // ── f16 path ──────────────────────────────────────────────────────
    let values16: Vec<[f16; 2]> = values64
        .iter()
        .map(|v| [f16::from_f32(v.re as f32), f16::from_f32(v.im as f32)])
        .collect();
    let represented16: Vec<Complex64> = values16
        .iter()
        .copied()
        .map(<[f16; 2]>::to_complex64)
        .collect();
    let expected16 = plan.type1(&positions, &represented16);
    let mut output16 = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; n_out];
    plan.type1_typed_into(
        &positions,
        &values16,
        &mut scratch,
        &mut output16,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("typed f16 type1");
    for (actual, expected) in output16.iter().zip(expected16.iter()) {
        let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        assert!(
            (f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound,
            "f16 re: actual={}, expected={}, bound={re_bound:.3e}",
            f64::from(actual[0].to_f32()),
            expected.re
        );
        assert!(
            (f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound,
            "f16 im: actual={}, expected={}, bound={im_bound:.3e}",
            f64::from(actual[1].to_f32()),
            expected.im
        );
    }

    // ── profile mismatch ──────────────────────────────────────────────
    let mut mismatch_output = vec![Complex32::new(0.0, 0.0); n_out];
    let err = plan
        .type1_typed_into(
            &positions,
            &values32,
            &mut scratch,
            &mut mismatch_output,
            PrecisionProfile::HIGH_ACCURACY_F64,
        )
        .expect_err("profile mismatch must fail");
    assert!(
        matches!(err, ApolloError::Validation { ref field, .. } if field == "precision_profile"),
        "expected Validation {{ field: \"precision_profile\" }}, got {err:?}"
    );
}

/// Invariant: `NufftPlan1D::type2_typed_into` produces value-identical output
/// for `Complex64`, within-f32-precision for `Complex32`, and within-f16-
/// quantization for `[f16; 2]`, and rejects profile mismatches.
///
/// **Verification strategy:**
/// - Use the f64 type1 output as type2 input (round-trip guarantee)
/// - f64 path matches allocating `type2` to machine precision
/// - f32 path matches f64 reference to within 1e-5 relative tolerance
/// - f16 path matches f64 reference to within `|v|·2⁻¹⁰ + 2⁻¹⁴`
/// - f32 storage with f64 profile returns `ApolloError::Validation { field: "precision_profile" }`
#[test]
fn typed_type2_1d_supports_complex64_complex32_and_f16_storage() {
    let domain = UniformDomain1D::new(8, 0.25).expect("domain");
    let sigma = DEFAULT_NUFFT_OVERSAMPLING;
    let kernel_width = DEFAULT_NUFFT_KERNEL_WIDTH;
    let plan = NufftPlan1D::new(domain, sigma, kernel_width);
    let m = sigma * domain.n;
    let positions = vec![0.1_f64, 0.5, 1.3, 1.8];

    // Build type2 input from a type1 forward pass
    let values64 = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.5, -0.3),
        Complex64::new(-0.25, 0.8),
        Complex64::new(0.1, 0.1),
    ];
    let coeffs64_array = plan.type1(&positions, &values64);

    // Allocating reference (Complex64 owner path)
    let expected64 = plan.type2(&coeffs64_array, &positions);

    // ── f64 path ──────────────────────────────────────────────────────
    let mut scratch = vec![Complex64::new(0.0, 0.0); m];
    let mut output64 = vec![Complex64::new(0.0, 0.0); positions.len()];
    plan.type2_typed_into(
        coeffs64_array.as_slice().unwrap(),
        &positions,
        &mut scratch,
        &mut output64,
        PrecisionProfile::HIGH_ACCURACY_F64,
    )
    .expect("typed complex64 type2");
    for (actual, expected) in output64.iter().zip(expected64.iter()) {
        eunomia::assert_abs_diff_eq!(actual.re, expected.re);
        eunomia::assert_abs_diff_eq!(actual.im, expected.im);
    }

    // ── f32 path ──────────────────────────────────────────────────────
    let coeffs32: Vec<Complex32> = coeffs64_array
        .iter()
        .map(|v| Complex32::new(v.re as f32, v.im as f32))
        .collect();
    let represented32: Vec<Complex64> = coeffs32
        .iter()
        .copied()
        .map(Complex32::to_complex64)
        .collect();
    let represented32_array = Array1::from_shape_vec([represented32.len()], represented32)
        .expect("invariant: represented f32 coefficient length matches Array1 shape");
    let expected32 = plan.type2(&represented32_array, &positions);
    let mut output32 = vec![Complex32::new(0.0, 0.0); positions.len()];
    plan.type2_typed_into(
        &coeffs32,
        &positions,
        &mut scratch,
        &mut output32,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed complex32 type2");
    let mut max_rel_err_f32 = 0.0_f64;
    for (actual, expected) in output32.iter().zip(expected32.iter()) {
        let denom_re = expected.re.abs().max(1e-30);
        let denom_im = expected.im.abs().max(1e-30);
        max_rel_err_f32 = max_rel_err_f32
            .max((f64::from(actual.re) - expected.re).abs() / denom_re)
            .max((f64::from(actual.im) - expected.im).abs() / denom_im);
    }
    assert!(
        max_rel_err_f32 < 1e-5,
        "f32 type2 max relative error {max_rel_err_f32:.3e} exceeds 1e-5"
    );

    // ── f16 path ──────────────────────────────────────────────────────
    let coeffs16: Vec<[f16; 2]> = coeffs64_array
        .iter()
        .map(|v| [f16::from_f32(v.re as f32), f16::from_f32(v.im as f32)])
        .collect();
    let represented16: Vec<Complex64> = coeffs16
        .iter()
        .copied()
        .map(<[f16; 2]>::to_complex64)
        .collect();
    let represented16_array = Array1::from_shape_vec([represented16.len()], represented16)
        .expect("invariant: represented f16 coefficient length matches Array1 shape");
    let expected16 = plan.type2(&represented16_array, &positions);
    let mut output16 = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; positions.len()];
    plan.type2_typed_into(
        &coeffs16,
        &positions,
        &mut scratch,
        &mut output16,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("typed f16 type2");
    for (actual, expected) in output16.iter().zip(expected16.iter()) {
        let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        assert!(
            (f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound,
            "f16 re: actual={}, expected={}, bound={re_bound:.3e}",
            f64::from(actual[0].to_f32()),
            expected.re
        );
        assert!(
            (f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound,
            "f16 im: actual={}, expected={}, bound={im_bound:.3e}",
            f64::from(actual[1].to_f32()),
            expected.im
        );
    }

    // ── profile mismatch ──────────────────────────────────────────────
    let mut mismatch_output = vec![Complex32::new(0.0, 0.0); positions.len()];
    let err = plan
        .type2_typed_into(
            &coeffs32,
            &positions,
            &mut scratch,
            &mut mismatch_output,
            PrecisionProfile::HIGH_ACCURACY_F64,
        )
        .expect_err("profile mismatch must fail");
    assert!(
        matches!(err, ApolloError::Validation { ref field, .. } if field == "precision_profile"),
        "expected Validation {{ field: \"precision_profile\" }}, got {err:?}"
    );
}
