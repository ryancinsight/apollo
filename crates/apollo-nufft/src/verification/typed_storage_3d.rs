use crate::{
    NufftComplexStorage, NufftPlan3D, UniformGrid3D, DEFAULT_NUFFT_KERNEL_WIDTH,
    DEFAULT_NUFFT_OVERSAMPLING,
};
use apollo_fft::{f16, ApolloError, Complex32, PrecisionProfile};
use eunomia::Complex64;
use leto::Array3;

/// Invariant: `NufftPlan3D::type1_typed_into` produces value-identical output
/// for `Complex64`, within-f32-precision for `Complex32`, and within-f16-
/// quantization for `[f16; 2]`, and rejects profile mismatches.
///
/// **Verification data** (3D DC-mode pattern):
/// - positions: `[(0.1,0.2,0.3), (0.5,0.6,0.7), (0.8,0.1,0.5)]`
/// - values: `[(1,0), (0.5,0.5), (-0.75,0.25)]`
/// - f64 path matches allocating `type1` to machine precision
/// - f32 path matches f64 reference to within 1e-5 relative tolerance
/// - f16 path matches f64 reference to within `|v|·2⁻¹⁰ + 2⁻¹⁴`
/// - f32 storage with f64 profile returns `ApolloError::Validation { field: "precision_profile" }`
#[test]
fn typed_type1_3d_supports_complex64_complex32_and_f16_storage() {
    let grid = UniformGrid3D::new(2, 2, 2, 1.0, 1.0, 1.0).expect("grid");
    let sigma = DEFAULT_NUFFT_OVERSAMPLING;
    let kernel_width = DEFAULT_NUFFT_KERNEL_WIDTH;
    let plan = NufftPlan3D::new(grid, sigma, kernel_width);
    // Buffer sizes: for n=2, sigma=2, kernel_width=6:
    //   oversampled = max(n*sigma, 2*kernel_width+1).next_power_of_two()
    //   = max(4, 13).next_power_of_two() = 16
    let mx = (grid.nx * sigma)
        .max(2 * kernel_width + 1)
        .next_power_of_two();
    let my = (grid.ny * sigma)
        .max(2 * kernel_width + 1)
        .next_power_of_two();
    let mz = (grid.nz * sigma)
        .max(2 * kernel_width + 1)
        .next_power_of_two();
    let w = kernel_width;
    let positions = vec![(0.1_f64, 0.2, 0.3), (0.5, 0.6, 0.7), (0.8, 0.1, 0.5)];
    let values64 = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.5, 0.5),
        Complex64::new(-0.75, 0.25),
    ];

    // Allocating reference (Complex64 owner path)
    let expected64 = plan.type1(&positions, &values64);

    // ── f64 path ──────────────────────────────────────────────────────
    let mut scratch_grid = Array3::<Complex64>::zeros([mx, my, mz]);
    let mut wx = vec![0.0_f64; 2 * w + 1];
    let mut wy = vec![0.0_f64; 2 * w + 1];
    let mut wz = vec![0.0_f64; 2 * w + 1];
    let mut output64 = Array3::<Complex64>::zeros([grid.nx, grid.ny, grid.nz]);
    plan.type1_typed_into(
        &positions,
        &values64,
        &mut scratch_grid,
        &mut wx,
        &mut wy,
        &mut wz,
        &mut output64,
        PrecisionProfile::HIGH_ACCURACY_F64,
    )
    .expect("typed complex64 3d type1");
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
    let mut output32 = Array3::<Complex32>::zeros([grid.nx, grid.ny, grid.nz]);
    plan.type1_typed_into(
        &positions,
        &values32,
        &mut scratch_grid,
        &mut wx,
        &mut wy,
        &mut wz,
        &mut output32,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed complex32 3d type1");
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
        "f32 3d type1 max relative error {max_rel_err_f32:.3e} exceeds 1e-5"
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
    let mut output16 = Array3::from_shape_fn([grid.nx, grid.ny, grid.nz], |_| {
        [f16::from_f32(0.0), f16::from_f32(0.0)]
    });
    plan.type1_typed_into(
        &positions,
        &values16,
        &mut scratch_grid,
        &mut wx,
        &mut wy,
        &mut wz,
        &mut output16,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("typed f16 3d type1");
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
    let mut mismatch_output = Array3::<Complex32>::zeros([grid.nx, grid.ny, grid.nz]);
    let err = plan
        .type1_typed_into(
            &positions,
            &values32,
            &mut scratch_grid,
            &mut wx,
            &mut wy,
            &mut wz,
            &mut mismatch_output,
            PrecisionProfile::HIGH_ACCURACY_F64,
        )
        .expect_err("profile mismatch must fail");
    assert!(
        matches!(err, ApolloError::Validation { ref field, .. } if field == "precision_profile"),
        "expected Validation {{ field: \"precision_profile\" }}, got {err:?}"
    );
}

/// Invariant: `NufftPlan3D::type2_typed_into` produces value-identical output
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
fn typed_type2_3d_supports_complex64_complex32_and_f16_storage() {
    let grid = UniformGrid3D::new(2, 2, 2, 1.0, 1.0, 1.0).expect("grid");
    let sigma = DEFAULT_NUFFT_OVERSAMPLING;
    let kernel_width = DEFAULT_NUFFT_KERNEL_WIDTH;
    let plan = NufftPlan3D::new(grid, sigma, kernel_width);
    let positions = vec![(0.1_f64, 0.2, 0.3), (0.5, 0.6, 0.7), (0.8, 0.1, 0.5)];

    // Build type2 input from a type1 forward pass
    let values64 = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.5, 0.5),
        Complex64::new(-0.75, 0.25),
    ];
    let coeffs64 = plan.type1(&positions, &values64);

    // Allocating reference (Complex64 owner path)
    let expected64 = plan.type2(&positions, &coeffs64);

    // ── f64 path ──────────────────────────────────────────────────────
    let mut output64 = vec![Complex64::new(0.0, 0.0); positions.len()];
    plan.type2_typed_into(
        &positions,
        &coeffs64,
        &mut output64,
        PrecisionProfile::HIGH_ACCURACY_F64,
    )
    .expect("typed complex64 3d type2");
    for (actual, expected) in output64.iter().zip(expected64.iter()) {
        eunomia::assert_abs_diff_eq!(actual.re, expected.re);
        eunomia::assert_abs_diff_eq!(actual.im, expected.im);
    }

    // ── f32 path ──────────────────────────────────────────────────────
    let coeffs32 = coeffs64.mapv(|v| Complex32::new(v.re as f32, v.im as f32));
    let represented32 = coeffs32.mapv(Complex32::to_complex64);
    let expected32 = plan.type2(&positions, &represented32);
    let mut output32 = vec![Complex32::new(0.0, 0.0); positions.len()];
    plan.type2_typed_into(
        &positions,
        &coeffs32,
        &mut output32,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed complex32 3d type2");
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
        "f32 3d type2 max relative error {max_rel_err_f32:.3e} exceeds 1e-5"
    );

    // ── f16 path ──────────────────────────────────────────────────────
    let coeffs16 = coeffs64.mapv(|v| [f16::from_f32(v.re as f32), f16::from_f32(v.im as f32)]);
    let represented16 = coeffs16.mapv(<[f16; 2]>::to_complex64);
    let expected16 = plan.type2(&positions, &represented16);
    let mut output16 = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; positions.len()];
    plan.type2_typed_into(
        &positions,
        &coeffs16,
        &mut output16,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("typed f16 3d type2");
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
            &positions,
            &coeffs32,
            &mut mismatch_output,
            PrecisionProfile::HIGH_ACCURACY_F64,
        )
        .expect_err("profile mismatch must fail");
    assert!(
        matches!(err, ApolloError::Validation { ref field, .. } if field == "precision_profile"),
        "expected Validation {{ field: \"precision_profile\" }}, got {err:?}"
    );
}
