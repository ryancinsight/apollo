use super::execute::{
    fill_type1_weight_lanes, fill_type2_weight_lanes, interleaved_lanes, nufft_type1_coefficient,
    nufft_type1_coefficient_hermes, nufft_type2_sample, nufft_type2_sample_hermes,
};
use super::*;
use crate::DEFAULT_NUFFT_KERNEL_WIDTH;
use eunomia::assert_abs_diff_eq;
#[test]
fn release_scratch_returns_pools_to_baseline_and_serves_again() {
    let domain = UniformDomain1D::new(8, 0.125).unwrap();
    let plan = NufftPlan1D::new(
        domain,
        DEFAULT_NUFFT_OVERSAMPLING,
        DEFAULT_NUFFT_KERNEL_WIDTH,
    );
    let positions = vec![0.0, 0.09, 0.21, 0.47];
    let values = vec![
        Complex32::new(1.0, 0.25),
        Complex32::new(-0.5, 0.75),
        Complex32::new(0.25, -0.1),
        Complex32::new(0.125, 0.5),
    ];
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.m];
    let mut first = vec![Complex32::new(0.0, 0.0); plan.n_out];
    plan.type1_typed_into(
        &positions,
        &values,
        &mut scratch,
        &mut first,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed type-1 grows the pools");
    assert!(
        super::dimension_1d_scratch_capacities() != (0, 0),
        "the typed type-1 must retain a workspace"
    );
    super::release_dimension_1d_scratch();
    assert_eq!(
        super::dimension_1d_scratch_capacities(),
        (0, 0),
        "release must return every 1-D pool to baseline"
    );
    let mut second = vec![Complex32::new(0.0, 0.0); plan.n_out];
    plan.type1_typed_into(
        &positions,
        &values,
        &mut scratch,
        &mut second,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed type-1 after release");
    assert!(
        super::dimension_1d_scratch_capacities() != (0, 0),
        "the pools must serve after release"
    );
    assert_eq!(first, second, "post-release output must match");
}

/// DC mode invariant: f_0 = sum(c_j) because exp(-2pi*i * 0 * x_j / L) = 1 for all j.
///
/// With values [1.0, 0.5-0.25i, -0.5+0.75i, 0.25+0.1i]:
/// sum = (1.0+0.5-0.5+0.25) + (0-0.25+0.75+0.1)i = 1.25 + 0.6i
#[test]
fn type1_dc_mode_equals_sum_of_values() {
    let domain = UniformDomain1D::new(8, 0.125).unwrap();
    let positions = vec![0.0, 0.125, 0.25, 0.375];
    let values = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.5, -0.25),
        Complex64::new(-0.5, 0.75),
        Complex64::new(0.25, 0.1),
    ];
    let output = nufft_type1_1d(&positions, &values, domain);
    assert_eq!(output.size(), 8);
    // DC mode: sum of all values (analytical)
    let expected_dc = Complex64::new(1.25, 0.6);
    let err = (output[[0]] - expected_dc).norm();
    assert!(
        err < 1e-12,
        "DC mode err={err}: got {:?}, expected {:?}",
        output[[0]],
        expected_dc
    );
    // All outputs must be finite
    for (k, v) in output.iter().enumerate() {
        assert!(v.norm().is_finite(), "mode {k} is non-finite: {v:?}");
    }
}

#[test]
fn hermes_type1_coefficients_match_scalar_formula_at_threshold() {
    let domain = UniformDomain1D::new(128, 1.0).unwrap();
    let sample_count = DIRECT_PAR_OP_THRESHOLD / domain.n;
    let positions = (0..sample_count)
        .map(|index| (index as f64 * 0.37).rem_euclid(domain.length()))
        .collect::<Vec<_>>();
    let values = (0..sample_count)
        .map(|index| Complex64::new((index as f64 * 0.125).sin(), (index as f64 * 0.25).cos()))
        .collect::<Vec<_>>();
    let value_lanes = interleaved_lanes(&values);
    let two_pi_over_l = 2.0 * PI / domain.length();

    for k in [0usize, 1, 17, 64, 96, 127] {
        let actual =
            nufft_type1_coefficient_hermes(k, &positions, value_lanes, domain.n, two_pi_over_l);
        let expected = nufft_type1_coefficient(k, &positions, &values, domain.n, two_pi_over_l);

        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-10);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-10);
    }
}

#[test]
fn hermes_type2_samples_match_scalar_formula_at_threshold() {
    let domain = UniformDomain1D::new(128, 1.0).unwrap();
    let coeffs = Array1::from_shape_fn([domain.n], |[index]| {
        Complex64::new((index as f64 * 0.125).sin(), (index as f64 * 0.25).cos())
    });
    let coeff_lanes = interleaved_lanes(coeffs.as_slice().expect("contiguous coeffs"));

    for x in [0.0_f64, 0.125, 0.37, 0.5, 0.875, 1.25] {
        let actual = nufft_type2_sample_hermes(x, coeff_lanes, domain);
        let expected = nufft_type2_sample(x, &coeffs, domain);

        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-10);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-10);
    }
}

#[test]
fn direct_weight_lanes_match_type1_and_type2_phasors() {
    let domain = UniformDomain1D::new(16, 0.5).unwrap();
    let positions = [0.0_f64, 0.125, 0.25, 0.375];
    let mut lanes = vec![0.0; positions.len() * 2];
    let two_pi_over_l = 2.0 * PI / domain.length();
    let k_signed = fft_signed_index(13, domain.n);

    fill_type1_weight_lanes(&mut lanes, &positions, k_signed, two_pi_over_l);
    for (index, x) in positions.iter().enumerate() {
        let angle = -two_pi_over_l * k_signed as f64 * x;
        assert_eq!(lanes[index * 2].to_bits(), angle.cos().to_bits());
        assert_eq!(lanes[index * 2 + 1].to_bits(), angle.sin().to_bits());
    }

    fill_type2_weight_lanes(&mut lanes, 0.25, domain);
    for index in 0..positions.len() {
        let signed = fft_signed_index(index, domain.n);
        let angle = two_pi_over_l * signed as f64 * 0.25;
        assert_eq!(lanes[index * 2].to_bits(), angle.cos().to_bits());
        assert_eq!(lanes[index * 2 + 1].to_bits(), angle.sin().to_bits());
    }
}

/// Fast vs exact: fixed non-uniform positions and values.
/// Tolerance 1e-5 for kernel_width=6, sigma=2.
#[test]
fn type1_fast_vs_exact_agreement_fixed_inputs() {
    let domain = UniformDomain1D::new(8, 0.125).unwrap();
    let positions = vec![0.0, 0.125, 0.25, 0.375];
    let values = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.5, -0.25),
        Complex64::new(-0.5, 0.75),
        Complex64::new(0.25, 0.1),
    ];
    let exact = nufft_type1_1d(&positions, &values, domain);
    let fast = nufft_type1_1d_fast(&positions, &values, domain, DEFAULT_NUFFT_KERNEL_WIDTH);
    let scale = exact
        .iter()
        .map(|v| v.norm())
        .fold(1.0_f64, f64::max)
        .max(1e-30);
    for (k, (e, f)) in exact.iter().zip(fast.iter()).enumerate() {
        let err = (e - f).norm() / scale;
        assert!(err < 1e-5, "k={k}: exact={e:?}, fast={f:?}, rel_err={err}");
    }
}

#[test]
fn fast_1d_tracks_exact() {
    let domain = UniformDomain1D::new(32, 0.05).unwrap();
    let positions: Vec<f64> = (0..20)
        .map(|i| ((i as f64 * 0.137) % domain.length()).abs())
        .collect();
    let values: Vec<Complex64> = (0..20)
        .map(|i| Complex64::new((i as f64 * 0.3).cos(), (i as f64 * 0.2).sin()))
        .collect();
    let exact = nufft_type1_1d(&positions, &values, domain);
    let fast = nufft_type1_1d_fast(&positions, &values, domain, DEFAULT_NUFFT_KERNEL_WIDTH);
    let scale = exact.iter().map(|value| value.norm()).fold(1.0, f64::max);
    let max_err = exact
        .iter()
        .zip(fast.iter())
        .map(|(lhs, rhs)| (lhs - rhs).norm())
        .fold(0.0, f64::max);
    assert!(max_err / scale < 1e-6);
}

#[test]
fn typed_1d_paths_support_complex64_complex32_and_mixed_f16_storage() {
    let domain = UniformDomain1D::new(8, 0.125).unwrap();
    let plan = NufftPlan1D::new(
        domain,
        DEFAULT_NUFFT_OVERSAMPLING,
        DEFAULT_NUFFT_KERNEL_WIDTH,
    );
    let positions = vec![0.0, 0.09, 0.21, 0.47];
    let values64 = vec![
        Complex64::new(1.0, 0.25),
        Complex64::new(-0.5, 0.75),
        Complex64::new(0.25, -0.1),
        Complex64::new(0.125, 0.5),
    ];
    let expected = plan.type1(&positions, &values64);
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.m];
    let mut output64 = vec![Complex64::new(0.0, 0.0); plan.n_out];
    plan.type1_typed_into(
        &positions,
        &values64,
        &mut scratch,
        &mut output64,
        PrecisionProfile::HIGH_ACCURACY_F64,
    )
    .expect("typed complex64 type1");
    for (actual, expected) in output64.iter().zip(expected.iter()) {
        assert!((*actual - *expected).norm() < 1.0e-12);
    }

    let values32: Vec<Complex32> = values64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect();
    let represented32: Vec<Complex64> = values32.iter().copied().map(CpuStorage::to_cpu).collect();
    let expected32 = plan.type1(&positions, &represented32);
    let mut output32 = vec![Complex32::new(0.0, 0.0); plan.n_out];
    plan.type1_typed_into(
        &positions,
        &values32,
        &mut scratch,
        &mut output32,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed complex32 type1");
    for (actual, expected) in output32.iter().zip(expected32.iter()) {
        assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
        assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
    }

    let mut recovered32 = vec![Complex32::new(0.0, 0.0); positions.len()];
    plan.type2_typed_into(
        &output32,
        &positions,
        &mut scratch,
        &mut recovered32,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed complex32 type2");
    let expected_type2 = plan.type2(&expected32, &positions);
    for (actual, expected) in recovered32.iter().zip(expected_type2.iter()) {
        assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-3);
        assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-3);
    }

    let values16: Vec<[F16; 2]> = values64
        .iter()
        .map(|value| {
            [
                F16::from_f32(value.re as f32),
                F16::from_f32(value.im as f32),
            ]
        })
        .collect();
    let represented16: Vec<Complex64> = values16.iter().copied().map(CpuStorage::to_cpu).collect();
    let expected16 = plan.type1(&positions, &represented16);
    let mut output16 = vec![[F16::from_f32(0.0), F16::from_f32(0.0)]; plan.n_out];
    plan.type1_typed_into(
        &positions,
        &values16,
        &mut scratch,
        &mut output16,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("typed F16 type1");
    for (actual, expected) in output16.iter().zip(expected16.iter()) {
        let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        assert!((f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound);
        assert!((f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound);
    }
}

#[test]
fn leto_type1_matches_slice_reference() {
    let domain = UniformDomain1D::new(8, 0.125).unwrap();
    let plan = NufftPlan1D::new(
        domain,
        DEFAULT_NUFFT_OVERSAMPLING,
        DEFAULT_NUFFT_KERNEL_WIDTH,
    );
    let positions = vec![0.0, 0.09, 0.21, 0.47];
    let values = vec![
        Complex64::new(1.0, 0.25),
        Complex64::new(-0.5, 0.75),
        Complex64::new(0.25, -0.1),
        Complex64::new(0.125, 0.5),
    ];
    let leto_positions =
        leto::Array1::from_shape_vec([positions.len()], positions.clone()).unwrap();
    let leto_values = leto::Array1::from_shape_vec([values.len()], values.clone()).unwrap();
    let expected = plan.type1(&positions, &values);

    let actual = plan
        .type1_leto(leto_positions.view(), leto_values.view())
        .expect("leto type1");
    let actual_view = actual.view();
    let actual = actual_view.as_slice().expect("contiguous leto output");

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((*actual - *expected).norm() < 1.0e-12);
    }
}

#[test]
fn leto_strided_type1_matches_slice_reference() {
    let domain = UniformDomain1D::new(8, 0.125).unwrap();
    let plan = NufftPlan1D::new(
        domain,
        DEFAULT_NUFFT_OVERSAMPLING,
        DEFAULT_NUFFT_KERNEL_WIDTH,
    );
    let positions = vec![0.0, 0.09, 0.21, 0.47];
    let values = vec![
        Complex64::new(1.0, 0.25),
        Complex64::new(-0.5, 0.75),
        Complex64::new(0.25, -0.1),
        Complex64::new(0.125, 0.5),
    ];
    let mut position_backing = Vec::with_capacity(positions.len() * 2);
    let mut value_backing = Vec::with_capacity(values.len() * 2);
    for (position, value) in positions.iter().copied().zip(values.iter().copied()) {
        position_backing.push(position);
        position_backing.push(99.0);
        value_backing.push(value);
        value_backing.push(Complex64::new(99.0, 99.0));
    }
    let leto_positions =
        leto::Array1::from_shape_vec([position_backing.len()], position_backing).unwrap();
    let leto_values = leto::Array1::from_shape_vec([value_backing.len()], value_backing).unwrap();
    let strided_positions = leto_positions
        .view()
        .slice(&[(0, positions.len() * 2, 2)])
        .expect("strided positions");
    let strided_values = leto_values
        .view()
        .slice(&[(0, values.len() * 2, 2)])
        .expect("strided values");
    let expected = plan.type1(&positions, &values);

    let actual = plan
        .type1_leto(strided_positions, strided_values)
        .expect("leto type1");
    let actual_view = actual.view();
    let actual = actual_view.as_slice().expect("contiguous leto output");

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((*actual - *expected).norm() < 1.0e-12);
    }
}

#[test]
fn leto_type2_matches_slice_reference() {
    let domain = UniformDomain1D::new(8, 0.125).unwrap();
    let plan = NufftPlan1D::new(
        domain,
        DEFAULT_NUFFT_OVERSAMPLING,
        DEFAULT_NUFFT_KERNEL_WIDTH,
    );
    let positions = vec![0.0, 0.09, 0.21, 0.47];
    let coefficient_values = (0..plan.n_out)
        .map(|i| Complex64::new((i as f64 * 0.2).cos(), (i as f64 * 0.3).sin()))
        .collect::<Vec<_>>();
    let coefficients = Array1::from_shape_vec([coefficient_values.len()], coefficient_values)
        .expect("invariant: generated coefficient vector length matches plan output size");
    let leto_positions =
        leto::Array1::from_shape_vec([positions.len()], positions.clone()).unwrap();
    let leto_coefficients =
        leto::Array1::from_shape_vec([coefficients.size()], coefficients.clone().into_vec())
            .unwrap();
    let expected = plan.type2(&coefficients, &positions);

    let actual = plan
        .type2_leto(leto_coefficients.view(), leto_positions.view())
        .expect("leto type2");
    let actual_view = actual.view();
    let actual = actual_view.as_slice().expect("contiguous leto output");

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((*actual - *expected).norm() < 1.0e-12);
    }
}

#[test]
fn typed_leto_type1_and_type2_match_slice_reference() {
    let domain = UniformDomain1D::new(8, 0.125).unwrap();
    let plan = NufftPlan1D::new(
        domain,
        DEFAULT_NUFFT_OVERSAMPLING,
        DEFAULT_NUFFT_KERNEL_WIDTH,
    );
    let positions = vec![0.0, 0.09, 0.21, 0.47];
    let values = vec![
        Complex32::new(1.0, 0.25),
        Complex32::new(-0.5, 0.75),
        Complex32::new(0.25, -0.1),
        Complex32::new(0.125, 0.5),
    ];
    let leto_positions =
        leto::Array1::from_shape_vec([positions.len()], positions.clone()).unwrap();
    let leto_values = leto::Array1::from_shape_vec([values.len()], values.clone()).unwrap();
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.m];
    let mut expected_type1 = vec![Complex32::new(0.0, 0.0); plan.n_out];
    plan.type1_typed_into(
        &positions,
        &values,
        &mut scratch,
        &mut expected_type1,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed slice type1");

    let actual_type1 = plan
        .type1_leto_typed(
            leto_positions.view(),
            leto_values.view(),
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed leto type1");
    let actual_type1_view = actual_type1.view();
    let actual_type1 = actual_type1_view
        .as_slice()
        .expect("contiguous leto output");
    for (actual, expected) in actual_type1.iter().zip(expected_type1.iter()) {
        assert_eq!(actual.re.to_bits(), expected.re.to_bits());
        assert_eq!(actual.im.to_bits(), expected.im.to_bits());
    }

    let leto_coefficients =
        leto::Array1::from_shape_vec([expected_type1.len()], expected_type1.clone()).unwrap();
    let mut expected_type2 = vec![Complex32::new(0.0, 0.0); positions.len()];
    plan.type2_typed_into(
        &expected_type1,
        &positions,
        &mut scratch,
        &mut expected_type2,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed slice type2");
    let actual_type2 = plan
        .type2_leto_typed(
            leto_coefficients.view(),
            leto_positions.view(),
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed leto type2");
    let actual_type2_view = actual_type2.view();
    let actual_type2 = actual_type2_view
        .as_slice()
        .expect("contiguous leto output");
    for (actual, expected) in actual_type2.iter().zip(expected_type2.iter()) {
        assert_eq!(actual.re.to_bits(), expected.re.to_bits());
        assert_eq!(actual.im.to_bits(), expected.im.to_bits());
    }
}

#[test]
fn typed_1d_paths_reject_profile_and_shape_mismatch() {
    let domain = UniformDomain1D::new(4, 0.25).unwrap();
    let plan = NufftPlan1D::new(
        domain,
        DEFAULT_NUFFT_OVERSAMPLING,
        DEFAULT_NUFFT_KERNEL_WIDTH,
    );
    let positions = vec![0.0, 0.25];
    let values = vec![Complex32::new(1.0, 0.0), Complex32::new(0.5, 0.25)];
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.m];
    let mut output = vec![Complex32::new(0.0, 0.0); plan.n_out];
    let err = plan
        .type1_typed_into(
            &positions,
            &values,
            &mut scratch,
            &mut output,
            PrecisionProfile::HIGH_ACCURACY_F64,
        )
        .expect_err("profile mismatch");
    assert!(matches!(
        err,
        ApolloError::Validation { field, .. } if field == "precision_profile"
    ));

    let mut short_output = vec![Complex32::new(0.0, 0.0); plan.n_out - 1];
    let err = plan
        .type1_typed_into(
            &positions,
            &values,
            &mut scratch,
            &mut short_output,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect_err("shape mismatch");
    assert!(matches!(err, ApolloError::ShapeMismatch { .. }));
}
