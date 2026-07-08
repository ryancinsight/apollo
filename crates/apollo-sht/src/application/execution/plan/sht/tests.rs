use super::helpers::{
    coefficient_lanes, interleaved_lanes, sht_forward_mode_sum, sht_forward_mode_sum_hermes,
    sht_inverse_sample, sht_inverse_sample_hermes, SHT_HERMES_DOT_LEN_THRESHOLD,
};
use super::typed::ShtComplexStorage;
use super::ShtPlan;
use crate::domain::contracts::error::ShtError;
use crate::domain::spectrum::coefficients::SphericalHarmonicCoefficients;
use crate::infrastructure::kernel::spherical_harmonic::spherical_harmonic;
use apollo_fft::{f16, PrecisionProfile};
use approx::assert_abs_diff_eq;
use eunomia::{Complex32, Complex64};
use leto::Array2;

fn coefficient_shape(plan: &ShtPlan) -> [usize; 2] {
    [
        plan.grid().max_degree() + 1,
        2 * plan.grid().max_degree() + 1,
    ]
}

#[test]
fn hermes_forward_mode_sum_matches_scalar_formula_at_threshold() {
    let plan = ShtPlan::new(8, SHT_HERMES_DOT_LEN_THRESHOLD, 3).expect("plan");
    let row = (0..plan.grid().longitudes())
        .map(|lon| {
            Complex64::new(
                (lon as f64 * 0.013).sin() + 0.125,
                (lon as f64 * 0.017).cos() - 0.25,
            )
        })
        .collect::<Vec<_>>();
    let lanes = interleaved_lanes(&row);
    let theta = plan.theta(3);

    for (degree, order) in [(0, 0), (1, -1), (2, 1), (3, 3)] {
        let expected = sht_forward_mode_sum(&row, degree, order, theta, plan.grid().longitudes());
        let actual =
            sht_forward_mode_sum_hermes(lanes, degree, order, theta, plan.grid().longitudes());
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-11);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-11);
    }
}

#[test]
fn hermes_inverse_sample_matches_scalar_formula_at_threshold() {
    let max_degree = 15;
    let plan = ShtPlan::new(16, 31, max_degree).expect("plan");
    let all_modes = (0..=max_degree)
        .flat_map(|degree| {
            (-(degree as isize)..=(degree as isize)).map(move |order| (degree, order))
        })
        .collect::<Vec<_>>();
    assert_eq!(all_modes.len(), SHT_HERMES_DOT_LEN_THRESHOLD);
    let mut coefficients = SphericalHarmonicCoefficients::zeros(max_degree);
    for &(degree, order) in &all_modes {
        coefficients.set(
            degree,
            order,
            Complex64::new(
                degree as f64 * 0.031 + order as f64 * 0.007,
                degree as f64 * -0.019 + order as f64 * 0.011,
            ),
        );
    }
    let lanes = coefficient_lanes(&coefficients, &all_modes);

    for (lat, lon) in [(0, 0), (5, 7), (15, 30)] {
        let theta = plan.theta(lat);
        let phi = plan.phi(lon);
        let expected = sht_inverse_sample(&coefficients, &all_modes, theta, phi);
        let actual = sht_inverse_sample_hermes(&lanes, &all_modes, theta, phi);
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-11);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-11);
    }
}

#[test]
fn typed_real_forward_supports_f64_f32_and_mixed_f16_storage() {
    let plan = ShtPlan::new(6, 13, 2).expect("plan");
    let constant = 1.0 / (4.0 * std::f64::consts::PI).sqrt();
    let samples64 = Array2::from_elem(
        [plan.grid().latitudes(), plan.grid().longitudes()],
        constant,
    );
    let expected = plan.forward_real(&samples64).expect("forward");
    let shape = coefficient_shape(&plan);

    let mut out64 = Array2::<Complex64>::zeros(shape);
    plan.forward_real_typed_into(
        &samples64,
        &mut out64,
        PrecisionProfile::HIGH_ACCURACY_F64,
        PrecisionProfile::HIGH_ACCURACY_F64,
    )
    .expect("typed f64 real forward");
    for (actual, expected) in out64.iter().zip(expected.values().iter()) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }

    let samples32 = samples64.mapv(|value| value as f32);
    let represented32 = samples32.mapv(f64::from);
    let expected32 = plan
        .forward_real(&represented32)
        .expect("represented f32 forward");
    let mut out32 = Array2::<Complex32>::zeros(shape);
    plan.forward_real_typed_into(
        &samples32,
        &mut out32,
        PrecisionProfile::LOW_PRECISION_F32,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed f32 real forward");
    for (actual, expected) in out32.iter().zip(expected32.values().iter()) {
        assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
        assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
    }

    let samples16 = samples64.mapv(|value| f16::from_f32(value as f32));
    let represented16 = samples16.mapv(|value| f64::from(value.to_f32()));
    let expected16 = plan
        .forward_real(&represented16)
        .expect("represented f16 forward");
    let mut out16 = Array2::from_elem(shape, [f16::from_f32(0.0), f16::from_f32(0.0)]);
    plan.forward_real_typed_into(
        &samples16,
        &mut out16,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
        PrecisionProfile::MIXED_PRECISION_F16_F32,
    )
    .expect("typed f16 real forward");
    for (actual, expected) in out16.iter().zip(expected16.values().iter()) {
        let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        assert!((f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound);
        assert!((f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound);
    }
}

#[test]
fn leto_real_forward_matches_leto_reference() {
    let plan = ShtPlan::new(6, 13, 2).expect("plan");
    let constant = 1.0 / (4.0 * std::f64::consts::PI).sqrt();
    let samples = Array2::from_elem(
        [plan.grid().latitudes(), plan.grid().longitudes()],
        constant,
    );
    let input = leto::Array2::from_shape_vec(
        [plan.grid().latitudes(), plan.grid().longitudes()],
        samples.iter().copied().collect(),
    )
    .expect("leto samples");
    let expected = plan.forward_real(&samples).expect("leto forward");

    let actual = plan
        .forward_real_leto(input.view())
        .expect("leto real forward");
    let actual_view = actual.view();
    let actual = actual_view.as_slice().expect("contiguous coefficients");
    for (actual, expected) in actual.iter().zip(expected.values().iter()) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_strided_real_forward_matches_leto_reference() {
    let plan = ShtPlan::new(6, 13, 2).expect("plan");
    let samples = Array2::from_shape_fn(
        [plan.grid().latitudes(), plan.grid().longitudes()],
        |[lat, lon]| (lat as f64 * 0.2).sin() + (lon as f64 * 0.1).cos(),
    );
    let mut backing = Vec::with_capacity(samples.size() * 2);
    for value in samples.iter().copied() {
        backing.push(value);
        backing.push(99.0);
    }
    let input = leto::Array2::from_shape_vec(
        [plan.grid().latitudes(), plan.grid().longitudes() * 2],
        backing,
    )
    .expect("leto samples");
    let strided = input
        .view()
        .slice(&[
            (0, plan.grid().latitudes(), 1),
            (0, plan.grid().longitudes() * 2, 2),
        ])
        .expect("strided samples");
    let expected = plan.forward_real(&samples).expect("leto forward");

    let actual = plan.forward_real_leto(strided).expect("leto real forward");
    let actual_view = actual.view();
    let actual = actual_view.as_slice().expect("contiguous coefficients");
    for (actual, expected) in actual.iter().zip(expected.values().iter()) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_complex_forward_and_inverse_match_leto_reference() {
    let plan = ShtPlan::new(6, 13, 2).expect("plan");
    let samples = Array2::from_shape_fn(
        [plan.grid().latitudes(), plan.grid().longitudes()],
        |[lat, lon]| spherical_harmonic(1, 1, plan.theta(lat), plan.phi(lon)),
    );
    let input = leto::Array2::from_shape_vec(
        [plan.grid().latitudes(), plan.grid().longitudes()],
        samples.iter().copied().collect(),
    )
    .expect("leto samples");
    let expected_coefficients = plan.forward_complex(&samples).expect("leto forward");

    let actual_coefficients = plan
        .forward_complex_leto(input.view())
        .expect("leto complex forward");
    let actual_coefficients_view = actual_coefficients.view();
    let actual_coefficients_slice = actual_coefficients_view
        .as_slice()
        .expect("contiguous coefficients");
    for (actual, expected) in actual_coefficients_slice
        .iter()
        .zip(expected_coefficients.values().iter())
    {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }

    let coefficients = leto::Array2::from_shape_vec(
        [
            plan.grid().max_degree() + 1,
            2 * plan.grid().max_degree() + 1,
        ],
        expected_coefficients.values().iter().copied().collect(),
    )
    .expect("leto coefficients");
    let expected_inverse = plan
        .inverse_complex(&expected_coefficients)
        .expect("leto inverse");
    let actual_inverse = plan
        .inverse_complex_leto(coefficients.view())
        .expect("leto inverse");
    let actual_inverse_view = actual_inverse.view();
    let actual_inverse = actual_inverse_view
        .as_slice()
        .expect("contiguous inverse samples");
    for (actual, expected) in actual_inverse.iter().zip(expected_inverse.iter()) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn typed_leto_forward_and_inverse_match_leto_reference() {
    let plan = ShtPlan::new(6, 13, 2).expect("plan");
    let samples = Array2::from_shape_fn(
        [plan.grid().latitudes(), plan.grid().longitudes()],
        |[lat, lon]| (lat as f32 * 0.2).sin() + (lon as f32 * 0.1).cos(),
    );
    let input = leto::Array2::from_shape_vec(
        [plan.grid().latitudes(), plan.grid().longitudes()],
        samples.iter().copied().collect(),
    )
    .expect("leto samples");
    let shape = coefficient_shape(&plan);
    let mut expected_coefficients = Array2::<Complex32>::zeros(shape);
    plan.forward_real_typed_into(
        &samples,
        &mut expected_coefficients,
        PrecisionProfile::LOW_PRECISION_F32,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed leto forward");

    let actual_coefficients = plan
        .forward_real_leto_typed::<f32, Complex32>(
            input.view(),
            PrecisionProfile::LOW_PRECISION_F32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed leto forward");
    let actual_coefficients_view = actual_coefficients.view();
    let actual_coefficients_slice = actual_coefficients_view
        .as_slice()
        .expect("contiguous coefficients");
    for (actual, expected) in actual_coefficients_slice
        .iter()
        .zip(expected_coefficients.iter())
    {
        assert_eq!(actual.re.to_bits(), expected.re.to_bits());
        assert_eq!(actual.im.to_bits(), expected.im.to_bits());
    }

    let coefficients = leto::Array2::from_shape_vec(
        [shape[0], shape[1]],
        expected_coefficients.iter().copied().collect(),
    )
    .expect("leto coefficients");
    let mut expected_samples =
        Array2::<f32>::zeros([plan.grid().latitudes(), plan.grid().longitudes()]);
    plan.inverse_real_typed_into(
        &expected_coefficients,
        &mut expected_samples,
        PrecisionProfile::LOW_PRECISION_F32,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed leto inverse");
    let actual_samples = plan
        .inverse_real_leto_typed::<Complex32, f32>(
            coefficients.view(),
            PrecisionProfile::LOW_PRECISION_F32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed leto inverse");
    let actual_samples_view = actual_samples.view();
    let actual_samples = actual_samples_view.as_slice().expect("contiguous samples");
    for (actual, expected) in actual_samples.iter().zip(expected_samples.iter()) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn typed_complex_forward_and_inverse_support_complex32_storage() {
    let plan = ShtPlan::new(6, 13, 2).expect("plan");
    let samples64 = Array2::from_shape_fn(
        [plan.grid().latitudes(), plan.grid().longitudes()],
        |[lat, lon]| spherical_harmonic(1, 1, plan.theta(lat), plan.phi(lon)),
    );
    let samples32 = samples64.mapv(|value| Complex32::new(value.re as f32, value.im as f32));
    let represented32 = samples32.mapv(Complex32::to_complex64);
    let expected = plan.forward_complex(&represented32).expect("forward");
    let shape = coefficient_shape(&plan);

    let mut coefficients32 = Array2::<Complex32>::zeros(shape);
    plan.forward_complex_typed_into(
        &samples32,
        &mut coefficients32,
        PrecisionProfile::LOW_PRECISION_F32,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed complex32 forward");
    for (actual, expected) in coefficients32.iter().zip(expected.values().iter()) {
        assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
        assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
    }

    let mut recovered32 =
        Array2::<Complex32>::zeros([plan.grid().latitudes(), plan.grid().longitudes()]);
    plan.inverse_complex_typed_into(
        &coefficients32,
        &mut recovered32,
        PrecisionProfile::LOW_PRECISION_F32,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed complex32 inverse");
    for (actual, expected) in recovered32.iter().zip(samples32.iter()) {
        assert!((actual.re - expected.re).abs() < 1.0e-4);
        assert!((actual.im - expected.im).abs() < 1.0e-4);
    }
}

#[test]
fn typed_real_inverse_and_mismatch_rejections_are_value_semantic() {
    let plan = ShtPlan::new(5, 11, 2).expect("plan");
    let mut coefficients = SphericalHarmonicCoefficients::zeros(plan.grid().max_degree());
    coefficients.set(0, 0, Complex64::new(1.0, 0.0));
    let coefficient_shape = coefficient_shape(&plan);
    let coefficients32 = coefficients
        .values()
        .mapv(|value| Complex32::new(value.re as f32, value.im as f32));
    let mut samples32 = Array2::<f32>::zeros([plan.grid().latitudes(), plan.grid().longitudes()]);

    plan.inverse_real_typed_into(
        &coefficients32,
        &mut samples32,
        PrecisionProfile::LOW_PRECISION_F32,
        PrecisionProfile::LOW_PRECISION_F32,
    )
    .expect("typed real inverse");
    let expected = plan.inverse_real(&coefficients).expect("inverse");
    for (actual, expected) in samples32.iter().zip(expected.iter()) {
        assert!((f64::from(*actual) - *expected).abs() < 1.0e-5);
    }

    let err = plan
        .inverse_real_typed_into(
            &coefficients32,
            &mut samples32,
            PrecisionProfile::HIGH_ACCURACY_F64,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect_err("profile mismatch");
    assert_eq!(err, ShtError::PrecisionMismatch);

    let bad_coefficients =
        Array2::<Complex32>::zeros([coefficient_shape[0], coefficient_shape[1] + 1]);
    let err = plan
        .inverse_real_typed_into(
            &bad_coefficients,
            &mut samples32,
            PrecisionProfile::LOW_PRECISION_F32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect_err("shape mismatch");
    assert_eq!(err, ShtError::CoefficientShapeMismatch);
}
