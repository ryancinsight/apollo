//! Independent ordering and normalization oracles for the 2048-point route.

use eunomia::{Complex32, Complex64};

const LENGTH: usize = 2048;

#[test]
fn impulse_and_constant_have_exact_spectra() {
    let plan = crate::FftPlan1D::<f32>::new(crate::Shape1D::new(LENGTH).unwrap());
    let zero = Complex32::new(0.0, 0.0);
    let amplitude = Complex32::new(1.0, -0.5);
    let mut impulse = vec![zero; LENGTH];
    impulse[0] = amplitude;
    let source = impulse.clone();
    plan.forward_complex_slice_inplace(&mut impulse);
    assert_eq!(impulse, vec![amplitude; LENGTH]);
    plan.inverse_complex_slice_inplace(&mut impulse);
    assert_eq!(impulse, source);

    let mut constant = vec![amplitude; LENGTH];
    plan.forward_complex_slice_inplace(&mut constant);
    let mut expected = vec![zero; LENGTH];
    expected[0] = Complex32::new(2048.0, -1024.0);
    assert_eq!(constant, expected);
    plan.inverse_complex_slice_inplace(&mut constant);
    assert_eq!(constant, vec![amplitude; LENGTH]);
}

#[test]
fn normalized_inverse_matches_independent_direct_sum() {
    let plan = crate::FftPlan1D::<f32>::new(crate::Shape1D::new(LENGTH).unwrap());
    let source: Vec<Complex32> = super::signal(LENGTH)
        .iter()
        .map(|v| Complex32::new(v.re as f32, v.im as f32))
        .collect();
    // Widen only the independent test oracle: the operation under test
    // receives and computes f32, while f64 removes the direct sum's O(N u)
    // rounding noise from the comparison.
    let reference_input: Vec<Complex64> = source
        .iter()
        .map(|v| Complex64::new(f64::from(v.re), f64::from(v.im)))
        .collect();
    let expected = super::dft(&reference_input, true);
    let mut actual = source;
    plan.inverse_complex_slice_inplace(&mut actual);
    // Sixteen rounding contributions per binary stage bound complex
    // arithmetic and twiddle rounding. The normalized inverse divides the
    // forward-error bound by N; powers-of-two scaling is exact here.
    let du = 16.0 * f64::from(LENGTH.ilog2()) * (f64::from(f32::EPSILON) / 2.0);
    let norm: f64 = reference_input.iter().map(|v| v.re.hypot(v.im)).sum();
    let bound = (du / (1.0 - du) * norm + super::tolerance(&reference_input)) / 2048.0;
    for (index, (value, reference)) in actual.iter().zip(expected).enumerate() {
        assert!(value.re.is_finite() && value.im.is_finite(), "bin {index}");
        let error = (f64::from(value.re) - reference.re / 2048.0)
            .hypot(f64::from(value.im) - reference.im / 2048.0);
        assert!(error <= bound, "bin {index}: {error:e} > {bound:e}");
    }
}
