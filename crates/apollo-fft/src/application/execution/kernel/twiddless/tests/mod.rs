//! Value-semantic verification of the twiddless instrument.
//!
//! Every tolerance derives from the forward-error theorem in the module
//! documentation. With `t` halving levels, twiddle relative error `μ ≤ 4u`
//! (direct `sin_cos` evaluation, at most two ulps per component, narrowing
//! included) and `η = μ + γ₄(1 + μ) ≤ 8u`, Higham's Theorem 24.2 gives
//! `‖X̂ − X‖₂ ≤ 8u·t / (1 − 8u·t) · ‖X‖₂`. The leaf adds one direct DFT of
//! length at most 5, whose rounding `γ₁₀ ≈ 10u` is covered by counting two
//! extra levels, and `tη ≪ 1` at every exercised length, so the instantiated
//! bound is [`forward_bound`]`(t) = 8u·(t + 2)`.

mod oracle;

use super::{
    ButterflyIterative, ButterflyRecursive, CompressedHalves, Schedule, TwiddlessLengthError,
    TwiddlessPlan, TwiddlessScalar,
};
use crate::application::execution::kernel::twiddle_table::twiddle_components;
use crate::{FftPlan1D, Shape1D};
use eunomia::{Complex, Complex64, FloatElement};
use oracle::{direct_dft, distance, norm_sqr, oracle_bound, U64};
use proptest::prelude::*;

/// Relative 2-norm forward-error bound for `levels` halvings in a precision
/// with unit roundoff `u`.
fn forward_bound(levels: u32, u: f64) -> f64 {
    8.0 * u * (f64::from(levels) + 2.0)
}

/// Every admitted length `c · 2^k` with `c ∈ {1, 3, 5}` and `k ≤ max_k`.
fn admitted_lengths(max_k: u32) -> Vec<usize> {
    let mut lengths: Vec<usize> = [1_usize, 3, 5]
        .into_iter()
        .flat_map(|c| (0..=max_k).map(move |k| c << k))
        .collect();
    lengths.sort_unstable();
    lengths
}

/// Deterministic complex signal with no structure a transform could exploit.
fn signal<F: TwiddlessScalar + FloatElement>(len: usize) -> Vec<Complex<F>> {
    (0..len)
        .map(|index| {
            let x = index as f64;
            Complex::new(
                F::from_f64((0.017 * x).sin() + 0.3 * (1.3 * x).cos()),
                F::from_f64(0.25 * (0.031 * x).cos() - 0.1 * (2.1 * x).sin()),
            )
        })
        .collect()
}

fn widen<F: TwiddlessScalar + FloatElement>(values: &[Complex<F>]) -> Vec<Complex64> {
    values
        .iter()
        .map(|z| Complex64::new(z.re.to_f64(), z.im.to_f64()))
        .collect()
}

fn run<F: TwiddlessScalar, S: Schedule>(
    plan: &TwiddlessPlan<F>,
    input: &[Complex<F>],
) -> Vec<Complex<F>> {
    let mut output = vec![Complex::<F>::default(); plan.transform_len()];
    let mut scratch = vec![Complex::<F>::default(); plan.scratch_len()];
    plan.forward::<S>(input, &mut output, &mut scratch);
    output
}

fn schedules_agree_bitwise<F: TwiddlessScalar + FloatElement>(max_k: u32) {
    for len in admitted_lengths(max_k) {
        let plan = TwiddlessPlan::<F>::new(len).expect("admitted length");
        let input = signal::<F>(len);
        let published = run::<F, CompressedHalves>(&plan, &input);
        let recursive = run::<F, ButterflyRecursive>(&plan, &input);
        let iterative = run::<F, ButterflyIterative>(&plan, &input);
        assert_eq!(
            published, recursive,
            "len {len}: recursive butterfly differs"
        );
        assert_eq!(
            published, iterative,
            "len {len}: iterative butterfly differs"
        );
    }
}

/// The three schedules execute one arithmetic body: identical outputs bit for
/// bit is the executed proof that Algorithm 1 is the DIF butterfly.
#[test]
fn schedules_agree_bitwise_at_every_admitted_length() {
    schedules_agree_bitwise::<f32>(12);
    schedules_agree_bitwise::<f64>(12);
}

fn matches_oracle_within_bound<F: TwiddlessScalar + FloatElement>(max_k: u32, u: f64) {
    for len in admitted_lengths(max_k) {
        let plan = TwiddlessPlan::<F>::new(len).expect("admitted length");
        let input = signal::<F>(len);
        let expected = direct_dft(&widen(&input));
        let actual = widen(&run::<F, CompressedHalves>(&plan, &input));
        let scale = norm_sqr(&expected).sqrt();
        let tolerance = (forward_bound(plan.levels(), u) + oracle_bound(len)) * scale;
        let error = distance(&actual, &expected);
        assert!(
            error <= tolerance,
            "len {len}: ‖X̂ − X‖₂ = {error:e} exceeds 8u(t+2)‖X‖₂ + oracle = {tolerance:e}"
        );
    }
}

/// Forward error against the compensated oracle stays inside the Higham
/// bound plus the oracle's own derived bound, in both compute precisions.
#[test]
fn forward_error_stays_within_derived_bound() {
    matches_oracle_within_bound::<f32>(8, f64::from(f32::EPSILON) / 2.0);
    matches_oracle_within_bound::<f64>(8, U64);
}

fn agrees_with_production_plan<F, P>(max_k: u32, u: f64, production: P)
where
    F: TwiddlessScalar + FloatElement,
    P: Fn(&mut [Complex<F>]),
{
    for len in admitted_lengths(max_k) {
        let plan = TwiddlessPlan::<F>::new(len).expect("admitted length");
        let input = signal::<F>(len);
        let mut expected = input.clone();
        production(&mut expected);
        let expected = widen(&expected);
        let actual = widen(&run::<F, CompressedHalves>(&plan, &input));
        let scale = norm_sqr(&expected).sqrt();
        // Both routes carry the forward bound independently; their distance
        // is at most the sum.
        let tolerance = 2.0 * forward_bound(plan.levels(), u) * scale;
        let error = distance(&actual, &expected);
        assert!(
            error <= tolerance,
            "len {len}: distance to production plan {error:e} exceeds 2·8u(t+2)‖X‖₂ = {tolerance:e}"
        );
    }
}

/// Differential check against Apollo's production route at lengths past the
/// oracle's reach; the two implementations share no code beyond the twiddle
/// authority.
#[test]
fn agrees_with_production_plan_within_summed_bounds() {
    agrees_with_production_plan::<f32, _>(12, f64::from(f32::EPSILON) / 2.0, |data| {
        FftPlan1D::<f32>::new(Shape1D::new(data.len()).expect("non-zero"))
            .forward_complex_slice_inplace(data);
    });
    agrees_with_production_plan::<f64, _>(12, U64, |data| {
        FftPlan1D::<f64>::new(Shape1D::new(data.len()).expect("non-zero"))
            .forward_complex_slice_inplace(data);
    });
}

fn analytic_spectra<F: TwiddlessScalar + FloatElement>(u: f64) {
    for len in admitted_lengths(7) {
        let plan = TwiddlessPlan::<F>::new(len).expect("admitted length");
        let n = len as f64;

        // A unit impulse at zero is exact at every level: every twiddle that
        // touches a non-zero value is W^0 = 1 + 0i exactly.
        let mut impulse = vec![Complex::<F>::default(); len];
        impulse[0] = Complex::new(F::ONE, F::ZERO);
        let spectrum = run::<F, CompressedHalves>(&plan, &impulse);
        assert!(
            spectrum.iter().all(|z| z.re == F::ONE && z.im == F::ZERO),
            "len {len}: impulse spectrum is not exactly all-ones"
        );

        // A single tone at bin m has spectrum N·e_m; ‖X‖₂ = N. Samples come
        // from the twiddle authority (exponent reduced modulo N before the
        // angle forms), so each carries the same perturbation as a twiddle:
        // the reduced angle θ ≤ 2π is rounded three times (the TAU constant,
        // the quotient, the product), |θ̂ − θ| ≤ 3u·2π < 19u, sin and cos add
        // at most one ulp each (2u), so each component is within 21u and the
        // complex sample within √2·21u < 30u of the exact unit value, plus
        // u_F/2 per component on narrowing. A relative input perturbation
        // passes through the unitary transform unchanged in norm, so it adds
        // (30u₆₄ + u_F)·‖X‖₂ to the forward bound.
        let tone_bin = len / 3;
        let tone: Vec<Complex<F>> = (0..len)
            .map(|index| {
                let (sin, cos) = twiddle_components(1.0, tone_bin * index, len);
                Complex::new(F::from_f64(cos), F::from_f64(sin))
            })
            .collect();
        let actual = widen(&run::<F, CompressedHalves>(&plan, &tone));
        let mut expected = vec![Complex64::default(); len];
        expected[tone_bin] = Complex64::new(n, 0.0);
        let input_bound = 30.0 * U64 + u;
        let tolerance = (forward_bound(plan.levels(), u) + input_bound) * n;
        let error = distance(&actual, &expected);
        assert!(
            error <= tolerance,
            "len {len}: tone spectrum error {error:e} exceeds {tolerance:e}"
        );
    }
}

/// Closed-form spectra: the impulse is exact and the tone lands on its bin.
#[test]
fn impulse_and_tone_spectra_match_closed_forms() {
    analytic_spectra::<f32>(f64::from(f32::EPSILON) / 2.0);
    analytic_spectra::<f64>(U64);
}

/// Lengths Algorithm 1 admits and rejects, with the reported odd part.
#[test]
fn length_admission_follows_the_published_base_case() {
    assert_eq!(
        TwiddlessPlan::<f64>::new(0).err(),
        Some(TwiddlessLengthError::Zero)
    );
    for (len, odd_part) in [
        (7_usize, 7_usize),
        (14, 7),
        (9, 9),
        (18, 9),
        (11, 11),
        (44, 11),
    ] {
        assert_eq!(
            TwiddlessPlan::<f64>::new(len).err(),
            Some(TwiddlessLengthError::OddPart { len, odd_part }),
            "len {len}"
        );
    }
    for (len, leaf, levels) in [
        (1_usize, 1_usize, 0_u32),
        (2, 2, 0),
        (3, 3, 0),
        (4, 4, 0),
        (5, 5, 0),
        (6, 3, 1),
        (8, 4, 1),
        (10, 5, 1),
        (12, 3, 2),
        (16, 4, 2),
        (20, 5, 2),
        (5120, 5, 10),
    ] {
        let plan = TwiddlessPlan::<f32>::new(len).expect("admitted length");
        assert_eq!(
            (plan.transform_len(), plan.leaf(), plan.levels()),
            (len, leaf, levels)
        );
        assert_eq!(plan.scratch_len(), len);
    }
}

/// Mismatched buffer lengths are contract violations, not silent truncation.
#[test]
#[should_panic(expected = "output length differs from plan")]
fn forward_rejects_mismatched_output_length() {
    let plan = TwiddlessPlan::<f64>::new(8).expect("admitted length");
    let input = vec![Complex64::default(); 8];
    let mut output = vec![Complex64::default(); 4];
    let mut scratch = vec![Complex64::default(); 8];
    plan.forward::<CompressedHalves>(&input, &mut output, &mut scratch);
}

fn parseval_holds<F: TwiddlessScalar + FloatElement>(
    len: usize,
    re: &[f64],
    im: &[f64],
    u: f64,
) -> Result<(), TestCaseError> {
    let plan = TwiddlessPlan::<F>::new(len).expect("admitted length");
    let input: Vec<Complex<F>> = re
        .iter()
        .zip(im)
        .map(|(&a, &b)| Complex::new(F::from_f64(a), F::from_f64(b)))
        .collect();
    let spectrum = widen(&run::<F, CompressedHalves>(&plan, &input));
    let energy_in = norm_sqr(&widen(&input)) * len as f64;
    let energy_out = norm_sqr(&spectrum);
    // ‖X̂‖₂ = ‖X‖₂(1 + δ) with |δ| ≤ bound, so the energies differ by at
    // most 2·bound + bound² relative; the compensated norms add 2u.
    let bound = forward_bound(plan.levels(), u);
    let tolerance = (2.0 * bound + bound * bound + 2.0 * U64) * energy_in;
    prop_assert!(
        (energy_out - energy_in).abs() <= tolerance,
        "len {len}: |‖X‖² − N‖x‖²| = {:e} exceeds {tolerance:e}",
        (energy_out - energy_in).abs()
    );
    Ok(())
}

fn admitted_length_strategy() -> impl Strategy<Value = usize> {
    (prop_oneof![Just(1_usize), Just(3), Just(5)], 0_u32..=9).prop_map(|(c, k)| c << k)
}

proptest! {
    /// Parseval's identity `‖X‖₂² = N·‖x‖₂²` is an oracle independent of any
    /// DFT implementation; it falsifies a wrong twiddle, a dropped element or
    /// a mis-gathered bin at every admitted length.
    #[test]
    fn parseval_identity_holds_at_random_lengths(
        (len, re, im) in admitted_length_strategy().prop_flat_map(|len| {
            (
                Just(len),
                prop::collection::vec(-1.0_f64..1.0, len),
                prop::collection::vec(-1.0_f64..1.0, len),
            )
        })
    ) {
        parseval_holds::<f32>(len, &re, &im, f64::from(f32::EPSILON) / 2.0)?;
        parseval_holds::<f64>(len, &re, &im, U64)?;
    }
}
