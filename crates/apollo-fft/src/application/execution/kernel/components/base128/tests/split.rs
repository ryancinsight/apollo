//! Independent ordering and normalization oracles for the column-first
//! routes: 2048 and 4096, and the chains from 8192 to 262144.

use eunomia::{Complex32, Complex64};

const LENGTH: usize = 2048;
/// Every length the column-first steps serve above the base.
const COLUMN_FIRST_LENGTHS: [usize; 8] =
    [2048, 4096, 8192, 16_384, 32_768, 65_536, 131_072, 262_144];

#[test]
fn impulse_and_constant_have_exact_spectra() {
    for length in COLUMN_FIRST_LENGTHS {
        let plan = crate::FftPlan1D::<f32>::new(crate::Shape1D::new(length).unwrap());
        let zero = Complex32::new(0.0, 0.0);
        let amplitude = Complex32::new(1.0, -0.5);
        let mut impulse = vec![zero; length];
        impulse[0] = amplitude;
        let source = impulse.clone();
        plan.forward_complex_slice_inplace(&mut impulse);
        assert_eq!(impulse, vec![amplitude; length], "impulse at {length}");
        plan.inverse_complex_slice_inplace(&mut impulse);
        assert_eq!(impulse, source, "impulse round trip at {length}");

        let mut constant = vec![amplitude; length];
        plan.forward_complex_slice_inplace(&mut constant);
        let mut expected = vec![zero; length];
        expected[0] = Complex32::new(length as f32, -(length as f32) / 2.0);
        assert_eq!(constant, expected, "constant at {length}");
        plan.inverse_complex_slice_inplace(&mut constant);
        assert_eq!(
            constant,
            vec![amplitude; length],
            "constant round trip at {length}"
        );
    }
}

/// The bins an independent direct sum checks at a length: the ends and
/// middle of the spectrum and a fixed pseudo-random spread, so the
/// forward transform is checked against `O(bins n)` work at 32768.
fn sparse_bins(n: usize) -> Vec<usize> {
    let mut bins = vec![0, 1, 2, 3, n / 2 - 1, n / 2, n / 2 + 1, n - 2, n - 1];
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    for _ in 0..56 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        bins.push(
            usize::try_from((state >> 33) % n as u64)
                .expect("invariant: a bin index below n fits usize"),
        );
    }
    bins
}

#[test]
fn column_first_forward_matches_the_direct_sum_at_sparse_bins() {
    for length in COLUMN_FIRST_LENGTHS {
        let plan = crate::FftPlan1D::<f32>::new(crate::Shape1D::new(length).unwrap());
        let source: Vec<Complex32> = super::signal(length)
            .iter()
            .map(|v| Complex32::new(v.re as f32, v.im as f32))
            .collect();
        // Widen only the independent test oracle (as the 2048 inverse below).
        let reference_input: Vec<Complex64> = source
            .iter()
            .map(|v| Complex64::new(f64::from(v.re), f64::from(v.im)))
            .collect();
        let mut actual = source.clone();
        plan.forward_complex_slice_inplace(&mut actual);
        // Sixteen rounding contributions per binary stage bound the route's
        // complex arithmetic and twiddle rounding at the input's L1 norm.
        let du = 16.0 * f64::from(length.ilog2()) * (f64::from(f32::EPSILON) / 2.0);
        let norm: f64 = reference_input.iter().map(|v| v.re.hypot(v.im)).sum();
        let bound = du / (1.0 - du) * norm + super::tolerance(&reference_input);
        for k in sparse_bins(length) {
            let reference = super::dft_bin(&reference_input, k);
            let value = actual[k];
            assert!(
                value.re.is_finite() && value.im.is_finite(),
                "n {length} bin {k}"
            );
            let error =
                (f64::from(value.re) - reference.re).hypot(f64::from(value.im) - reference.im);
            assert!(error <= bound, "n {length} bin {k}: {error:e} > {bound:e}");
        }
        plan.inverse_complex_slice_inplace(&mut actual);
        let round_trip_bound = 2.0 * bound;
        for (index, (value, expected)) in actual.iter().zip(&source).enumerate() {
            let error = (value.re - expected.re).hypot(value.im - expected.im);
            assert!(
                f64::from(error) <= round_trip_bound,
                "n {length} round trip at {index}: {error:e} > {round_trip_bound:e}"
            );
        }
    }
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

/// The `BLOCKS`-block interleave against the scalar reference: block `q`
/// at `k` lands at `BLOCKS k + q`. The pass moves values, so both widths
/// must match bit-exactly where they handle the request.
fn assert_interleave_matches_reference<T, const BLOCKS: usize>()
where
    T: crate::application::execution::kernel::mixed_radix::MixedRadixScalar
        + hermes_simd::LaneScalar,
{
    let n = BLOCKS * 256;
    let lanes: Vec<T> = (0..2 * n).map(|i| T::from_precise(i as f64)).collect();
    let mut reference = vec![T::from_precise(0.0); 2 * n];
    for q in 0..BLOCKS {
        for k in 0..256 {
            reference[(k * BLOCKS + q) * 2] = lanes[(q * 256 + k) * 2];
            reference[(k * BLOCKS + q) * 2 + 1] = lanes[(q * 256 + k) * 2 + 1];
        }
    }
    let mut narrow = vec![T::from_precise(0.0); 2 * n];
    let mut wide = vec![T::from_precise(0.0); 2 * n];
    let narrow_handled =
        hermes_simd::vectorize_lanes::<4, T, _>(super::super::split_boundary::InterleaveBlocks::<
            T,
            BLOCKS,
        > {
            src: &lanes,
            dst: &mut narrow,
            block_lanes: 512,
        })
        .unwrap_or(false);
    let wide_handled =
        hermes_simd::vectorize_lanes::<8, T, _>(super::super::split_boundary::InterleaveBlocks::<
            T,
            BLOCKS,
        > {
            src: &lanes,
            dst: &mut wide,
            block_lanes: 512,
        })
        .unwrap_or(false);
    assert!(narrow_handled, "four-lane interleave must be handled");
    assert_eq!(narrow, reference, "four-lane interleave output mismatch");
    let plan_is_wide = super::super::instance_major::Plan8x16::<T>::new_if_supported::<false>()
        .is_some_and(|plan| plan.native_eight_lanes());
    if plan_is_wide {
        assert!(
            wide_handled,
            "the eight-lane interleave must handle where the plan is eight-lane"
        );
        assert_eq!(wide, reference, "eight-lane interleave output mismatch");
    }
}

#[test]
fn interleave_matches_the_strided_reference_at_both_widths() {
    assert_interleave_matches_reference::<f64, 8>();
    assert_interleave_matches_reference::<f32, 8>();
    assert_interleave_matches_reference::<f64, 4>();
    assert_interleave_matches_reference::<f32, 4>();
    assert_interleave_matches_reference::<f64, 3>();
    assert_interleave_matches_reference::<f32, 3>();
}
