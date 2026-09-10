//! The assembled transform against the direct DFT and against RustFFT.

use super::super::boundary::staged_transpose_applies;
use super::super::driver::{four_step_batched_by, TransposeRoute};
use super::super::lane_order::LaneOrder;
use super::super::plane::plane_geometry;
use super::super::{four_step_batched, planar_applies, scratch_len, BatchedPlanCache};
use super::oracle::{dft, signal, tolerance};
use eunomia::{Complex, Complex32, Complex64};

#[test]
fn forward_matches_the_direct_transform() {
    // Even powers only: the four-step gate admits square splits.
    for k in [2u32, 4, 6, 8, 10, 12] {
        let n = 1usize << k;
        let src = signal(n);
        let expected = dft(&src, false);
        let mut data = src.clone();
        let mut scratch = vec![Complex64::default(); scratch_len(n)];
        four_step_batched::<f64, false>(&mut data, &mut scratch);

        let bound = tolerance(n, &src);
        let worst = data
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a.re - b.re).hypot(a.im - b.im))
            .fold(0.0f64, f64::max);
        assert!(
            worst <= bound,
            "N={n}: forward differs by {worst:.3e} > {bound:.3e}"
        );
    }
}

#[test]
fn inverse_matches_the_direct_transform() {
    for k in [2u32, 4, 6, 8, 10] {
        let n = 1usize << k;
        let src = signal(n);
        let expected = dft(&src, true);
        let mut data = src.clone();
        let mut scratch = vec![Complex64::default(); scratch_len(n)];
        four_step_batched::<f64, true>(&mut data, &mut scratch);

        let bound = tolerance(n, &src);
        let worst = data
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a.re - b.re).hypot(a.im - b.im))
            .fold(0.0f64, f64::max);
        assert!(
            worst <= bound,
            "N={n}: inverse differs by {worst:.3e} > {bound:.3e}"
        );
    }
}

#[test]
fn forward_then_inverse_recovers_the_input() {
    for k in [4u32, 6, 8, 10, 12] {
        let n = 1usize << k;
        let src = signal(n);
        let mut data = src.clone();
        let mut scratch = vec![Complex64::default(); scratch_len(n)];
        four_step_batched::<f64, false>(&mut data, &mut scratch);
        four_step_batched::<f64, true>(&mut data, &mut scratch);

        // The unnormalized round trip scales by N.
        let bound = tolerance(n, &src) * n as f64;
        let worst = data
            .iter()
            .zip(src.iter())
            .map(|(a, b)| (a.re - b.re * n as f64).hypot(a.im - b.im * n as f64))
            .fold(0.0f64, f64::max);
        assert!(
            worst <= bound,
            "N={n}: round trip differs by {worst:.3e} > {bound:.3e}"
        );
    }
}

#[test]
fn f32_forward_matches_the_direct_transform() {
    for k in [4u32, 6, 8] {
        let n = 1usize << k;
        let src64 = signal(n);
        let src: Vec<Complex32> = src64
            .iter()
            .map(|v| Complex32::new(v.re as f32, v.im as f32))
            .collect();
        let expected = dft(&src64, false);
        let mut data = src.clone();
        let mut scratch = vec![Complex32::default(); scratch_len(n)];
        four_step_batched::<f32, false>(&mut data, &mut scratch);

        let l1: f64 = src64.iter().map(|v| v.re.hypot(v.im)).sum();
        let stages = f64::from(u32::try_from(n.trailing_zeros()).expect("fits u32"));
        let bound = 16.0 * stages * f64::from(f32::EPSILON / 2.0) * l1;
        let worst = data
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| {
                f64::from(a.re)
                    .hypot(0.0)
                    .mul_add(0.0, (f64::from(a.re) - b.re).hypot(f64::from(a.im) - b.im))
            })
            .fold(0.0f64, f64::max);
        assert!(
            worst <= bound,
            "N={n} f32: differs by {worst:.3e} > {bound:.3e}"
        );
    }
}

/// The odd-power split route runs both stage sets without seams, and a
/// seam that reads its staged source whenever stage 2 is present, rather
/// than when a source exists, escaped the impulse and round-trip checks:
/// the output was a consistent permutation, which an impulse cannot see and
/// an inverse undoes. The split lengths are therefore held to the direct
/// transform, 2048 with two sweeps per half and 8192, the Bluestein padding
/// of the prime squares that first exposed it.
#[test]
fn odd_lengths_match_the_direct_transform() {
    for n in [512usize, 2048, 8192] {
        let input = signal(n);
        let mut data = input.clone();
        let mut scratch = vec![Complex64::default(); scratch_len(n)];
        four_step_batched::<f64, false>(&mut data, &mut scratch);
        let expected = dft(&input, false);
        let err = data
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a.re - b.re).hypot(a.im - b.im))
            .fold(0.0_f64, f64::max);
        let bound = tolerance(n, &input);
        assert!(err <= bound, "n={n}: {err:.3e} > {bound:.3e}");
    }
}

#[test]
fn f32_n32768_public_plan_matches_an_impulse_and_round_trip() {
    let n = 32_768usize;
    let source_index = 13usize;
    let source = Complex32::new(0.75, -0.25);
    let mut actual = vec![Complex32::default(); n];
    actual[source_index] = source;
    let plan = crate::FftPlan1D::<f32>::new(
        crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
    );

    plan.forward_complex_slice_inplace(&mut actual);

    let stages = n.trailing_zeros() as f32;
    // Each radix level contributes at most one complex twiddle multiply and
    // two butterfly additions. Bounding each complex operation by sixteen
    // roundoffs, then doubling for twiddle construction and comparison,
    // gives 64 * log2(N) * epsilon * |source|.
    let forward_bound = 64.0 * stages * f32::EPSILON * source.norm();
    let forward_error = actual
        .iter()
        .enumerate()
        .map(|(frequency, got)| {
            let phase =
                -core::f32::consts::TAU * ((frequency * source_index) % n) as f32 / n as f32;
            let (sin, cos) = phase.sin_cos();
            let expected = Complex32::new(
                source.re.mul_add(cos, -(source.im * sin)),
                source.re.mul_add(sin, source.im * cos),
            );
            (*got - expected).norm()
        })
        .fold(0.0_f32, f32::max);
    assert!(
        forward_error <= forward_bound,
        "N={n} f32 impulse differs by {forward_error:.3e} > {forward_bound:.3e}"
    );

    plan.inverse_complex_slice_inplace(&mut actual);

    // Forward and inverse each satisfy the bound above; normalization adds
    // one exactly representable power-of-two scale at this length.
    let round_trip_bound = 128.0 * stages * f32::EPSILON * source.norm();
    let round_trip_error = actual
        .iter()
        .enumerate()
        .map(|(index, got)| {
            let expected = if index == source_index {
                source
            } else {
                Complex32::default()
            };
            (*got - expected).norm()
        })
        .fold(0.0_f32, f32::max);
    assert!(
        round_trip_error <= round_trip_bound,
        "N={n} f32 round trip differs by {round_trip_error:.3e} > {round_trip_bound:.3e}"
    );
}
/// Differential check at the lengths the planar domain gained: RustFFT is an
/// independent implementation whose forward error carries the same
/// `O(log N · u)` bound, so the distance between the two is at most twice
/// [`tolerance`] scaled to the precision under test.
fn large_lengths_agree_with_rustfft<F>(unit_roundoff: f64)
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<Complex = Complex<F>>
        + crate::application::orchestration::cache::plans::PlanCacheProvider<PlanScalar = F>
        + eunomia::FloatElement
        + rustfft::FftNum,
{
    for k in [16u32, 18] {
        let n = 1usize << k;
        assert!(planar_applies(n), "n = {n} is inside the planar domain");
        let source = signal(n);
        let input: Vec<Complex<F>> = source
            .iter()
            .map(|z| {
                Complex::new(
                    <F as eunomia::FloatElement>::from_f64(z.re),
                    <F as eunomia::FloatElement>::from_f64(z.im),
                )
            })
            .collect();

        let mut actual = input.clone();
        crate::FftPlan1D::<F>::new(
            crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
        )
        .forward_complex_slice_inplace(&mut actual);

        let mut expected: Vec<rustfft::num_complex::Complex<F>> = input
            .iter()
            .map(|z| rustfft::num_complex::Complex::new(z.re, z.im))
            .collect();
        rustfft::FftPlanner::<F>::new()
            .plan_fft_forward(n)
            .process(&mut expected);

        let l1: f64 = source.iter().map(|v| v.re.hypot(v.im)).sum();
        let stages = f64::from(k);
        let bound = 2.0 * 16.0 * stages * unit_roundoff * l1;
        for (bin, (a, e)) in actual.iter().zip(&expected).enumerate() {
            let error = (Complex64::new(a.re.to_f64(), a.im.to_f64())
                - Complex64::new(e.re.to_f64(), e.im.to_f64()))
            .norm();
            assert!(error <= bound, "n={n}, bin={bin}: {error:e} > {bound:e}");
        }
    }
}

#[test]
fn large_planar_lengths_agree_with_rustfft_in_both_precisions() {
    large_lengths_agree_with_rustfft::<f32>(f64::from(f32::EPSILON) / 2.0);
    large_lengths_agree_with_rustfft::<f64>(f64::EPSILON / 2.0);
}

/// The staged transpose hands the second set the values the transpose pass
/// wrote, in the same registers, so the two routes agree bit for bit.
fn staged_transpose_matches_the_transpose_pass<T>()
where
    T: BatchedPlanCache<Complex = Complex<T>>
        + eunomia::FloatElement
        + PartialEq
        + core::fmt::Debug,
{
    let mut compared = 0;
    for n in [512usize, 2048, 8192, 32768] {
        let (n1, n2) = plane_geometry(n);
        let lanes = LaneOrder::for_batch::<T>(n1).lanes();
        if !staged_transpose_applies::<T>(n1, n2, lanes) {
            continue;
        }
        let input: Vec<Complex<T>> = signal(n)
            .iter()
            .map(|z| Complex::new(T::from_f64(z.re), T::from_f64(z.im)))
            .collect();
        let mut scratch = vec![Complex::<T>::default(); scratch_len(n)];
        let mut staged = input.clone();
        four_step_batched_by::<T, false>(&mut staged, &mut scratch, Some(TransposeRoute::Staged));
        let mut passed = input;
        four_step_batched_by::<T, false>(&mut passed, &mut scratch, Some(TransposeRoute::Pass));
        if let Some(at) = staged.iter().zip(&passed).position(|(a, b)| a != b) {
            panic!(
                "n={n}: element {at} reads {:?} staged against {:?} through the pass",
                staged[at], passed[at]
            );
        }
        compared += 1;
    }
    // Every dispatched register width covers 512 to 8192.
    let lanes = LaneOrder::for_batch::<T>(16).lanes();
    assert!(
        compared >= 3 || !matches!(lanes, 2 | 4 | 8 | 16),
        "the staged route applied to {compared} lengths at {lanes} lanes"
    );
}

#[test]
fn staged_transpose_matches_the_transpose_pass_in_both_precisions() {
    staged_transpose_matches_the_transpose_pass::<f32>();
    staged_transpose_matches_the_transpose_pass::<f64>();
}
