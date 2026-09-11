//! The 180 route and its 36-point base against the codelet and the direct
//! transform.

use super::{transform_180, State180};
use crate::application::execution::kernel::components::winograd::composite::{
    dft36_kernel, Twiddles36,
};
use crate::application::execution::kernel::mixed_radix::scalar::simd::avx::vector_frame_available;
use crate::application::execution::kernel::mixed_radix::traits::ShortDft;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::FftPlan1D;
use core::mem::MaybeUninit;
use eunomia::{Complex32, Complex64};
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel};

/// A signal with no symmetry a wrong output slot could hide behind.
fn signal(n: usize) -> Vec<Complex32> {
    (0..n)
        .map(|index| {
            let phase = index as f64;
            Complex32::new(
                ((phase * 0.37).sin() * 1.25) as f32,
                ((phase * 0.11).cos() - 0.5) as f32,
            )
        })
        .collect()
}

/// The direct transform in double precision, the oracle both routes meet.
fn direct(input: &[Complex32], inverse: bool) -> Vec<Complex64> {
    let n = input.len();
    (0..n)
        .map(|k| {
            let mut sum = Complex64::new(0.0, 0.0);
            for (j, x) in input.iter().enumerate() {
                let angle = core::f64::consts::TAU * ((j * k) % n) as f64 / n as f64;
                let sign = if inverse { 1.0 } else { -1.0 };
                let w = Complex64::new(angle.cos(), sign * angle.sin());
                sum += Complex64::new(f64::from(x.re), f64::from(x.im)) * w;
            }
            sum
        })
        .collect()
}

/// Every output is a sum of `n` unit-twiddled terms rounded through `levels`
/// butterfly and twiddle stages; each level contributes at most one rounding
/// of each partial at `f32::EPSILON / 2` relative, so the absolute error is
/// bounded by `levels * EPSILON * Σ|x|` with the twiddle rounding folded
/// into one more level.
fn bound(input: &[Complex32], levels: f64) -> f64 {
    let l1: f64 = input
        .iter()
        .map(|x| f64::from(x.re).abs() + f64::from(x.im).abs())
        .sum();
    (levels + 1.0) * f64::from(f32::EPSILON) * l1
}

fn assert_close(got: &[Complex32], expected: &[Complex64], bound: f64, what: &str) {
    for (index, (g, e)) in got.iter().zip(expected).enumerate() {
        let error = ((f64::from(g.re) - e.re).powi(2) + (f64::from(g.im) - e.im).powi(2)).sqrt();
        assert!(
            error <= bound,
            "{what}: output {index} is ({}, {}) against ({}, {}), error {error:e} over the bound {bound:e}",
            g.re,
            g.im,
            e.re,
            e.im
        );
    }
}

struct Kernel36<'a, const INVERSE: bool> {
    twiddles: &'a Twiddles36<f32>,
    input: &'a [Complex32; 36],
    output: &'a mut [MaybeUninit<Complex32>; 36],
}

impl<const INVERSE: bool> LaneKernel<f32> for Kernel36<'_, INVERSE> {
    type Output = ();

    fn call<A: SimdArch + SimdKernel<f32>>(self, simd: Simd<f32, A>) {
        dft36_kernel::<f32, A, INVERSE>(simd, self.twiddles, self.input, self.output);
    }
}

#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
unsafe fn run_36<const INVERSE: bool>(input: &[Complex32; 36]) -> [Complex32; 36] {
    let twiddles = Twiddles36::<f32>::new(INVERSE);
    let mut output = [MaybeUninit::<Complex32>::uninit(); 36];
    // SAFETY: the caller established the frame.
    unsafe {
        hermes_simd::vectorize_in_frame::<f32, _>(Kernel36::<INVERSE> {
            twiddles: &twiddles,
            input,
            output: &mut output,
        });
    }
    // SAFETY: the kernel wrote all thirty-six outputs.
    unsafe { core::mem::transmute::<[MaybeUninit<Complex32>; 36], [Complex32; 36]>(output) }
}

/// Whether this host runs the route: the frame, at four complexes a register.
fn route_runs() -> bool {
    <f32 as LaneScalar>::FRAME_LANES == 8 && vector_frame_available()
}

fn kernel_matches_the_codelet<const INVERSE: bool>(input: &[Complex32; 36]) {
    // SAFETY: `route_runs` probed the frame.
    let got = unsafe { run_36::<INVERSE>(input) };
    let mut expected = *input;
    <f32 as ShortDft<36>>::dft::<INVERSE>(&mut expected);
    let reference = direct(input, INVERSE);
    // The codelet and the kernel each sit inside the bound of the direct
    // transform; five levels: radix-4, twiddle, radix-3, twiddle, radix-3.
    let bound = bound(input, 5.0);
    assert_close(&got, &reference, bound, "the 36-point kernel");
    assert_close(&expected, &reference, bound, "the 36-point codelet");
}

#[test]
fn thirty_six_kernel_matches_the_codelet_and_the_direct_transform() {
    if !route_runs() {
        return;
    }
    let input: [Complex32; 36] = signal(36).try_into().expect("thirty-six samples");
    kernel_matches_the_codelet::<false>(&input);
    kernel_matches_the_codelet::<true>(&input);
}

#[test]
fn thirty_six_kernel_places_every_impulse() {
    if !route_runs() {
        return;
    }
    for position in 0..36 {
        let mut input = [Complex32::new(0.0, 0.0); 36];
        input[position] = Complex32::new(1.0, 0.0);
        kernel_matches_the_codelet::<false>(&input);
    }
}

#[test]
fn route_matches_the_direct_transform_at_180() {
    if !route_runs() {
        return;
    }
    let state = State180::<f32>::new_if_supported().expect("the frame was probed");
    let input = signal(180);
    let mut forward = input.clone();
    // SAFETY: `route_runs` probed the frame, and the buffer is 180 samples.
    unsafe { transform_180::<f32, false, false>(&mut forward, &state) };
    // Eight levels: radix-5, twiddle, radix-4, twiddle, radix-3, twiddle,
    // radix-3, and the interleave, which rounds nothing.
    assert_close(
        &forward,
        &direct(&input, false),
        bound(&input, 7.0),
        "the 180 route forward",
    );

    let mut composite = input.clone();
    <f32 as MixedRadixScalar>::composite_forward(&mut composite, &[4, 3, 3, 5]);
    assert_close(
        &composite,
        &direct(&input, false),
        bound(&input, 7.0),
        "the composite route",
    );

    let mut round_trip = forward.clone();
    // SAFETY: as above.
    unsafe { transform_180::<f32, true, true>(&mut round_trip, &state) };
    let back: Vec<Complex64> = input
        .iter()
        .map(|x| Complex64::new(f64::from(x.re), f64::from(x.im)))
        .collect();
    // The forward output carries `n` times the signal's scale, so the round
    // trip's bound is the forward bound taken through the inverse: twice
    // the levels over the forward output's magnitude, divided by `n`.
    let round_bound = bound(&forward, 7.0) / 180.0 + bound(&input, 7.0);
    assert_close(&round_trip, &back, round_bound, "the 180 round trip");

    let mut unnormalized = forward.clone();
    // SAFETY: as above.
    unsafe { transform_180::<f32, true, false>(&mut unnormalized, &state) };
    let scaled: Vec<Complex64> = back
        .iter()
        .map(|x| *x * Complex64::new(180.0, 0.0))
        .collect();
    assert_close(
        &unnormalized,
        &scaled,
        round_bound * 180.0,
        "the 180 unnormalized inverse",
    );
}

#[test]
fn plan_at_180_takes_the_column_route() {
    let plan = FftPlan1D::<f32>::new(
        crate::domain::metadata::shape::Shape1D::new(180).expect("180 is a valid length"),
    );
    assert_eq!(
        plan.column180.is_some(),
        route_runs(),
        "the plan builds the route state exactly where the route runs"
    );
    let input = signal(180);
    let mut forward = input.clone();
    plan.forward_complex_slice_inplace(&mut forward);
    assert_close(
        &forward,
        &direct(&input, false),
        bound(&input, 7.0),
        "the plan at 180",
    );
    let mut round_trip = forward.clone();
    plan.inverse_complex_slice_inplace(&mut round_trip);
    let back: Vec<Complex64> = input
        .iter()
        .map(|x| Complex64::new(f64::from(x.re), f64::from(x.im)))
        .collect();
    assert_close(
        &round_trip,
        &back,
        bound(&forward, 7.0) / 180.0 + bound(&input, 7.0),
        "the plan round trip at 180",
    );
}
