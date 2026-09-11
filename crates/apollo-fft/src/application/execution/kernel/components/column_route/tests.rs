//! The 180 route and its 36-point base against the direct transform, at both
//! scalars.

use super::{transform_180, State180};
use crate::application::execution::kernel::components::winograd::composite::{
    dft36_kernel, Twiddles36,
};
use crate::application::execution::kernel::mixed_radix::scalar::simd::avx::vector_frame_available;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::FftPlan1D;
use core::mem::MaybeUninit;
use eunomia::{Complex, Complex64};
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel};

/// The scalars the route runs at on this host.
trait RouteScalar:
    MixedRadixScalar<Complex = Complex<Self>> + LaneScalar + eunomia::layout::Pod
where
    Complex<Self>: eunomia::layout::Pod,
{
    /// Whether this host runs the route at this scalar: the frame, at four
    /// or two complexes a register.
    fn route_runs() -> bool {
        matches!(Self::FRAME_LANES, 4 | 8) && vector_frame_available()
    }
}

impl RouteScalar for f32 {}
impl RouteScalar for f64 {}

/// A signal with no symmetry a wrong output slot could hide behind.
fn signal<F: RouteScalar>(n: usize) -> Vec<Complex<F>>
where
    Complex<F>: eunomia::layout::Pod,
{
    (0..n)
        .map(|index| {
            let phase = index as f64;
            Complex::new(
                F::from_precise((phase * 0.37).sin() * 1.25),
                F::from_precise((phase * 0.11).cos() - 0.5),
            )
        })
        .collect()
}

fn widen<F: RouteScalar>(x: &Complex<F>) -> Complex64
where
    Complex<F>: eunomia::layout::Pod,
{
    Complex64::new(x.re.to_f64(), x.im.to_f64())
}

/// The direct transform in double precision, the oracle both routes meet.
fn direct<F: RouteScalar>(input: &[Complex<F>], inverse: bool) -> Vec<Complex64>
where
    Complex<F>: eunomia::layout::Pod,
{
    let n = input.len();
    (0..n)
        .map(|k| {
            let mut sum = Complex64::new(0.0, 0.0);
            for (j, x) in input.iter().enumerate() {
                let angle = core::f64::consts::TAU * ((j * k) % n) as f64 / n as f64;
                let sign = if inverse { 1.0 } else { -1.0 };
                let w = Complex64::new(angle.cos(), sign * angle.sin());
                sum += widen(x) * w;
            }
            sum
        })
        .collect()
}

/// Every output is a sum of `n` unit-twiddled terms rounded through `levels`
/// butterfly and twiddle stages; each level contributes at most one rounding
/// of each partial at `EPSILON / 2` relative, so the absolute error is
/// bounded by `levels * EPSILON * Σ|x|`, with the twiddle rounding folded
/// into one more level. In double precision the direct reference's own
/// rounding is inside the same bound.
fn bound<F: RouteScalar>(input: &[Complex<F>], levels: f64) -> f64
where
    Complex<F>: eunomia::layout::Pod,
{
    let l1: f64 = input
        .iter()
        .map(|x| x.re.to_f64().abs() + x.im.to_f64().abs())
        .sum();
    (levels + 1.0) * F::EPSILON.to_f64() * l1
}

fn assert_close<F: RouteScalar>(got: &[Complex<F>], expected: &[Complex64], bound: f64, what: &str)
where
    Complex<F>: eunomia::layout::Pod,
{
    for (index, (g, e)) in got.iter().zip(expected).enumerate() {
        let g = widen(g);
        let error = ((g.re - e.re).powi(2) + (g.im - e.im).powi(2)).sqrt();
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

struct Kernel36<'a, F, const INVERSE: bool> {
    twiddles: &'a Twiddles36<F>,
    input: &'a [Complex<F>; 36],
    output: &'a mut [MaybeUninit<Complex<F>>; 36],
}

impl<F: RouteScalar, const INVERSE: bool> LaneKernel<F> for Kernel36<'_, F, INVERSE>
where
    Complex<F>: eunomia::layout::Pod,
{
    type Output = ();

    fn call<A: SimdArch + SimdKernel<F>>(self, simd: Simd<F, A>) {
        dft36_kernel::<F, A, INVERSE>(simd, self.twiddles, self.input, self.output);
    }
}

#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
unsafe fn run_36<F: RouteScalar, const INVERSE: bool>(input: &[Complex<F>; 36]) -> [Complex<F>; 36]
where
    Complex<F>: eunomia::layout::Pod,
{
    let twiddles = Twiddles36::<F>::new(INVERSE);
    let mut output = [MaybeUninit::<Complex<F>>::uninit(); 36];
    // SAFETY: the caller established the frame.
    unsafe {
        hermes_simd::vectorize_in_frame::<F, _>(Kernel36::<F, INVERSE> {
            twiddles: &twiddles,
            input,
            output: &mut output,
        });
    }
    // SAFETY: the kernel wrote all thirty-six outputs.
    output.map(|sample| unsafe { sample.assume_init() })
}

fn kernel_matches_the_direct_transform<F: RouteScalar, const INVERSE: bool>(
    input: &[Complex<F>; 36],
) where
    Complex<F>: eunomia::layout::Pod,
{
    // SAFETY: `route_runs` probed the frame.
    let got = unsafe { run_36::<F, INVERSE>(input) };
    let reference = direct(input, INVERSE);
    // Five levels: radix-4, twiddle, radix-3, twiddle, radix-3 (radix-6,
    // twiddle, radix-6 at two complexes a register).
    assert_close(&got, &reference, bound(input, 5.0), "the 36-point kernel");
}

fn kernel_at<F: RouteScalar>()
where
    Complex<F>: eunomia::layout::Pod,
{
    if !F::route_runs() {
        return;
    }
    let input: [Complex<F>; 36] = signal::<F>(36).try_into().expect("thirty-six samples");
    kernel_matches_the_direct_transform::<F, false>(&input);
    kernel_matches_the_direct_transform::<F, true>(&input);
    for position in 0..36 {
        let mut impulse = [Complex::new(F::from_precise(0.0), F::from_precise(0.0)); 36];
        impulse[position] = Complex::new(F::from_precise(1.0), F::from_precise(0.0));
        kernel_matches_the_direct_transform::<F, false>(&impulse);
    }
}

#[test]
fn thirty_six_kernel_matches_the_direct_transform_at_every_impulse() {
    kernel_at::<f32>();
    kernel_at::<f64>();
}

fn route_at<F: RouteScalar>()
where
    Complex<F>: eunomia::layout::Pod,
{
    if !F::route_runs() {
        return;
    }
    let state = State180::<F>::new_if_supported().expect("the frame was probed");
    let input = signal::<F>(180);
    let mut forward = input.clone();
    // SAFETY: `route_runs` probed the frame, and the buffer is 180 samples.
    unsafe { transform_180::<F, false, false>(&mut forward, &state) };
    // Seven levels: radix-5, twiddle, and the kernel's five; the interleave
    // rounds nothing.
    assert_close(
        &forward,
        &direct(&input, false),
        bound(&input, 7.0),
        "the 180 route forward",
    );

    let mut composite = input.clone();
    <F as MixedRadixScalar>::composite_forward(&mut composite, &[4, 3, 3, 5]);
    assert_close(
        &composite,
        &direct(&input, false),
        bound(&input, 7.0),
        "the composite route",
    );

    let mut round_trip = forward.clone();
    // SAFETY: as above.
    unsafe { transform_180::<F, true, true>(&mut round_trip, &state) };
    let back: Vec<Complex64> = input.iter().map(widen).collect();
    // The forward output carries `n` times the signal's scale, so the round
    // trip's bound is the forward bound taken through the inverse: the
    // levels over the forward output's magnitude, divided by `n`.
    let round_bound = bound(&forward, 7.0) / 180.0 + bound(&input, 7.0);
    assert_close(&round_trip, &back, round_bound, "the 180 round trip");

    let mut unnormalized = forward.clone();
    // SAFETY: as above.
    unsafe { transform_180::<F, true, false>(&mut unnormalized, &state) };
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
fn route_matches_the_direct_transform_at_180() {
    route_at::<f32>();
    route_at::<f64>();
}

fn plan_at<F: RouteScalar>()
where
    Complex<F>: eunomia::layout::Pod,
{
    let plan = FftPlan1D::<F>::new(
        crate::domain::metadata::shape::Shape1D::new(180).expect("180 is a valid length"),
    );
    assert_eq!(
        plan.column180.is_some(),
        F::route_runs(),
        "the plan builds the route state exactly where the route runs"
    );
    let input = signal::<F>(180);
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
    let back: Vec<Complex64> = input.iter().map(widen).collect();
    assert_close(
        &round_trip,
        &back,
        bound(&forward, 7.0) / 180.0 + bound(&input, 7.0),
        "the plan round trip at 180",
    );
}

#[test]
fn plan_at_180_takes_the_column_route() {
    plan_at::<f32>();
    plan_at::<f64>();
}
