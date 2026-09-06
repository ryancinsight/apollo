//! Caller-owned workspace reuse across independently submitted transforms.

use super::{window, MnemosyneHooks, GLOBAL_ALLOCATIONS, MNEMOSYNE_ALLOCATIONS};
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::kernel::worker_quiescence;
use crate::{FftPlan2D, FftPlan3D, Shape2D, Shape3D, StaticFftPlan2D, StaticFftPlan3D};
use eunomia::{Complex, Complex64};
use leto::{ArrayViewMut, Layout};
use std::sync::atomic::Ordering;

#[derive(Clone, Copy)]
enum Planning {
    Static,
    Dynamic,
}

fn coefficients<const D: usize>() -> [Complex64; D] {
    let mut amplitude = -0.25;
    std::array::from_fn(|_| {
        let coefficient = Complex64::new(amplitude, -amplitude / 2.0);
        amplitude /= 2.0;
        coefficient
    })
}

fn sparse_input<F: Copy + From<f32>, const D: usize>(shape: [usize; D]) -> Vec<Complex<F>> {
    let mut input = vec![Complex::new(F::from(0.0), F::from(0.0)); shape.iter().product()];
    input[0] = Complex::new(F::from(1.0), F::from(0.5));
    let mut amplitude = -0.25_f32;
    for (axis, length) in shape.iter().enumerate() {
        let stride: usize = shape[axis + 1..].iter().product();
        input[length / 4 * stride] = Complex::new(F::from(amplitude), F::from(-amplitude / 2.0));
        amplitude /= 2.0;
    }
    input
}

fn assert_spectrum<F: Copy + Into<f64>, const D: usize>(
    signal: &[Complex<F>],
    shape: [usize; D],
    bound: f64,
) {
    let amplitudes = coefficients::<D>();
    for (index, actual) in signal.iter().enumerate() {
        let mut expected = Complex64::new(1.0, 0.5);
        let mut residual = index;
        for (&length, coefficient) in shape.iter().zip(&amplitudes).rev() {
            let phase = residual % length % 4;
            residual /= length;
            expected += match phase {
                0 => *coefficient,
                1 => Complex64::new(coefficient.im, -coefficient.re),
                2 => -*coefficient,
                3 => Complex64::new(-coefficient.im, coefficient.re),
                _ => unreachable!("phase is reduced modulo four"),
            };
        }
        let error = (Complex64::new(actual.re.into(), actual.im.into()) - expected).norm();
        assert!(
            error <= bound,
            "bin {index}: error {error:e} exceeds {bound:e}"
        );
    }
}

fn assert_recovery<F: Copy + Into<f64>>(signal: &[Complex<F>], input: &[Complex<F>], bound: f64) {
    for (index, (actual, expected)) in signal.iter().zip(input).enumerate() {
        let error = (Complex64::new(actual.re.into(), actual.im.into())
            - Complex64::new(expected.re.into(), expected.im.into()))
        .norm();
        assert!(
            error <= bound,
            "sample {index}: error {error:e} exceeds {bound:e}"
        );
    }
}

fn assert_no_allocations() {
    assert_eq!(
        GLOBAL_ALLOCATIONS.allocations.load(Ordering::Relaxed),
        0,
        "a warmed independent transform allocated through the global allocator"
    );
    assert_eq!(
        MNEMOSYNE_ALLOCATIONS.allocations.load(Ordering::Relaxed),
        0,
        "a warmed independent transform allocated directly through Mnemosyne"
    );
}

fn check_submissions<F, const D: usize>(
    shape: [usize; D],
    unit_roundoff: f64,
    forward: impl Fn(&mut [Complex<F>]),
    inverse: impl Fn(&mut [Complex<F>]),
) where
    F: Copy + From<f32> + Into<f64>,
{
    let _hooks = MnemosyneHooks::install();
    worker_quiescence::arm();
    crate::ensure_thread_local_scratch_hook_registered();
    moirai::register_idle_hook(worker_quiescence::observe_idle)
        .expect("invariant: submission observer fits Moirai's hook registry");
    let input = sparse_input(shape);
    let mut signal = input.clone();
    // At most sixteen rounded real operations per radix stage bound each
    // output path by gamma_k times the input L1 norm, as in the 1-D sparse
    // oracle. Separability adds the stage counts across axes. The exact
    // dyadic coefficients and fourth roots require no trigonometric oracle.
    let stages: u32 = shape.iter().map(|length| length.ilog2()).sum();
    let ku = 16.0 * f64::from(stages) * unit_roundoff;
    let gamma = ku / (1.0 - ku);
    let input_norm = Complex64::new(1.0, 0.5).norm()
        + coefficients::<D>()
            .iter()
            .map(|value| value.norm())
            .sum::<f64>();
    let forward_bound = gamma * input_norm;
    // The normalized inverse has infinity norm one. Propagating the first
    // transform's error through the inverse gives (2 gamma + gamma^2)||x||_1;
    // division by each power-of-two axis length is exact.
    let recovery_bound = (2.0 * gamma + gamma * gamma) * input_norm;

    worker_quiescence::begin_phase();
    forward(&mut signal);
    worker_quiescence::wait_for_phase();
    assert_spectrum(&signal, shape, forward_bound);
    worker_quiescence::begin_phase();
    inverse(&mut signal);
    worker_quiescence::wait_for_phase();
    assert_recovery(&signal, &input, recovery_bound);

    signal.copy_from_slice(&input);
    worker_quiescence::begin_phase();
    window("independent forward", || {
        forward(&mut signal);
        worker_quiescence::wait_for_phase();
    });
    assert_no_allocations();
    assert_spectrum(&signal, shape, forward_bound);
    worker_quiescence::begin_phase();
    window("independent inverse", || {
        inverse(&mut signal);
        worker_quiescence::wait_for_phase();
    });
    assert_no_allocations();
    assert_recovery(&signal, &input, recovery_bound);
}

fn check_matrix<F>(planning: Planning, unit_roundoff: f64)
where
    F: MixedRadixScalar<Complex = Complex<F>> + From<f32> + Into<f64>,
    Complex<F>: PlanScratch,
{
    let shape = [4096, 16];
    let layout = Layout::c_contiguous(shape).expect("valid matrix layout");
    let dynamic = FftPlan2D::<F>::new(Shape2D::new(4096, 16).expect("valid matrix shape"));
    let static_plan = StaticFftPlan2D::<F, 4096, 16>::new();
    check_submissions(
        shape,
        unit_roundoff,
        |signal| {
            let view = ArrayViewMut::try_new(layout, signal).expect("signal fits matrix layout");
            match planning {
                Planning::Dynamic => dynamic.forward_complex_leto_inplace(view),
                Planning::Static => static_plan.forward_complex_leto_inplace(view),
            }
        },
        |signal| {
            let view = ArrayViewMut::try_new(layout, signal).expect("signal fits matrix layout");
            match planning {
                Planning::Dynamic => dynamic.inverse_complex_leto_inplace(view),
                Planning::Static => static_plan.inverse_complex_leto_inplace(view),
            }
        },
    );
}

fn check_volume<F>(planning: Planning, unit_roundoff: f64)
where
    F: MixedRadixScalar<Complex = Complex<F>> + From<f32> + Into<f64>,
    Complex<F>: PlanScratch,
{
    let shape = [4096, 4, 4];
    let layout = Layout::c_contiguous(shape).expect("valid volume layout");
    let dynamic = FftPlan3D::<F>::new(Shape3D::new(4096, 4, 4).expect("valid volume shape"));
    let static_plan = StaticFftPlan3D::<F, 4096, 4, 4>::new();
    check_submissions(
        shape,
        unit_roundoff,
        |signal| {
            let view = ArrayViewMut::try_new(layout, signal).expect("signal fits volume layout");
            match planning {
                Planning::Dynamic => dynamic.forward_complex_leto_inplace(view),
                Planning::Static => static_plan.forward_complex_leto_inplace(view),
            }
        },
        |signal| {
            let view = ArrayViewMut::try_new(layout, signal).expect("signal fits volume layout");
            match planning {
                Planning::Dynamic => dynamic.inverse_complex_leto_inplace(view),
                Planning::Static => static_plan.inverse_complex_leto_inplace(view),
            }
        },
    );
}

#[test]
fn dynamic_matrix_reuses_workspace_after_worker_idle() {
    check_matrix::<f32>(Planning::Dynamic, f64::from(f32::EPSILON) / 2.0);
    check_matrix::<f64>(Planning::Dynamic, f64::EPSILON / 2.0);
}

#[test]
fn static_matrix_reuses_workspace_after_worker_idle() {
    check_matrix::<f32>(Planning::Static, f64::from(f32::EPSILON) / 2.0);
    check_matrix::<f64>(Planning::Static, f64::EPSILON / 2.0);
}

#[test]
fn dynamic_volume_reuses_workspace_after_worker_idle() {
    check_volume::<f32>(Planning::Dynamic, f64::from(f32::EPSILON) / 2.0);
    check_volume::<f64>(Planning::Dynamic, f64::EPSILON / 2.0);
}

#[test]
fn static_volume_reuses_workspace_after_worker_idle() {
    check_volume::<f32>(Planning::Static, f64::from(f32::EPSILON) / 2.0);
    check_volume::<f64>(Planning::Static, f64::EPSILON / 2.0);
}
