//! Sparse analytical transforms cover incomplete lane groups and view staging.

use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::{FftPlan2D, FftPlan3D, Shape2D, Shape3D, StaticFftPlan2D, StaticFftPlan3D};
use eunomia::{Complex, Complex64};
use leto::{ArrayView, ArrayViewMut, Layout};

#[derive(Clone, Copy)]
enum Storage {
    Dense,
    Interleaved,
}

impl Storage {
    fn layout<const D: usize>(self, shape: [usize; D]) -> (Layout<D>, usize) {
        let count = shape.iter().product::<usize>();
        match self {
            Self::Dense => (
                Layout::c_contiguous(shape).expect("valid dense layout"),
                count,
            ),
            Self::Interleaved => {
                let mut strides = [0; D];
                let mut stride = 2;
                for (axis, length) in shape.iter().enumerate().rev() {
                    strides[axis] = isize::try_from(stride).expect("test stride fits isize");
                    stride *= length;
                }
                (
                    Layout::try_new(shape, strides, 1).expect("valid interleaved layout"),
                    2 * count + 1,
                )
            }
        }
    }

    fn assert_padding<F: Copy + Into<f64>>(self, storage: &[Complex<F>]) {
        if let Self::Interleaved = self {
            for (index, value) in storage.iter().step_by(2).enumerate() {
                assert_eq!(
                    (value.re.into(), value.im.into()),
                    (9.0, -3.0),
                    "view staging changed padding at physical index {}",
                    2 * index
                );
            }
        }
    }
}

fn shifts<const D: usize>(shape: [usize; D]) -> [usize; D] {
    shape.map(|length| if length >= 4 { length / 4 } else { 1 })
}

fn sparse_input<F: Copy + From<f32>, const D: usize>(shape: [usize; D]) -> Vec<Complex<F>> {
    let mut input = vec![Complex::new(F::from(0.0), F::from(0.0)); shape.iter().product()];
    input[0] = Complex::new(F::from(1.0), F::from(0.5));
    let mut amplitude = -0.25_f32;
    for (axis, shift) in shifts(shape).iter().enumerate() {
        let stride: usize = shape[axis + 1..].iter().product();
        input[shift * stride] = Complex::new(F::from(amplitude), F::from(-amplitude / 2.0));
        amplitude /= 2.0;
    }
    input
}

fn expected<const D: usize>(index: usize, shape: [usize; D]) -> Complex64 {
    let mut spectrum = Complex64::new(1.0, 0.5);
    let mut amplitude = -0.25;
    for (axis, length) in shape.iter().enumerate() {
        let stride: usize = shape[axis + 1..].iter().product();
        let frequency = index / stride % length;
        let root = match length {
            2 => Complex64::new(if frequency == 0 { 1.0 } else { -1.0 }, 0.0),
            3 => match frequency {
                0 => Complex64::new(1.0, 0.0),
                1 => Complex64::new(-0.5, -3.0_f64.sqrt() / 2.0),
                2 => Complex64::new(-0.5, 3.0_f64.sqrt() / 2.0),
                _ => unreachable!("frequency is reduced modulo three"),
            },
            _ => match frequency % 4 {
                0 => Complex64::new(1.0, 0.0),
                1 => Complex64::new(0.0, -1.0),
                2 => Complex64::new(-1.0, 0.0),
                3 => Complex64::new(0.0, 1.0),
                _ => unreachable!("frequency is reduced modulo four"),
            },
        };
        spectrum += Complex64::new(amplitude, -amplitude / 2.0) * root;
        amplitude /= 2.0;
    }
    spectrum
}

fn check_layout<F, const D: usize>(
    shape: [usize; D],
    storage_kind: Storage,
    unit_roundoff: f64,
    forward: impl Fn(ArrayViewMut<'_, Complex<F>, D>),
    inverse: impl Fn(ArrayViewMut<'_, Complex<F>, D>),
) where
    F: Copy + From<f32> + Into<f64>,
{
    let input = sparse_input(shape);
    let dense_layout = Layout::c_contiguous(shape).expect("valid dense reference layout");
    let (layout, storage_len) = storage_kind.layout(shape);
    let mut storage = vec![Complex::new(F::from(9.0), F::from(-3.0)); storage_len];
    ArrayViewMut::try_new(layout, &mut storage)
        .expect("test storage fits layout")
        .assign(&ArrayView::try_new(dense_layout, &input).expect("reference fits dense layout"));

    // Sixteen rounded real operations per binary stage cover the radix
    // butterflies; ceil(log2(3)) also covers the short radix-three butterfly.
    // Add two operations per axis for inverse normalization, including the
    // rounded reciprocal and multiplication at a length-three axis.
    let stages: u32 = shape
        .iter()
        .map(|length| length.next_power_of_two().ilog2())
        .sum();
    let rank = f64::from(u32::try_from(D).expect("test rank fits u32"));
    let ku = (16.0 * f64::from(stages) + 2.0 * rank) * unit_roundoff;
    let gamma = ku / (1.0 - ku);
    let mut amplitude = -0.25;
    let input_norm = Complex64::new(1.0, 0.5).norm()
        + (0..D)
            .map(|_| {
                let norm = Complex64::new(amplitude, -amplitude / 2.0).norm();
                amplitude /= 2.0;
                norm
            })
            .sum::<f64>();
    // The oracle uses one rounded sqrt for third roots, then at most eight
    // real operations per axis. Fourth roots and dyadic coefficients are
    // exact. Account separately for its binary64 arithmetic.
    let oracle_bound = 8.0 * rank * f64::EPSILON * input_norm;
    let forward_bound = gamma * input_norm + oracle_bound;
    let recovery_bound = (2.0 * gamma + gamma * gamma) * input_norm;

    forward(ArrayViewMut::try_new(layout, &mut storage).expect("forward view fits storage"));
    for (index, actual) in ArrayView::try_new(layout, &storage)
        .expect("forward result fits storage")
        .iter()
        .enumerate()
    {
        let error =
            (Complex64::new(actual.re.into(), actual.im.into()) - expected(index, shape)).norm();
        assert!(
            error <= forward_bound,
            "bin {index}: {error:e} exceeds {forward_bound:e}"
        );
    }
    storage_kind.assert_padding(&storage);

    inverse(ArrayViewMut::try_new(layout, &mut storage).expect("inverse view fits storage"));
    for (index, (actual, expected)) in ArrayView::try_new(layout, &storage)
        .expect("recovered input fits storage")
        .iter()
        .zip(&input)
        .enumerate()
    {
        let error = (Complex64::new(actual.re.into(), actual.im.into())
            - Complex64::new(expected.re.into(), expected.im.into()))
        .norm();
        assert!(
            error <= recovery_bound,
            "sample {index}: {error:e} exceeds {recovery_bound:e}"
        );
    }
    storage_kind.assert_padding(&storage);
}

fn check_matrix<F, const NX: usize, const NY: usize>(storage: Storage, unit_roundoff: f64)
where
    F: MixedRadixScalar<Complex = Complex<F>> + From<f32> + Into<f64>,
    Complex<F>: PlanScratch,
{
    let dynamic = FftPlan2D::<F>::new(Shape2D::new(NX, NY).expect("valid test matrix"));
    let static_plan = StaticFftPlan2D::<F, NX, NY>::new();
    check_layout(
        [NX, NY],
        storage,
        unit_roundoff,
        |view| dynamic.forward_complex_leto_inplace(view),
        |view| dynamic.inverse_complex_leto_inplace(view),
    );
    check_layout(
        [NX, NY],
        storage,
        unit_roundoff,
        |view| static_plan.forward_complex_leto_inplace(view),
        |view| static_plan.inverse_complex_leto_inplace(view),
    );
}

fn check_volume<F>(storage: Storage, unit_roundoff: f64)
where
    F: MixedRadixScalar<Complex = Complex<F>> + From<f32> + Into<f64>,
    Complex<F>: PlanScratch,
{
    let dynamic = FftPlan3D::<F>::new(Shape3D::new(3, 2, 4096).expect("valid test volume"));
    let static_plan = StaticFftPlan3D::<F, 3, 2, 4096>::new();
    check_layout(
        [3, 2, 4096],
        storage,
        unit_roundoff,
        |view| dynamic.forward_complex_leto_inplace(view),
        |view| dynamic.inverse_complex_leto_inplace(view),
    );
    check_layout(
        [3, 2, 4096],
        storage,
        unit_roundoff,
        |view| static_plan.forward_complex_leto_inplace(view),
        |view| static_plan.inverse_complex_leto_inplace(view),
    );
}

#[test]
fn incomplete_lane_groups_preserve_every_matrix_lane() {
    check_matrix::<f32, 4096, 3>(Storage::Dense, f64::from(f32::EPSILON) / 2.0);
    check_matrix::<f64, 4096, 3>(Storage::Dense, f64::EPSILON / 2.0);
}

#[test]
fn contiguous_lanes_borrow_rank_disjoint_staging() {
    for storage in [Storage::Dense, Storage::Interleaved] {
        check_matrix::<f32, 3, 4096>(storage, f64::from(f32::EPSILON) / 2.0);
        check_matrix::<f64, 3, 4096>(storage, f64::EPSILON / 2.0);
        check_volume::<f32>(storage, f64::from(f32::EPSILON) / 2.0);
        check_volume::<f64>(storage, f64::EPSILON / 2.0);
    }
}

#[test]
fn recursive_workspace_preserves_incomplete_lane_groups() {
    check_matrix::<f32, 131_072, 3>(Storage::Dense, f64::from(f32::EPSILON) / 2.0);
    check_matrix::<f64, 131_072, 3>(Storage::Dense, f64::EPSILON / 2.0);
}
