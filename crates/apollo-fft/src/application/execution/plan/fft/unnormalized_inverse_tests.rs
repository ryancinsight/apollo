//! The unnormalized 2-D and 3-D inverses (ADR 0067 slice 1) against the direct
//! inverse sum `y[j] = Σ_k X[k] exp(+2πi j·k / n)` taken axis by axis in f64.
//!
//! ## Bound
//!
//! Compared in the 2-norm, relative to `‖y‖₂ = √N ‖X‖₂` (each unnormalized
//! axis pass scales the 2-norm by `√n`):
//!
//! - Each axis pass of the plan is within `c log₂ p · u` of its exact
//!   transform (Higham, *Accuracy and Stability of Numerical Algorithms*,
//!   2nd ed., §24.1), where `p < 4n` covers Bluestein's padded length and
//!   `u` is the plan precision's unit roundoff. `c` is taken as
//!   [`TOLERANCE_FACTOR`], the two-engine constant of the real-split tests,
//!   which over-covers one engine, and Bluestein's three transforms give the
//!   factor 3. The axis errors add to first order.
//! - The oracle's direct sum along an axis of length `n` is within
//!   `(n + 20) u₆₄ Σ|x_k|` per output: `γ_{n-1}` for the summation (Higham
//!   §3.1), `√2 γ₂` for the complex product (§3.6), at most `2π · 2.35 u`
//!   (about `14.8u`) from the rounded argument `TAU · r / n` of each root,
//!   and one rounding each for its cosine and sine, so at most
//!   `(n + 20) √n · u₆₄` relative in the 2-norm, since `‖x‖₁ ≤ √n ‖x‖₂`.
//!
//! The inputs are narrowed to the plan precision before either side sees
//! them, so input rounding is common to both. Every compared element is held
//! to that 2-norm bound, which bounds each element.
//!
//! ## Sampling
//!
//! The direct sum along a 4096 axis costs `O(N n)`, several times the test
//! budget on the CI runner in a debug build. The axis sums commute, so the
//! oracle takes the longest axis last and, above [`SAMPLED_AXIS`], evaluates
//! it only at `j = m s` and at `n - 1`, where `s` is the least integer of at
//! least `n / SAMPLES` coprime to `n`. A route of that length factors `n`
//! into passes of some `q | n`, and a defective twiddle for residue `r` of
//! `q` corrupts the outputs `j ≡ r (mod q)`. Since `gcd(s, q) = 1`, the
//! samples `m s` for `m < q` cover every residue of every `q ≤ n / s`, over
//! 120 at the lengths here, the four-step factor 64 included. A
//! step sharing a factor with `q` would see only the residue classes of
//! that factor's multiples, where the four-step twiddle is 1.

use super::dimension_2d::FftPlan2D;
use super::dimension_3d::FftPlan3D;
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::domain::metadata::shape::{Shape2D, Shape3D};
use eunomia::{Complex, Complex64};
use leto::{Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3, Layout};
use std::f64::consts::TAU;

/// A plan precision, with its unit roundoff and exact widening.
trait Sample: MixedRadixScalar<Complex = Complex<Self>> + Copy {
    const UNIT_ROUNDOFF: f64;
    fn narrow(value: f64) -> Self;
    fn widen(self) -> f64;
}

impl Sample for f64 {
    const UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;
    fn narrow(value: f64) -> Self {
        value
    }
    fn widen(self) -> f64 {
        self
    }
}

impl Sample for f32 {
    const UNIT_ROUNDOFF: f64 = f32::EPSILON as f64 / 2.0;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "narrowing the test signal to the plan precision is the intent"
    )]
    fn narrow(value: f64) -> Self {
        value as f32
    }
    fn widen(self) -> f64 {
        f64::from(self)
    }
}

/// `2c` of the forward-error bound, as in `tests/real_split_parity.rs`.
const TOLERANCE_FACTOR: f64 = 16.0;

/// A spectrum in C order, narrowed to `F`.
fn spectrum<F: Sample>(len: usize) -> Vec<Complex<F>> {
    (0..len)
        .map(|index| {
            let x = index as f64;
            Complex::new(
                F::narrow((0.17 * x).sin() + 0.3),
                F::narrow(0.23 * (0.31 * x).cos()),
            )
        })
        .collect()
}

/// Axes longer than this are evaluated at about [`SAMPLES`] outputs.
const SAMPLED_AXIS: usize = 512;

/// Outputs evaluated along a sampled axis, at least.
const SAMPLES: usize = 128;

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// The unnormalized inverse of C-order `data` of shape `dims`, axis by axis,
/// the longest axis last and sampled above [`SAMPLED_AXIS`]; unsampled
/// outputs stay `None`.
fn direct_inverse(data: &[Complex64], dims: &[usize]) -> Vec<Option<Complex64>> {
    let mut order: Vec<usize> = (0..dims.len()).collect();
    order.sort_by_key(|&axis| dims[axis]);
    let mut current: Vec<Option<Complex64>> = data.iter().copied().map(Some).collect();
    for (position, &axis) in order.iter().enumerate() {
        let n = dims[axis];
        let step = if position + 1 == order.len() && n > SAMPLED_AXIS {
            (n / SAMPLES..n)
                .find(|&s| gcd(s, n) == 1)
                .expect("invariant: n - 1 is coprime to n")
        } else {
            1
        };
        let stride: usize = dims[axis + 1..].iter().product();
        let roots: Vec<Complex64> = (0..n)
            .map(|r| Complex64::from_polar(1.0, TAU * r as f64 / n as f64))
            .collect();
        let mut next = vec![None; current.len()];
        for outer in 0..current.len() / (n * stride) {
            for inner in 0..stride {
                let base = outer * n * stride + inner;
                for j in (0..n).filter(|&j| j % step == 0 || j + 1 == n) {
                    let mut acc = Complex64::default();
                    for k in 0..n {
                        acc += current[base + k * stride]
                            .expect("invariant: every axis but the last is complete")
                            * roots[(j * k) % n];
                    }
                    next[base + j * stride] = Some(acc);
                }
            }
        }
        current = next;
    }
    current
}

/// Asserts `actual` (C order) is the unnormalized inverse of `input`.
fn assert_unnormalized_inverse<F: Sample>(
    label: &str,
    dims: &[usize],
    input: &[Complex<F>],
    actual: impl Iterator<Item = Complex<F>>,
) {
    let wide: Vec<Complex64> = input
        .iter()
        .map(|v| Complex64::new(v.re.widen(), v.im.widen()))
        .collect();
    let expected = direct_inverse(&wide, dims);
    let norm = wide.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    let count: usize = dims.iter().product();
    let scale = (count as f64).sqrt() * norm;
    let relative: f64 = dims
        .iter()
        .map(|&n| {
            let log_p = f64::from((4 * n).next_power_of_two().trailing_zeros());
            3.0 * TOLERANCE_FACTOR * log_p * F::UNIT_ROUNDOFF
                + (n as f64 + 20.0) * (n as f64).sqrt() * (f64::EPSILON / 2.0)
        })
        .sum();
    let bound = relative * scale;
    let mut seen = 0usize;
    for (index, (a, e)) in actual.zip(&expected).enumerate() {
        seen += 1;
        if let Some(e) = e {
            let error = (Complex64::new(a.re.widen(), a.im.widen()) - *e).norm();
            assert!(
                error <= bound,
                "{label} element {index}: {error:.3e} > {bound:.3e}"
            );
        }
    }
    assert_eq!(seen, count, "{label}: the transform returned every element");
}

const SHAPES_2D: [[usize; 2]; 6] = [[6, 10], [97, 47], [4096, 3], [3, 4096], [1, 16], [361, 1]];

const SHAPES_3D: [[usize; 3]; 5] = [[4, 6, 8], [47, 47, 31], [16, 1, 2048], [1, 1, 7], [5, 6, 7]];

fn plane<F: Sample>()
where
    Complex<F>: PlanScratch,
{
    for [nx, ny] in SHAPES_2D {
        let plan = FftPlan2D::<F>::new(Shape2D::new(nx, ny).expect("non-zero"));
        let input = spectrum::<F>(nx * ny);

        let mut array = Array2::from_shape_vec([nx, ny], input.clone()).expect("C-order shape");
        plan.inverse_complex_unnorm_inplace(&mut array);
        assert_unnormalized_inverse(
            &format!("{nx}x{ny} array"),
            &[nx, ny],
            &input,
            array.iter().copied(),
        );

        let layout = Layout::f_contiguous([nx, ny]).expect("valid Fortran layout");
        let mut storage = vec![Complex::<F>::default(); nx * ny];
        let source = Array2::from_shape_vec([nx, ny], input.clone()).expect("C-order shape");
        ArrayViewMut2::try_new(layout, &mut storage)
            .expect("layout fits storage")
            .assign(&source.view());
        plan.inverse_complex_unnorm_leto_inplace(
            ArrayViewMut2::try_new(layout, &mut storage).expect("layout fits storage"),
        );
        let view = ArrayView2::try_new(layout, &storage).expect("layout fits storage");
        assert_unnormalized_inverse(
            &format!("{nx}x{ny} Fortran view"),
            &[nx, ny],
            &input,
            view.iter().copied(),
        );
    }
}

fn volume<F: Sample>()
where
    Complex<F>: PlanScratch,
{
    for [nx, ny, nz] in SHAPES_3D {
        let plan = FftPlan3D::<F>::new(Shape3D::new(nx, ny, nz).expect("non-zero"));
        let input = spectrum::<F>(nx * ny * nz);

        let mut array = Array3::from_shape_vec([nx, ny, nz], input.clone()).expect("C-order shape");
        plan.inverse_complex_unnorm_inplace(&mut array);
        assert_unnormalized_inverse(
            &format!("{nx}x{ny}x{nz} array"),
            &[nx, ny, nz],
            &input,
            array.iter().copied(),
        );

        let layout = Layout::f_contiguous([nx, ny, nz]).expect("valid Fortran layout");
        let mut storage = vec![Complex::<F>::default(); nx * ny * nz];
        let source = Array3::from_shape_vec([nx, ny, nz], input.clone()).expect("C-order shape");
        ArrayViewMut3::try_new(layout, &mut storage)
            .expect("layout fits storage")
            .assign(&source.view());
        plan.inverse_complex_unnorm_leto_inplace(
            ArrayViewMut3::try_new(layout, &mut storage).expect("layout fits storage"),
        );
        let view = ArrayView3::try_new(layout, &storage).expect("layout fits storage");
        assert_unnormalized_inverse(
            &format!("{nx}x{ny}x{nz} Fortran view"),
            &[nx, ny, nz],
            &input,
            view.iter().copied(),
        );
    }
}

#[test]
fn plane_unnormalized_inverse_is_the_direct_sum() {
    plane::<f64>();
    plane::<f32>();
}

#[test]
fn volume_unnormalized_inverse_is_the_direct_sum() {
    volume::<f64>();
    volume::<f32>();
}

/// With every axis a power of two, each normalized pass divides by its
/// length and the unnormalized pass does not, and both are exact binary
/// rescales absent overflow and underflow (IEEE 754), so the unnormalized
/// inverse is the normalized one times `N`, bit for bit.
fn power_of_two_scaling<F: Sample>()
where
    Complex<F>: PlanScratch,
{
    for [nx, ny] in [[8, 16], [4, 2048], [1, 64]] {
        let plan = FftPlan2D::<F>::new(Shape2D::new(nx, ny).expect("non-zero"));
        let input = spectrum::<F>(nx * ny);
        let mut unnorm = Array2::from_shape_vec([nx, ny], input.clone()).expect("C-order shape");
        plan.inverse_complex_unnorm_inplace(&mut unnorm);
        let mut normalized = Array2::from_shape_vec([nx, ny], input).expect("C-order shape");
        plan.inverse_complex_inplace(&mut normalized);
        let scale = F::narrow((nx * ny) as f64);
        for (u, v) in unnorm.iter().zip(normalized.iter()) {
            assert!(
                u.re.widen().to_bits() == (v.re * scale).widen().to_bits()
                    && u.im.widen().to_bits() == (v.im * scale).widen().to_bits(),
                "{nx}x{ny}: {u:?} is not {v:?} times {}",
                nx * ny
            );
        }
    }
    for [nx, ny, nz] in [[4, 8, 8], [2, 4, 4096], [1, 1, 32]] {
        let plan = FftPlan3D::<F>::new(Shape3D::new(nx, ny, nz).expect("non-zero"));
        let input = spectrum::<F>(nx * ny * nz);
        let mut unnorm =
            Array3::from_shape_vec([nx, ny, nz], input.clone()).expect("C-order shape");
        plan.inverse_complex_unnorm_inplace(&mut unnorm);
        let mut normalized = Array3::from_shape_vec([nx, ny, nz], input).expect("C-order shape");
        plan.inverse_complex_inplace(&mut normalized);
        let scale = F::narrow((nx * ny * nz) as f64);
        for (u, v) in unnorm.iter().zip(normalized.iter()) {
            assert!(
                u.re.widen().to_bits() == (v.re * scale).widen().to_bits()
                    && u.im.widen().to_bits() == (v.im * scale).widen().to_bits(),
                "{nx}x{ny}x{nz}: {u:?} is not {v:?} times {}",
                nx * ny * nz
            );
        }
    }
}

#[test]
fn power_of_two_unnormalized_inverse_is_the_normalized_one_times_n_exactly() {
    power_of_two_scaling::<f64>();
    power_of_two_scaling::<f32>();
}
