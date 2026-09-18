use super::{FftPlan2D, StaticFftPlan2D};
use crate::domain::metadata::shape::Shape2D;
use eunomia::Complex64;
use leto::{Array2, ArrayView2, ArrayViewMut2, Layout};
use std::f64::consts::PI;

fn signal<const NX: usize, const NY: usize>() -> Array2<Complex64> {
    Array2::from_shape_fn([NX, NY], |[i, j]| {
        let x = (i * NY + j) as f64;
        Complex64::new(
            (0.17 * x).sin() + 0.11 * (0.07 * x).cos(),
            0.23 * (0.31 * x).cos(),
        )
    })
}

fn direct_forward<const NX: usize, const NY: usize>(
    input: &Array2<Complex64>,
) -> Array2<Complex64> {
    let mut out = Array2::from_elem([NX, NY], Complex64::new(0.0, 0.0));
    for kx in 0..NX {
        for ky in 0..NY {
            let mut acc = Complex64::new(0.0, 0.0);
            for x in 0..NX {
                for y in 0..NY {
                    let phase =
                        -2.0 * PI * ((kx * x) as f64 / NX as f64 + (ky * y) as f64 / NY as f64);
                    acc += input[[x, y]] * Complex64::from_polar(1.0, phase);
                }
            }
            out[[kx, ky]] = acc;
        }
    }
    out
}

fn max_err(a: &Array2<Complex64>, b: &Array2<Complex64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0, f64::max)
}

fn assert_view_matches(
    label: &str,
    stage: &str,
    actual: &ArrayView2<'_, Complex64>,
    expected: &Array2<Complex64>,
) {
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "{label} {stage} mismatch at index {index}"
        );
    }
}

fn assert_view_layout<const NX: usize, const NY: usize>(
    label: &str,
    layout: Layout<2>,
    storage_len: usize,
    forward: impl for<'view> Fn(ArrayViewMut2<'view, Complex64>),
    inverse: impl for<'view> Fn(ArrayViewMut2<'view, Complex64>),
) {
    let input = signal::<NX, NY>();
    let mut expected = input.clone();
    forward(expected.view_mut());
    let mut storage = vec![Complex64::default(); storage_len];
    ArrayViewMut2::try_new(layout, &mut storage)
        .expect("test layout fits storage")
        .assign(&input.view());

    forward(ArrayViewMut2::try_new(layout, &mut storage).expect("test layout fits storage"));
    let transformed = ArrayView2::try_new(layout, &storage).expect("test layout fits storage");
    assert_view_matches(label, "forward", &transformed, &expected);

    inverse(expected.view_mut());
    inverse(ArrayViewMut2::try_new(layout, &mut storage).expect("test layout fits storage"));
    let recovered = ArrayView2::try_new(layout, &storage).expect("test layout fits storage");
    assert_view_matches(label, "roundtrip", &recovered, &expected);
}

fn exercise_nonstandard_layouts(
    plan_name: &str,
    forward: impl for<'view> Fn(ArrayViewMut2<'view, Complex64>),
    inverse: impl for<'view> Fn(ArrayViewMut2<'view, Complex64>),
) {
    const NX: usize = 3;
    const NY: usize = 4;
    let cases = [
        (
            "offset C-order",
            Layout::try_new([NX, NY], [NY as isize, 1], 3).expect("valid offset layout"),
            NX * NY + 3,
        ),
        (
            "Fortran-order",
            Layout::f_contiguous([NX, NY]).expect("valid Fortran layout"),
            NX * NY,
        ),
        (
            "strided",
            Layout::try_new([NX, NY], [10, 2], 1).expect("valid strided layout"),
            28,
        ),
    ];

    for (layout_name, layout, storage_len) in cases {
        let label = format!("{plan_name} {layout_name}");
        assert_view_layout::<NX, NY>(&label, layout, storage_len, &forward, &inverse);
    }
}

#[test]
fn static_fft_2d_plan_is_zero_sized() {
    assert_eq!(std::mem::size_of::<StaticFftPlan2D<f64, 4, 5>>(), 0);
    assert_eq!(StaticFftPlan2D::<f64, 4, 5>::new().shape(), (4, 5));
}

#[test]
fn static_fft_2d_forward_matches_direct() {
    let plan = StaticFftPlan2D::<f64, 4, 5>::new();
    let input = signal::<4, 5>();
    let expected = direct_forward::<4, 5>(&input);
    let mut actual = input;
    plan.forward_complex_inplace(&mut actual);
    let err = max_err(&actual, &expected);
    assert!(err <= 1.0e-10, "static 2D forward mismatch err={err:.2e}");
}

#[test]
fn static_fft_2d_inverse_roundtrip_recovers_input() {
    let plan = StaticFftPlan2D::<f64, 4, 5>::new();
    let input = signal::<4, 5>();
    let mut actual = input.clone();
    plan.forward_complex_inplace(&mut actual);
    plan.inverse_complex_inplace(&mut actual);
    let err = max_err(&actual, &input);
    assert!(err <= 1.0e-10, "static 2D roundtrip mismatch err={err:.2e}");
}

#[test]
fn static_fft_2d_preserves_logical_view_order() {
    let plan = StaticFftPlan2D::<f64, 3, 4>::new();
    exercise_nonstandard_layouts(
        "static",
        |view| plan.forward_complex_leto_inplace(view),
        |view| plan.inverse_complex_leto_inplace(view),
    );
}

#[test]
fn dynamic_fft_2d_preserves_logical_view_order() {
    let plan =
        FftPlan2D::<f64>::new(Shape2D::new(3, 4).expect("invariant: shape lengths are non-zero"));
    exercise_nonstandard_layouts(
        "dynamic",
        |view| plan.forward_complex_leto_inplace(view),
        |view| plan.inverse_complex_leto_inplace(view),
    );
}

/// The axis passes are the halves of the whole-plane transform, in its
/// order (rows then columns forward, columns then rows inverse), so they
/// compose to it bit for bit; a forward then an inverse along one axis is
/// the identity within the 1-D round trip's rounding.
#[test]
fn axis_passes_compose_to_the_plane_and_invert_per_axis() {
    for (nx, ny) in [(6usize, 10usize), (8, 16), (5, 1), (1, 7)] {
        let plan = FftPlan2D::<f64>::new(Shape2D::new(nx, ny).expect("non-zero"));
        let original = Array2::from_shape_fn([nx, ny], |[i, j]| {
            let x = (i * ny + j) as f64;
            Complex64::new((0.17 * x).sin() + 0.3, 0.23 * (0.31 * x).cos())
        });
        let mut full = original.clone();
        plan.forward_complex_inplace(&mut full);
        let mut composed = original.clone();
        plan.forward_axis_complex_inplace(&mut composed, 1);
        plan.forward_axis_complex_inplace(&mut composed, 0);
        assert!(
            full.iter().zip(composed.iter()).all(|(a, b)| a == b),
            "{nx}x{ny}: forward differs from rows then columns"
        );
        let mut full_inverse = full.clone();
        plan.inverse_complex_inplace(&mut full_inverse);
        plan.inverse_axis_complex_inplace(&mut composed, 0);
        plan.inverse_axis_complex_inplace(&mut composed, 1);
        assert!(
            full_inverse
                .iter()
                .zip(composed.iter())
                .all(|(a, b)| a == b),
            "{nx}x{ny}: inverse differs from columns then rows"
        );
        for axis in 0..2 {
            let mut data = original.clone();
            plan.forward_axis_complex_inplace(&mut data, axis);
            plan.inverse_axis_complex_inplace(&mut data, axis);
            // A forward and a normalized inverse FFT of length `n` stay
            // within `γ √n max|x|` per sample, `γ = 8 ⌈log₂ n⌉ ε` (Higham,
            // Accuracy and Stability of Numerical Algorithms, §24.1, both
            // transforms), plus one `ε` for the normalization.
            let n = if axis == 0 { nx } else { ny } as f64;
            let peak = original.iter().fold(0.0f64, |m, v| m.max(v.norm()));
            let bound = (8.0 * n.log2().ceil() * n.sqrt() + 1.0) * f64::EPSILON * peak;
            assert!(
                max_err(&data, &original) <= bound,
                "{nx}x{ny} axis {axis}: round trip beyond {bound:e}"
            );
        }
    }
}

/// The column pass computes each column's DFT: checked against the direct
/// sum along axis 0 only.
#[test]
fn column_pass_matches_the_direct_column_sum() {
    const NX: usize = 6;
    const NY: usize = 4;
    let plan = FftPlan2D::<f64>::new(Shape2D::new(NX, NY).expect("non-zero"));
    let input = signal::<NX, NY>();
    let mut actual = input.clone();
    plan.forward_axis_complex_inplace(&mut actual, 0);
    let mut expected = Array2::from_elem([NX, NY], Complex64::new(0.0, 0.0));
    for k in 0..NX {
        for y in 0..NY {
            let mut acc = Complex64::new(0.0, 0.0);
            for x in 0..NX {
                let phase = -2.0 * PI * (k * x) as f64 / NX as f64;
                acc += input[[x, y]] * Complex64::from_polar(1.0, phase);
            }
            expected[[k, y]] = acc;
        }
    }
    // The FFT is within `8 ⌈log₂ N⌉ ε √N max|x|` of the exact column DFT
    // (Higham §24.1), and the direct sum within `N (1 + 2 ε) ε max|x|` for
    // its `N` products and additions with unit-modulus phases rounded once.
    let peak = input.iter().fold(0.0f64, |m, v| m.max(v.norm()));
    let n = NX as f64;
    let bound = (8.0 * n.log2().ceil() * n.sqrt() + 2.0 * n) * f64::EPSILON * peak;
    assert!(max_err(&actual, &expected) <= bound, "beyond {bound:e}");
}

#[test]
fn axis_passes_preserve_logical_view_order() {
    let plan =
        FftPlan2D::<f64>::new(Shape2D::new(3, 4).expect("invariant: shape lengths are non-zero"));
    for axis in 0..2 {
        exercise_nonstandard_layouts(
            "axis",
            |view| plan.forward_axis_complex_leto_inplace(view, axis),
            |view| plan.inverse_axis_complex_leto_inplace(view, axis),
        );
    }
}

#[test]
#[should_panic(expected = "axis must be 0 or 1")]
fn axis_beyond_the_plane_is_refused() {
    let plan = FftPlan2D::<f64>::new(Shape2D::new(3, 4).expect("non-zero"));
    let mut data = Array2::from_elem([3, 4], Complex64::new(0.0, 0.0));
    plan.forward_axis_complex_inplace(&mut data, 2);
}
