use super::*;
use eunomia::Complex64;
use leto::{ArrayView2, Layout};
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
