use crate::application::execution::plan::fft::dimension_3d::StaticFftPlan3D;
use eunomia::Complex64;
use leto::{Array3, ArrayView3, ArrayViewMut3, Layout};
use std::f64::consts::PI;

fn signal<const NX: usize, const NY: usize, const NZ: usize>() -> Array3<Complex64> {
    Array3::from_shape_fn([NX, NY, NZ], |[i, j, k]| {
        let x = ((i * NY + j) * NZ + k) as f64;
        Complex64::new(
            (0.17 * x).sin() + 0.11 * (0.07 * x).cos(),
            0.23 * (0.31 * x).cos(),
        )
    })
}

fn direct_forward<const NX: usize, const NY: usize, const NZ: usize>(
    input: &Array3<Complex64>,
) -> Array3<Complex64> {
    let mut out = Array3::from_elem([NX, NY, NZ], Complex64::new(0.0, 0.0));
    for kx in 0..NX {
        for ky in 0..NY {
            for kz in 0..NZ {
                let mut acc = Complex64::new(0.0, 0.0);
                for x in 0..NX {
                    for y in 0..NY {
                        for z in 0..NZ {
                            let phase = -2.0
                                * PI
                                * ((kx * x) as f64 / NX as f64
                                    + (ky * y) as f64 / NY as f64
                                    + (kz * z) as f64 / NZ as f64);
                            acc += input[[x, y, z]] * Complex64::from_polar(1.0, phase);
                        }
                    }
                }
                out[[kx, ky, kz]] = acc;
            }
        }
    }
    out
}

fn max_err(a: &Array3<Complex64>, b: &Array3<Complex64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0, f64::max)
}

fn assert_view_matches(
    label: &str,
    stage: &str,
    actual: &ArrayView3<'_, Complex64>,
    expected: &Array3<Complex64>,
) {
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "{label} {stage} mismatch at index {index}"
        );
    }
}

fn assert_view_layout<const NX: usize, const NY: usize, const NZ: usize>(
    label: &str,
    layout: Layout<3>,
    storage_len: usize,
    forward: impl for<'view> Fn(ArrayViewMut3<'view, Complex64>),
    inverse: impl for<'view> Fn(ArrayViewMut3<'view, Complex64>),
) {
    let input = signal::<NX, NY, NZ>();
    let mut expected = input.clone();
    forward(expected.view_mut());
    let mut storage = vec![Complex64::default(); storage_len];
    ArrayViewMut3::try_new(layout, &mut storage)
        .expect("test layout fits storage")
        .assign(&input.view());

    forward(ArrayViewMut3::try_new(layout, &mut storage).expect("test layout fits storage"));
    let transformed = ArrayView3::try_new(layout, &storage).expect("test layout fits storage");
    assert_view_matches(label, "forward", &transformed, &expected);

    inverse(expected.view_mut());
    inverse(ArrayViewMut3::try_new(layout, &mut storage).expect("test layout fits storage"));
    let recovered = ArrayView3::try_new(layout, &storage).expect("test layout fits storage");
    assert_view_matches(label, "roundtrip", &recovered, &expected);
}

fn exercise_nonstandard_layouts(
    plan_name: &str,
    forward: impl for<'view> Fn(ArrayViewMut3<'view, Complex64>),
    inverse: impl for<'view> Fn(ArrayViewMut3<'view, Complex64>),
) {
    const NX: usize = 2;
    const NY: usize = 3;
    const NZ: usize = 4;
    let cases = [
        (
            "offset C-order",
            Layout::try_new([NX, NY, NZ], [(NY * NZ) as isize, NZ as isize, 1], 3)
                .expect("valid offset layout"),
            NX * NY * NZ + 3,
        ),
        (
            "Fortran-order",
            Layout::f_contiguous([NX, NY, NZ]).expect("valid Fortran layout"),
            NX * NY * NZ,
        ),
        (
            "strided",
            Layout::try_new([NX, NY, NZ], [40, 10, 2], 1).expect("valid strided layout"),
            68,
        ),
    ];

    for (layout_name, layout, storage_len) in cases {
        let label = format!("{plan_name} {layout_name}");
        assert_view_layout::<NX, NY, NZ>(&label, layout, storage_len, &forward, &inverse);
    }
}

#[test]
fn static_fft_3d_plan_is_zero_sized() {
    assert_eq!(std::mem::size_of::<StaticFftPlan3D<f64, 3, 4, 5>>(), 0);
    assert_eq!(StaticFftPlan3D::<f64, 3, 4, 5>::new().shape(), (3, 4, 5));
    assert_eq!(StaticFftPlan3D::<f64, 3, 4, 5>::new().nz_c(), 3);
}

#[test]
fn static_fft_3d_forward_matches_direct() {
    let plan = StaticFftPlan3D::<f64, 3, 4, 5>::new();
    let input = signal::<3, 4, 5>();
    let expected = direct_forward::<3, 4, 5>(&input);
    let mut actual = input;
    plan.forward_complex_inplace(&mut actual);
    let err = max_err(&actual, &expected);
    assert!(err <= 1.0e-10, "static 3D forward mismatch err={err:.2e}");
}

#[test]
fn static_fft_3d_inverse_roundtrip_recovers_input() {
    let plan = StaticFftPlan3D::<f64, 3, 4, 5>::new();
    let input = signal::<3, 4, 5>();
    let mut actual = input.clone();
    plan.forward_complex_inplace(&mut actual);
    plan.inverse_complex_inplace(&mut actual);
    let err = max_err(&actual, &input);
    assert!(err <= 1.0e-10, "static 3D roundtrip mismatch err={err:.2e}");
}

#[test]
fn axis_passes_compose_to_full_forward_and_roundtrip_per_axis() {
    use crate::application::execution::plan::fft::dimension_3d::FftPlan3D;
    use crate::domain::metadata::shape::Shape3D;

    let (nx, ny, nz) = (6usize, 4usize, 8usize);
    let plan = FftPlan3D::<f64>::new(
        Shape3D::new(nx, ny, nz).expect("invariant: shape lengths are non-zero"),
    );
    let original = Array3::from_shape_fn([nx, ny, nz], |[i, j, k]| {
        let x = ((i * ny + j) * nz + k) as f64;
        Complex64::new((0.17 * x).sin() + 0.3, 0.23 * (0.31 * x).cos())
    });

    // Sequential per-axis forward (z, y, x) equals the full separable forward.
    let mut full = original.clone();
    plan.forward_complex_inplace(&mut full);
    let mut composed = original.clone();
    plan.forward_axis_complex_inplace(&mut composed, 2);
    plan.forward_axis_complex_inplace(&mut composed, 1);
    plan.forward_axis_complex_inplace(&mut composed, 0);
    let err = composed
        .iter()
        .zip(full.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        err <= 1.0e-10,
        "axis compose != full forward, err={err:.2e}"
    );

    // forward_axis then inverse_axis along the same axis is the identity.
    for axis in 0..3 {
        let mut d = original.clone();
        plan.forward_axis_complex_inplace(&mut d, axis);
        plan.inverse_axis_complex_inplace(&mut d, axis);
        let err = d
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        assert!(
            err <= 1.0e-10,
            "axis {axis} roundtrip not identity, err={err:.2e}"
        );
    }
}

#[test]
fn static_fft_3d_preserves_logical_view_order() {
    let plan = StaticFftPlan3D::<f64, 2, 3, 4>::new();
    exercise_nonstandard_layouts(
        "static",
        |view| plan.forward_complex_leto_inplace(view),
        |view| plan.inverse_complex_leto_inplace(view),
    );
}

#[test]
fn dynamic_fft_3d_preserves_logical_view_order() {
    use crate::application::execution::plan::fft::dimension_3d::FftPlan3D;
    use crate::domain::metadata::shape::Shape3D;

    let plan = FftPlan3D::<f64>::new(
        Shape3D::new(2, 3, 4).expect("invariant: shape lengths are non-zero"),
    );
    exercise_nonstandard_layouts(
        "dynamic",
        |view| plan.forward_complex_leto_inplace(view),
        |view| plan.inverse_complex_leto_inplace(view),
    );
}
