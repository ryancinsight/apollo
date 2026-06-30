use crate::application::execution::plan::fft::dimension_3d::StaticFftPlan3D;
use leto::Array3;
use eunomia::Complex64;
use std::f64::consts::PI;

fn signal<const NX: usize, const NY: usize, const NZ: usize>() -> Array3<Complex64> {
    Array3::from_shape_fn((NX, NY, NZ), |(i, j, k)| {
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
                            acc += input[(x, y, z)] * Complex64::from_polar(1.0, phase);
                        }
                    }
                }
                out[(kx, ky, kz)] = acc;
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

#[test]
fn static_fft_3d_plan_is_zero_sized() {
    assert_eq!(std::mem::size_of::<StaticFftPlan3D<f64, 3, 4, 5>>(), 0);
    assert_eq!(StaticFftPlan3D::<f64, 3, 4, 5>::new().shape(), [3, 4, 5]);
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
    let plan = FftPlan3D::<f64>::new(Shape3D { nx, ny, nz });
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
    assert!(err <= 1.0e-10, "axis compose != full forward, err={err:.2e}");

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
        assert!(err <= 1.0e-10, "axis {axis} roundtrip not identity, err={err:.2e}");
    }
}
