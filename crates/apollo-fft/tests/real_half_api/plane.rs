//! The `(nx, ny/2 + 1)` half-spectrum 2-D pair.

use crate::{count_allocations, l1, tolerance, Sample};
use apollo_fft::{PlanCacheProvider, PlanScratch, F16};
use eunomia::{Complex, Complex64};
use leto::{Array2, Array3};

/// Shapes exercising the split (`ny` a multiple of four), each refusal (an odd
/// `ny`, an even one that is not a multiple of four, one below four), an extent
/// of one on each axis in turn, and one plane whose rows and columns both
/// cross the lane helpers' parallel thresholds.
const SHAPES: [[usize; 2]; 11] = [
    [8, 12],
    [16, 16],
    [32, 16],
    [32, 32],
    [3, 8],
    [7, 9],
    [6, 10],
    [4, 2],
    [1, 8],
    [4, 1],
    [2048, 64],
];

fn field<T: Sample>([nx, ny]: [usize; 2]) -> Array2<T> {
    Array2::from_shape_fn([nx, ny], |[i, j]| {
        let x = (i * ny + j) as f64;
        T::from_f64((0.013 * x).sin() + 0.35 * (0.071 * x).cos() - 0.05)
    })
}

/// Distance between two bins, in f64.
fn distance<P: Copy + Into<f64>>(a: Complex<P>, b: Complex<P>) -> f64 {
    let (a_re, a_im, b_re, b_im): (f64, f64, f64, f64) =
        (a.re.into(), a.im.into(), b.re.into(), b.im.into());
    (a_re - b_re).hypot(a_im - b_im)
}

/// The forward against the full spectrum and the pair's round trip, for
/// storage scalar `T`.
fn pair<T>()
where
    T: Sample,
    T::PlanScalar: PlanCacheProvider + Into<f64>,
    Complex<T::PlanScalar>: PlanScratch,
{
    for shape @ [nx, ny] in SHAPES {
        let depth = ny / 2 + 1;
        let real = field::<T>(shape);
        let stored: Vec<f64> = real.iter().map(|&v| v.to_f64()).collect();
        // One transform's per-bin bound over the whole plane: the separable
        // passes compose to log2(nx) + log2(ny) stages.
        let one = tolerance(nx * ny, l1(&stored), T::PLAN_UNIT);

        let full = apollo_fft::fft_2d_array::<T>(&real);
        let mut half = Array2::from_elem([nx, depth], Complex::<T::PlanScalar>::default());
        apollo_fft::fft_2d_array_half_into(&real, &mut half);
        for i in 0..nx {
            for j in 0..depth {
                // Two routes to one bin, each within `one` of the exact value.
                let error = distance(half[[i, j]], full[[i, j]]);
                assert!(
                    error <= 2.0 * one,
                    "{} {shape:?} bin ({i}, {j}): {error:.3e} from the full spectrum (bound {:.3e})",
                    std::any::type_name::<T>(),
                    2.0 * one
                );
            }
        }

        let mut back = Array2::from_elem(shape, T::from_f64(0.0));
        apollo_fft::ifft_2d_array_half_into(&mut half, &mut back);
        for (index, (got, want)) in back.iter().zip(&stored).enumerate() {
            let got = got.to_f64();
            // The forward and the normalized inverse each within `one`, then
            // one rounding to storage.
            let bound = 2.0 * one + T::STORAGE_UNIT * (want.abs() + 2.0 * one);
            assert!(
                (got - want).abs() <= bound,
                "{} {shape:?} sample {index}: {got} against {want} (bound {bound:.2e})",
                std::any::type_name::<T>()
            );
        }
    }
}

#[test]
fn the_half_forward_is_the_full_spectrum_truncated_and_the_pair_round_trips() {
    pair::<f64>();
    pair::<f32>();
    pair::<F16>();
}

/// A plane is a volume of unit x extent whose halved axis is the same last
/// axis, so the 3-D pair on `[1, nx, ny]` computes the same bins and samples
/// through its own passes.
fn agrees_with_the_volume_pair<T>()
where
    T: Sample,
    T::PlanScalar: PlanCacheProvider + Into<f64>,
    Complex<T::PlanScalar>: PlanScratch,
{
    for shape @ [nx, ny] in SHAPES {
        let depth = ny / 2 + 1;
        let real = field::<T>(shape);
        let stored: Vec<f64> = real.iter().map(|&v| v.to_f64()).collect();
        let one = tolerance(nx * ny, l1(&stored), T::PLAN_UNIT);
        let volume = Array3::from_shape_fn([1, nx, ny], |[_, i, j]| real[[i, j]]);

        let mut half = Array2::from_elem([nx, depth], Complex::<T::PlanScalar>::default());
        let mut half_volume =
            Array3::from_elem([1, nx, depth], Complex::<T::PlanScalar>::default());
        apollo_fft::fft_2d_array_half_into(&real, &mut half);
        apollo_fft::fft_3d_array_half_into(&volume, &mut half_volume);
        for i in 0..nx {
            for j in 0..depth {
                let error = distance(half[[i, j]], half_volume[[0, i, j]]);
                assert!(
                    error <= 2.0 * one,
                    "{} {shape:?} bin ({i}, {j}): {error:.3e} from the volume pair's (bound {:.3e})",
                    std::any::type_name::<T>(),
                    2.0 * one
                );
            }
        }

        let mut back = Array2::from_elem(shape, T::from_f64(0.0));
        let mut back_volume = Array3::from_elem([1, nx, ny], T::from_f64(0.0));
        apollo_fft::ifft_2d_array_half_into(&mut half, &mut back);
        apollo_fft::ifft_3d_array_half_into(&mut half_volume, &mut back_volume);
        for (index, (plane, volume)) in back.iter().zip(back_volume.iter()).enumerate() {
            let (plane, volume) = (plane.to_f64(), volume.to_f64());
            // Each inverse within `one` of the exact sample, each then rounded
            // once to storage.
            let bound = 2.0 * one + 2.0 * T::STORAGE_UNIT * (volume.abs() + one);
            assert!(
                (plane - volume).abs() <= bound,
                "{} {shape:?} sample {index}: {plane} against the volume pair's {volume} (bound {bound:.2e})",
                std::any::type_name::<T>()
            );
        }
    }
}

#[test]
fn the_pair_agrees_with_the_volume_pair_on_a_unit_depth_volume() {
    agrees_with_the_volume_pair::<f64>();
    agrees_with_the_volume_pair::<f32>();
    agrees_with_the_volume_pair::<F16>();
}

#[test]
fn the_half_inverse_is_the_real_part_of_the_hermitian_completion() {
    for shape @ [nx, ny] in SHAPES {
        let depth = ny / 2 + 1;
        // An arbitrary half spectrum, not the forward of any real field: its
        // zero and Nyquist columns carry imaginary parts and lack the
        // in-column symmetry a real field's spectrum has.
        let half = Array2::from_shape_fn([nx, depth], |[i, j]| {
            let x = (i * depth + j) as f64;
            Complex64::new((0.29 * x).sin(), (0.43 * x).cos() - 0.2)
        });
        let mut completed = Array2::from_shape_fn(shape, |[i, j]| {
            if j < depth {
                half[[i, j]]
            } else {
                let bin = half[[(nx - i) % nx, ny - j]];
                Complex64::new(bin.re, -bin.im)
            }
        });
        apollo_fft::ifft_2d_complex_inplace::<f64>(&mut completed);

        let mut scratch = half.clone();
        let mut ours = Array2::from_elem(shape, 0.0_f64);
        apollo_fft::ifft_2d_array_half_into(&mut scratch, &mut ours);

        // Two normalized inverses of spectra whose 1-norm is at most twice the
        // half's; each errs by at most the bound for that norm over the count.
        let total = nx * ny;
        let half_l1: f64 = half.iter().map(|z| z.re.abs() + z.im.abs()).sum();
        let bound = 2.0 * tolerance(total, 2.0 * half_l1 / total as f64, f64::EPSILON / 2.0);
        for (index, (a, b)) in ours.iter().zip(completed.iter()).enumerate() {
            assert!(
                (a - b.re).abs() <= bound,
                "{shape:?} sample {index}: {a} against the completion's {} (bound {bound:.2e})",
                b.re
            );
        }
    }
}

#[test]
fn the_pair_allocates_nothing_once_warm() {
    // Under moirai's parallel threshold, so every row runs on this thread and
    // the thread-local count sees all of the work.
    let shape @ [nx, ny] = [8, 12];
    let real = field::<f64>(shape);
    let mut half = Array2::from_elem([nx, ny / 2 + 1], Complex64::default());
    let mut back = Array2::from_elem(shape, 0.0_f64);

    // One warm pair: the plan, its tables, and the scratch the x pass borrows
    // are built on first use, and those are not per-call costs.
    apollo_fft::fft_2d_array_half_into(&real, &mut half);
    apollo_fft::ifft_2d_array_half_into(&mut half, &mut back);

    let ((), observed) = count_allocations(|| {
        apollo_fft::fft_2d_array_half_into(&real, &mut half);
        apollo_fft::ifft_2d_array_half_into(&mut half, &mut back);
    });
    assert_eq!(
        observed, 0,
        "the pair allocated {observed} times on a warm plan"
    );
}

/// Refused-length shapes (`ny` not a positive multiple of four): one under
/// every shipped scalar's parallel byte threshold for the widened fallback
/// pairing, and one that clears it even for the narrowest scalar (`F16`'s
/// `Complex<f32>` bins beside `F16` reals) — `lanes::PARALLEL_BYTES` is sized
/// off `f64`-width lane data, so a narrower scalar needs more rows to reach it.
const FALLBACK_PLANE_SHAPES: [[usize; 2]; 2] = [[8, 6], [4096, 30]];

/// The widened row fallback (`ny` not a positive multiple of four) reuses
/// thread-local scratch, sharing the retained-memory bound the x pass already
/// keeps, instead of allocating one `ny`-length buffer per scheduled task.
/// Regression coverage for `half_plane::forward`/`half_plane::inverse`.
fn the_fallback_allocates_nothing_once_warm<T>()
where
    T: Sample,
    T::PlanScalar: PlanCacheProvider + Into<f64>,
    Complex<T::PlanScalar>: PlanScratch,
{
    for shape @ [nx, ny] in FALLBACK_PLANE_SHAPES {
        assert!(
            !T::real_split_applies(ny),
            "{shape:?}: ny={ny} must refuse the split to exercise the widened fallback"
        );
        let depth = ny / 2 + 1;
        let real = field::<T>(shape);
        let stored: Vec<f64> = real.iter().map(|&v| v.to_f64()).collect();
        let one = tolerance(nx * ny, l1(&stored), T::PLAN_UNIT);

        let mut half = Array2::from_elem([nx, depth], Complex::<T::PlanScalar>::default());
        let mut back = Array2::from_elem(shape, T::from_f64(0.0));

        // One warm pair: the plan, its tables, and every thread-local scratch
        // role a call touches are built on first use, not a per-call cost.
        apollo_fft::fft_2d_array_half_into(&real, &mut half);
        apollo_fft::ifft_2d_array_half_into(&mut half, &mut back);

        let ((), observed) = count_allocations(|| {
            apollo_fft::fft_2d_array_half_into(&real, &mut half);
            apollo_fft::ifft_2d_array_half_into(&mut half, &mut back);
        });
        assert_eq!(
            observed,
            0,
            "{} {shape:?}: the refused-length fallback allocated {observed} times on a warm plan",
            std::any::type_name::<T>()
        );

        // The round trip still reproduces the input within the derived bound:
        // the scratch reuse must not change what the fallback computes.
        for (index, (got, want)) in back.iter().zip(&stored).enumerate() {
            let got = got.to_f64();
            let bound = 2.0 * one + T::STORAGE_UNIT * (want.abs() + 2.0 * one);
            assert!(
                (got - want).abs() <= bound,
                "{} {shape:?} sample {index}: {got} against {want} (bound {bound:.2e})",
                std::any::type_name::<T>()
            );
        }
    }
}

#[test]
fn the_refused_length_fallback_allocates_nothing_once_warm() {
    the_fallback_allocates_nothing_once_warm::<f64>();
    the_fallback_allocates_nothing_once_warm::<f32>();
    the_fallback_allocates_nothing_once_warm::<F16>();
}

#[test]
#[should_panic(expected = "the half spectrum must be (nx, ny/2 + 1)")]
fn a_full_size_spectrum_is_rejected() {
    let real = field::<f64>([4, 8]);
    let mut out = Array2::from_elem([4, 8], Complex64::default());
    apollo_fft::fft_2d_array_half_into(&real, &mut out);
}
