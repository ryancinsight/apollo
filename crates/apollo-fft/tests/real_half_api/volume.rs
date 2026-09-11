//! The `(nx, ny, nz/2 + 1)` half-spectrum 3-D pair.

use crate::{count_allocations, l1, tolerance, Sample};
use apollo_fft::{PlanCacheProvider, PlanScratch, F16};
use eunomia::{Complex, Complex64};
use leto::Array3;

/// Shapes exercising the split (`nz` a multiple of four), each refusal (an odd
/// `nz`, an even one that is not a multiple of four, one below four), and an
/// extent of one on each axis in turn.
const SHAPES: [[usize; 3]; 9] = [
    [8, 6, 12],
    [16, 16, 16],
    [3, 5, 8],
    [7, 5, 9],
    [6, 4, 10],
    [4, 3, 2],
    [1, 4, 8],
    [4, 1, 8],
    [4, 6, 1],
];

fn field<T: Sample>([nx, ny, nz]: [usize; 3]) -> Array3<T> {
    Array3::from_shape_fn([nx, ny, nz], |[i, j, k]| {
        let x = ((i * ny + j) * nz + k) as f64;
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
    for shape @ [nx, ny, nz] in SHAPES {
        let depth = nz / 2 + 1;
        let real = field::<T>(shape);
        let stored: Vec<f64> = real.iter().map(|&v| v.to_f64()).collect();
        // One transform's per-bin bound over the whole volume: the separable
        // passes compose to log2(nx) + log2(ny) + log2(nz) stages.
        let one = tolerance(nx * ny * nz, l1(&stored), T::PLAN_UNIT);

        let full = apollo_fft::fft_3d_array::<T>(&real);
        let mut half = Array3::from_elem([nx, ny, depth], Complex::<T::PlanScalar>::default());
        apollo_fft::fft_3d_array_half_into(&real, &mut half);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..depth {
                    // Two routes to one bin, each within `one` of the exact value.
                    let error = distance(half[[i, j, k]], full[[i, j, k]]);
                    assert!(
                        error <= 2.0 * one,
                        "{} {shape:?} bin ({i}, {j}, {k}): {error:.3e} from the full spectrum (bound {:.3e})",
                        std::any::type_name::<T>(),
                        2.0 * one
                    );
                }
            }
        }

        let mut back = Array3::from_elem(shape, T::from_f64(0.0));
        apollo_fft::ifft_3d_array_half_into(&mut half, &mut back);
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

#[test]
fn the_half_inverse_is_the_real_part_of_the_hermitian_completion() {
    for shape @ [nx, ny, nz] in SHAPES {
        let depth = nz / 2 + 1;
        // An arbitrary half spectrum, not the forward of any real field: its
        // zero and Nyquist planes carry imaginary parts and lack the in-plane
        // symmetry a real field's spectrum has.
        let half = Array3::from_shape_fn([nx, ny, depth], |[i, j, k]| {
            let x = ((i * ny + j) * depth + k) as f64;
            Complex64::new((0.29 * x).sin(), (0.43 * x).cos() - 0.2)
        });
        let mut completed = Array3::from_shape_fn(shape, |[i, j, k]| {
            if k < depth {
                half[[i, j, k]]
            } else {
                let bin = half[[(nx - i) % nx, (ny - j) % ny, nz - k]];
                Complex64::new(bin.re, -bin.im)
            }
        });
        apollo_fft::ifft_3d_complex_inplace::<f64>(&mut completed);

        let mut scratch = half.clone();
        let mut ours = Array3::from_elem(shape, 0.0_f64);
        apollo_fft::ifft_3d_array_half_into(&mut scratch, &mut ours);

        // Two normalized inverses of spectra whose 1-norm is at most twice the
        // half's; each errs by at most the bound for that norm over the count.
        let total = nx * ny * nz;
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
    // Under moirai's parallel threshold, so every lane runs on this thread and
    // the thread-local count sees all of the work.
    let shape @ [nx, ny, nz] = [8, 6, 12];
    let real = field::<f64>(shape);
    let mut half = Array3::from_elem([nx, ny, nz / 2 + 1], Complex64::default());
    let mut back = Array3::from_elem(shape, 0.0_f64);

    // One warm pair: the plan, its tables, and the scratch the x and y passes
    // borrow are built on first use, and those are not per-call costs.
    apollo_fft::fft_3d_array_half_into(&real, &mut half);
    apollo_fft::ifft_3d_array_half_into(&mut half, &mut back);

    let ((), observed) = count_allocations(|| {
        apollo_fft::fft_3d_array_half_into(&real, &mut half);
        apollo_fft::ifft_3d_array_half_into(&mut half, &mut back);
    });
    assert_eq!(
        observed, 0,
        "the pair allocated {observed} times on a warm plan"
    );
}

#[test]
#[should_panic(expected = "the half spectrum must be (nx, ny, nz/2 + 1)")]
fn a_full_size_spectrum_is_rejected() {
    let real = field::<f64>([4, 4, 8]);
    let mut out = Array3::from_elem([4, 4, 8], Complex64::default());
    apollo_fft::fft_3d_array_half_into(&real, &mut out);
}
