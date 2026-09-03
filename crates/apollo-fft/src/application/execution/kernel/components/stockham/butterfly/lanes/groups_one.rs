//! The final Stockham stage (`groups == 1`) as a lane kernel.
//!
//! For `n = 2R` the last stage reads adjacent pairs and writes halves:
//! `dst[j] = src[2j] + W_n^j · src[2j+1]`, `dst[R + j] = src[2j] − W_n^j · src[2j+1]`.
//! A register of `L` consecutive `j` needs the even and odd samples of `2L`
//! consecutive inputs, which one pair-deinterleave of two loaded registers
//! yields; the twiddles for those `j` are contiguous. The ragged tail runs
//! the scalar recurrence.

use core::ops::{Add, Mul, Sub};

use eunomia::Complex;
use hermes_simd::{ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, Vector};

struct GroupsOneStage<'a, T> {
    src: &'a [Complex<T>],
    dst: &'a mut [Complex<T>],
    radix: usize,
    twiddles: &'a [Complex<T>],
}

impl<T> LaneKernel<T> for GroupsOneStage<'_, T>
where
    T: LaneScalar + eunomia::layout::Pod,
    Complex<T>: eunomia::layout::Pod
        + Add<Output = Complex<T>>
        + Sub<Output = Complex<T>>
        + Mul<Output = Complex<T>>,
{
    type Output = ();

    #[inline]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
        let Self {
            src,
            dst,
            radix,
            twiddles,
        } = self;
        let half_n = radix;
        let per_register = A::LANE_COUNT / 2;
        let vector_end = radix & !(per_register - 1);

        if vector_end > 0 {
            let src_view = simd.view(eunomia::layout::cast_slice::<Complex<T>, T>(src));
            let twiddle_view = simd.view(eunomia::layout::cast_slice::<Complex<T>, T>(twiddles));
            let mut dst_view = simd.view_mut(eunomia::layout::cast_slice_mut::<Complex<T>, T>(dst));
            for j in (0..vector_end).step_by(per_register) {
                // `2j` and `2j + per_register` are register indices because
                // `j` steps by `per_register`; `j` and `half_n + j` are too,
                // since `half_n = radix >= vector_end >= per_register` is a
                // power of two once the loop runs.
                let lower = Vector::from_view_chunk(&src_view, (2 * j) / per_register);
                let upper = Vector::from_view_chunk(&src_view, (2 * j) / per_register + 1);
                let (evens, odds) = lower.deinterleave_pairs(upper);
                let w = ComplexReg::from_interleaved(Vector::from_view_chunk(
                    &twiddle_view,
                    j / per_register,
                ));
                let a = ComplexReg::from_interleaved(evens);
                let product = ComplexReg::from_interleaved(odds) * w;
                (a + product)
                    .into_interleaved()
                    .store_to_view_chunk(&mut dst_view, j / per_register);
                (a - product)
                    .into_interleaved()
                    .store_to_view_chunk(&mut dst_view, (half_n + j) / per_register);
            }
        }

        for j in vector_end..radix {
            let a = src[2 * j];
            let product = src[2 * j + 1] * twiddles[j];
            dst[j] = a + product;
            dst[half_n + j] = a - product;
        }
    }
}

/// Runs the final Stockham stage (`groups == 1`, `src.len() == 2 * radix`)
/// at `LANES` scalar lanes per register on the hardware backend of that width.
///
/// Returns `false`, having touched nothing, when this host has no hardware
/// backend at `LANES`; the caller then takes its scalar route.
#[inline]
pub(crate) fn stage_groups_one_lanes<T, const LANES: usize>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    radix: usize,
    twiddles: &[Complex<T>],
) -> bool
where
    T: LaneScalar + eunomia::layout::Pod,
    Complex<T>: eunomia::layout::Pod
        + Add<Output = Complex<T>>
        + Sub<Output = Complex<T>>
        + Mul<Output = Complex<T>>,
{
    hermes_simd::vectorize_hardware_lanes::<LANES, T, _>(GroupsOneStage {
        src,
        dst,
        radix,
        twiddles,
    })
    .is_some()
}
