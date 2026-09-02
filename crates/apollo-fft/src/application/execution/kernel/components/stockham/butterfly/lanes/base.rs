//! The single Stockham stage as a lane kernel: `dst = a ± w·b` per group.

use core::ops::{Add, Mul, Sub};

use eunomia::Complex;
use hermes_simd::{ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, Vector};

/// One radix-2 Stockham stage over `src.len() / (2 * radix)` groups.
///
/// Each group `j` combines `a = src[2jg + k]` with `w_j · src[2jg + g + k]`
/// for `k < g` (where `g` is the group count), writing the sum to
/// `dst[jg + k]` and the difference to `dst[jg + n/2 + k]`.
struct BaseStage<'a, T> {
    src: &'a [Complex<T>],
    dst: &'a mut [Complex<T>],
    radix: usize,
    twiddles: &'a [Complex<T>],
}

impl<T> LaneKernel<T> for BaseStage<'_, T>
where
    T: LaneScalar + bytemuck::Pod,
    Complex<T>: bytemuck::Pod
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
        let n = src.len();
        let half_n = n >> 1;
        let groups = n / (radix << 1);
        let per_register = A::LANE_COUNT / 2;
        let vector_end = groups & !(per_register - 1);
        let src_view = simd.view(bytemuck::cast_slice::<Complex<T>, T>(src));

        for j in 0..radix {
            let w = twiddles[j];
            let src_base = j * groups * 2;
            let dst_base = j * groups;

            if vector_end > 0 {
                // The vector loop runs only when `groups >= per_register`; both
                // are powers of two, so `src_base`, `dst_base`, `groups`,
                // `half_n`, and each `k` are whole register indices.
                let wv = ComplexReg::<T, A>::splat(w);
                let load = |offset: usize| {
                    ComplexReg::from_interleaved(Vector::from_view_chunk(
                        &src_view,
                        offset / per_register,
                    ))
                };
                let mut dst_view = simd.view_mut(bytemuck::cast_slice_mut::<Complex<T>, T>(dst));
                for k in (0..vector_end).step_by(per_register) {
                    let a = load(src_base + k);
                    let product = load(src_base + groups + k) * wv;
                    (a + product)
                        .into_interleaved()
                        .store_to_view_chunk(&mut dst_view, (dst_base + k) / per_register);
                    (a - product)
                        .into_interleaved()
                        .store_to_view_chunk(&mut dst_view, (dst_base + half_n + k) / per_register);
                }
            }

            for k in vector_end..groups {
                let a = src[src_base + k];
                let product = src[src_base + groups + k] * w;
                dst[dst_base + k] = a + product;
                dst[dst_base + half_n + k] = a - product;
            }
        }
    }
}

/// Runs one Stockham stage at `LANES` scalar lanes per register on the
/// hardware backend of that width.
///
/// Returns `false`, having touched nothing, when this host has no hardware
/// backend at `LANES`; the caller then takes its scalar route. Any group
/// count is accepted — a count below the register width runs the scalar
/// recurrence inside the kernel — but callers route `groups == 1` to its
/// dedicated stage before reaching here.
#[inline]
pub(crate) fn stage_lanes<T, const LANES: usize>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    radix: usize,
    twiddles: &[Complex<T>],
) -> bool
where
    T: LaneScalar + bytemuck::Pod,
    Complex<T>: bytemuck::Pod
        + Add<Output = Complex<T>>
        + Sub<Output = Complex<T>>
        + Mul<Output = Complex<T>>,
{
    hermes_simd::vectorize_hardware_lanes::<LANES, T, _>(BaseStage {
        src,
        dst,
        radix,
        twiddles,
    })
    .is_some()
}
