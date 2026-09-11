//! The half-spectrum 3-D pair for real storage.
//!
//! A real field's 3-D spectrum satisfies `X[i, j, k] = conj(X[-i, -j, -k])`, so
//! the bins with `k > nz/2` repeat the others and the `(nx, ny, nz/2 + 1)` half
//! volume holds all of it. The forward runs the z lanes first, through the real
//! split, straight into that half volume, then x and y on it; the inverse runs
//! x and y first and the z lanes last. The x and y passes move half the data
//! the full complex transform moves.
//!
//! After the x and y inverses every z lane is the half spectrum of a real lane.
//! Writing `G` for the volume they leave, the inverse over x and y maps the
//! repeated bins `conj(X[-i, -j, nz - k])` to `conj(G[x, y, nz - k])`, so
//! `G[x, y, nz - k] = conj(G[x, y, k])` lane by lane and the per-lane real
//! inverse applies. It reads only the real parts of each lane's zero and
//! Nyquist bins — which is also what taking the real part of the full complex
//! inverse of the Hermitian completion does, so the two agree for any half
//! spectrum, not only for one a real field produced.

use super::{split, RealFftData};
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::plan::fft::dimension_3d::FftPlan3D;
use crate::application::execution::plan::fft::lanes;
use apollo_leto_interop::view_cow;
use eunomia::Complex;
use leto::Array3;

/// See [`RealFftData::forward_3d_half_into`].
pub(super) fn forward<T>(
    plan: &FftPlan3D<T::PlanScalar>,
    input: &Array3<T>,
    output: &mut Array3<Complex<T::PlanScalar>>,
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (nx, ny, nz) = plan.dimensions();
    let depth = plan.nz_c();
    assert_eq!(
        input.shape(),
        [nx, ny, nz],
        "forward_3d_half_into: the input must be the plan's shape"
    );
    assert_eq!(
        output.shape(),
        [nx, ny, depth],
        "forward_3d_half_into: the half spectrum must be (nx, ny, nz/2 + 1)"
    );
    let source = view_cow(&input.view());
    let spectrum = output
        .as_slice_mut()
        .expect("forward_3d_half_into: the half spectrum must be C-contiguous");

    if T::real_split_applies(nz) {
        let half_lane = plan.half_z_lane::<true>();
        lanes::paired(
            spectrum,
            depth,
            &source,
            nz,
            || (),
            |(), bins, reals| {
                split::forward(
                    reals,
                    bins,
                    plan.split_twiddles().iter().copied(),
                    &half_lane,
                );
            },
        );
    } else {
        let z_lane = plan.z_lane::<true>();
        lanes::paired(
            spectrum,
            depth,
            &source,
            nz,
            || vec![Complex::<T::PlanScalar>::default(); nz],
            |widened, bins, reals| {
                for (slot, &value) in widened.iter_mut().zip(reals) {
                    *slot = value.to_spectrum();
                }
                if nz > 1 {
                    z_lane(widened.as_mut_slice());
                }
                bins.copy_from_slice(&widened[..depth]);
            },
        );
    }
    plan.xy_axes_inplace::<true>(spectrum, depth);
}

/// See [`RealFftData::inverse_3d_half_into`].
pub(super) fn inverse<T>(
    plan: &FftPlan3D<T::PlanScalar>,
    spectrum: &mut Array3<Complex<T::PlanScalar>>,
    output: &mut Array3<T>,
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (nx, ny, nz) = plan.dimensions();
    let depth = plan.nz_c();
    assert_eq!(
        spectrum.shape(),
        [nx, ny, depth],
        "inverse_3d_half_into: the half spectrum must be (nx, ny, nz/2 + 1)"
    );
    assert_eq!(
        output.shape(),
        [nx, ny, nz],
        "inverse_3d_half_into: the output must be the plan's shape"
    );
    let bins = spectrum
        .as_slice_mut()
        .expect("inverse_3d_half_into: the half spectrum must be C-contiguous");
    let values = output
        .as_slice_mut()
        .expect("inverse_3d_half_into: the output must be C-contiguous");

    plan.xy_axes_inplace::<false>(bins, depth);
    if T::real_split_applies(nz) {
        // Two passes, because the lanes on the two sides differ in length and
        // both are written: the retangle and half-length inverse in place on
        // the spectrum's lanes, then the unpack into the output's.
        let half_lane = plan.half_z_lane::<false>();
        lanes::each(bins, depth, |lane| {
            split::inverse_packed::<T>(lane, nz, plan.split_twiddles().iter().copied(), &half_lane);
        });
        let packed = nz / 2;
        lanes::paired(
            values,
            nz,
            &*bins,
            depth,
            || (),
            |(), reals, lane| {
                split::unpack(&lane[..packed], reals);
            },
        );
    } else {
        let z_lane = plan.z_lane::<false>();
        let mirrored = nz - depth;
        lanes::paired(
            values,
            nz,
            &*bins,
            depth,
            || vec![Complex::<T::PlanScalar>::default(); nz],
            |full, reals, half| {
                full[..depth].copy_from_slice(half);
                // Bin `nz - k` of a real lane is the conjugate of bin `k`.
                for (slot, source) in full[depth..]
                    .iter_mut()
                    .zip(half[1..=mirrored].iter().rev())
                {
                    *slot = Complex::new(source.re, -source.im);
                }
                if nz > 1 {
                    z_lane(full.as_mut_slice());
                }
                for (value, &sample) in reals.iter_mut().zip(full.iter()) {
                    *value = T::from_spectrum(sample);
                }
            },
        );
    }
}
