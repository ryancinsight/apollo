//! The half-spectrum 2-D pair for real storage.
//!
//! A real field's 2-D spectrum satisfies `X[i, j] = conj(X[-i, -j])`, so the
//! bins with `j > ny/2` repeat the others and the `(nx, ny/2 + 1)` half plane
//! holds all of it. The forward runs the rows first, through the real split,
//! straight into that half plane, then x on it; the inverse runs x first and
//! the rows last. The x pass moves half the data the full complex transform
//! moves.
//!
//! The 3-D pair's argument ([`half_volume`](super::half_volume)) carries over
//! with one axis fewer: after the x inverse every row is the half spectrum of
//! a real row, so the per-row real inverse applies, and it agrees with the
//! real part of the full complex inverse of the Hermitian completion for any
//! half spectrum, not only for one a real field produced.

use super::{split, RealFftData};
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::{
    with_view_staging, PlanScratch,
};
use crate::application::execution::plan::fft::dimension_2d::FftPlan2D;
use crate::application::execution::plan::fft::lanes;
use apollo_leto_interop::view_cow;
use eunomia::Complex;
use leto::Array2;

/// See [`RealFftData::forward_2d_half_into`].
pub(super) fn forward<T>(
    plan: &FftPlan2D<T::PlanScalar>,
    input: &Array2<T>,
    output: &mut Array2<Complex<T::PlanScalar>>,
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (nx, ny) = plan.dimensions();
    let depth = plan.ny_c();
    assert_eq!(
        input.shape(),
        [nx, ny],
        "forward_2d_half_into: the input must be the plan's shape"
    );
    assert_eq!(
        output.shape(),
        [nx, depth],
        "forward_2d_half_into: the half spectrum must be (nx, ny/2 + 1)"
    );
    let source = view_cow(&input.view());
    let spectrum = output
        .as_slice_mut()
        .expect("forward_2d_half_into: the half spectrum must be C-contiguous");

    if T::real_split_applies(ny) {
        let half_lane = plan.half_y_lane::<true>();
        lanes::paired(spectrum, depth, &source, ny, |bins_group, reals_group| {
            for (bins, reals) in bins_group
                .chunks_exact_mut(depth)
                .zip(reals_group.chunks_exact(ny))
            {
                split::forward(
                    reals,
                    bins,
                    plan.split_twiddles().iter().copied(),
                    &half_lane,
                );
            }
        });
    } else {
        let y_lane = plan.y_lane::<true>();
        lanes::paired(spectrum, depth, &source, ny, |bins_group, reals_group| {
            // The rank-two staging role is one no pass of this plan borrows
            // (the x pass below uses the 2-D role), and every task in this
            // group is disjoint and runs on its own thread at most once at a
            // time, so this borrow never nests with a sibling task's.
            with_view_staging::<Complex<T::PlanScalar>, 2, _>(ny, |widened| {
                for (bins, reals) in bins_group
                    .chunks_exact_mut(depth)
                    .zip(reals_group.chunks_exact(ny))
                {
                    for (slot, &value) in widened.iter_mut().zip(reals.iter()) {
                        *slot = value.to_spectrum();
                    }
                    if ny > 1 {
                        y_lane(widened);
                    }
                    bins.copy_from_slice(&widened[..depth]);
                }
            });
        });
    }
    plan.x_axis_inplace::<true>(spectrum, depth);
}

/// See [`RealFftData::inverse_2d_half_into`].
pub(super) fn inverse<T>(
    plan: &FftPlan2D<T::PlanScalar>,
    spectrum: &mut Array2<Complex<T::PlanScalar>>,
    output: &mut Array2<T>,
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (nx, ny) = plan.dimensions();
    let depth = plan.ny_c();
    assert_eq!(
        spectrum.shape(),
        [nx, depth],
        "inverse_2d_half_into: the half spectrum must be (nx, ny/2 + 1)"
    );
    assert_eq!(
        output.shape(),
        [nx, ny],
        "inverse_2d_half_into: the output must be the plan's shape"
    );
    let bins = spectrum
        .as_slice_mut()
        .expect("inverse_2d_half_into: the half spectrum must be C-contiguous");
    let values = output
        .as_slice_mut()
        .expect("inverse_2d_half_into: the output must be C-contiguous");

    plan.x_axis_inplace::<false>(bins, depth);
    if T::real_split_applies(ny) {
        // Two passes, because the rows on the two sides differ in length and
        // both are written: the retangle and half-length inverse in place on
        // the spectrum's rows, then the unpack into the output's.
        let half_lane = plan.half_y_lane::<false>();
        lanes::each(bins, depth, |lane| {
            split::inverse_packed::<T>(lane, ny, plan.split_twiddles().iter().copied(), &half_lane);
        });
        let packed = ny / 2;
        lanes::paired(values, ny, &*bins, depth, |reals_group, lanes_group| {
            for (reals, lane) in reals_group
                .chunks_exact_mut(ny)
                .zip(lanes_group.chunks_exact(depth))
            {
                split::unpack(&lane[..packed], reals);
            }
        });
    } else {
        let y_lane = plan.y_lane::<false>();
        let mirrored = ny - depth;
        lanes::paired(values, ny, &*bins, depth, |reals_group, halves_group| {
            // Runs after the x pass above has returned, and each task's own
            // borrow of the rank-two staging role never nests with a sibling
            // task's on another thread.
            with_view_staging::<Complex<T::PlanScalar>, 2, _>(ny, |full| {
                for (reals, half) in reals_group
                    .chunks_exact_mut(ny)
                    .zip(halves_group.chunks_exact(depth))
                {
                    full[..depth].copy_from_slice(half);
                    // Bin `ny - k` of a real row is the conjugate of bin `k`.
                    for (slot, source) in full[depth..]
                        .iter_mut()
                        .zip(half[1..=mirrored].iter().rev())
                    {
                        *slot = Complex::new(source.re, -source.im);
                    }
                    if ny > 1 {
                        y_lane(full);
                    }
                    for (value, &sample) in reals.iter_mut().zip(full.iter()) {
                        *value = T::from_spectrum(sample);
                    }
                }
            });
        });
    }
}
