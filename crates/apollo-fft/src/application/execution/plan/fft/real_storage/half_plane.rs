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
//!
//! The full-spectrum entries route through the pair where the split admits
//! `ny`. The forwards compute the half plane in the rank-one staging role and
//! write the `(nx, ny)` output once through [`expand`](super::expand) — the
//! same pass whether the output is caller storage or fresh capacity, so the
//! owned form never fills its plane twice (at short rows a zero-fill of a
//! fresh plane costs as much as the transform, and the page faults a fresh
//! plane pays are what the parallel pass spreads). The inverses pack the
//! lower `ny/2 + 1` bins of each row and read nothing else.

use super::{expand, split, RealFftData};
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::{
    with_view_staging, PlanScratch,
};
use crate::application::execution::plan::fft::dimension_2d::FftPlan2D;
use crate::application::execution::plan::fft::lanes;
use crate::application::execution::plan::fft::lanes::{Forward, Inverse};
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
    assert_eq!(
        input.shape(),
        [nx, ny],
        "forward_2d_half_into: the input must be the plan's shape"
    );
    assert_eq!(
        output.shape(),
        [nx, plan.ny_c()],
        "forward_2d_half_into: the half spectrum must be (nx, ny/2 + 1)"
    );
    let source = view_cow(&input.view());
    let spectrum = output
        .as_slice_mut()
        .expect("forward_2d_half_into: the half spectrum must be C-contiguous");
    forward_half(plan, &source, spectrum);
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
    assert_eq!(
        spectrum.shape(),
        [nx, plan.ny_c()],
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
    inverse_half(plan, bins, values);
}

/// The forward of the `(nx, ny)` real plane `source` into the C-order
/// `(nx, ny/2 + 1)` half plane `spectrum`, both of the plan's shape.
fn forward_half<T>(
    plan: &FftPlan2D<T::PlanScalar>,
    source: &[T],
    spectrum: &mut [Complex<T::PlanScalar>],
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (_, ny) = plan.dimensions();
    let depth = plan.ny_c();
    if T::real_split_applies(ny) {
        let half_lane = plan.half_y_lane::<Forward>();
        lanes::paired(spectrum, depth, source, ny, |bins_group, reals_group| {
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
        let y_lane = plan.y_lane::<Forward>();
        lanes::paired(spectrum, depth, source, ny, |bins_group, reals_group| {
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
    plan.x_axis_inplace::<Forward>(spectrum, depth);
}

/// The inverse of the C-order `(nx, ny/2 + 1)` half plane `bins`, consumed as
/// scratch, into the `(nx, ny)` real plane `values`.
fn inverse_half<T>(
    plan: &FftPlan2D<T::PlanScalar>,
    bins: &mut [Complex<T::PlanScalar>],
    values: &mut [T],
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (_, ny) = plan.dimensions();
    let depth = plan.ny_c();
    plan.x_axis_inplace::<Inverse>(bins, depth);
    if T::real_split_applies(ny) {
        // Two passes, because the rows on the two sides differ in length and
        // both are written: the retangle and half-length inverse in place on
        // the spectrum's rows, then the unpack into the output's.
        let half_lane = plan.half_y_lane::<Inverse>();
        lanes::each(bins, depth, |_, lane| {
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
        let y_lane = plan.y_lane::<Inverse>();
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

/// The full `(nx, ny)` forward through the pair into caller storage: the half
/// plane is computed in the rank-one staging role and `output` written once
/// from it, so every row's repeated half costs a copy rather than a transform.
///
/// Returns whether it ran, which it does when the split admits `ny` and
/// `output` is a C-contiguous array of the plan's shape; a caller takes the
/// widening path otherwise without probing admissibility itself.
pub(crate) fn forward_full_via_split<T>(
    plan: &FftPlan2D<T::PlanScalar>,
    input: &Array2<T>,
    output: &mut Array2<Complex<T::PlanScalar>>,
) -> bool
where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (nx, ny) = plan.dimensions();
    let depth = plan.ny_c();
    if !T::real_split_applies(ny) || input.shape() != [nx, ny] || output.shape() != [nx, ny] {
        return false;
    }
    let Some(full) = output.as_slice_mut() else {
        return false;
    };
    let source = view_cow(&input.view());
    with_staged_half(plan, &source, nx * depth, |half| {
        expand::write_expanded(full, half, ny, depth, |i| (nx - i) % nx);
    });
    true
}

/// The full `(nx, ny)` forward through the pair into an owned plane, written
/// exactly once into fresh capacity: the call allocates only its returned
/// plane and never fills it twice.
///
/// Returns `None` where the split does not admit `ny`, `input` is not the
/// plan's shape, or the plane is under [`expand::OWNED_ROUTE_BYTES`]: unlike the
/// caller-owned form this one pays the half plane's staging round trip on
/// top of the write, and a fresh plane's page faults, which the parallel
/// pass spreads over the workers, are what buys that back — so below the
/// floor the widened route's single fused write stays ahead.
pub(crate) fn forward_owned_via_split<T>(
    plan: &FftPlan2D<T::PlanScalar>,
    input: &Array2<T>,
) -> Option<Array2<Complex<T::PlanScalar>>>
where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (nx, ny) = plan.dimensions();
    let depth = plan.ny_c();
    if !T::real_split_applies(ny)
        || input.shape() != [nx, ny]
        || nx * ny * core::mem::size_of::<Complex<T::PlanScalar>>() < expand::OWNED_ROUTE_BYTES
    {
        return None;
    }
    let source = view_cow(&input.view());
    let full = with_staged_half(plan, &source, nx * depth, |half| {
        expand::fresh_expanded(half, nx, ny, depth, |i| (nx - i) % nx)
    });
    Some(Array2::from_shape_vec([nx, ny], full).expect("invariant: nx * ny elements were written"))
}

/// Runs `consumer` over the half plane of `source`, held in the rank-one
/// staging role.
///
/// No pass of the 2-D plan borrows that role (the x pass takes the 2-D role,
/// the split rows none, the refused-length rows the rank-two role), so the
/// half stays live while the output is written from it, without a fourth
/// full-plane scratch.
fn with_staged_half<T, R>(
    plan: &FftPlan2D<T::PlanScalar>,
    source: &[T],
    len: usize,
    consumer: impl FnOnce(&[Complex<T::PlanScalar>]) -> R,
) -> R
where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    with_view_staging::<Complex<T::PlanScalar>, 1, _>(len, |half| {
        forward_half(plan, source, half);
        consumer(half)
    })
}

/// The full inverse reading only the lower `ny/2 + 1` bins of each row of
/// `spectrum`, which it consumes as scratch: the rows are packed in place into
/// the half plane at the front and the pair's inverse runs on it.
///
/// Returns whether it ran (the split admits `ny`, both arrays C-contiguous and
/// of the plan's shape).
pub(crate) fn inverse_spectrum_via_split<T>(
    plan: &FftPlan2D<T::PlanScalar>,
    spectrum: &mut Array2<Complex<T::PlanScalar>>,
    output: &mut Array2<T>,
) -> bool
where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (nx, ny) = plan.dimensions();
    let depth = plan.ny_c();
    if !T::real_split_applies(ny) || spectrum.shape() != [nx, ny] || output.shape() != [nx, ny] {
        return false;
    }
    let (Some(full), Some(values)) = (spectrum.as_slice_mut(), output.as_slice_mut()) else {
        return false;
    };
    expand::pack_lanes_in_place(full, ny, depth);
    inverse_half(plan, &mut full[..nx * depth], values);
    true
}

/// [`inverse_spectrum_via_split`] over a borrowed spectrum, packing the lower
/// bins into the front of the caller's `scratch`.
pub(crate) fn inverse_into_via_split<T>(
    plan: &FftPlan2D<T::PlanScalar>,
    input: &Array2<Complex<T::PlanScalar>>,
    output: &mut Array2<T>,
    scratch: &mut Array2<Complex<T::PlanScalar>>,
) -> bool
where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (nx, ny) = plan.dimensions();
    let depth = plan.ny_c();
    if !T::real_split_applies(ny)
        || input.shape() != [nx, ny]
        || output.shape() != [nx, ny]
        || scratch.shape() != [nx, ny]
    {
        return false;
    }
    let (Some(full), Some(values), Some(work)) = (
        input.as_slice(),
        output.as_slice_mut(),
        scratch.as_slice_mut(),
    ) else {
        return false;
    };
    let half = &mut work[..nx * depth];
    expand::pack_lanes(full, half, ny, depth);
    inverse_half(plan, half, values);
    true
}

/// [`inverse_spectrum_via_split`] over a borrowed spectrum into an owned real
/// plane: the lower bins are packed into the rank-one staging role, so the
/// call allocates exactly its returned plane.
///
/// Returns `None` where the split does not admit `ny` or `input` is not a
/// C-contiguous array of the plan's shape.
pub(crate) fn inverse_owned_via_split<T>(
    plan: &FftPlan2D<T::PlanScalar>,
    input: &Array2<Complex<T::PlanScalar>>,
) -> Option<Array2<T>>
where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (nx, ny) = plan.dimensions();
    let depth = plan.ny_c();
    if !T::real_split_applies(ny) || input.shape() != [nx, ny] {
        return None;
    }
    let full = input.as_slice()?;
    let mut output = Array2::from_elem([nx, ny], T::from_spectrum(Complex::default()));
    let values = output
        .as_slice_mut()
        .expect("invariant: a freshly built plane is C-contiguous");
    // The rank-one staging role is free here for the reason `with_staged_half`
    // gives, and the inverse borrows nothing of it beyond the pack.
    with_view_staging::<Complex<T::PlanScalar>, 1, _>(nx * depth, |half| {
        expand::pack_lanes(full, half, ny, depth);
        inverse_half(plan, half, values);
    });
    Some(output)
}
