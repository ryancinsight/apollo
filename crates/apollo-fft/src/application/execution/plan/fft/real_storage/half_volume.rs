//! The half-spectrum 3-D pair for real storage.
//!
//! A real field's 3-D spectrum satisfies `X[i, j, k] = conj(X[-i, -j, -k])`, so
//! the bins with `k > nz/2` repeat the others and the `(nx, ny, nz/2 + 1)` half
//! volume holds all of it. The forward runs the z lanes first, through the real
//! split, straight into that half volume, then x and y on it; the inverse runs
//! y and x first and the z lanes last. Where both of `nx` and `ny` exceed
//! one the z sweeps write and read the `(y, z, x)` order the x lanes need, so
//! each direction makes two full-volume moves rather than three. The x and y
//! passes move half the data the full complex transform moves.
//!
//! After the x and y inverses every z lane is the half spectrum of a real lane.
//! Writing `G` for the volume they leave, the inverse over x and y maps the
//! repeated bins `conj(X[-i, -j, nz - k])` to `conj(G[x, y, nz - k])`, so
//! `G[x, y, nz - k] = conj(G[x, y, k])` lane by lane and the per-lane real
//! inverse applies. It reads only the real parts of each lane's zero and
//! Nyquist bins — which is also what taking the real part of the full complex
//! inverse of the Hermitian completion does, so the two agree for any half
//! spectrum, not only for one a real field produced.
//!
//! The full-spectrum entries route through the pair where the split admits
//! `nz`, as the 2-D ones do ([`half_plane`](super::half_plane)): the forwards
//! compute the half volume in the 2-D staging role — free once the split
//! applies, since the x and y passes hold the two 3-D roles and only the
//! refused-length z lanes borrow the 2-D one — and write the `(nx, ny, nz)`
//! output once through [`expand`](super::expand); the inverses pack the
//! lower `nz/2 + 1` bins of each lane and read nothing else. The caller-owned
//! routes take the pair from [`routed::ROUTED_LANE_FLOOR`] lanes, the owned forward
//! from [`super::expand::OWNED_ROUTE_BYTES`] of output.

use super::{split, RealFftData};
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::{
    with_3d_x_scratch, with_view_staging, PlanScratch,
};
use crate::application::execution::plan::fft::dimension_3d::FftPlan3D;
use crate::application::execution::plan::fft::lanes;
use crate::application::execution::plan::fft::lanes::{Forward, Inverse};
use apollo_leto_interop::view_cow;
use eunomia::Complex;
use leto::Array3;

pub(crate) mod routed;

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
    assert_eq!(
        input.shape(),
        [nx, ny, nz],
        "forward_3d_half_into: the input must be the plan's shape"
    );
    assert_eq!(
        output.shape(),
        [nx, ny, plan.nz_c()],
        "forward_3d_half_into: the half spectrum must be (nx, ny, nz/2 + 1)"
    );
    let source = view_cow(&input.view());
    let spectrum = output
        .as_slice_mut()
        .expect("forward_3d_half_into: the half spectrum must be C-contiguous");
    forward_half(plan, &source, spectrum);
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
    assert_eq!(
        spectrum.shape(),
        [nx, ny, plan.nz_c()],
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
    inverse_half(plan, bins, values);
}

/// The forward of the `(nx, ny, nz)` real volume `source` into the C-order
/// `(nx, ny, nz/2 + 1)` half volume `spectrum`, both of the plan's shape.
fn forward_half<T>(
    plan: &FftPlan3D<T::PlanScalar>,
    source: &[T],
    spectrum: &mut [Complex<T::PlanScalar>],
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (nx, ny, nz) = plan.dimensions();
    let depth = plan.nz_c();
    if T::real_split_applies(nz) && nx > 1 && ny > 1 {
        forward_z_split_x_last(plan, source, spectrum);
        plan.xy_axes_from_x_last::<Forward>(spectrum, depth);
        return;
    }
    if T::real_split_applies(nz) {
        let half_lane = plan.half_z_lane::<Forward>();
        lanes::paired(spectrum, depth, source, nz, |bins_group, reals_group| {
            for (bins, reals) in bins_group
                .chunks_exact_mut(depth)
                .zip(reals_group.chunks_exact(nz))
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
        let z_lane = plan.z_lane::<Forward>();
        lanes::paired(spectrum, depth, source, nz, |bins_group, reals_group| {
            // Every task in this group is disjoint and each runs on its own
            // thread at most once at a time, so the thread-local widened-lane
            // role it borrows here never nests with the one the x/y passes
            // borrow afterward (`Self::xy_axes_inplace` below) or with a
            // sibling task's borrow on another thread.
            with_view_staging::<Complex<T::PlanScalar>, 3, _>(nz, |widened| {
                for (bins, reals) in bins_group
                    .chunks_exact_mut(depth)
                    .zip(reals_group.chunks_exact(nz))
                {
                    for (slot, &value) in widened.iter_mut().zip(reals.iter()) {
                        *slot = value.to_spectrum();
                    }
                    if nz > 1 {
                        z_lane(widened);
                    }
                    bins.copy_from_slice(&widened[..depth]);
                }
            });
        });
    }
    plan.xy_axes_inplace::<Forward>(spectrum, depth);
}

/// The z lanes of `source` through the real split into `spectrum` in
/// `(y, z, x)` order, the order the x lanes need, so the chain after it skips
/// the move that would make axis 0 contiguous.
///
/// A task takes whole `y` slabs, `depth * nx` bins each: every lane of a slab
/// is transformed in one cached lane buffer and scattered into the slab at
/// stride `nx`, so the writes stay inside the slab while the reads visit the
/// slab's `nx` source lanes, lane `x * ny + y` for each `x`. The arithmetic
/// per lane is the C-order sweep's, and the move it replaces only places
/// values, so the chain's result is the same bits.
fn forward_z_split_x_last<T>(
    plan: &FftPlan3D<T::PlanScalar>,
    source: &[T],
    spectrum: &mut [Complex<T::PlanScalar>],
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (nx, ny, nz) = plan.dimensions();
    let depth = plan.nz_c();
    let slab = depth * nx;
    let slab_bytes =
        slab * core::mem::size_of::<Complex<T::PlanScalar>>() + nx * nz * core::mem::size_of::<T>();
    let half_lane = plan.half_z_lane::<Forward>();
    lanes::units(spectrum, slab, slab_bytes, |first_y, slabs| {
        with_3d_x_scratch::<Complex<T::PlanScalar>, _>(depth, |lane| {
            for (y, bins) in (first_y..).zip(slabs.chunks_exact_mut(slab)) {
                for (x, reals) in source.chunks_exact(nz).skip(y).step_by(ny).enumerate() {
                    split::forward(
                        reals,
                        lane,
                        plan.split_twiddles().iter().copied(),
                        &half_lane,
                    );
                    for (row, &bin) in bins.chunks_exact_mut(nx).zip(lane.iter()) {
                        row[x] = bin;
                    }
                }
            }
        });
    });
}

/// The inverse of the C-order `(nx, ny, nz/2 + 1)` half volume `bins`,
/// consumed as scratch, into the `(nx, ny, nz)` real volume `values`.
fn inverse_half<T>(
    plan: &FftPlan3D<T::PlanScalar>,
    bins: &mut [Complex<T::PlanScalar>],
    values: &mut [T],
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (nx, ny, nz) = plan.dimensions();
    let depth = plan.nz_c();
    if T::real_split_applies(nz) && nx > 1 && ny > 1 {
        plan.xy_axes_leaving_x_last::<Inverse>(bins, depth);
        inverse_z_split_x_last(plan, bins, values);
        return;
    }
    plan.xy_axes_inplace::<Inverse>(bins, depth);
    if T::real_split_applies(nz) {
        inverse_z_split(plan, bins, values);
    } else {
        let z_lane = plan.z_lane::<Inverse>();
        let mirrored = nz - depth;
        lanes::paired(values, nz, &*bins, depth, |reals_group, halves_group| {
            // Runs after `Self::xy_axes_inplace` above has already returned,
            // so this thread-local widened-lane borrow never nests with the
            // x/y passes' transpose scratch; each task's own borrow never
            // nests with a sibling task's on another thread either.
            with_view_staging::<Complex<T::PlanScalar>, 3, _>(nz, |full| {
                for (reals, half) in reals_group
                    .chunks_exact_mut(nz)
                    .zip(halves_group.chunks_exact(depth))
                {
                    full[..depth].copy_from_slice(half);
                    // Bin `nz - k` of a real lane is the conjugate of bin `k`.
                    for (slot, source) in full[depth..]
                        .iter_mut()
                        .zip(half[1..=mirrored].iter().rev())
                    {
                        *slot = Complex::new(source.re, -source.im);
                    }
                    if nz > 1 {
                        z_lane(full);
                    }
                    for (value, &sample) in reals.iter_mut().zip(full.iter()) {
                        *value = T::from_spectrum(sample);
                    }
                }
            });
        });
    }
}

/// The z inverse of a half volume stored in `(y, z, x)` order, the order
/// [`FftPlan3D::xy_axes_leaving_x_last`] leaves, into the C-order `values`.
///
/// A task takes whole `x` slabs of the output, `ny * nz` reals each, and
/// gathers each of its lanes from the `y` slab that holds it: lane `(x, y)` is
/// every `nx`-th bin of slab `y` from offset `x`, a constant stride the
/// prefetcher follows. Each lane is then retangled, inverted at half length
/// and unpacked into the output as [`inverse_z_split`] does.
fn inverse_z_split_x_last<T>(
    plan: &FftPlan3D<T::PlanScalar>,
    bins: &[Complex<T::PlanScalar>],
    values: &mut [T],
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (nx, ny, nz) = plan.dimensions();
    let depth = plan.nz_c();
    let packed = nz / 2;
    let y_slab = depth * nx;
    let x_slab = ny * nz;
    let x_slab_bytes = x_slab * core::mem::size_of::<T>()
        + ny * depth * core::mem::size_of::<Complex<T::PlanScalar>>();
    let half_lane = plan.half_z_lane::<Inverse>();
    lanes::units(values, x_slab, x_slab_bytes, |first_x, slabs| {
        // The x and y passes have returned, so the 3-D X role is free on
        // every thread; the 2-D role may be an owned caller's half volume.
        with_3d_x_scratch::<Complex<T::PlanScalar>, _>(depth, |lane| {
            for (x, output) in (first_x..).zip(slabs.chunks_exact_mut(x_slab)) {
                for (reals, slab) in output.chunks_exact_mut(nz).zip(bins.chunks_exact(y_slab)) {
                    for (slot, row) in lane.iter_mut().zip(slab.chunks_exact(nx)) {
                        *slot = row[x];
                    }
                    split::inverse_packed::<T>(
                        lane,
                        nz,
                        plan.split_twiddles().iter().copied(),
                        &half_lane,
                    );
                    split::unpack(&lane[..packed], reals);
                }
            }
        });
    });
}

/// The z inverse of the half pair through the real split, in one sweep.
///
/// Each lane's `nz/2 + 1` bins are copied into an L1-resident buffer,
/// retangled, inverted at half length and unpacked straight into the output
/// lane. The lanes on the two sides differ in length, so no scheduler hands a
/// task both of them mutably; run as two sweeps instead — the split in place
/// on the spectrum, then the unpack from it — the second sweep re-reads the
/// whole spectrum after the first has left it, which at 64³ is 2 MiB read
/// cold to save a 528-byte copy per lane.
fn inverse_z_split<T>(
    plan: &FftPlan3D<T::PlanScalar>,
    bins: &[Complex<T::PlanScalar>],
    values: &mut [T],
) where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (_, _, nz) = plan.dimensions();
    let depth = plan.nz_c();
    let packed = nz / 2;
    let half_lane = plan.half_z_lane::<Inverse>();
    lanes::paired(values, nz, bins, depth, |reals_group, lanes_group| {
        // The x and y passes have returned, so the 3-D X role is free on every
        // thread here; the 2-D role is not, since an owned caller holds the
        // half volume in it.
        with_3d_x_scratch::<Complex<T::PlanScalar>, _>(depth, |lane| {
            for (reals, source) in reals_group
                .chunks_exact_mut(nz)
                .zip(lanes_group.chunks_exact(depth))
            {
                lane.copy_from_slice(source);
                split::inverse_packed::<T>(
                    lane,
                    nz,
                    plan.split_twiddles().iter().copied(),
                    &half_lane,
                );
                split::unpack(&lane[..packed], reals);
            }
        });
    });
}

#[cfg(test)]
mod phases;
#[cfg(test)]
mod tests;
