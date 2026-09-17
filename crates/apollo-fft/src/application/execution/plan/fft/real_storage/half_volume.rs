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
//!
//! The full-spectrum entries route through the pair where the split admits
//! `nz`, as the 2-D ones do ([`half_plane`](super::half_plane)): the forwards
//! compute the half volume in the 2-D staging role — free once the split
//! applies, since the x and y passes hold the two 3-D roles and only the
//! refused-length z lanes borrow the 2-D one — and write the `(nx, ny, nz)`
//! output once through [`expand`](super::expand); the inverses pack the
//! lower `nz/2 + 1` bins of each lane and read nothing else. The caller-owned
//! routes take the pair from [`ROUTED_LANE_FLOOR`] lanes, the owned forward
//! from [`expand::OWNED_ROUTE_BYTES`] of output.

use super::{expand, split, RealFftData};
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::{
    with_3d_x_scratch, with_view_staging, PlanScratch,
};
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
    let (_, _, nz) = plan.dimensions();
    let depth = plan.nz_c();
    if T::real_split_applies(nz) {
        let half_lane = plan.half_z_lane::<true>();
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
        let z_lane = plan.z_lane::<true>();
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
    plan.xy_axes_inplace::<true>(spectrum, depth);
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
    let (_, _, nz) = plan.dimensions();
    let depth = plan.nz_c();
    plan.xy_axes_inplace::<false>(bins, depth);
    if T::real_split_applies(nz) {
        inverse_z_split(plan, bins, values);
    } else {
        let z_lane = plan.z_lane::<false>();
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
    let half_lane = plan.half_z_lane::<false>();
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

/// The z-lane length from which the caller-owned routes take the pair.
///
/// A caller-owned route pays the expansion or the pack on top of the pair,
/// and the pair saves little on the shortest lanes the split admits: measured
/// on the census host (`output/probe3d`, 2026-09-15) at `nz = 4` the routed
/// forward ran 0.83x and the routed inverse 0.91-0.96x of the widened route,
/// at `nz = 8` 1.30-1.41x and 2.0-2.3x across `nx * ny`. The owned inverse,
/// which also drops the widened form's full-volume copy, wins at every lane
/// length (2.4x at `nz = 4`) and takes no floor; the owned forward takes the
/// byte floor [`expand::OWNED_ROUTE_BYTES`].
const ROUTED_LANE_FLOOR: usize = 8;

/// The lane of the half volume whose conjugate mirror completes lane `l`:
/// the lane at the negated x and y indices.
fn partner(nx: usize, ny: usize) -> impl Fn(usize) -> usize + Send + Sync {
    move |l| ((nx - l / ny) % nx) * ny + (ny - l % ny) % ny
}

/// The full `(nx, ny, nz)` forward through the pair into caller storage: the
/// half volume is computed in the 2-D staging role and `output` written once
/// from it, so every lane's repeated half costs a copy rather than a
/// transform.
///
/// Returns whether it ran, which it does when the split admits `nz`, the
/// lanes reach [`ROUTED_LANE_FLOOR`] and `output` is a C-contiguous array of
/// the plan's shape; a caller takes the widening path otherwise without
/// probing admissibility itself.
pub(crate) fn forward_full_via_split<T>(
    plan: &FftPlan3D<T::PlanScalar>,
    input: &Array3<T>,
    output: &mut Array3<Complex<T::PlanScalar>>,
) -> bool
where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (nx, ny, nz) = plan.dimensions();
    let depth = plan.nz_c();
    if !T::real_split_applies(nz)
        || nz < ROUTED_LANE_FLOOR
        || input.shape() != [nx, ny, nz]
        || output.shape() != [nx, ny, nz]
    {
        return false;
    }
    let Some(full) = output.as_slice_mut() else {
        return false;
    };
    let source = view_cow(&input.view());
    with_staged_half(plan, &source, nx * ny * depth, |half| {
        expand::write_expanded(full, half, nz, depth, partner(nx, ny));
    });
    true
}

/// The full `(nx, ny, nz)` forward through the pair into an owned volume,
/// written exactly once into fresh capacity: the call allocates only its
/// returned volume and never fills it twice.
///
/// Returns `None` where the split does not admit `nz`, `input` is not the
/// plan's shape, or the volume is under [`expand::OWNED_ROUTE_BYTES`], the
/// floor below which the staging round trip an owned form pays outweighs the
/// fault spread that repays it.
pub(crate) fn forward_owned_via_split<T>(
    plan: &FftPlan3D<T::PlanScalar>,
    input: &Array3<T>,
) -> Option<Array3<Complex<T::PlanScalar>>>
where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (nx, ny, nz) = plan.dimensions();
    let depth = plan.nz_c();
    if !T::real_split_applies(nz)
        || input.shape() != [nx, ny, nz]
        || nx * ny * nz * core::mem::size_of::<Complex<T::PlanScalar>>() < expand::OWNED_ROUTE_BYTES
    {
        return None;
    }
    let source = view_cow(&input.view());
    let full = with_staged_half(plan, &source, nx * ny * depth, |half| {
        expand::fresh_expanded(half, nx * ny, nz, depth, partner(nx, ny))
    });
    Some(
        Array3::from_shape_vec([nx, ny, nz], full)
            .expect("invariant: nx * ny * nz elements were written"),
    )
}

/// Runs `consumer` over the half volume of `source`, held in the 2-D staging
/// role.
///
/// Once the split applies no pass of the pair borrows that role: the x and y
/// passes take the two 3-D roles and only refused-length z lanes stage
/// through the 2-D one, so the half stays live while the output is written
/// from it, without a fourth full-volume scratch.
fn with_staged_half<T, R>(
    plan: &FftPlan3D<T::PlanScalar>,
    source: &[T],
    len: usize,
    consumer: impl FnOnce(&[Complex<T::PlanScalar>]) -> R,
) -> R
where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    with_view_staging::<Complex<T::PlanScalar>, 3, _>(len, |half| {
        forward_half(plan, source, half);
        consumer(half)
    })
}

/// The full inverse reading only the lower `nz/2 + 1` bins of each lane of
/// `spectrum`, which it consumes as scratch: the lanes are packed in place
/// into the half volume at the front and the pair's inverse runs on it.
///
/// Returns whether it ran (the split admits `nz`, the lanes reach
/// [`ROUTED_LANE_FLOOR`], both arrays C-contiguous and of the plan's shape).
pub(crate) fn inverse_spectrum_via_split<T>(
    plan: &FftPlan3D<T::PlanScalar>,
    spectrum: &mut Array3<Complex<T::PlanScalar>>,
    output: &mut Array3<T>,
) -> bool
where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (nx, ny, nz) = plan.dimensions();
    let depth = plan.nz_c();
    if !T::real_split_applies(nz)
        || nz < ROUTED_LANE_FLOOR
        || spectrum.shape() != [nx, ny, nz]
        || output.shape() != [nx, ny, nz]
    {
        return false;
    }
    let (Some(full), Some(values)) = (spectrum.as_slice_mut(), output.as_slice_mut()) else {
        return false;
    };
    expand::pack_lanes_in_place(full, nz, depth);
    inverse_half(plan, &mut full[..nx * ny * depth], values);
    true
}

/// [`inverse_spectrum_via_split`] over a borrowed spectrum, packing the lower
/// bins into the front of the caller's `scratch`.
pub(crate) fn inverse_into_via_split<T>(
    plan: &FftPlan3D<T::PlanScalar>,
    input: &Array3<Complex<T::PlanScalar>>,
    output: &mut Array3<T>,
    scratch: &mut Array3<Complex<T::PlanScalar>>,
) -> bool
where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (nx, ny, nz) = plan.dimensions();
    let depth = plan.nz_c();
    if !T::real_split_applies(nz)
        || nz < ROUTED_LANE_FLOOR
        || input.shape() != [nx, ny, nz]
        || output.shape() != [nx, ny, nz]
        || scratch.shape() != [nx, ny, nz]
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
    let half = &mut work[..nx * ny * depth];
    expand::pack_lanes(full, half, nz, depth);
    inverse_half(plan, half, values);
    true
}

/// [`inverse_spectrum_via_split`] over a borrowed spectrum into an owned real
/// volume: the lower bins are packed into the 2-D staging role, so the call
/// allocates exactly its returned volume.
///
/// Returns `None` where the split does not admit `nz` or `input` is not a
/// C-contiguous array of the plan's shape.
pub(crate) fn inverse_owned_via_split<T>(
    plan: &FftPlan3D<T::PlanScalar>,
    input: &Array3<Complex<T::PlanScalar>>,
) -> Option<Array3<T>>
where
    T: RealFftData,
    Complex<T::PlanScalar>: PlanScratch,
{
    let (nx, ny, nz) = plan.dimensions();
    let depth = plan.nz_c();
    if !T::real_split_applies(nz) || input.shape() != [nx, ny, nz] {
        return None;
    }
    let full = input.as_slice()?;
    let mut output = Array3::from_elem([nx, ny, nz], T::from_spectrum(Complex::default()));
    let values = output
        .as_slice_mut()
        .expect("invariant: a freshly built volume is C-contiguous");
    // The 2-D staging role is free here for the reason `with_staged_half`
    // gives, and the inverse borrows nothing of it beyond the pack.
    with_view_staging::<Complex<T::PlanScalar>, 3, _>(nx * ny * depth, |half| {
        expand::pack_lanes(full, half, nz, depth);
        inverse_half(plan, half, values);
    });
    Some(output)
}

#[cfg(test)]
mod phases;
