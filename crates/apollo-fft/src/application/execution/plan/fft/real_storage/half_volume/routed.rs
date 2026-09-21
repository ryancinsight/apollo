//! Full-volume routes built on the half-spectrum pair.

use super::super::{RealFftData, expand};
use super::{forward_half, inverse_half};
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::{
    PlanScratch, with_view_staging,
};
use crate::application::execution::plan::fft::dimension_3d::FftPlan3D;
use apollo_leto_interop::view_cow;
use eunomia::Complex;
use leto::Array3;

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
pub(super) const ROUTED_LANE_FLOOR: usize = 8;

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
