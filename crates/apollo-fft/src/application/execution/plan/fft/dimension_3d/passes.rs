//! The axis passes of a 3-D transform, and the three of them as one chain.
//!
//! A pass along a non-contiguous axis transposes the volume so that axis is
//! contiguous, runs its lanes there, and transposes back — two full-volume
//! moves, of which the second only restores the caller's layout. Run one at
//! a time, as the per-axis entry points do, that is what a pass must be. Run
//! all three, the moves chain instead: axis 2 in C order, one transpose so
//! axis 0 is contiguous, one more so axis 1 is, and one back to C order —
//! three single-matrix moves for the four the separate passes pay, the last
//! landing in the caller's storage because the chain crosses both 3-D scratch
//! roles.
//!
//! Every transform here is a lane pass over one contiguous axis, so the
//! arithmetic of a forward and an inverse is the per-axis kernel's whatever
//! the order the axes are visited in; the inverse visits them in the reverse
//! of the forward so each axis's pair composes as its own round trip.

use super::super::lanes;
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::{
    with_3d_x_scratch, with_3d_y_scratch, PlanScratch,
};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::plan::fft::layout::transpose_matrices;
use eunomia::Complex;

/// One direction's lane transform for each axis, over that axis's length.
pub(super) struct AxisLanes<X, Y, Z> {
    pub(super) x: X,
    pub(super) y: Y,
    pub(super) z: Z,
}

/// Transforms every axis-2 lane of a C-order `[_, _, nz]` volume in place.
pub(super) fn axis2<F, const FORWARD: bool>(
    data: &mut [F::Complex],
    nz: usize,
    lane: impl Fn(&mut [F::Complex]) + Send + Sync,
) where
    F: MixedRadixScalar<Complex = Complex<F>>,
    F::Complex: PlanScratch,
{
    if nz <= 1 {
        return;
    }
    lanes::contiguous::<F, FORWARD, 3>(data, nz, lane);
}

/// Transforms every axis-1 lane of a C-order `[nx, ny, nz]` volume in place:
/// `nx` matrices of `[ny, nz]` into the Y scratch, lanes there, and back.
pub(super) fn axis1<F, const FORWARD: bool>(
    data: &mut [F::Complex],
    [nx, ny, nz]: [usize; 3],
    lane: impl Fn(&mut [F::Complex]) + Send + Sync,
) where
    F: MixedRadixScalar<Complex = Complex<F>>,
    F::Complex: PlanScratch,
{
    if ny <= 1 {
        return;
    }
    with_3d_y_scratch::<F::Complex, _>(nx * ny * nz, |scratch| {
        transpose_matrices(data, scratch, nx, ny, nz);
        lanes::execute::<F, FORWARD>(scratch, data, ny, lane);
        transpose_matrices(scratch, data, nx, nz, ny);
    });
}

/// Transforms every axis-0 lane of a C-order `[nx, ny, nz]` volume in place:
/// one `[nx, ny * nz]` matrix into the X scratch, lanes there, and back.
pub(super) fn axis0<F, const FORWARD: bool>(
    data: &mut [F::Complex],
    [nx, ny, nz]: [usize; 3],
    lane: impl Fn(&mut [F::Complex]) + Send + Sync,
) where
    F: MixedRadixScalar<Complex = Complex<F>>,
    F::Complex: PlanScratch,
{
    if nx <= 1 {
        return;
    }
    with_3d_x_scratch::<F::Complex, _>(nx * ny * nz, |scratch| {
        transpose_matrices(data, scratch, 1, nx, ny * nz);
        lanes::execute::<F, FORWARD>(scratch, data, nx, lane);
        transpose_matrices(scratch, data, 1, ny * nz, nx);
    });
}

/// Transforms a C-order `[nx, ny, nz]` volume along all three axes in place.
///
/// With two non-contiguous axes of length above one the passes chain through
/// both scratch roles in three moves; with one, that axis takes its own pass;
/// axis 2 runs in place first for a forward and last for an inverse.
pub(super) fn all_axes<F, const FORWARD: bool, X, Y, Z>(
    data: &mut [F::Complex],
    shape @ [nx, ny, nz]: [usize; 3],
    lanes: AxisLanes<X, Y, Z>,
) where
    F: MixedRadixScalar<Complex = Complex<F>>,
    F::Complex: PlanScratch,
    X: Fn(&mut [F::Complex]) + Send + Sync,
    Y: Fn(&mut [F::Complex]) + Send + Sync,
    Z: Fn(&mut [F::Complex]) + Send + Sync,
{
    if FORWARD {
        axis2::<F, FORWARD>(data, nz, &lanes.z);
    }
    match (nx > 1, ny > 1) {
        (false, false) => {}
        (true, false) => axis0::<F, FORWARD>(data, shape, &lanes.x),
        (false, true) => axis1::<F, FORWARD>(data, shape, &lanes.y),
        (true, true) => chain::<F, FORWARD>(data, shape, &lanes.x, &lanes.y),
    }
    if !FORWARD {
        axis2::<F, FORWARD>(data, nz, &lanes.z);
    }
}

/// Axes 0 and 1 of a C-order `[nx, ny, nz]` volume in three moves.
///
/// `(x, y, z)` transposes as one `[nx, ny * nz]` matrix into the X scratch,
/// giving `(y, z, x)` with axis 0 contiguous; that transposes as one
/// `[ny, nz * nx]` matrix into the Y scratch, giving `(z, x, y)` with axis 1
/// contiguous; and that transposes as one `[nz, nx * ny]` matrix back into
/// `data`, restoring `(x, y, z)`. The caller's storage is dead between the
/// first and the last move, which is what lets it serve as the lane passes'
/// four-step companion.
fn chain<F, const FORWARD: bool>(
    data: &mut [F::Complex],
    [nx, ny, nz]: [usize; 3],
    lane_x: impl Fn(&mut [F::Complex]) + Send + Sync,
    lane_y: impl Fn(&mut [F::Complex]) + Send + Sync,
) where
    F: MixedRadixScalar<Complex = Complex<F>>,
    F::Complex: PlanScratch,
{
    let volume = nx * ny * nz;
    with_3d_x_scratch::<F::Complex, _>(volume, |staged_x| {
        with_3d_y_scratch::<F::Complex, _>(volume, |staged_y| {
            transpose_matrices(data, staged_x, 1, nx, ny * nz);
            lanes::execute::<F, FORWARD>(staged_x, data, nx, lane_x);
            transpose_matrices(staged_x, staged_y, 1, ny, nz * nx);
            lanes::execute::<F, FORWARD>(staged_y, data, ny, lane_y);
            transpose_matrices(staged_y, data, 1, nz, nx * ny);
        });
    });
}
