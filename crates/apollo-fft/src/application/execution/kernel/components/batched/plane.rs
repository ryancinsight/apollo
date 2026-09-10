//! The padded planar layout: its geometry, the scratch it needs, the
//! route's domain, and the scalar transposes between the two stage sets'
//! planes.

use super::lane_order::LaneOrder;
use super::sweep;
use eunomia::Complex;

/// Extra elements appended to each plane row, so the row stride is `m + 8`
/// rather than `m`.
///
/// With an unpadded power-of-two stride, all of it aliases: every plane row
/// maps to the same L1 sets, `re` and `im` (a power-of-two apart) share sets,
/// and the three same-sized buffers a four-step holds live — caller data,
/// scratch planes, twiddle matrix — collide by allocation accident. Eight
/// f64 elements are one cache line, so consecutive rows shift by a full line
/// and the uniform aliasing is gone; for f32 the shift is half a line, which
/// still rotates the sets. The pad is a spacer, never computed on: every loop
/// bounds itself by the live column count and multiplies row indices by the
/// stride.
pub(crate) const ROW_PAD: usize = 8;

/// Scratch length, in complex elements, that [`super::driver::four_step_batched`] requires
/// for a transform of length `n`: the padded plane pair of the first stage
/// set, the second pair an odd power transposes into, and the seam staging
/// block.
///
/// The single definition of the padded-plane requirement, so callers and the
/// driver cannot disagree about it.
pub(crate) fn scratch_len(n: usize) -> usize {
    let (n1, n2) = plane_geometry(n);
    let second = if n1 == n2 { 0 } else { n2 * (n1 + ROW_PAD) };
    PLANE_ALIGN_SLACK + n1 * (n2 + ROW_PAD) + second + sweep::STAGING_LEN
}

/// Complexes of slack that let the planes start on a cache line whatever
/// the scratch's own alignment: one line of the narrowest complex.
///
/// The allocator aligns to sixteen bytes, so three allocations in four put
/// the planes part way into a line, where every register load and store
/// whose row offset lands in that part straddles two lines. Measured on
/// the pinned performance core with the scratch stepped through a line
/// (`output/apollo-planar-rectangular/gapsweep.txt`, `pagesweep.txt`), the
/// transform cost half again at 2048 to 8192 `f64` when it did (19.1k
/// against 12.4k cycles at 2048, 86k against 56k at 8192) and nothing
/// when the scratch sat on a line, whatever its page offset.
pub(crate) const PLANE_ALIGN_SLACK: usize = 64 / 8;

/// Largest even power the planar route serves, and half the largest odd
/// one; longer transforms fall to the generic four-step, whose rows thread
/// through Moirai.
///
/// The bound used to be the generic route's threading threshold (65536), on
/// the premise that threaded rows beat a sequential SIMD pass from there.
/// Measured on the pinned performance core against that premise (ADR 0053),
/// the generic route at 65536 cost 2.7 to 4.5 times RustFFT while this route
/// one length below sat at 1.25 times; at 65536 this route measured 208 to
/// 228 µs against the generic route's 466 to 767 across four runs, and at
/// 262144 it halved `f32` while leaving `f64` level. At 1048576, once the
/// seams were staged and the fold folded from a two-level table (ADR 0058,
/// ADR 0059), a quiet replicated census read this route at 5.6 to 6.6 ms
/// `f64` against the generic route's 9.9 to 10.2 and RustFFT's 6.4 to 7.1,
/// and 2.6 to 2.8 ms `f32` against 5.1 to 6.0 and PhastFT's 3.1 to 3.4; the
/// split at 2097152 halved likewise. The value binds to one host's cache
/// hierarchy and Moirai's dispatch cost; re-measure before moving it in
/// either direction.
pub(crate) const PLANAR_MAX_LEN: usize = 1 << 20;

/// Whether [`super::driver::four_step_batched`] covers a transform of length `n`.
///
/// The single definition of the planar route's domain: an even power of two
/// from 4 up to [`PLANAR_MAX_LEN`], run as a square, or an odd power of two
/// from 512 whose longer side is within it, run as the rectangle
/// `N1 × 2 N1` (ADR 0060). The odd lower bound is the one the decimated
/// route carried: below it the plan hands these lengths to the base kernels
/// and codelets, which never reach here.
pub(crate) fn planar_applies(n: usize) -> bool {
    if !n.is_power_of_two() {
        return false;
    }
    if n.trailing_zeros() % 2 == 0 {
        n >= 4 && n <= PLANAR_MAX_LEN
    } else {
        n >= 512 && n / 2 <= PLANAR_MAX_LEN
    }
}

/// Plane geometry for a length-`n` planar transform: the first stage set's
/// `n1` rows of `n2` columns, `n = n1 · n2` with `n2 = n1` for an even
/// power and `n2 = 2 n1` for an odd one. Input index `n2 · r + c` sits at
/// plane cell `(r, c)` of that set and output index `k1 + n1 · k2` at cell
/// `(k2, k1)` of the second set's `n2 × n1` planes; each pair pads its rows
/// by [`ROW_PAD`].
///
/// # Panics
///
/// Panics if `n` is not a power of two of at least four.
pub(super) fn plane_geometry(n: usize) -> (usize, usize) {
    let k = n.trailing_zeros();
    assert!(
        n.is_power_of_two() && n >= 4,
        "requires a power of two of at least 4"
    );
    (1usize << (k / 2), 1usize << (k - k / 2))
}

/// Splits a plane buffer into its real and imaginary halves.
///
/// One complex scratch element is two reals, so `m * stride` complexes hold
/// the two padded planes. The pad breaks the power-of-two row stride that
/// makes every plane row alias to one L1 set (see [`ROW_PAD`]).
///
/// # Panics
///
/// Panics if `scratch` is shorter than `plane`.
pub(super) fn split_plane<T>(scratch: &mut [Complex<T>], plane: usize) -> (&mut [T], &mut [T])
where
    T: eunomia::layout::Pod,
    Complex<T>: eunomia::layout::Pod,
{
    assert!(
        scratch.len() >= plane,
        "scratch must hold two padded planes"
    );
    let flat: &mut [T] = eunomia::layout::cast_slice_mut(&mut scratch[..plane]);
    flat.split_at_mut(plane)
}

/// Transposes both `m x m` planes in place: the scalar reference for the
/// vector tile transpose, and the route where no vector width applies.
///
/// Pure exchange: the four-step twiddle that used to ride this pass as a
/// scalar multiply — 26% of the driver at N = 256 — now rides stage-set-2's
/// first-stage vector loads instead, which the twiddle matrix's symmetry
/// (`W^(j*b)` equals its own transpose) makes exactly equivalent.
///
/// Plane cell `(r, c)` holds logical column `order(c)` of row `r`, so its
/// transpose partner is cell `(order(c), order(r))`; the map is an
/// involution and each pair swaps once. Rows leave in natural order and
/// columns in plane order, which is what both stage sets consume.
pub(super) fn transpose_planes<T: Copy>(
    re: &mut [T],
    im: &mut [T],
    m: usize,
    stride: usize,
    order: LaneOrder,
) {
    debug_assert!(stride >= m && re.len() >= m * stride);
    for r in 0..m {
        for c in 0..m {
            // Plane cell `(r, c)` holds memory `(r, order(c))`, which the
            // transpose puts at memory `(order(c), r)`: plane cell
            // `(order(c), plane(r))`. The map is an involution on cells,
            // so each pair swaps once.
            let partner = (order.column(c), order.plane(r));
            if partner > (r, c) {
                let (pr, pc) = partner;
                re.swap(r * stride + c, pr * stride + pc);
                im.swap(r * stride + c, pr * stride + pc);
            }
        }
    }
}

/// One padded plane pair with its shape, as a transpose source.
pub(super) struct PlaneView<'a, T> {
    pub(super) re: &'a [T],
    pub(super) im: &'a [T],
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) stride: usize,
}

/// Reference form of [`super::boundary::TransposePlanesInto`]: `rows × cols` planes
/// into `cols × rows` planes, plane cell `(r, c)` landing at `(order(c),
/// plane(r))` as the in-place transpose moves it ([`transpose_planes`]).
pub(super) fn transpose_planes_into<T: Copy>(
    src: PlaneView<'_, T>,
    dst_re: &mut [T],
    dst_im: &mut [T],
    dst_stride: usize,
    order: LaneOrder,
) {
    debug_assert!(src.stride >= src.cols && dst_stride >= src.rows);
    for r in 0..src.rows {
        for c in 0..src.cols {
            let from = r * src.stride + c;
            let to = order.column(c) * dst_stride + order.plane(r);
            dst_re[to] = src.re[from];
            dst_im[to] = src.im[from];
        }
    }
}
