//! The row phase of the instance-major base kernel at four lanes: each row
//! pair runs from its loads to its staging stores in registers.
//!
//! A register holds sample `b` of two rows — `[x[ROWS b + 2p], x[ROWS b +
//! 2p + 1]]`, one contiguous source load — so every row twiddle is a
//! broadcast scalar and the trivial ones cost a rotation, a sign, or the
//! `sqrt(2)/2` scaling. The row is `ROW_LEN = (ROW_LEN / 4) x 4` with
//! `b = 4 b1 + b0` and output `k = (ROW_LEN / 4) q + m`: a radix-`ROW_LEN /
//! 4` over `b1` within each stride-4 group, the `W_ROW_LEN^{b0 m}` layer,
//! then radix-4 over `b0`. Natural order in and out, so no bit reversal.
//!
//! The two stages meet in a full transpose (every second-stage group takes
//! one value from each first-stage group), so the whole row pair is live at
//! the crossover — thirty-two registers for 32-sample rows. The allocator
//! spills what the file cannot hold; the earlier explicit spill plane
//! (`zbuf`, a store and a reload for every value) cost 73 stack moves a
//! row pair on the asm census against the 43 the reference kernel pays for
//! the same crossover (`output/apollo-base128/base256_2026-09-11.md`), and
//! its four-arm `match` over a runtime `b0` stayed a loop.
//!
//! Every helper here is a function, never a closure, and every group and
//! pair is its own monomorphization over a constant index: a closure or an
//! `array::from_fn` in this body compiles outside the dispatcher's
//! target-feature frame, so the intrinsics inside it become calls and the
//! closure itself stays out of line (45 calls a row pair on the first
//! draft's census). The row pair is straight-line code the census reads per
//! iteration.

use super::super::cmul::cmul_chunk;
use super::{radix4, radix8, root2_twiddle, rot90};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use hermes_simd::{
    Alignment, ComplexReg, ExecutionMode, LaneScalar, Simd, SimdArch, SimdKernel, SimdView, Vector,
};

/// Runs the row phase over `data` into `staging`: the `ROWS / 2` row pairs
/// of `ROW_LEN` samples, each pair's spectra landing in staging as chunk
/// `row * (ROW_LEN / 2) + g` holding samples `2g` and `2g + 1` of that row.
/// `layer` is the table chunk of the layer's first broadcast twiddle.
#[expect(
    clippy::inline_always,
    reason = "the pass must fold into the dispatcher's target-feature frame"
)]
#[inline(always)]
pub(super) fn row_pass<T, A, const INVERSE: bool, const ROWS: usize, const ROW_LEN: usize>(
    simd: Simd<T, A>,
    data: &[T],
    table: &[T],
    staging: &mut [T],
    layer: usize,
) where
    T: LaneScalar + MixedRadixScalar,
    A: SimdArch + SimdKernel<T>,
{
    let data = simd.view(data);
    let tab = simd.view(table);
    let mut stg = simd.view_mut(staging);
    let half_root2 =
        Vector::<T, A>::from_view_chunk(&tab, layer + super::layer_chunks(ROW_LEN) - 1);
    // Blend mask selecting the high complex of a register.
    let zero = T::from_precise(0.0);
    let neg = T::from_precise(-1.0);
    let mask = [zero, zero, neg, neg];
    let hi_mask = Vector::<T, A>::from_view_chunk(&simd.view(&mask), 0);
    for p in 0..ROWS / 2 {
        let z = [
            group::<T, A, INVERSE, ROWS, ROW_LEN, 0, _, _, _>(
                simd, &data, &tab, p, layer, half_root2,
            ),
            group::<T, A, INVERSE, ROWS, ROW_LEN, 1, _, _, _>(
                simd, &data, &tab, p, layer, half_root2,
            ),
            group::<T, A, INVERSE, ROWS, ROW_LEN, 2, _, _, _>(
                simd, &data, &tab, p, layer, half_root2,
            ),
            group::<T, A, INVERSE, ROWS, ROW_LEN, 3, _, _, _>(
                simd, &data, &tab, p, layer, half_root2,
            ),
        ];
        pair::<T, A, INVERSE, ROW_LEN, 0, _, _>(&mut stg, p, hi_mask, &z);
        pair::<T, A, INVERSE, ROW_LEN, 1, _, _>(&mut stg, p, hi_mask, &z);
        if ROW_LEN == 32 {
            pair::<T, A, INVERSE, ROW_LEN, 2, _, _>(&mut stg, p, hi_mask, &z);
            pair::<T, A, INVERSE, ROW_LEN, 3, _, _>(&mut stg, p, hi_mask, &z);
        }
    }
}

/// Sample `B0 + 4 b1` of row pair `p`: the two rows' samples are adjacent
/// in the source, `ROWS / 2` chunks apart per sample.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn sample<T, A, const ROWS: usize, const B0: usize, Align, Mode, Ref>(
    data: &SimdView<'_, T, A, Align, Mode, Ref>,
    p: usize,
    b1: usize,
) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Mode: ExecutionMode,
    Ref: core::ops::Deref<Target = [T]>,
{
    ComplexReg::from_interleaved(Vector::from_view_chunk(
        data,
        (ROWS / 2) * (B0 + 4 * b1) + p,
    ))
}

/// The register negated: the `W^{ROW_LEN / 2}` twiddle.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn negate<T, A>(v: ComplexReg<T, A>) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    ComplexReg::from_interleaved(-v.into_interleaved())
}

/// First stage of one row pair for group `B0`: the radix-`ROW_LEN / 4`
/// over `b1` on samples `B0 + 4 b1`, then the `W_ROW_LEN^{B0 m}` layer.
/// Sixteen-sample rows fill the first four slots and leave the rest zero,
/// which the constant-indexed second stage never reads.
///
/// The layer's general multiplies are the table's broadcasts; every other
/// twiddle is one of those under `W^{ROW_LEN / 4}` (a rotation) or
/// `W^{ROW_LEN / 2}` (a sign), or an eighth's `sqrt(2)/2` scaling.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn group<
    T,
    A,
    const INVERSE: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const B0: usize,
    Align,
    Mode,
    Ref,
>(
    simd: Simd<T, A>,
    data: &SimdView<'_, T, A, Align, Mode, Ref>,
    tab: &SimdView<'_, T, A, Align, Mode, Ref>,
    p: usize,
    layer: usize,
    half_root2: Vector<T, A>,
) -> [ComplexReg<T, A>; 8]
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Mode: ExecutionMode,
    Ref: core::ops::Deref<Target = [T]>,
{
    let rot = rot90::<T, A, INVERSE>;
    let eighth = root2_twiddle::<T, A, INVERSE, false>;
    let three_eighths = root2_twiddle::<T, A, INVERSE, true>;
    if ROW_LEN == 32 {
        // Broadcasts `W_32^{1,3,5,7}` then `W_16^{1,3}`.
        let (w1, w3, w5, w7, v1, v3) = (
            layer,
            layer + 2,
            layer + 4,
            layer + 6,
            layer + 8,
            layer + 10,
        );
        let y = radix8::<T, A, INVERSE>(
            [
                sample::<T, A, ROWS, B0, _, _, _>(data, p, 0),
                sample::<T, A, ROWS, B0, _, _, _>(data, p, 1),
                sample::<T, A, ROWS, B0, _, _, _>(data, p, 2),
                sample::<T, A, ROWS, B0, _, _, _>(data, p, 3),
                sample::<T, A, ROWS, B0, _, _, _>(data, p, 4),
                sample::<T, A, ROWS, B0, _, _, _>(data, p, 5),
                sample::<T, A, ROWS, B0, _, _, _>(data, p, 6),
                sample::<T, A, ROWS, B0, _, _, _>(data, p, 7),
            ],
            half_root2,
        );
        match B0 {
            0 => y,
            1 => [
                y[0],
                cmul_chunk(tab, y[1], w1),
                cmul_chunk(tab, y[2], v1),
                cmul_chunk(tab, y[3], w3),
                eighth(y[4], half_root2),
                cmul_chunk(tab, y[5], w5),
                cmul_chunk(tab, y[6], v3),
                cmul_chunk(tab, y[7], w7),
            ],
            2 => [
                y[0],
                cmul_chunk(tab, y[1], v1),
                eighth(y[2], half_root2),
                cmul_chunk(tab, y[3], v3),
                rot(y[4]),
                rot(cmul_chunk(tab, y[5], v1)),
                three_eighths(y[6], half_root2),
                rot(cmul_chunk(tab, y[7], v3)),
            ],
            _ => [
                y[0],
                cmul_chunk(tab, y[1], w3),
                cmul_chunk(tab, y[2], v3),
                rot(cmul_chunk(tab, y[3], w1)),
                three_eighths(y[4], half_root2),
                rot(cmul_chunk(tab, y[5], w7)),
                negate(cmul_chunk(tab, y[6], v1)),
                negate(cmul_chunk(tab, y[7], w5)),
            ],
        }
    } else {
        // Broadcasts `W_16^1`, `W_16^3`, `-W_16^1`.
        let (w1, w3, n1) = (layer, layer + 2, layer + 4);
        let y = radix4::<T, A, INVERSE>([
            sample::<T, A, ROWS, B0, _, _, _>(data, p, 0),
            sample::<T, A, ROWS, B0, _, _, _>(data, p, 1),
            sample::<T, A, ROWS, B0, _, _, _>(data, p, 2),
            sample::<T, A, ROWS, B0, _, _, _>(data, p, 3),
        ]);
        let z = match B0 {
            0 => y,
            1 => [
                y[0],
                cmul_chunk(tab, y[1], w1),
                eighth(y[2], half_root2),
                cmul_chunk(tab, y[3], w3),
            ],
            2 => [
                y[0],
                eighth(y[1], half_root2),
                rot(y[2]),
                three_eighths(y[3], half_root2),
            ],
            _ => [
                y[0],
                cmul_chunk(tab, y[1], w3),
                three_eighths(y[2], half_root2),
                cmul_chunk(tab, y[3], n1),
            ],
        };
        // Through the dispatch token: `Vector::zero()` re-probes the host.
        let zero = ComplexReg::from_interleaved(simd.zero());
        [z[0], z[1], z[2], z[3], zero, zero, zero, zero]
    }
}

/// Second stage for the output pair `MH`: radix-4 over `b0` at `m = 2 MH`
/// and `2 MH + 1`, then the pair transpose that hands the column pass its
/// sample-major registers. Output `(ROW_LEN / 4) q + m`; the two `m` of the
/// pair land as consecutive samples of chunk `(ROW_LEN / 8) q + MH` of each
/// row, row `r` at chunk `r * (ROW_LEN / 2)`.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn pair<'a, T, A, const INVERSE: bool, const ROW_LEN: usize, const MH: usize, Align, Mode>(
    stg: &mut SimdView<'a, T, A, Align, Mode, &'a mut [T]>,
    p: usize,
    hi_mask: Vector<T, A>,
    z: &[[ComplexReg<T, A>; 8]; 4],
) where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Mode: ExecutionMode,
{
    let o0 = radix4::<T, A, INVERSE>([z[0][2 * MH], z[1][2 * MH], z[2][2 * MH], z[3][2 * MH]]);
    let o1 = radix4::<T, A, INVERSE>([
        z[0][2 * MH + 1],
        z[1][2 * MH + 1],
        z[2][2 * MH + 1],
        z[3][2 * MH + 1],
    ]);
    let pairs = ROW_LEN / 8;
    let row0 = 2 * p * (ROW_LEN / 2);
    let row1 = row0 + ROW_LEN / 2;
    transpose_store(stg, hi_mask, o0[0], o1[0], row0 + MH, row1 + MH);
    transpose_store(
        stg,
        hi_mask,
        o0[1],
        o1[1],
        row0 + pairs + MH,
        row1 + pairs + MH,
    );
    transpose_store(
        stg,
        hi_mask,
        o0[2],
        o1[2],
        row0 + 2 * pairs + MH,
        row1 + 2 * pairs + MH,
    );
    transpose_store(
        stg,
        hi_mask,
        o0[3],
        o1[3],
        row0 + 3 * pairs + MH,
        row1 + 3 * pairs + MH,
    );
}

/// Stores `[a0, b0]` to chunk `chunk0` and `[a1, b1]` to chunk `chunk1`
/// from `a = [a0, a1]` and `b = [b0, b1]`: one `swap_pairs` and one blend
/// per output register.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn transpose_store<'a, T, A, Align, Mode>(
    stg: &mut SimdView<'a, T, A, Align, Mode, &'a mut [T]>,
    hi_mask: Vector<T, A>,
    a: ComplexReg<T, A>,
    b: ComplexReg<T, A>,
    chunk0: usize,
    chunk1: usize,
) where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Mode: ExecutionMode,
{
    let a = a.into_interleaved();
    let b = b.into_interleaved();
    hi_mask
        .blend(b.swap_pairs(), a)
        .store_to_view_chunk(stg, chunk0);
    hi_mask
        .blend(b, a.swap_pairs())
        .store_to_view_chunk(stg, chunk1);
}
