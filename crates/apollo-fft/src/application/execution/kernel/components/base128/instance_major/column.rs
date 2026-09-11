//! The lane-wise `ROWS`-point DIF column pass, shared by both register
//! widths of the instance-major base kernel.
//!
//! Both widths stage the row spectra as `ROWS x groups` chunks — the
//! four-lane layout holds two interleaved complex samples per chunk over
//! eight groups, the eight-lane layout four samples over four groups — and
//! the pass below is identical modulo that `groups` factor: load column
//! chunk `a * groups + g`, apply the mixed-radix twiddle from the
//! dup-split table, run the `ROWS`-point DIF across the row index, and
//! store register `q` to output row `rev(q)` through the selected
//! [`StoreSink`]. Keeping it in one place keeps the sink family available
//! to every width for the same reason it keeps the arithmetic identical.
//!
//! Every column register is a named binding and every index a constant,
//! and the in-network twiddles (`W_8^{1,3}`, `W_16^{1,3,5,7}`) are
//! dup-split register pairs, one shuffle a multiply where the general form
//! paid three. The census reads the eight-row group at 84 instructions and
//! 15 shuffles from 83 and 17, the eight stack moves a group unchanged: the
//! compiler keeps the twiddled columns on the stack across the distance-4
//! stage whatever the source form, and the group's critical path (about 67
//! cycles for 20 cycles of issue) is what the pass costs — the
//! out-of-order window overlaps groups, and the meter reads about 37
//! cycles a group.
//!
//! The sixteen-row form (the 512-point base at eight lanes) runs one
//! distance-8 stage under `W_16^a` and then the eight-point network on each
//! half: the sums hold the even spectral rows, the twiddled differences the
//! odd ones, so the low half's register `q` lands on row `2 rev3(q)` and
//! the high half's on `2 rev3(q) + 1`. Sixteen columns and their
//! broadcasts exceed the AVX2 file, so the group spills — 223
//! instructions, 39 shuffles and 35 stack moves, a chain of about 104
//! cycles for 54 of issue — and still runs 512 as one block in 12% less
//! wall clock than two gathered 256-blocks under the combining sink (ADR
//! 0061, the revision of 2026-09-11).

use super::store::StoreSink;
use super::{MIX_CH, REV2, REV3};
use hermes_simd::{
    Alignment, ComplexReg, ExecutionMode, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage,
    SimdView, Vector,
};

/// A broadcast twiddle as a dup-split register pair `([re; n], [im; n])`.
type Broadcast<T, A> = (Vector<T, A>, Vector<T, A>);

/// The column network's twiddles, splatted once per transform: `W_8^{1,3}`
/// for the distance-4 stage and `W_16^{1,3,5,7}` for the sixteen-row
/// form's distance-8 stage (its `W_16^{2,6}` are the eighths, its
/// `W_16^4` the quarter turn).
struct ColumnTwiddles<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    w8_1: Broadcast<T, A>,
    w8_3: Broadcast<T, A>,
    w16: [Broadcast<T, A>; 4],
}

/// Runs the column pass over `staging`, storing every register through
/// `sink` into `out`, the kernel's output surface. `col` carries `W_8^1`,
/// `W_8^3`, then `W_16^{1,3,5,7}` as complex values: the four-row form
/// reads none of them, the eight-row form the first two, the sixteen-row
/// form all six.
///
/// Inlined into the dispatcher's target-feature frame; the caller's
/// fixed-size arrays keep the view bounds foldable through that inlining.
#[expect(
    clippy::inline_always,
    reason = "the pass must fold into the dispatcher's target-feature frame"
)]
#[inline(always)]
pub(super) fn column_pass<T, A, S, const INVERSE: bool, const ROWS: usize, const ROW_LEN: usize>(
    simd: Simd<T, A>,
    staging: &[T],
    table: &[T],
    col: &[[T; 2]; 6],
    out: &mut [T],
    mut sink: S,
) where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    S: StoreSink<T>,
{
    // Chunks per row at this width: the row's samples over the register's.
    let groups = 2 * ROW_LEN / <A as SimdStorage<T>>::LANE_COUNT;
    let tab = simd.view(table);
    let stg = simd.view(staging);
    let splat = |pair: [T; 2]| (simd.splat(pair[0]), simd.splat(pair[1]));
    let tw = ColumnTwiddles {
        w8_1: splat(col[0]),
        w8_3: splat(col[1]),
        w16: [splat(col[2]), splat(col[3]), splat(col[4]), splat(col[5])],
    };
    for g in 0..groups {
        if ROWS == 16 {
            sixteen_rows::<T, A, S, INVERSE, _, _, _>(
                simd, &stg, &tab, &tw, groups, g, out, &mut sink,
            );
        } else if ROWS == 8 {
            eight_rows::<T, A, S, INVERSE, _, _, _>(
                simd, &stg, &tab, &tw, groups, g, out, &mut sink,
            );
        } else {
            four_rows::<T, A, S, INVERSE, _, _, _>(simd, &stg, &tab, groups, g, out, &mut sink);
        }
    }
}

/// Column chunk `a` of group `g`: staging chunk `a * groups + g`.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn column<T, A, Align, Mode, Ref>(
    stg: &SimdView<'_, T, A, Align, Mode, Ref>,
    groups: usize,
    g: usize,
    a: usize,
) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Mode: ExecutionMode,
    Ref: core::ops::Deref<Target = [T]>,
{
    ComplexReg::from_interleaved(Vector::from_view_chunk(stg, a * groups + g))
}

/// Column chunk `a` of group `g` times its mixed-radix twiddle
/// `W_N^{a k}`, dup-split at chunk pair `MIX_CH + 2 ((a - 1) groups + g)`.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn twiddled_column<T, A, Align, Mode, Ref>(
    stg: &SimdView<'_, T, A, Align, Mode, Ref>,
    tab: &SimdView<'_, T, A, Align, Mode, Ref>,
    groups: usize,
    g: usize,
    a: usize,
) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Mode: ExecutionMode,
    Ref: core::ops::Deref<Target = [T]>,
{
    super::super::cmul::cmul_chunk(
        tab,
        column(stg, groups, g, a),
        MIX_CH + 2 * ((a - 1) * groups + g),
    )
}

/// `v` times the broadcast `(re, im)`: one swap, one multiply, one
/// `fmaddsub`.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn broadcast_mul<T, A>(v: ComplexReg<T, A>, (re, im): Broadcast<T, A>) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let vi = v.into_interleaved();
    ComplexReg::from_interleaved(vi.fmaddsub(re, vi.swap_adjacent() * im))
}

/// `-+ i v`: the quarter-turn twiddle of every stage.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn quarter<T, A, const INVERSE: bool>(v: ComplexReg<T, A>) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    if INVERSE {
        v.mul_i()
    } else {
        v.mul_neg_i()
    }
}

/// The eight-point DIF network over eight column registers: distance 4
/// under `W_8^a`, distance 2 under `[1, -+i]`, distance 1 under one.
/// Output register `q` holds spectral row `rev3(q)` of its eight.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn dif8<T, A, const INVERSE: bool>(
    [c0, c1, c2, c3, c4, c5, c6, c7]: [ComplexReg<T, A>; 8],
    tw: &ColumnTwiddles<T, A>,
) -> [ComplexReg<T, A>; 8]
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    // Distance 4 (L = 8): `W_8^a` on the difference.
    let (c0, c4) = c0.butterfly(c4);
    let (c1, c5) = c1.butterfly(c5);
    let c5 = broadcast_mul(c5, tw.w8_1);
    let (c2, c6) = c2.butterfly(c6);
    let c6 = quarter::<T, A, INVERSE>(c6);
    let (c3, c7) = c3.butterfly(c7);
    let c7 = broadcast_mul(c7, tw.w8_3);
    // Distance 2 (L = 4): twiddles `[1, -+i]` in each half.
    let (c0, c2) = c0.butterfly(c2);
    let (c1, c3) = c1.butterfly(c3);
    let c3 = quarter::<T, A, INVERSE>(c3);
    let (c4, c6) = c4.butterfly(c6);
    let (c5, c7) = c5.butterfly(c7);
    let c7 = quarter::<T, A, INVERSE>(c7);
    // Distance 1 (L = 2): twiddle one.
    let (c0, c1) = c0.butterfly(c1);
    let (c2, c3) = c2.butterfly(c3);
    let (c4, c5) = c4.butterfly(c5);
    let (c6, c7) = c6.butterfly(c7);
    [c0, c1, c2, c3, c4, c5, c6, c7]
}

/// One eight-row column group: the DIF over the row index, register `q`
/// stored to row `rev3(q)`.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "one column group's operands; bundling them would be the pass itself"
)]
#[inline(always)]
fn eight_rows<T, A, S, const INVERSE: bool, Align, Mode, Ref>(
    simd: Simd<T, A>,
    stg: &SimdView<'_, T, A, Align, Mode, Ref>,
    tab: &SimdView<'_, T, A, Align, Mode, Ref>,
    tw: &ColumnTwiddles<T, A>,
    groups: usize,
    g: usize,
    out: &mut [T],
    sink: &mut S,
) where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    S: StoreSink<T>,
    Align: Alignment,
    Mode: ExecutionMode,
    Ref: core::ops::Deref<Target = [T]>,
{
    let [c0, c1, c2, c3, c4, c5, c6, c7] = dif8::<T, A, INVERSE>(
        [
            column(stg, groups, g, 0),
            twiddled_column(stg, tab, groups, g, 1),
            twiddled_column(stg, tab, groups, g, 2),
            twiddled_column(stg, tab, groups, g, 3),
            twiddled_column(stg, tab, groups, g, 4),
            twiddled_column(stg, tab, groups, g, 5),
            twiddled_column(stg, tab, groups, g, 6),
            twiddled_column(stg, tab, groups, g, 7),
        ],
        tw,
    );
    sink.store(&simd, c0, REV3[0] * groups + g, out);
    sink.store(&simd, c1, REV3[1] * groups + g, out);
    sink.store(&simd, c2, REV3[2] * groups + g, out);
    sink.store(&simd, c3, REV3[3] * groups + g, out);
    sink.store(&simd, c4, REV3[4] * groups + g, out);
    sink.store(&simd, c5, REV3[5] * groups + g, out);
    sink.store(&simd, c6, REV3[6] * groups + g, out);
    sink.store(&simd, c7, REV3[7] * groups + g, out);
}

/// One sixteen-row column group: the distance-8 stage under `W_16^a`,
/// then the eight-point network on each half. The low half's register `q`
/// holds spectral row `2 rev3(q)`, the high half's `2 rev3(q) + 1`.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "one column group's operands; bundling them would be the pass itself"
)]
#[inline(always)]
fn sixteen_rows<T, A, S, const INVERSE: bool, Align, Mode, Ref>(
    simd: Simd<T, A>,
    stg: &SimdView<'_, T, A, Align, Mode, Ref>,
    tab: &SimdView<'_, T, A, Align, Mode, Ref>,
    tw: &ColumnTwiddles<T, A>,
    groups: usize,
    g: usize,
    out: &mut [T],
    sink: &mut S,
) where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    S: StoreSink<T>,
    Align: Alignment,
    Mode: ExecutionMode,
    Ref: core::ops::Deref<Target = [T]>,
{
    // Distance 8 (L = 16): `W_16^a` on the difference — the odd multiples
    // from the broadcasts, `W_16^{2,6}` the eighths, `W_16^4` the quarter
    // turn. Each butterfly loads its own pair so no more than the network
    // needs is live at once.
    let (c0, c8) = column(stg, groups, g, 0).butterfly(twiddled_column(stg, tab, groups, g, 8));
    let (c1, c9) =
        twiddled_column(stg, tab, groups, g, 1).butterfly(twiddled_column(stg, tab, groups, g, 9));
    let c9 = broadcast_mul(c9, tw.w16[0]);
    let (c2, c10) =
        twiddled_column(stg, tab, groups, g, 2).butterfly(twiddled_column(stg, tab, groups, g, 10));
    let c10 = broadcast_mul(c10, tw.w8_1);
    let (c3, c11) =
        twiddled_column(stg, tab, groups, g, 3).butterfly(twiddled_column(stg, tab, groups, g, 11));
    let c11 = broadcast_mul(c11, tw.w16[1]);
    let (c4, c12) =
        twiddled_column(stg, tab, groups, g, 4).butterfly(twiddled_column(stg, tab, groups, g, 12));
    let c12 = quarter::<T, A, INVERSE>(c12);
    let (c5, c13) =
        twiddled_column(stg, tab, groups, g, 5).butterfly(twiddled_column(stg, tab, groups, g, 13));
    let c13 = broadcast_mul(c13, tw.w16[2]);
    let (c6, c14) =
        twiddled_column(stg, tab, groups, g, 6).butterfly(twiddled_column(stg, tab, groups, g, 14));
    let c14 = broadcast_mul(c14, tw.w8_3);
    let (c7, c15) =
        twiddled_column(stg, tab, groups, g, 7).butterfly(twiddled_column(stg, tab, groups, g, 15));
    let c15 = broadcast_mul(c15, tw.w16[3]);
    let [e0, e1, e2, e3, e4, e5, e6, e7] =
        dif8::<T, A, INVERSE>([c0, c1, c2, c3, c4, c5, c6, c7], tw);
    sink.store(&simd, e0, 2 * REV3[0] * groups + g, out);
    sink.store(&simd, e1, 2 * REV3[1] * groups + g, out);
    sink.store(&simd, e2, 2 * REV3[2] * groups + g, out);
    sink.store(&simd, e3, 2 * REV3[3] * groups + g, out);
    sink.store(&simd, e4, 2 * REV3[4] * groups + g, out);
    sink.store(&simd, e5, 2 * REV3[5] * groups + g, out);
    sink.store(&simd, e6, 2 * REV3[6] * groups + g, out);
    sink.store(&simd, e7, 2 * REV3[7] * groups + g, out);
    let [o0, o1, o2, o3, o4, o5, o6, o7] =
        dif8::<T, A, INVERSE>([c8, c9, c10, c11, c12, c13, c14, c15], tw);
    sink.store(&simd, o0, (2 * REV3[0] + 1) * groups + g, out);
    sink.store(&simd, o1, (2 * REV3[1] + 1) * groups + g, out);
    sink.store(&simd, o2, (2 * REV3[2] + 1) * groups + g, out);
    sink.store(&simd, o3, (2 * REV3[3] + 1) * groups + g, out);
    sink.store(&simd, o4, (2 * REV3[4] + 1) * groups + g, out);
    sink.store(&simd, o5, (2 * REV3[5] + 1) * groups + g, out);
    sink.store(&simd, o6, (2 * REV3[6] + 1) * groups + g, out);
    sink.store(&simd, o7, (2 * REV3[7] + 1) * groups + g, out);
}

/// One four-row column group: the DIF over the row index in two
/// distances, register `q` stored to row `rev2(q)`.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn four_rows<T, A, S, const INVERSE: bool, Align, Mode, Ref>(
    simd: Simd<T, A>,
    stg: &SimdView<'_, T, A, Align, Mode, Ref>,
    tab: &SimdView<'_, T, A, Align, Mode, Ref>,
    groups: usize,
    g: usize,
    out: &mut [T],
    sink: &mut S,
) where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    S: StoreSink<T>,
    Align: Alignment,
    Mode: ExecutionMode,
    Ref: core::ops::Deref<Target = [T]>,
{
    let c0 = column(stg, groups, g, 0);
    let c1 = twiddled_column(stg, tab, groups, g, 1);
    let c2 = twiddled_column(stg, tab, groups, g, 2);
    let c3 = twiddled_column(stg, tab, groups, g, 3);
    // Distance 2 (L = 4): twiddles `[1, -+i]`.
    let (c0, c2) = c0.butterfly(c2);
    let (c1, c3) = c1.butterfly(c3);
    let c3 = quarter::<T, A, INVERSE>(c3);
    // Distance 1 (L = 2): twiddle one; register `q` lands on row `rev2(q)`.
    let (c0, c1) = c0.butterfly(c1);
    let (c2, c3) = c2.butterfly(c3);
    sink.store(&simd, c0, REV2[0] * groups + g, out);
    sink.store(&simd, c1, REV2[1] * groups + g, out);
    sink.store(&simd, c2, REV2[2] * groups + g, out);
    sink.store(&simd, c3, REV2[3] * groups + g, out);
}
