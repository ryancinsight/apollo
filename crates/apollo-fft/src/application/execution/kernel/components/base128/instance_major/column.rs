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
//! and the distance-4 twiddles `W_8^{1,3}` are dup-split register pairs,
//! one shuffle a multiply where the general form paid three. The census
//! reads the group at 84 instructions and 15 shuffles from 83 and 17, the
//! eight stack moves a group unchanged: the compiler keeps the twiddled
//! columns on the stack across the distance-4 stage whatever the source
//! form, and the group's critical path (about 67 cycles for 20 cycles of
//! issue) is what the pass costs — the out-of-order window overlaps
//! groups, and the meter reads about 37 cycles a group.

use super::store::StoreSink;
use super::{MIX_CH, REV2, REV3};
use hermes_simd::{
    Alignment, ComplexReg, ExecutionMode, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage,
    SimdView, Vector,
};

/// Runs the column pass over `staging`, storing every register through
/// `sink` into `out`, the kernel's output surface. `col` carries `W_8^1`
/// and `W_8^3` for the eight-row form's distance-4 stage; the four-row
/// form never reads them.
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
    col: &[[T; 2]; 2],
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
    let w8_1 = (simd.splat(col[0][0]), simd.splat(col[0][1]));
    let w8_3 = (simd.splat(col[1][0]), simd.splat(col[1][1]));
    for g in 0..groups {
        if ROWS == 8 {
            eight_rows::<T, A, S, INVERSE, _, _, _>(
                simd, &stg, &tab, w8_1, w8_3, groups, g, out, &mut sink,
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
fn broadcast_mul<T, A>(
    v: ComplexReg<T, A>,
    (re, im): (Vector<T, A>, Vector<T, A>),
) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let vi = v.into_interleaved();
    ComplexReg::from_interleaved(vi.fmaddsub(re, vi.swap_adjacent() * im))
}

/// `-+ i v`: the distance-2 stage's odd twiddle.
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

/// One eight-row column group: the DIF over the row index in three
/// distances, register `q` stored to row `rev3(q)`.
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
    w8_1: (Vector<T, A>, Vector<T, A>),
    w8_3: (Vector<T, A>, Vector<T, A>),
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
    let c4 = twiddled_column(stg, tab, groups, g, 4);
    let c5 = twiddled_column(stg, tab, groups, g, 5);
    let c6 = twiddled_column(stg, tab, groups, g, 6);
    let c7 = twiddled_column(stg, tab, groups, g, 7);
    // Distance 4 (L = 8): `W_8^a` on the difference.
    let (c0, c4) = c0.butterfly(c4);
    let (c1, c5) = c1.butterfly(c5);
    let c5 = broadcast_mul(c5, w8_1);
    let (c2, c6) = c2.butterfly(c6);
    let c6 = quarter::<T, A, INVERSE>(c6);
    let (c3, c7) = c3.butterfly(c7);
    let c7 = broadcast_mul(c7, w8_3);
    // Distance 2 (L = 4): twiddles `[1, -+i]` in each half.
    let (c0, c2) = c0.butterfly(c2);
    let (c1, c3) = c1.butterfly(c3);
    let c3 = quarter::<T, A, INVERSE>(c3);
    let (c4, c6) = c4.butterfly(c6);
    let (c5, c7) = c5.butterfly(c7);
    let c7 = quarter::<T, A, INVERSE>(c7);
    // Distance 1 (L = 2): twiddle one; register `q` lands on row `rev3(q)`.
    let (c0, c1) = c0.butterfly(c1);
    let (c2, c3) = c2.butterfly(c3);
    let (c4, c5) = c4.butterfly(c5);
    let (c6, c7) = c6.butterfly(c7);
    sink.store(&simd, c0, REV3[0] * groups + g, out);
    sink.store(&simd, c1, REV3[1] * groups + g, out);
    sink.store(&simd, c2, REV3[2] * groups + g, out);
    sink.store(&simd, c3, REV3[3] * groups + g, out);
    sink.store(&simd, c4, REV3[4] * groups + g, out);
    sink.store(&simd, c5, REV3[5] * groups + g, out);
    sink.store(&simd, c6, REV3[6] * groups + g, out);
    sink.store(&simd, c7, REV3[7] * groups + g, out);
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
