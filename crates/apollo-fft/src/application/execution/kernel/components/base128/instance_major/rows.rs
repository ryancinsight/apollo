//! The row phase of the instance-major base kernel: each row group runs
//! from its loads to its staging stores in registers, at either native
//! width, from a contiguous block or straight out of a split's parent.
//!
//! A register holds sample `b` of `S` rows — `[x[ROWS b + S g], ..,
//! x[ROWS b + S g + S - 1]]`, one contiguous source load, `S = 2` at four
//! lanes and `S = 4` at eight — so every row twiddle is a broadcast and the
//! trivial ones cost a rotation, a sign, or the `sqrt(2)/2` scaling. The
//! row is `ROW_LEN = (ROW_LEN / 4) x 4` with `b = 4 b1 + b0` and output
//! `k = (ROW_LEN / 4) q + m`: a radix-`ROW_LEN / 4` over `b1` within each
//! stride-4 group, the `W_ROW_LEN^{b0 m}` layer, then radix-4 over `b0`.
//! Natural order in and out, so no bit reversal. The two widths differ only
//! in where the layer's broadcasts come from ([`LayerTwiddles`]) and in the
//! `S x S` sample transpose that hands the column pass its sample-major
//! registers.
//!
//! The two stages meet in a full transpose (every second-stage group takes
//! one value from each first-stage group), so the whole row group is live
//! at the crossover — thirty-two registers for 32-sample rows. The
//! allocator spills what the file cannot hold; the earlier explicit spill
//! plane (`zbuf`, a store and a reload for every value) cost 73 stack moves
//! a row pair on the asm census against the 43 the reference kernel pays
//! for the same crossover (`output/apollo-base128/base256_2026-09-11.md`),
//! and its four-arm `match` over a runtime `b0` stayed a loop.
//!
//! Every helper here is a function, never a closure, and every group and
//! tile is its own monomorphization over a constant index: a closure or an
//! `array::from_fn` in this body compiles outside the dispatcher's
//! target-feature frame, so the intrinsics inside it become calls and the
//! closure itself stays out of line (45 calls a row pair on the first
//! draft's census), and `Vector::zero()` re-probes the host, so zeros come
//! from the dispatch token. The row group is straight-line code the census
//! reads per iteration.

use super::super::cmul::cmul_chunk;
use super::{radix4, radix8, rot90, DupSplitEighths, Eighths};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use hermes_simd::{
    Alignment, ComplexReg, ExecutionMode, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage,
    SimdView, Vector,
};

/// Where a block's samples come from: chunk `c` holds block samples
/// `S c .. S c + S`, which sit either contiguously in the block itself or
/// `BLOCKS` apart in a split's parent from sample `OFFSET`.
///
/// Reading the parent directly is what deletes the split's gather pass:
/// block `OFFSET` of a `BLOCKS`-way split is the stride-`BLOCKS`
/// subsequence, and its rows load that subsequence as they go. The loads
/// are `S`-sample windows aligned to `S` in the parent (never past its
/// end), and the needed samples are picked out of them with the same
/// pair-and-half shuffles the row transpose uses: at `S = 2` one
/// `interleave_halves` of two windows; at `S = 4` one `deinterleave_pairs`
/// of two windows for a two-way split, or the two-stage transpose of four
/// for a four-way one. Two to four loads and one to three shuffles a
/// register, against the gather's load, shuffle, and store per chunk plus
/// the block's own load.
pub(crate) trait BlockSource<T: LaneScalar> {
    /// Blocks of the parent this source reads at stride: one for a
    /// contiguous block.
    const BLOCKS: usize;

    /// Lanes the parent holds, for the kernel's entry assertion.
    fn parent_lanes(&self, own: &[T]) -> usize;

    /// Chunk `c` of the block, `S` samples a register.
    fn chunk<A, const S: usize>(&self, simd: Simd<T, A>, own: &[T], c: usize) -> Vector<T, A>
    where
        A: SimdArch + SimdKernel<T>;
}

/// The kernel's own output surface is the parent: block `OFFSET` of its
/// `BLOCKS`-way split, or the whole surface when `BLOCKS = 1`. The rows
/// read it before the column pass writes it.
pub(crate) struct SelfSplit<const BLOCKS: usize, const OFFSET: usize>;

impl<T: LaneScalar, const BLOCKS: usize, const OFFSET: usize> BlockSource<T>
    for SelfSplit<BLOCKS, OFFSET>
{
    const BLOCKS: usize = BLOCKS;

    fn parent_lanes(&self, own: &[T]) -> usize {
        own.len()
    }

    #[expect(
        clippy::inline_always,
        reason = "register kernels must retain their caller's target-feature scope"
    )]
    #[inline(always)]
    fn chunk<A, const S: usize>(&self, simd: Simd<T, A>, own: &[T], c: usize) -> Vector<T, A>
    where
        A: SimdArch + SimdKernel<T>,
    {
        strided::<T, A, S, BLOCKS, OFFSET>(simd, own, c)
    }
}

/// Block `OFFSET` of a `BLOCKS`-way split of a parent the kernel does not
/// write: the direct blocks of the split, whose spectra land in scratch.
pub(crate) struct ParentSplit<'a, T, const BLOCKS: usize, const OFFSET: usize>(pub(crate) &'a [T]);

impl<T: LaneScalar, const BLOCKS: usize, const OFFSET: usize> BlockSource<T>
    for ParentSplit<'_, T, BLOCKS, OFFSET>
{
    const BLOCKS: usize = BLOCKS;

    fn parent_lanes(&self, _own: &[T]) -> usize {
        self.0.len()
    }

    #[expect(
        clippy::inline_always,
        reason = "register kernels must retain their caller's target-feature scope"
    )]
    #[inline(always)]
    fn chunk<A, const S: usize>(&self, simd: Simd<T, A>, _own: &[T], c: usize) -> Vector<T, A>
    where
        A: SimdArch + SimdKernel<T>,
    {
        strided::<T, A, S, BLOCKS, OFFSET>(simd, self.0, c)
    }
}

/// One register of `S` parent samples from sample `sample` on.
///
/// The kernel asserts the parent's length at entry and every start below
/// is an `S`-aligned window inside it, so the bound is a debug assertion.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn window<T, A>(simd: Simd<T, A>, parent: &[T], sample: usize) -> Vector<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let lanes = <A as SimdStorage<T>>::LANE_COUNT;
    debug_assert!(
        2 * sample + lanes <= parent.len(),
        "invariant: source window in bounds"
    );
    let _ = simd;
    // SAFETY: `simd` proves the host supports `A`; the kernel's entry
    // assertion on the parent's length and the aligned-window arithmetic
    // of `strided` keep `2 * sample + LANE_COUNT <= parent.len()`, so the
    // load reads a full vector in bounds.
    unsafe { Vector::load_unaligned(parent.as_ptr().add(2 * sample)) }
}

/// Chunk `c` of block `OFFSET` of a `BLOCKS`-way split: parent samples
/// `BLOCKS (S c + s) + OFFSET` for `s < S`, gathered from `S`-aligned
/// windows. `BLOCKS = 1` is the contiguous block, one load.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn strided<T, A, const S: usize, const BLOCKS: usize, const OFFSET: usize>(
    simd: Simd<T, A>,
    parent: &[T],
    c: usize,
) -> Vector<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(OFFSET < BLOCKS && S * 2 == <A as SimdStorage<T>>::LANE_COUNT);
    if BLOCKS == 1 {
        window(simd, parent, S * c)
    } else if BLOCKS == 3 && S == 2 {
        // Two samples three apart from `6 c + OFFSET`: the low sample of
        // the window there and the high sample of the window two samples
        // on (a window at `+ 3` would read one sample past the parent on
        // the last chunk), the latter swapped to its low half so one
        // interleave pairs them.
        let base = 6 * c + OFFSET;
        let low = window(simd, parent, base);
        let high = ComplexReg::from_interleaved(window(simd, parent, base + 2))
            .swap_samples()
            .into_interleaved();
        low.interleave_halves(high).0
    } else if BLOCKS == 3 {
        // Four samples three apart from `12 c + OFFSET`: column 0 of the
        // four windows there and three, six, nine samples on. The parent
        // ends inside the last chunk's window nine on, so that chunk's
        // windows step back three samples and take column 3.
        let base = 12 * c + OFFSET;
        let (start, col) = if 2 * (base + 13) <= parent.len() {
            (base, 0)
        } else {
            (base - 3, 3)
        };
        let (x0, x1) =
            window(simd, parent, start).interleave_halves(window(simd, parent, start + 3));
        let (y0, y1) =
            window(simd, parent, start + 6).interleave_halves(window(simd, parent, start + 9));
        let (x, y) = if col / 2 == 0 { (x0, y0) } else { (x1, y1) };
        let (even, odd) = x.deinterleave_pairs(y);
        if col % 2 == 0 {
            even
        } else {
            odd
        }
    } else if S == 2 {
        // The two samples sit `BLOCKS` apart from `2 BLOCKS c + OFFSET`;
        // the even-aligned windows holding them share the sample's parity.
        let base = 2 * BLOCKS * c + 2 * (OFFSET / 2);
        let (even, odd) =
            window(simd, parent, base).interleave_halves(window(simd, parent, base + BLOCKS));
        if OFFSET % 2 == 0 {
            even
        } else {
            odd
        }
    } else {
        // Four samples four apart from `16 c + OFFSET`: sample `OFFSET` of
        // each of the four windows from `16 c`, one column of their
        // transpose.
        let base = 16 * c;
        let (x0, x1) = window(simd, parent, base).interleave_halves(window(simd, parent, base + 4));
        let (y0, y1) =
            window(simd, parent, base + 8).interleave_halves(window(simd, parent, base + 12));
        let (x, y) = if OFFSET / 2 == 0 { (x0, y0) } else { (x1, y1) };
        let (even, odd) = x.deinterleave_pairs(y);
        if OFFSET % 2 == 0 {
            even
        } else {
            odd
        }
    }
}

/// The layer's broadcast twiddles: index `k` names the `k`th broadcast the
/// plan pushes (`W_32^{1,3,5,7}`, `W_16^{1,3}`, `W_32^{9,15,21}`,
/// `W_16^{5,7,9}` for 32-sample rows; `W_16^1`, `W_16^3`, `-W_16^1` for
/// sixteen), and the eighths `W_8^1`, `W_8^3` close both lists.
pub(super) trait LayerTwiddles<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    /// `v` times broadcast `k`.
    fn mul(&self, v: ComplexReg<T, A>, k: usize) -> ComplexReg<T, A>;
    /// The eighths `W_8^{1,3}` as dup-split register pairs.
    fn eighths(&self) -> DupSplitEighths<T, A>;
}

/// The four-lane layer: the plan's dup-split broadcast chunks, read in
/// place by [`cmul_chunk`].
pub(super) struct TableLayer<'v, T, A, Align, Mode, Ref>
where
    A: SimdArch,
    Align: Alignment,
    Mode: ExecutionMode,
{
    /// The plan table.
    pub(super) table: &'v SimdView<'v, T, A, Align, Mode, Ref>,
    /// Chunk of the first broadcast.
    pub(super) layer: usize,
    /// Index of the `W_8^1` broadcast; `W_8^3` follows it.
    pub(super) eighth: usize,
}

impl<T, A, Align, Mode, Ref> LayerTwiddles<T, A> for TableLayer<'_, T, A, Align, Mode, Ref>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Mode: ExecutionMode,
    Ref: core::ops::Deref<Target = [T]>,
{
    #[expect(
        clippy::inline_always,
        reason = "register kernels must retain their caller's target-feature scope"
    )]
    #[inline(always)]
    fn mul(&self, v: ComplexReg<T, A>, k: usize) -> ComplexReg<T, A> {
        cmul_chunk(self.table, v, self.layer + 2 * k)
    }

    #[expect(
        clippy::inline_always,
        reason = "register kernels must retain their caller's target-feature scope"
    )]
    #[inline(always)]
    fn eighths(&self) -> DupSplitEighths<T, A> {
        let pair = |k: usize| {
            (
                Vector::from_view_chunk(self.table, self.layer + 2 * k),
                Vector::from_view_chunk(self.table, self.layer + 2 * k + 1),
            )
        };
        DupSplitEighths {
            one: pair(self.eighth),
            three: pair(self.eighth + 1),
        }
    }
}

/// The eight-lane layer: the plan's broadcasts are four-lane chunks, so
/// their scalars are splatted once per transform into dup-split register
/// pairs (`[re; 8]`, `[im; 8]`), the multiply then one `swap_adjacent`,
/// one multiply, and one `fmaddsub` as at four lanes. Slots past the
/// row length's broadcast count hold zero and are never read.
pub(super) struct SplatLayer<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    re: [Vector<T, A>; 14],
    im: [Vector<T, A>; 14],
    /// Index of the `W_8^1` broadcast; `W_8^3` follows it.
    eighth: usize,
}

impl<T, A> SplatLayer<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    /// Splats the `count` broadcasts the plan pushed from lane
    /// `layer_lane` on, the eighths last: each broadcast is `[re; 4]` then
    /// `[im; 4]`.
    #[expect(
        clippy::inline_always,
        reason = "register kernels must retain their caller's target-feature scope"
    )]
    #[inline(always)]
    pub(super) fn new(simd: Simd<T, A>, table: &[T], layer_lane: usize, count: usize) -> Self {
        let mut re = [simd.zero(); 14];
        let mut im = [simd.zero(); 14];
        for k in 0..count {
            re[k] = simd.splat(table[layer_lane + 8 * k]);
            im[k] = simd.splat(table[layer_lane + 8 * k + 4]);
        }
        Self {
            re,
            im,
            eighth: count - 2,
        }
    }
}

impl<T, A> LayerTwiddles<T, A> for SplatLayer<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    #[expect(
        clippy::inline_always,
        reason = "register kernels must retain their caller's target-feature scope"
    )]
    #[inline(always)]
    fn mul(&self, v: ComplexReg<T, A>, k: usize) -> ComplexReg<T, A> {
        let vi = v.into_interleaved();
        ComplexReg::from_interleaved(vi.fmaddsub(self.re[k], vi.swap_adjacent() * self.im[k]))
    }

    #[expect(
        clippy::inline_always,
        reason = "register kernels must retain their caller's target-feature scope"
    )]
    #[inline(always)]
    fn eighths(&self) -> DupSplitEighths<T, A> {
        DupSplitEighths {
            one: (self.re[self.eighth], self.im[self.eighth]),
            three: (self.re[self.eighth + 1], self.im[self.eighth + 1]),
        }
    }
}

/// Runs the row phase from `source` (over `own`, the kernel's output
/// surface) into `staging` with `S` samples a register: the `ROWS / S` row
/// groups of `ROW_LEN` samples, each group's spectra landing in staging as
/// chunk `row * (ROW_LEN / S) + c` holding samples `S c .. S c + S` of that
/// row.
#[expect(
    clippy::inline_always,
    reason = "the pass must fold into the dispatcher's target-feature frame"
)]
#[inline(always)]
pub(super) fn row_pass<
    T,
    A,
    Src,
    W,
    const INVERSE: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const S: usize,
>(
    simd: Simd<T, A>,
    source: &Src,
    own: &[T],
    staging: &mut [T],
    layer: &W,
) where
    T: LaneScalar + MixedRadixScalar,
    A: SimdArch + SimdKernel<T>,
    Src: BlockSource<T>,
    W: LayerTwiddles<T, A>,
{
    let mut stg = simd.view_mut(staging);
    // Blend mask selecting the high sample of each register half: the
    // pattern repeats per 128 bits, so one array serves both widths.
    let zero = T::from_precise(0.0);
    let neg = T::from_precise(-1.0);
    let mask = [zero, zero, neg, neg, zero, zero, neg, neg];
    let hi_mask = Vector::<T, A>::from_view_chunk(&simd.view(&mask), 0);
    for g in 0..ROWS / S {
        let z = [
            group::<T, A, Src, W, INVERSE, ROWS, ROW_LEN, S, 0>(simd, source, own, layer, g),
            group::<T, A, Src, W, INVERSE, ROWS, ROW_LEN, S, 1>(simd, source, own, layer, g),
            group::<T, A, Src, W, INVERSE, ROWS, ROW_LEN, S, 2>(simd, source, own, layer, g),
            group::<T, A, Src, W, INVERSE, ROWS, ROW_LEN, S, 3>(simd, source, own, layer, g),
        ];
        if S == 2 {
            pair_tile::<T, A, INVERSE, ROW_LEN, 0, _, _>(&mut stg, g, hi_mask, &z);
            pair_tile::<T, A, INVERSE, ROW_LEN, 1, _, _>(&mut stg, g, hi_mask, &z);
            if ROW_LEN == 32 {
                pair_tile::<T, A, INVERSE, ROW_LEN, 2, _, _>(&mut stg, g, hi_mask, &z);
                pair_tile::<T, A, INVERSE, ROW_LEN, 3, _, _>(&mut stg, g, hi_mask, &z);
            }
        } else {
            quad_tile::<T, A, INVERSE, ROW_LEN, 0, _, _>(&mut stg, g, hi_mask, &z);
            if ROW_LEN == 32 {
                quad_tile::<T, A, INVERSE, ROW_LEN, 1, _, _>(&mut stg, g, hi_mask, &z);
            }
        }
    }
}

/// Sample `B0 + 4 b1` of row group `g`: the group's rows are adjacent in
/// the block, `ROWS / S` chunks apart per sample.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn sample<T, A, Src, const ROWS: usize, const S: usize, const B0: usize>(
    simd: Simd<T, A>,
    source: &Src,
    own: &[T],
    g: usize,
    b1: usize,
) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    Src: BlockSource<T>,
{
    ComplexReg::from_interleaved(source.chunk::<A, S>(simd, own, (ROWS / S) * (B0 + 4 * b1) + g))
}

/// First stage of one row group for `B0`: the radix-`ROW_LEN / 4` over
/// `b1` on samples `B0 + 4 b1`, then the `W_ROW_LEN^{B0 m}` layer.
/// Sixteen-sample rows fill the first four slots and leave the rest zero,
/// which the constant-indexed second stage never reads.
///
/// The layer's general multiplies are the broadcasts, pre-rotated so no
/// rotation or sign follows one; the eighths are dup-split multiplies too,
/// and the pure rotation `W^{ROW_LEN / 4}` is the only other twiddle.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn group<
    T,
    A,
    Src,
    W,
    const INVERSE: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const S: usize,
    const B0: usize,
>(
    simd: Simd<T, A>,
    source: &Src,
    own: &[T],
    layer: &W,
    g: usize,
) -> [ComplexReg<T, A>; 8]
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    Src: BlockSource<T>,
    W: LayerTwiddles<T, A>,
{
    let rot = rot90::<T, A, INVERSE>;
    let e = layer.eighths();
    if ROW_LEN == 32 {
        // Broadcasts `W_32^{1,3,5,7}`, `W_16^{1,3}`, then the pre-rotated
        // `W_32^{9,15,21}` and `W_16^{5,7,9}` the later groups reach.
        let (w1, w3, w5, w7, v1, v3) = (0, 1, 2, 3, 4, 5);
        let (w9, w15, w21, v5, v7, v9) = (6, 7, 8, 9, 10, 11);
        let y = radix8::<T, A, INVERSE, _>(
            [
                sample::<T, A, Src, ROWS, S, B0>(simd, source, own, g, 0),
                sample::<T, A, Src, ROWS, S, B0>(simd, source, own, g, 1),
                sample::<T, A, Src, ROWS, S, B0>(simd, source, own, g, 2),
                sample::<T, A, Src, ROWS, S, B0>(simd, source, own, g, 3),
                sample::<T, A, Src, ROWS, S, B0>(simd, source, own, g, 4),
                sample::<T, A, Src, ROWS, S, B0>(simd, source, own, g, 5),
                sample::<T, A, Src, ROWS, S, B0>(simd, source, own, g, 6),
                sample::<T, A, Src, ROWS, S, B0>(simd, source, own, g, 7),
            ],
            &e,
        );
        match B0 {
            0 => y,
            1 => [
                y[0],
                layer.mul(y[1], w1),
                layer.mul(y[2], v1),
                layer.mul(y[3], w3),
                e.one(y[4]),
                layer.mul(y[5], w5),
                layer.mul(y[6], v3),
                layer.mul(y[7], w7),
            ],
            2 => [
                y[0],
                layer.mul(y[1], v1),
                e.one(y[2]),
                layer.mul(y[3], v3),
                rot(y[4]),
                layer.mul(y[5], v5),
                e.three(y[6]),
                layer.mul(y[7], v7),
            ],
            _ => [
                y[0],
                layer.mul(y[1], w3),
                layer.mul(y[2], v3),
                layer.mul(y[3], w9),
                e.three(y[4]),
                layer.mul(y[5], w15),
                layer.mul(y[6], v9),
                layer.mul(y[7], w21),
            ],
        }
    } else {
        // Broadcasts `W_16^1`, `W_16^3`, `-W_16^1`.
        let (w1, w3, n1) = (0, 1, 2);
        let y = radix4::<T, A, INVERSE>([
            sample::<T, A, Src, ROWS, S, B0>(simd, source, own, g, 0),
            sample::<T, A, Src, ROWS, S, B0>(simd, source, own, g, 1),
            sample::<T, A, Src, ROWS, S, B0>(simd, source, own, g, 2),
            sample::<T, A, Src, ROWS, S, B0>(simd, source, own, g, 3),
        ]);
        let z = match B0 {
            0 => y,
            1 => [y[0], layer.mul(y[1], w1), e.one(y[2]), layer.mul(y[3], w3)],
            2 => [y[0], e.one(y[1]), rot(y[2]), e.three(y[3])],
            _ => [
                y[0],
                layer.mul(y[1], w3),
                e.three(y[2]),
                layer.mul(y[3], n1),
            ],
        };
        // Through the dispatch token: `Vector::zero()` re-probes the host.
        let zero = ComplexReg::from_interleaved(simd.zero());
        [z[0], z[1], z[2], z[3], zero, zero, zero, zero]
    }
}

/// Radix-4 over `b0` at `m`.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn second<T, A, const INVERSE: bool>(
    z: &[[ComplexReg<T, A>; 8]; 4],
    m: usize,
) -> [ComplexReg<T, A>; 4]
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    radix4::<T, A, INVERSE>([z[0][m], z[1][m], z[2][m], z[3][m]])
}

/// Second stage at four lanes for output pair `MH`: radix-4 over `b0` at
/// `m = 2 MH` and `2 MH + 1`, then the pair transpose. Output
/// `(ROW_LEN / 4) q + m`; the two `m` land as consecutive samples of chunk
/// `(ROW_LEN / 8) q + MH` of each row, row `r` at chunk `r * (ROW_LEN / 2)`.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn pair_tile<'a, T, A, const INVERSE: bool, const ROW_LEN: usize, const MH: usize, Align, Mode>(
    stg: &mut SimdView<'a, T, A, Align, Mode, &'a mut [T]>,
    g: usize,
    hi_mask: Vector<T, A>,
    z: &[[ComplexReg<T, A>; 8]; 4],
) where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Mode: ExecutionMode,
{
    let o0 = second::<T, A, INVERSE>(z, 2 * MH);
    let o1 = second::<T, A, INVERSE>(z, 2 * MH + 1);
    let tiles = ROW_LEN / 8;
    let row0 = 2 * g * (ROW_LEN / 2);
    // Constant indices throughout: a register array indexed by a loop
    // variable round-trips the stack.
    pair_out(stg, hi_mask, o0[0], o1[0], row0 + MH, ROW_LEN / 2);
    pair_out(stg, hi_mask, o0[1], o1[1], row0 + tiles + MH, ROW_LEN / 2);
    pair_out(
        stg,
        hi_mask,
        o0[2],
        o1[2],
        row0 + 2 * tiles + MH,
        ROW_LEN / 2,
    );
    pair_out(
        stg,
        hi_mask,
        o0[3],
        o1[3],
        row0 + 3 * tiles + MH,
        ROW_LEN / 2,
    );
}

/// Transposes one output pair and stores it to chunk `chunk` of the
/// group's first row and the same chunk `row_stride` on.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn pair_out<'a, T, A, Align, Mode>(
    stg: &mut SimdView<'a, T, A, Align, Mode, &'a mut [T]>,
    hi_mask: Vector<T, A>,
    a: ComplexReg<T, A>,
    b: ComplexReg<T, A>,
    chunk: usize,
    row_stride: usize,
) where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Mode: ExecutionMode,
{
    let (a, b) = pair_transpose(hi_mask, a, b);
    a.store_to_view_chunk(stg, chunk);
    b.store_to_view_chunk(stg, chunk + row_stride);
}

/// Second stage at eight lanes for output quartet `MQ`: radix-4 over `b0`
/// at `m = 4 MQ .. 4 MQ + 4`, then the four-sample transpose. The four `m`
/// land as consecutive samples of chunk `(ROW_LEN / 16) q + MQ` of each
/// row, row `r` at chunk `r * (ROW_LEN / 4)`.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn quad_tile<'a, T, A, const INVERSE: bool, const ROW_LEN: usize, const MQ: usize, Align, Mode>(
    stg: &mut SimdView<'a, T, A, Align, Mode, &'a mut [T]>,
    g: usize,
    hi_mask: Vector<T, A>,
    z: &[[ComplexReg<T, A>; 8]; 4],
) where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Mode: ExecutionMode,
{
    let o0 = second::<T, A, INVERSE>(z, 4 * MQ);
    let o1 = second::<T, A, INVERSE>(z, 4 * MQ + 1);
    let o2 = second::<T, A, INVERSE>(z, 4 * MQ + 2);
    let o3 = second::<T, A, INVERSE>(z, 4 * MQ + 3);
    let tiles = ROW_LEN / 16;
    let row0 = 4 * g * (ROW_LEN / 4);
    quad_out(
        stg,
        hi_mask,
        [o0[0], o1[0], o2[0], o3[0]],
        row0 + MQ,
        ROW_LEN / 4,
    );
    quad_out(
        stg,
        hi_mask,
        [o0[1], o1[1], o2[1], o3[1]],
        row0 + tiles + MQ,
        ROW_LEN / 4,
    );
    quad_out(
        stg,
        hi_mask,
        [o0[2], o1[2], o2[2], o3[2]],
        row0 + 2 * tiles + MQ,
        ROW_LEN / 4,
    );
    quad_out(
        stg,
        hi_mask,
        [o0[3], o1[3], o2[3], o3[3]],
        row0 + 3 * tiles + MQ,
        ROW_LEN / 4,
    );
}

/// Transposes one output quartet and stores it to chunk `chunk` of the
/// group's first row and the same chunk of the next three rows,
/// `row_stride` apart.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn quad_out<'a, T, A, Align, Mode>(
    stg: &mut SimdView<'a, T, A, Align, Mode, &'a mut [T]>,
    hi_mask: Vector<T, A>,
    [a, b, c, d]: [ComplexReg<T, A>; 4],
    chunk: usize,
    row_stride: usize,
) where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Mode: ExecutionMode,
{
    let (x0, x1) = pair_transpose(hi_mask, a, b);
    let (y0, y1) = pair_transpose(hi_mask, c, d);
    let (r0, r2) = x0.interleave_halves(y0);
    let (r1, r3) = x1.interleave_halves(y1);
    r0.store_to_view_chunk(stg, chunk);
    r1.store_to_view_chunk(stg, chunk + row_stride);
    r2.store_to_view_chunk(stg, chunk + 2 * row_stride);
    r3.store_to_view_chunk(stg, chunk + 3 * row_stride);
}

/// Within each 128-bit half, `[a0, a1]` and `[b0, b1]` become `[a0, b0]`
/// and `[a1, b1]`: one `swap_pairs` and one blend per output. At four
/// lanes this is the whole pair transpose; at eight it is the first of the
/// two stages, `interleave_halves` finishing across the halves.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn pair_transpose<T, A>(
    hi_mask: Vector<T, A>,
    a: ComplexReg<T, A>,
    b: ComplexReg<T, A>,
) -> (Vector<T, A>, Vector<T, A>)
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let a = a.into_interleaved();
    let b = b.into_interleaved();
    (
        hi_mask.blend(b.swap_pairs(), a),
        hi_mask.blend(b, a.swap_pairs()),
    )
}
