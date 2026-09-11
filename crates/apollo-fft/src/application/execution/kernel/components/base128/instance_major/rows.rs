//! The row phase of the instance-major base kernel: each row group runs
//! from its loads to its staging stores in registers, at either native
//! width.
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
use super::{radix4, radix8, root2_twiddle, rot90};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use hermes_simd::{
    Alignment, ComplexReg, ExecutionMode, LaneScalar, Simd, SimdArch, SimdKernel, SimdView, Vector,
};

/// The layer's broadcast twiddles: index `k` names the `k`th broadcast the
/// plan pushes (`W_32^{1,3,5,7}`, `W_16^{1,3}` for 32-sample rows;
/// `W_16^1`, `W_16^3`, `-W_16^1` for sixteen), and the real `sqrt(2)/2`
/// broadcast follows them.
pub(super) trait LayerTwiddles<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    /// `v` times broadcast `k`.
    fn mul(&self, v: ComplexReg<T, A>, k: usize) -> ComplexReg<T, A>;
    /// The real `sqrt(2)/2` in every lane.
    fn half_root2(&self) -> Vector<T, A>;
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
    /// Chunks the broadcasts span, the `sqrt(2)/2` chunk last.
    pub(super) chunks: usize,
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
    fn half_root2(&self) -> Vector<T, A> {
        Vector::from_view_chunk(self.table, self.layer + self.chunks - 1)
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
    re: [Vector<T, A>; 6],
    im: [Vector<T, A>; 6],
    half_root2: Vector<T, A>,
}

impl<T, A> SplatLayer<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    /// Splats the `count` broadcasts and the `sqrt(2)/2` that the plan
    /// pushed from lane `layer_lane` on: each broadcast is `[re; 4]` then
    /// `[im; 4]`.
    #[expect(
        clippy::inline_always,
        reason = "register kernels must retain their caller's target-feature scope"
    )]
    #[inline(always)]
    pub(super) fn new(simd: Simd<T, A>, table: &[T], layer_lane: usize, count: usize) -> Self {
        let mut re = [simd.zero(); 6];
        let mut im = [simd.zero(); 6];
        for k in 0..count {
            re[k] = simd.splat(table[layer_lane + 8 * k]);
            im[k] = simd.splat(table[layer_lane + 8 * k + 4]);
        }
        Self {
            re,
            im,
            half_root2: simd.splat(table[layer_lane + 8 * count]),
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
    fn half_root2(&self) -> Vector<T, A> {
        self.half_root2
    }
}

/// Runs the row phase over `data` into `staging` with `S` samples a
/// register: the `ROWS / S` row groups of `ROW_LEN` samples, each group's
/// spectra landing in staging as chunk `row * (ROW_LEN / S) + c` holding
/// samples `S c .. S c + S` of that row.
#[expect(
    clippy::inline_always,
    reason = "the pass must fold into the dispatcher's target-feature frame"
)]
#[inline(always)]
pub(super) fn row_pass<
    T,
    A,
    W,
    const INVERSE: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const S: usize,
>(
    simd: Simd<T, A>,
    data: &[T],
    staging: &mut [T],
    layer: &W,
) where
    T: LaneScalar + MixedRadixScalar,
    A: SimdArch + SimdKernel<T>,
    W: LayerTwiddles<T, A>,
{
    let data = simd.view(data);
    let mut stg = simd.view_mut(staging);
    // Blend mask selecting the high sample of each register half: the
    // pattern repeats per 128 bits, so one array serves both widths.
    let zero = T::from_precise(0.0);
    let neg = T::from_precise(-1.0);
    let mask = [zero, zero, neg, neg, zero, zero, neg, neg];
    let hi_mask = Vector::<T, A>::from_view_chunk(&simd.view(&mask), 0);
    for g in 0..ROWS / S {
        let z = [
            group::<T, A, W, INVERSE, ROWS, ROW_LEN, S, 0, _, _, _>(simd, &data, layer, g),
            group::<T, A, W, INVERSE, ROWS, ROW_LEN, S, 1, _, _, _>(simd, &data, layer, g),
            group::<T, A, W, INVERSE, ROWS, ROW_LEN, S, 2, _, _, _>(simd, &data, layer, g),
            group::<T, A, W, INVERSE, ROWS, ROW_LEN, S, 3, _, _, _>(simd, &data, layer, g),
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
/// the source, `ROWS / S` chunks apart per sample.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn sample<T, A, const ROWS: usize, const S: usize, const B0: usize, Align, Mode, Ref>(
    data: &SimdView<'_, T, A, Align, Mode, Ref>,
    g: usize,
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
        (ROWS / S) * (B0 + 4 * b1) + g,
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

/// First stage of one row group for `B0`: the radix-`ROW_LEN / 4` over
/// `b1` on samples `B0 + 4 b1`, then the `W_ROW_LEN^{B0 m}` layer.
/// Sixteen-sample rows fill the first four slots and leave the rest zero,
/// which the constant-indexed second stage never reads.
///
/// The layer's general multiplies are the broadcasts; every other twiddle
/// is one of those under `W^{ROW_LEN / 4}` (a rotation) or
/// `W^{ROW_LEN / 2}` (a sign), or an eighth's `sqrt(2)/2` scaling.
#[expect(
    clippy::inline_always,
    reason = "register kernels must retain their caller's target-feature scope"
)]
#[inline(always)]
fn group<
    T,
    A,
    W,
    const INVERSE: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const S: usize,
    const B0: usize,
    Align,
    Mode,
    Ref,
>(
    simd: Simd<T, A>,
    data: &SimdView<'_, T, A, Align, Mode, Ref>,
    layer: &W,
    g: usize,
) -> [ComplexReg<T, A>; 8]
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    W: LayerTwiddles<T, A>,
    Align: Alignment,
    Mode: ExecutionMode,
    Ref: core::ops::Deref<Target = [T]>,
{
    let rot = rot90::<T, A, INVERSE>;
    let eighth = root2_twiddle::<T, A, INVERSE, false>;
    let three_eighths = root2_twiddle::<T, A, INVERSE, true>;
    let half_root2 = layer.half_root2();
    if ROW_LEN == 32 {
        // Broadcasts `W_32^{1,3,5,7}` then `W_16^{1,3}`.
        let (w1, w3, w5, w7, v1, v3) = (0, 1, 2, 3, 4, 5);
        let y = radix8::<T, A, INVERSE>(
            [
                sample::<T, A, ROWS, S, B0, _, _, _>(data, g, 0),
                sample::<T, A, ROWS, S, B0, _, _, _>(data, g, 1),
                sample::<T, A, ROWS, S, B0, _, _, _>(data, g, 2),
                sample::<T, A, ROWS, S, B0, _, _, _>(data, g, 3),
                sample::<T, A, ROWS, S, B0, _, _, _>(data, g, 4),
                sample::<T, A, ROWS, S, B0, _, _, _>(data, g, 5),
                sample::<T, A, ROWS, S, B0, _, _, _>(data, g, 6),
                sample::<T, A, ROWS, S, B0, _, _, _>(data, g, 7),
            ],
            half_root2,
        );
        match B0 {
            0 => y,
            1 => [
                y[0],
                layer.mul(y[1], w1),
                layer.mul(y[2], v1),
                layer.mul(y[3], w3),
                eighth(y[4], half_root2),
                layer.mul(y[5], w5),
                layer.mul(y[6], v3),
                layer.mul(y[7], w7),
            ],
            2 => [
                y[0],
                layer.mul(y[1], v1),
                eighth(y[2], half_root2),
                layer.mul(y[3], v3),
                rot(y[4]),
                rot(layer.mul(y[5], v1)),
                three_eighths(y[6], half_root2),
                rot(layer.mul(y[7], v3)),
            ],
            _ => [
                y[0],
                layer.mul(y[1], w3),
                layer.mul(y[2], v3),
                rot(layer.mul(y[3], w1)),
                three_eighths(y[4], half_root2),
                rot(layer.mul(y[5], w7)),
                negate(layer.mul(y[6], v1)),
                negate(layer.mul(y[7], w5)),
            ],
        }
    } else {
        // Broadcasts `W_16^1`, `W_16^3`, `-W_16^1`.
        let (w1, w3, n1) = (0, 1, 2);
        let y = radix4::<T, A, INVERSE>([
            sample::<T, A, ROWS, S, B0, _, _, _>(data, g, 0),
            sample::<T, A, ROWS, S, B0, _, _, _>(data, g, 1),
            sample::<T, A, ROWS, S, B0, _, _, _>(data, g, 2),
            sample::<T, A, ROWS, S, B0, _, _, _>(data, g, 3),
        ]);
        let z = match B0 {
            0 => y,
            1 => [
                y[0],
                layer.mul(y[1], w1),
                eighth(y[2], half_root2),
                layer.mul(y[3], w3),
            ],
            2 => [
                y[0],
                eighth(y[1], half_root2),
                rot(y[2]),
                three_eighths(y[3], half_root2),
            ],
            _ => [
                y[0],
                layer.mul(y[1], w3),
                three_eighths(y[2], half_root2),
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
