//! Batched power-of-two sub-transforms with the transform index in the lane.
//!
//! ## Why this layout
//!
//! A butterfly vectorized *within* one transform must gather its two operands
//! from positions `j` and `j + half`, and a complex multiply on interleaved data
//! then needs cross-lane shuffles to separate the real and imaginary parts. Both
//! costs vanish when the lane position holds the *transform* index instead:
//! with `B` independent transforms laid out so element `j` of transform `b` sits
//! at `j * B + b`, a butterfly reads two contiguous runs of `B` values, the
//! twiddle is one scalar broadcast across every lane, and no shuffle occurs
//! anywhere.
//!
//! This is the arrangement [`FftPlanarMut`] documents — "lane `c` across all
//! rows is one independent transform instance and no cross-lane shuffle is
//! required" — and which nothing previously used.
//!
//! Measured against the other kernel shapes tried in this crate, all of which
//! sat between 3.4 and 6.1 flops/ns, this reaches 10.2 to 10.7 across batch and
//! length combinations, using the same `hermes_simd` `Vector` operations. The
//! layout is the variable.
//!
//! ## Where it applies
//!
//! The four-step decomposition already splits `N = N1 x N2` and transforms
//! along each axis in turn. Writing `i = j * N2 + b`, the input is *already*
//! batch-major for the first axis, so those `N2` transforms of length `N1` need
//! no transpose at all; and the output index `k2 * N1 + k1` falls out of the
//! second batched pass, so there is no final transpose either. One transpose
//! sits between them, in place because the four-step gate admits only square
//! splits.
//!
//! [`FftPlanarMut`]: crate::domain::storage::FftPlanarMut

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;

#[cfg(all(test, windows, target_arch = "x86_64"))]
macro_rules! sect {
    ($label:expr, $body:block) => {{
        let t0 = unsafe { core::arch::x86_64::_rdtsc() };
        let out = $body;
        let t1 = unsafe { core::arch::x86_64::_rdtsc() };
        crate::application::execution::kernel::components::batched::sections::record(
            $label,
            t1 - t0,
        );
        out
    }};
}
#[cfg(not(all(test, windows, target_arch = "x86_64")))]
macro_rules! sect {
    ($label:expr, $body:block) => {{
        let _label: &str = $label;
        $body
    }};
}

mod cache;
mod lane_order;
mod radix;
mod sweep;
pub(crate) use cache::BatchedPlanCache;
use lane_order::LaneOrder;

/// Section labels of the time-decimated stage set's sweeps, by sweep index;
/// the attribution probe reports them beneath `stages1`.
const TIME_SWEEPS: [&str; 3] = ["t1", "t2", "t3"];
/// Section labels of the frequency-decimated stage set's sweeps, by sweep
/// index; reported beneath `stages2`.
const FREQUENCY_SWEEPS: [&str; 3] = ["f1", "f2", "f3"];

use eunomia::Complex;
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

/// Tile side for the in-place transpose, in elements.
///
/// Per-length batched-transform plan: the stage-major twiddle table, a
/// build-time cost.
///
/// Both stage sets read this one table. The time-decimated set walks its
/// stages upward and the frequency-decimated set downward, over the same
/// values.
pub(crate) struct BatchedPlan<T> {
    len: usize,
    /// Stage-major twiddles: stage `s` (sub-transform length `2^(s+1)`) occupies
    /// `2^s` entries, so the table totals `len - 1`.
    tw: Vec<(T, T)>,
}

impl<T: MixedRadixScalar> BatchedPlan<T> {
    fn new<const INVERSE: bool>(len: usize) -> Self {
        assert!(
            len.is_power_of_two(),
            "batched plan requires a power of two"
        );
        let sign = if INVERSE { 1.0_f64 } else { -1.0_f64 };
        let mut tw = Vec::with_capacity(len.saturating_sub(1));
        let mut l = 2usize;
        while l <= len {
            for j in 0..l / 2 {
                // Direct evaluation per entry: a recurrence here would carry
                // O(N·u) twiddle error, which the forward-error bound this
                // crate documents does not admit.
                let (sin, cos) = (sign * core::f64::consts::TAU * j as f64 / l as f64).sin_cos();
                tw.push((T::from_precise(cos), T::from_precise(sin)));
            }
            l <<= 1;
        }
        Self { len, tw }
    }
}

/// All stages of `batch` independent length-`len` transforms over planar data.
///
/// Dispatch happens once for the whole stage set, which is the placement Hermes
/// ADR 016 requires: outside the innermost loop, and never wrapping a
/// thread-spawning call.
struct BatchedStages<'a, T> {
    re: &'a mut [T],
    im: &'a mut [T],
    tw: &'a [(T, T)],
    /// Interleaved input read by the first pass in place of the planes, or
    /// `None` when the planes already hold it. Rows of `batch` complexes as
    /// `2 * batch` reals in natural order; plane row `p` reads source row
    /// `rev(p)`, which is the row map the deinterleave pass used to apply.
    /// Reading here deletes that pass: the first stage pair is the one that
    /// loads every element exactly once, and two interleaved vector loads
    /// plus one register deinterleave replace the two plane loads.
    source: Option<&'a [T]>,
    /// Live columns per row — the loop bound.
    batch: usize,
    /// Elements per row including [`ROW_PAD`] — the index multiplier.
    stride: usize,
    len: usize,
}

impl<T> LaneKernel<T> for BatchedStages<'_, T>
where
    T: LaneScalar + MixedRadixScalar + radix::Lane,
{
    type Output = ();

    #[expect(
        clippy::inline_always,
        reason = "large LaneKernel::call body must fold into the dispatcher's target-feature scope"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
        let Self {
            re,
            im,
            tw,
            source,
            batch: b,
            stride: s,
            len,
        } = self;
        // Stages ascend from 2; each sweep applies up to `SWEEP_STAGES` of
        // them per trip through the planes, two per pass while two remain
        // and then one, over tiles that stay in L1 between its passes
        // (see [`sweep`]). The four-step twiddle and the interleaved source
        // both ride the pass over stage 2, the one pass that loads every
        // element exactly once. The per-element operation order is the
        // single-stage one, so results are bitwise those of any other
        // grouping.
        let mut l0 = 2usize;
        for (index, stages) in sweep::sweep_lengths(len.trailing_zeros()).enumerate() {
            sect!(TIME_SWEEPS[index], {
                sweep::sweep_time(re, im, tw, source, b, s, len, l0, stages, simd);
            });
            l0 <<= stages;
        }
    }
}

/// One-vector load at `at`, which the caller has proved is in bounds.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line               call here reintroduces the ADR 009 penalty this kernel exists to avoid"
)]
#[inline(always)]
pub(super) fn load<T, A>(data: &[T], at: usize) -> Vector<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(at + <A as SimdStorage<T>>::LANE_COUNT <= data.len());
    // SAFETY: the caller's loop condition bounds `at + LANE_COUNT` by the slice
    // length, and `BatchedStages::call` receives the capability proving that
    // the host executes `A`. The checked wrapper revalidates both per call,
    // which measured as 45% of this kernel's time; the bound here is
    // loop-invariant.
    unsafe { Vector::<T, A>::load_unaligned(data.as_ptr().add(at)) }
}

/// One-vector store at `at`, which the caller has proved is in bounds.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line               call here reintroduces the ADR 009 penalty this kernel exists to avoid"
)]
#[inline(always)]
pub(super) fn store<T, A>(v: Vector<T, A>, data: &mut [T], at: usize)
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(at + <A as SimdStorage<T>>::LANE_COUNT <= data.len());
    // SAFETY: as `load` above.
    unsafe { v.store_unaligned(data.as_mut_ptr().add(at)) }
}

/// Two adjacent vectors of interleaved complexes at `at` (in reals), split
/// into their real and imaginary lanes in the plane column order
/// ([`LaneOrder`]): the sub-lane unpack alone, no cross-lane permute.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope, as `load`"
)]
#[inline(always)]
pub(super) fn load_interleaved<T, A>(data: &[T], at: usize) -> (Vector<T, A>, Vector<T, A>)
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let lanes = <A as SimdStorage<T>>::LANE_COUNT;
    load::<T, A>(data, at).deinterleave_sublanes(load::<T, A>(data, at + lanes))
}

/// The counterpart of [`load_interleaved`]: real and imaginary lanes in plane
/// column order stored as two adjacent vectors of interleaved complexes at
/// `at` (in reals), in memory order.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope, as `store`"
)]
#[inline(always)]
pub(super) fn store_interleaved<T, A>(re: Vector<T, A>, im: Vector<T, A>, data: &mut [T], at: usize)
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let lanes = <A as SimdStorage<T>>::LANE_COUNT;
    let (lo, hi) = re.interleave_sublanes(im);
    store::<T, A>(lo, data, at);
    store::<T, A>(hi, data, at + lanes);
}

/// Bit-reverses a row index within `bits` bits; the map between plane rows
/// and interleaved rows on both sides of the stage sets.
#[inline]
pub(super) fn reverse_row(row: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        row.reverse_bits() >> (usize::BITS - bits)
    }
}

/// Square in-place transpose of an `m x m` plane, tiled for locality.
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
fn transpose_planes<T: Copy>(
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

/// Plans keyed by `(length, inverse)`: the two directions carry conjugate
/// twiddles and cannot share an entry.
/// Shortest transform whose fold table is two-level.
///
/// The compact table costs a second complex multiply per element and two
/// broadcasts per row per register, which the fold sweep pays back only
/// where the full matrix would stream from beyond L2: at 262144 `f64` the
/// fold sweep fell from 635k to 369k cycles, at 65536 and 16384 it moved
/// within noise or lost (48k to 58k at 65536 `f32`; ADR 0059). One host's
/// 3 MiB L2 sets the bound; re-measure before moving it.
const COMPACT_FOLD_MIN_LEN: usize = 1 << 18;

/// The four-step twiddle `W_N^(p k)` for data row `p` of `rows` and plane
/// column `k` of `cols` (the second stage set's plane shape, `cols × rows`
/// for an odd power),
/// as two tables whose product is the entry, or as the full matrix below
/// [`COMPACT_FOLD_MIN_LEN`] (the fine table one full row wide, the coarse
/// table one).
///
/// Column `k` holds memory column `c = order(k)`, and `c = G F + f` with `F`
/// the lane group and `f = order(k mod F)`, so `W_N^(p c) = W_N^(p F G) ·
/// W_N^(p f)`: a coarse table of one scalar per row and lane group and a
/// fine table of one register per row, `m (F + m / F)` entries in place of
/// the `m^2` the full matrix held. At 65536 `f64` that is 128 KiB against
/// 1 MiB, the size of the data itself, which the fold sweep streamed from
/// L2 on every call (ADR 0059). Each entry is one direct evaluation, so the
/// product carries the two factors' roundings and the multiply's own:
/// within `4u` of the exact twiddle, against `u` for the full matrix, which
/// the transform's `O(log N · u)` bound absorbs.
pub(crate) struct FourStepFold<T> {
    /// Lanes per group, `F`: the fine table's row width.
    pub(crate) lanes: usize,
    /// `W_N^(p order(f))` for `f < F`, row-major by `p`.
    pub(crate) fine_re: Box<[T]>,
    pub(crate) fine_im: Box<[T]>,
    /// `W_N^(p F G)` for `G < cols / F`, row-major by `p`.
    pub(crate) coarse_re: Box<[T]>,
    pub(crate) coarse_im: Box<[T]>,
}

impl<T: MixedRadixScalar> FourStepFold<T> {
    fn new<const INVERSE: bool>(n: usize, rows: usize, cols: usize, order: LaneOrder) -> Self {
        use crate::application::execution::kernel::twiddle_table::twiddle_components;
        let sign = if INVERSE { 1.0_f64 } else { -1.0_f64 };
        // The two levels pay a second complex multiply per element, which
        // wins only once the full matrix would stream from beyond L2; below
        // that the fine table is the full row and the coarse table is one.
        let lanes = if n >= COMPACT_FOLD_MIN_LEN {
            order.lanes().min(cols)
        } else {
            cols
        };
        let groups = cols / lanes;
        // Direct evaluation per entry through the shared authority, as the
        // full matrix was built: mod-`n` reduction and one `sin_cos` each.
        let entry = |exponent: usize| {
            let (sin, cos) = twiddle_components(sign, exponent, n);
            (T::from_precise(cos), T::from_precise(sin))
        };
        let (fine_re, fine_im): (Vec<T>, Vec<T>) = (0..rows)
            .flat_map(|p| (0..lanes).map(move |f| (p, f)))
            .map(|(p, f)| entry(p * order.column(f)))
            .unzip();
        let (coarse_re, coarse_im): (Vec<T>, Vec<T>) = (0..rows)
            .flat_map(|p| (0..groups).map(move |g| (p, g)))
            .map(|(p, g)| entry(p * lanes * g))
            .unzip();
        Self {
            lanes,
            fine_re: fine_re.into_boxed_slice(),
            fine_im: fine_im.into_boxed_slice(),
            coarse_re: coarse_re.into_boxed_slice(),
            coarse_im: coarse_im.into_boxed_slice(),
        }
    }
}

/// Runs the stage set of `batch` transforms of length `plan.len` over planar
/// `re`/`im`, whose rows the caller has already bit-reversed — which the
/// driver's deinterleave does for free by writing each row to its reversed
/// position.
fn run_batched<T>(
    re: &mut [T],
    im: &mut [T],
    plan: &BatchedPlan<T>,
    source: Option<&[T]>,
    batch: usize,
    stride: usize,
) where
    T: LaneScalar + MixedRadixScalar + radix::Lane,
{
    hermes_simd::vectorize(BatchedStages {
        re,
        im,
        tw: &plan.tw,
        source,
        batch,
        stride,
        len: plan.len,
    });
}

/// Runs the same stage set decimated in frequency: rows arrive in natural
/// order and leave bit-reversed, which is the pairing that lets the sink
/// absorb the permutation the time-decimated set needed a whole pass to
/// repair (see [`dif`]).
fn run_batched_dif<T>(
    re: &mut [T],
    im: &mut [T],
    plan: &BatchedPlan<T>,
    fold: Option<&FourStepFold<T>>,
    sink: Option<&mut [T]>,
    staging: &mut [T],
    batch: usize,
    stride: usize,
) where
    T: LaneScalar + MixedRadixScalar + radix::Lane,
{
    hermes_simd::vectorize(dif::BatchedStagesDif {
        re,
        im,
        tw: &plan.tw,
        fold,
        sink,
        staging,
        batch,
        stride,
        len: plan.len,
    });
}

/// Scratch length, in complex elements, that [`four_step_batched`] requires
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

/// Whether [`four_step_batched`] covers a transform of length `n`.
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

#[cfg(all(test, windows, target_arch = "x86_64"))]
pub(crate) mod sections;

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
fn plane_geometry(n: usize) -> (usize, usize) {
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
fn split_plane<T>(scratch: &mut [Complex<T>], plane: usize) -> (&mut [T], &mut [T])
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

/// Four-step FFT over the padded planar layout.
///
/// Three steps and no more: the first stage set reads the caller's rows in
/// bit-reversed row order straight out of `data`, one transpose moves the
/// planes to the second axis (in place for a square, into the second plane
/// pair for the rectangle an odd power runs as, ADR 0060), and the second
/// stage set folds the four-step twiddle into its first loads and writes
/// `data` back from its last. The per-element operation order is the same
/// whatever the shape.
///
/// # Panics
///
/// Panics if `data.len()` is not a length [`planar_applies`] admits, or if
/// `scratch` is shorter than [`scratch_len`].
pub(crate) fn four_step_batched<T, const INVERSE: bool>(
    data: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
) where
    T: BatchedPlanCache<Complex = Complex<T>>,
{
    let n = data.len();
    assert!(planar_applies(n), "requires a planar power of two");
    let (n1, n2) = plane_geometry(n);
    let (stride_a, stride_b) = (n2 + ROW_PAD, n1 + ROW_PAD);
    let plane_a = n1 * stride_a;
    let plane_b = if n1 == n2 { 0 } else { n2 * stride_b };
    assert!(
        scratch.len() >= scratch_len(n),
        "scratch must hold the padded planes and the staging block"
    );
    // The planes start on a cache line; the slack in `scratch_len` absorbs
    // the shift (`PLANE_ALIGN_SLACK`).
    let lead = (64 - (scratch.as_ptr() as usize) % 64) % 64 / core::mem::size_of::<Complex<T>>();
    let scratch = &mut scratch[lead..];
    let (a, rest) = scratch.split_at_mut(plane_a);
    let (b, rest) = rest.split_at_mut(plane_b);
    let staging: &mut [T] = eunomia::layout::cast_slice_mut(&mut rest[..sweep::STAGING_LEN]);
    let data: &mut [T] = eunomia::layout::cast_slice_mut(data);
    // Both sets' batches are at least a register wide from 512 up, so one
    // order serves both; below that the square's order is the identity.
    let order = LaneOrder::for_batch::<T>(n1.min(n2));
    let (a_re, a_im) = split_plane(a, plane_a);

    // 1. `n2` transforms of length `n1` along the first axis; the caller's
    //    rows are batch-major for this direction, so the first pass reads
    //    them in place and no transpose is needed.
    let plan = T::cached_plan::<INVERSE>(n1);
    sect!("stages1", {
        run_batched(a_re, a_im, plan.as_ref(), Some(&*data), n2, stride_a)
    });

    // 2. Transpose so the second axis becomes batch-major. Pure exchange:
    //    the four-step twiddle rides stage set 2's first loads below. The
    //    same selector as the stage sets, so the tile width and the plane
    //    column order are the one backend's.
    let (re, im, stride) = if n1 == n2 {
        sect!("transpose", {
            let handled = hermes_simd::vectorize(boundary::TransposePlanes {
                re: &mut *a_re,
                im: &mut *a_im,
                m: n1,
                stride: stride_a,
            });
            if !handled {
                transpose_planes(a_re, a_im, n1, stride_a, order);
            }
        });
        (a_re, a_im, stride_a)
    } else {
        let (b_re, b_im) = split_plane(b, plane_b);
        sect!("transpose", {
            let handled = hermes_simd::vectorize(boundary::TransposePlanesInto {
                src_re: &*a_re,
                src_im: &*a_im,
                rows: n1,
                cols: n2,
                src_stride: stride_a,
                dst_re: &mut *b_re,
                dst_im: &mut *b_im,
                dst_stride: stride_b,
            });
            if !handled {
                transpose_planes_into(
                    PlaneView {
                        re: a_re,
                        im: a_im,
                        rows: n1,
                        cols: n2,
                        stride: stride_a,
                    },
                    b_re,
                    b_im,
                    stride_b,
                    order,
                );
            }
        });
        (b_re, b_im, stride_b)
    };

    // 3. `n1` transforms of length `n2` along the second axis, with the
    //    four-step twiddle `W_N^(n2 k1)` folded into the first stage's loads
    //    from a table with the planes' own shape. This set is decimated in
    //    frequency: the transpose leaves natural row order, which is what
    //    DIF consumes, and it leaves its output bit-reversed, which the sink
    //    absorbs by writing plane row `p` to row `rev(p)` — the mirror of
    //    the source's free permutation. Output `k1 + n1 k2` lands at row
    //    `k2`, column `k1`.
    let plan = if n1 == n2 {
        plan
    } else {
        T::cached_plan::<INVERSE>(n2)
    };
    let fold = T::cached_four_step_fold::<INVERSE>(n, n2, n1);
    sect!("stages2", {
        run_batched_dif(
            re,
            im,
            plan.as_ref(),
            Some(fold.as_ref()),
            Some(data),
            staging,
            n1,
            stride,
        )
    });
}

/// One padded plane pair with its shape, as a transpose source.
struct PlaneView<'a, T> {
    re: &'a [T],
    im: &'a [T],
    rows: usize,
    cols: usize,
    stride: usize,
}

/// Reference form of [`boundary::TransposePlanesInto`]: `rows × cols` planes
/// into `cols × rows` planes, plane cell `(r, c)` landing at `(order(c),
/// plane(r))` as the in-place transpose moves it ([`transpose_planes`]).
fn transpose_planes_into<T: Copy>(
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

pub(crate) mod boundary;
mod dif;

// Test-gated deliberately: the interleaved in-place kernel is correct and
// measured — and slower than the planar sibling by 16 to 37% pinned on an
// E-core at every covered size, because the shuffle cost of interleaved
// butterflies outweighs the planar seams it removes. It stays as the
// independent-implementation differential oracle for this module; the
// pinned probe that declined it is committed beside it.
#[cfg(test)]
pub(crate) mod interleaved;

// Windows-gated: pins threads through Win32 to control the hybrid scheduler.
#[cfg(all(test, windows))]
mod pinned_ladder;

// Additionally x86-gated: the pass totals come from `_rdtsc`.
#[cfg(all(test, windows, target_arch = "x86_64"))]
mod pinned_sections;

#[cfg(test)]
mod tests;
