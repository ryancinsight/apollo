//! The instance-major base butterfly: `ROWS` stride-`ROWS` subsequences of
//! `ROW_LEN` samples (`8 x 16` is the 128-point transform, `4 x 16` the
//! 64-point one, `8 x 32` the 256-point one), register-resident row pairs,
//! one staging buffer, no gathers and no scattered accesses.
//!
//! Decomposition `x[ROWS b + a]`, `a = 0..ROWS`, `b = 0..ROW_LEN`:
//!
//! 1. **Rows.** A register holds sample `b` of two rows, `[x[ROWS b + 2p],
//!    x[ROWS b + 2p + 1]]`, one contiguous source load, so the row twiddles
//!    are broadcast scalars. Each row pair runs the `(ROW_LEN / 4) x 4`
//!    transform in registers ([`rows`]) and stores its natural-order
//!    spectra into staging through a pair transpose, row-major with two
//!    samples a chunk.
//! 2. **Columns.** For each natural sample pair, `ROWS` registers load
//!    contiguously, rows `1..ROWS` multiply by the mixed-radix twiddle
//!    `W_N^{a k}`, a lane-wise `ROWS`-point DIF runs across the row index,
//!    and register `q` stores to output row `rev(q)` ([`column`]) — natural
//!    output order, 32-byte contiguous stores.
//!
//! The register map has native layouts for four scalar lanes (two interleaved
//! complex samples) and eight scalar lanes (four interleaved complex samples).
//! The plan selects the widest supported layout once, before execution; other
//! native widths decline before touching the input so production routing keeps
//! its incumbent path.

use crate::application::execution::kernel::components::register_butterfly::{
    radix4, radix8, rot90, DupSplitEighths, Eighths,
};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::Complex;
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage};

mod column;
mod plan;
mod rows;
mod store;

use crate::application::execution::kernel::components::aligned::CacheLineAligned;
use plan::{native_width, BaseLaneWidth};
pub(crate) use plan::{BasePlan, BasePlanState};
pub(crate) use rows::{BlockSource, ParentSplit, SelfSplit};
pub(crate) use store::{DirectSink, FinalRadix3Sink, FinalRadix4Sink, SplitSinks, StoreSink};

/// Per-phase TSC accumulators for the separately instantiated attribution
/// instrument.
#[cfg(all(test, windows, target_arch = "x86_64"))]
pub(crate) mod phase_meter {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub(crate) static PHASES: [AtomicU64; 3] =
        [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    pub(crate) static CALLS: AtomicU64 = AtomicU64::new(0);
    /// The split construction's phases per whole transform: the gather,
    /// the base transforms with their sinks, the combine levels past them.
    pub(crate) static OUTER: [AtomicU64; 3] =
        [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    pub(crate) static OUTER_CALLS: AtomicU64 = AtomicU64::new(0);
    #[expect(
        clippy::inline_always,
        reason = "a timing stamp outlined from the measured region distorts it"
    )]
    #[inline(always)]
    pub(crate) fn stamp() -> u64 {
        // SAFETY: x86_64 guarantees SSE2, so LFENCE is available. It orders
        // prior work before RDTSC; these stamps exist only in the separately
        // monomorphized attribution variant, never in the comparison kernel.
        unsafe {
            core::arch::x86_64::_mm_lfence();
            core::arch::x86_64::_rdtsc()
        }
    }
    #[expect(
        clippy::inline_always,
        reason = "a timing accumulator outlined from the measured region distorts it"
    )]
    #[inline(always)]
    pub(crate) fn add(phase: usize, dt: u64) {
        PHASES[phase].fetch_add(dt, Ordering::Relaxed);
    }
    #[expect(
        clippy::inline_always,
        reason = "a timing accumulator outlined from the measured region distorts it"
    )]
    #[inline(always)]
    pub(crate) fn add_outer(phase: usize, dt: u64) {
        OUTER[phase].fetch_add(dt, Ordering::Relaxed);
    }
}

/// Output row permutation of the column pass: DIF register `q` holds
/// spectral `REV3[q]`.
const REV3: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];
/// Two-bit reversal for the four-row column pass.
const REV2: [usize; 4] = [0, 2, 1, 3];

/// Twiddle tables for one direction, in dup-split form: every pair twiddle
/// `[W_a, W_b]` is stored as two four-lane chunks `[re_a, re_a, re_b, re_b]`
/// and `[im_a, im_a, im_b, im_b]`, so a complex multiply is one
/// `swap_adjacent`, one multiply, and one `fmaddsub` — a single shuffle
/// where the general interleaved multiply pays three. Chunk layout:
/// `W_8^{0..4}` pairs at chunk 0, `W_16^{0..8}` pairs at chunk 4, then the
/// mixed-radix pairs `[W_128^{a*2g}, W_128^{a*(2g+1)}]` for `a = 1..8`,
/// `g = 0..8` at chunk 12.
/// First chunk of the mixed-radix twiddles: the table opens with them.
const MIX_CH: usize = 0;

/// Chunk of the row layer's first broadcast twiddle (`W_16^1` for
/// sixteen-sample rows; `W_16^3`, `-W_16^1`, and the real `sqrt(2)/2`
/// broadcast follow at `+2`, `+4`, and `+6`), after the `(rows - 1) *
/// row_len` mixed-radix chunks.
const fn layer_ch(rows: usize, row_len: usize) -> usize {
    (rows - 1) * row_len
}

/// Chunks of the row layer's broadcasts: sixteen-sample rows carry
/// `W_16^1`, `W_16^3`, `-W_16^1`; 32-sample rows the twelve pre-rotated
/// general twiddles of the `8 x 4` layer (`W_32^{1,3,5,7,9,15,21}`,
/// `W_16^{1,3,5,7,9}`); both end with the eighths `W_8^1`, `W_8^3`, which
/// the radix-8 and the layer apply as dup-split multiplies.
const fn layer_chunks(row_len: usize) -> usize {
    if row_len == 32 {
        28
    } else {
        10
    }
}

/// Lane count of the table for `rows` subsequences of `row_len` samples:
/// `(rows - 1) * row_len / 2` mixed-radix dup-split pairs, then the row
/// layer's broadcasts ([`layer_chunks`]).
pub(crate) const fn table_lanes(rows: usize, row_len: usize) -> usize {
    ((rows - 1) * row_len + layer_chunks(row_len)) * 4
}

/// The base transform as a lane kernel over interleaved samples: `ROWS`
/// stride-`ROWS` subsequences of `ROW_LEN`, so `ROWS = 8` over sixteen is
/// the 128-point transform, `ROWS = 4` the 64-point one, `ROWS = 8` over
/// thirty-two the 256-point one (ADR 0061), and `ROWS = 16` over
/// thirty-two the 512-point one. The row machinery is
/// identical at every length; the column pass is a lane-wise DIF of length
/// `ROWS`.
///
/// `out` is the surface the sink writes — the block itself, or the two or
/// four blocks a combining sink fills — and `source` says where the block's
/// samples are read from: `out` itself, contiguous or as one
/// stride-`BLOCKS` subsequence of it, or a parent the kernel never writes.
/// The rows read every sample before the columns store the first, so one
/// `&mut` surface serves both roles in sequence, and a split's blocks load
/// the parent directly instead of through a gather pass.
pub(crate) struct BaseTransform<
    'a,
    T,
    const INVERSE: bool,
    const MEASURE_PHASES: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const LANES: usize,
    const TABLE_LANES: usize,
    Src,
    S,
> {
    /// The output surface, `S::OUT_BLOCKS` blocks of `LANES` lanes.
    pub(crate) out: &'a mut [T],
    /// Where the block's samples come from.
    pub(crate) source: Src,
    pub(crate) plan: &'a BasePlan<T, ROWS, ROW_LEN, TABLE_LANES>,
    /// Type-selected output strategy. Direct, pair, and four-block-final
    /// stores are separate monomorphizations, so no mode branch reaches the
    /// column loop.
    sink: S,
}

impl<
        T,
        const INVERSE: bool,
        const MEASURE_PHASES: bool,
        const ROWS: usize,
        const ROW_LEN: usize,
        const LANES: usize,
        const TABLE_LANES: usize,
        Src,
        S,
    > LaneKernel<T>
    for BaseTransform<'_, T, INVERSE, MEASURE_PHASES, ROWS, ROW_LEN, LANES, TABLE_LANES, Src, S>
where
    T: LaneScalar + MixedRadixScalar,
    Src: BlockSource<T>,
    S: StoreSink<T>,
{
    /// Whether the dispatched width handled the transform.
    type Output = bool;

    #[expect(
        clippy::inline_always,
        reason = "the body must inline into the dispatcher's target-feature \
                  frame (hermes LaneKernel contract for large bodies)"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) -> bool {
        let lanes = <A as SimdStorage<T>>::LANE_COUNT;
        if lanes != 4 && lanes != 8 {
            return false;
        }
        debug_assert_eq!(
            lanes == 8,
            matches!(self.plan.lane_width, BaseLaneWidth::Eight),
            "the plan's table geometry is the dispatched width's"
        );
        let out = self.out;
        // The two entry assertions are what the sinks' and sources' raw
        // chunk accesses rest on.
        assert_eq!(
            out.len(),
            S::OUT_BLOCKS * LANES,
            "invariant: the output surface spans the sink's blocks"
        );
        assert_eq!(
            self.source.parent_lanes(out),
            Src::BLOCKS * LANES,
            "invariant: the source parent spans its split"
        );
        #[cfg(all(test, windows, target_arch = "x86_64"))]
        let t0 = if MEASURE_PHASES {
            phase_meter::stamp()
        } else {
            0
        };
        // Phases 1 and 2, the row groups (`rows`); the staging buffer is
        // written in full by them before phase 3 reads a lane of it, so
        // zero-filling it first was pure waste — a 2 KB `memset` the
        // disassembly showed costing about 7% of the transform at every
        // size this kernel serves (gap_audit.md#base-kernel-memset).
        debug_assert!(LANES == 2 * ROW_LEN * ROWS && TABLE_LANES == table_lanes(ROWS, ROW_LEN));
        let mut staging_uninit = core::mem::MaybeUninit::<CacheLineAligned<[T; LANES]>>::uninit();
        // SAFETY: the reference is used only for writes until every lane is
        // initialized. The row pass stores chunk `row * (ROW_LEN / S) + c`
        // for every row in `0..ROWS` and every `c in 0..ROW_LEN / S` — all
        // `ROWS * ROW_LEN / S` chunks of `2 S` lanes (`LANES` lanes) —
        // before phase 3 performs the first read. Every `LaneScalar`
        // implementor is an IEEE float with no validity niche. Coverage is
        // enforced, not just argued: debug builds poison the buffer with
        // NaN below, so a lane read before it is written poisons the output
        // and fails every value-semantic oracle in the debug suite. Miri
        // cannot reach this body (the dispatcher only selects it on AVX2
        // hardware); the NaN poison plus the analytical oracles are the
        // substitute coverage.
        let staging: &mut [T; LANES] = unsafe { &mut (*staging_uninit.as_mut_ptr()).0 };
        #[cfg(debug_assertions)]
        staging.fill(T::from_precise(f64::NAN));
        let table = self.plan.table.0.as_slice();
        if lanes == 4 {
            let tab = simd.view(table);
            let layer = rows::TableLayer {
                table: &tab,
                layer: layer_ch(ROWS, ROW_LEN),
                eighth: layer_chunks(ROW_LEN) / 2 - 2,
            };
            rows::row_pass::<T, A, Src, _, INVERSE, ROWS, ROW_LEN, 2>(
                simd,
                &self.source,
                out,
                staging.as_mut_slice(),
                &layer,
            );
        } else {
            // The broadcasts are four-lane chunks whatever the width; the
            // eight-lane layer splats their scalars once per transform.
            let layer = rows::SplatLayer::new(
                simd,
                table,
                layer_ch(ROWS, ROW_LEN) * 4,
                layer_chunks(ROW_LEN) / 2,
            );
            rows::row_pass::<T, A, Src, _, INVERSE, ROWS, ROW_LEN, 4>(
                simd,
                &self.source,
                out,
                staging.as_mut_slice(),
                &layer,
            );
        }
        #[cfg(all(test, windows, target_arch = "x86_64"))]
        let t2m = if MEASURE_PHASES {
            let t = phase_meter::stamp();
            phase_meter::add(0, t - t0);
            t
        } else {
            0
        };
        // Phase 3: the shared lane-wise `ROWS`-point DIF column pass.
        column::column_pass::<T, A, S, INVERSE, ROWS, ROW_LEN>(
            simd,
            staging.as_slice(),
            table,
            &self.plan.col,
            out,
            self.sink,
        );
        #[cfg(all(test, windows, target_arch = "x86_64"))]
        if MEASURE_PHASES {
            let t = phase_meter::stamp();
            phase_meter::add(2, t - t2m);
            phase_meter::CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        true
    }
}

/// Runs the base kernel over `out` from `source` through `sink` at the
/// plan's native width.
///
/// # Panics
///
/// If `out` does not span the sink's blocks or the source's parent its
/// split.
pub(crate) fn transform_base<
    T,
    const INVERSE: bool,
    const MEASURE_PHASES: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const LANES: usize,
    const TABLE_LANES: usize,
    Src,
    S,
>(
    out: &mut [Complex<T>],
    source: Src,
    plan: &BasePlan<T, ROWS, ROW_LEN, TABLE_LANES>,
    sink: S,
) -> bool
where
    T: MixedRadixScalar,
    Complex<T>: eunomia::layout::Pod,
    Src: BlockSource<T>,
    S: StoreSink<T>,
{
    // One kernel body at either width: the plan fixed the register layout
    // (and its table geometry) once, before execution.
    let kernel =
        BaseTransform::<T, INVERSE, MEASURE_PHASES, ROWS, ROW_LEN, LANES, TABLE_LANES, Src, S> {
            out: eunomia::layout::cast_slice_mut(out),
            source,
            plan,
            sink,
        };
    match plan.lane_width {
        BaseLaneWidth::Four => hermes_simd::vectorize_lanes::<4, T, _>(kernel).unwrap_or(false),
        BaseLaneWidth::Eight => hermes_simd::vectorize_lanes::<8, T, _>(kernel).unwrap_or(false),
    }
}

/// Runs one base transform of `ROWS x ROW_LEN` samples in place.
///
/// # Panics
///
/// If `data.len() * 2` is not the base lane count.
pub(crate) fn transform_block<
    T,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [Complex<T>],
    plan: &BasePlan<T, ROWS, ROW_LEN, TABLE_LANES>,
) -> bool
where
    T: MixedRadixScalar,
    Complex<T>: eunomia::layout::Pod,
{
    transform_base::<T, INVERSE, MEASURE, ROWS, ROW_LEN, LANES, TABLE_LANES, _, _>(
        data,
        SelfSplit::<1, 0>,
        plan,
        DirectSink,
    )
}

#[cfg(test)]
/// The 512-point base plan: sixteen rows of thirty-two.
pub(crate) type Plan512<T> = BasePlan<T, 16, 32, { table_lanes(16, 32) }>;
#[cfg(test)]
pub(crate) type Plan256<T> = BasePlan<T, 8, 32, { table_lanes(8, 32) }>;
#[cfg(test)]
/// The 128-point base plan at four lanes, eight rows of sixteen: the
/// width probe of the gather test.
pub(crate) type Plan8x16<T> = BasePlan<T, 8, 16, { table_lanes(8, 16) }>;
/// The 64-point base plan: four rows of sixteen.
pub(crate) type Plan64<T> = BasePlan<T, 4, 16, { table_lanes(4, 16) }>;
/// Directional state for the 512-point base.
pub(crate) type State512<T> = BasePlanState<T, 16, 32, { table_lanes(16, 32) }>;
/// Directional state for the 256-point base and the split routes over it.
pub(crate) type State256<T> = BasePlanState<T, 8, 32, { table_lanes(8, 32) }>;

/// Directional state for the 128-point base, whose shape is the plan
/// width's measured choice (ADR 0061): eight 16-sample rows at four
/// lanes, four 32-sample rows at eight, where the shorter column pass
/// (two levels, no multiply) outweighs the wider row layer.
pub(crate) enum State128<T> {
    /// Eight rows of sixteen, the four-lane form.
    EightRows(BasePlanState<T, 8, 16, { table_lanes(8, 16) }>),
    /// Four rows of thirty-two, the eight-lane form.
    FourRows(BasePlanState<T, 4, 32, { table_lanes(4, 32) }>),
}

impl<T> State128<T>
where
    T: MixedRadixScalar<Complex = Complex<T>>,
{
    /// Builds the state for the shape this host's native width selects, or
    /// none where neither width is native.
    pub(crate) fn new_if_supported(n: usize) -> Option<Self> {
        match native_width::<T>()? {
            BaseLaneWidth::Eight => BasePlanState::new_if_supported(n).map(Self::FourRows),
            BaseLaneWidth::Four => BasePlanState::new_if_supported(n).map(Self::EightRows),
        }
    }

    #[cfg(test)]
    pub(crate) fn inverse_is_initialized(&self) -> bool {
        match self {
            Self::EightRows(state) => state.inverse_is_initialized(),
            Self::FourRows(state) => state.inverse_is_initialized(),
        }
    }
}
/// Directional state for the 64-point base.
pub(crate) type State64<T> = BasePlanState<T, 4, 16, { table_lanes(4, 16) }>;

#[cfg(test)]
/// Runs the 128-point base butterfly in the shape its state selected.
///
/// # Panics
///
/// If `data` is not exactly 128 samples.
pub(crate) fn transform_128<T, const INVERSE: bool, const MEASURE: bool>(
    data: &mut [Complex<T>],
    state: &State128<T>,
) -> bool
where
    T: MixedRadixScalar<Complex = Complex<T>>,
    Complex<T>: eunomia::layout::Pod,
{
    match state {
        State128::EightRows(state) => {
            let plan = if INVERSE {
                state.inverse()
            } else {
                state.forward()
            };
            transform_block::<T, INVERSE, MEASURE, 8, 16, 256, { table_lanes(8, 16) }>(data, plan)
        }
        State128::FourRows(state) => {
            let plan = if INVERSE {
                state.inverse()
            } else {
                state.forward()
            };
            transform_block::<T, INVERSE, MEASURE, 4, 32, 256, { table_lanes(4, 32) }>(data, plan)
        }
    }
}

#[cfg(test)]
/// Runs the 256-point base butterfly: eight stride-8 subsequences of
/// thirty-two, the row phases over `4 x 8` and the eight-point column pass
/// (ADR 0061, the two-pass base under one radix step).
///
/// # Panics
///
/// If `data` is not exactly 256 samples.
pub(crate) fn transform_256<T, const INVERSE: bool, const MEASURE: bool>(
    data: &mut [Complex<T>],
    plan: &Plan256<T>,
) -> bool
where
    T: MixedRadixScalar,
    Complex<T>: eunomia::layout::Pod,
{
    transform_base::<T, INVERSE, MEASURE, 8, 32, 512, { table_lanes(8, 32) }, _, _>(
        data,
        SelfSplit::<1, 0>,
        plan,
        DirectSink,
    )
}

#[cfg(test)]
/// Runs the 512-point base butterfly: sixteen stride-16 subsequences of
/// thirty-two, the row phases over `4 x 8` and the sixteen-point column
/// pass.
///
/// # Panics
///
/// If `data` is not exactly 512 samples.
pub(crate) fn transform_512<T, const INVERSE: bool, const MEASURE: bool>(
    data: &mut [Complex<T>],
    plan: &Plan512<T>,
) -> bool
where
    T: MixedRadixScalar,
    Complex<T>: eunomia::layout::Pod,
{
    transform_base::<T, INVERSE, MEASURE, 16, 32, 1024, { table_lanes(16, 32) }, _, _>(
        data,
        SelfSplit::<1, 0>,
        plan,
        DirectSink,
    )
}

/// Runs the 64-point base butterfly: the same construction over four
/// stride-4 subsequences of sixteen.
///
/// # Panics
///
/// If `data` is not exactly 64 samples.
pub(crate) fn transform_64<T, const INVERSE: bool>(
    data: &mut [Complex<T>],
    plan: &Plan64<T>,
) -> bool
where
    T: MixedRadixScalar,
    Complex<T>: eunomia::layout::Pod,
{
    transform_base::<T, INVERSE, false, 4, 16, 128, { table_lanes(4, 16) }, _, _>(
        data,
        SelfSplit::<1, 0>,
        plan,
        DirectSink,
    )
}
