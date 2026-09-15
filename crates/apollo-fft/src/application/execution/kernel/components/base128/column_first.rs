//! The column-first routes over a base — RustFFT's shape above its
//! butterflies (ADR 0061): a radix pass over the parent in place
//! ([`split_boundary::ColumnPass`] — `R` registers a block apart, the
//! register radix, the twiddle `W_n^{q c}` after the butterfly from one
//! chunk-major stream), each slice transformed out of place into
//! contiguous blocks, and one pass of pair interleaves that transposes the
//! blocks back into natural order ([`split_boundary::InterleaveBlocks`]).
//! `R + 1` streams a pass.
//!
//! One step ([`step`]) serves both placements: in place, the interleave
//! writes back into the parent the pass ran over; out of place, into a
//! separate output, so a step can be the slice transform of the step above
//! it. That nesting is the radix chain ([`chain`]): a chain of `k` levels
//! `[R_0, ..., R_{k-1}]` (outermost first, the innermost always eight over
//! the base) runs the outer pass over the parent, each slice out of place
//! into scratch through the level below, and the outer interleave back;
//! the placements alternate down the chain, so no level copies. It is
//! RustFFT's `8xn` chain over its 256 and 512 butterflies: `[8]` at 2048
//! and 4096, `[4, 8]` at 8192, `[8, 8]` at 16384 and 32768, `[4, 8, 8]` at
//! 65536, `[8, 8, 8]` at 131072 and 262144.
//!
//! For `n = c + B r` and `k = q + R p`, the DFT phase factors as
//! `rq/R + cq/(R B) + cp/B`: the radix across blocks, its twiddle, then
//! one transform of length `B`. Scratch block `q` holds `X[R p + q]` and
//! the interleave restores natural order. The inverse changes the phase
//! sign; normalization belongs to the public plan. Independent direct-sum
//! and exact-permutation tests cover these conventions at every length the
//! plan selects.

use super::{at_plan_width, block, instance_major, lanes, lanes_mut, split_boundary};

/// The radix-`radix` pass over `parent` in place, `block_lanes` scalar
/// lanes a block, from the chunk-major `rows`; false where the width or
/// the radix is not served.
fn column_pass<F, const INVERSE: bool>(
    eight_lanes: bool,
    radix: usize,
    parent: &mut [F::Complex],
    rows: &[F],
    block_lanes: usize,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let dir = if INVERSE { 1.0_f64 } else { -1.0_f64 };
    let eighth = |j: f64| -> [F; 2] {
        let (s, c) = (dir * core::f64::consts::TAU * j / 8.0).sin_cos();
        [F::from_precise(c), F::from_precise(s)]
    };
    // The radix selects its monomorphized pass once per level.
    match radix {
        3 => at_plan_width::<F, _>(
            eight_lanes,
            split_boundary::ColumnPass::<F, _, 3, INVERSE> {
                source: instance_major::SelfSplit::<1, 0>,
                dst: lanes_mut::<F>(parent),
                twiddles: rows,
                block_lanes,
                eighth_one: eighth(1.0),
                eighth_three: eighth(3.0),
            },
        ),
        4 => at_plan_width::<F, _>(
            eight_lanes,
            split_boundary::ColumnPass::<F, _, 4, INVERSE> {
                source: instance_major::SelfSplit::<1, 0>,
                dst: lanes_mut::<F>(parent),
                twiddles: rows,
                block_lanes,
                eighth_one: eighth(1.0),
                eighth_three: eighth(3.0),
            },
        ),
        8 => at_plan_width::<F, _>(
            eight_lanes,
            split_boundary::ColumnPass::<F, _, 8, INVERSE> {
                source: instance_major::SelfSplit::<1, 0>,
                dst: lanes_mut::<F>(parent),
                twiddles: rows,
                block_lanes,
                eighth_one: eighth(1.0),
                eighth_three: eighth(3.0),
            },
        ),
        _ => false,
    }
}

/// The `blocks`-block interleave from `src` into `dst`, `block_lanes`
/// scalar lanes a block; false where the width or the count is not served.
fn interleave<F>(
    eight_lanes: bool,
    blocks: usize,
    src: &[F::Complex],
    dst: &mut [F::Complex],
    block_lanes: usize,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    match blocks {
        3 => at_plan_width::<F, _>(
            eight_lanes,
            split_boundary::InterleaveBlocks::<F, 3> {
                src: lanes::<F>(src),
                dst: lanes_mut::<F>(dst),
                block_lanes,
            },
        ),
        4 => at_plan_width::<F, _>(
            eight_lanes,
            split_boundary::InterleaveBlocks::<F, 4> {
                src: lanes::<F>(src),
                dst: lanes_mut::<F>(dst),
                block_lanes,
            },
        ),
        8 => at_plan_width::<F, _>(
            eight_lanes,
            split_boundary::InterleaveBlocks::<F, 8> {
                src: lanes::<F>(src),
                dst: lanes_mut::<F>(dst),
                block_lanes,
            },
        ),
        _ => false,
    }
}

/// One column-first step over `parent`: the radix pass in place, its
/// slices through `transform` (the slice, then its destination block) into
/// `blocks`, and the interleave into `out` — `parent` itself where `out`
/// is none. `rows` are the level's `(radix - 1)` registers a chunk. Under
/// `MEASURE` (the separately instantiated attribution variant) each phase
/// stamps the meter at `level`.
fn step<F, const INVERSE: bool, const MEASURE: bool>(
    eight_lanes: bool,
    level: usize,
    radix: usize,
    parent: &mut [F::Complex],
    blocks: &mut [F::Complex],
    out: Option<&mut [F::Complex]>,
    rows: &[F],
    mut transform: impl FnMut(&mut [F::Complex], &mut [F::Complex]) -> bool,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    #[cfg(not(all(test, windows, target_arch = "x86_64")))]
    let _ = level;
    let n = parent.len();
    debug_assert!(n % radix == 0 && blocks.len() == n);
    let block_len = n / radix;
    #[cfg(all(test, windows, target_arch = "x86_64"))]
    let t0 = if MEASURE {
        instance_major::phase_meter::stamp()
    } else {
        0
    };
    if !column_pass::<F, INVERSE>(eight_lanes, radix, parent, rows, 2 * block_len) {
        return false;
    }
    #[cfg(all(test, windows, target_arch = "x86_64"))]
    let t1 = if MEASURE {
        let t = instance_major::phase_meter::stamp();
        instance_major::phase_meter::add_chain(level, 0, t - t0);
        t
    } else {
        0
    };
    let transformed = parent
        .chunks_exact_mut(block_len)
        .zip(blocks.chunks_exact_mut(block_len))
        .all(|(slice, dst)| transform(slice, dst));
    #[cfg(all(test, windows, target_arch = "x86_64"))]
    let t2 = if MEASURE {
        let t = instance_major::phase_meter::stamp();
        instance_major::phase_meter::add_chain(level, 1, t - t1);
        t
    } else {
        0
    };
    let interleaved = transformed
        && interleave::<F>(
            eight_lanes,
            radix,
            blocks,
            out.unwrap_or(parent),
            2 * block_len,
        );
    #[cfg(all(test, windows, target_arch = "x86_64"))]
    if MEASURE {
        let t = instance_major::phase_meter::stamp();
        instance_major::phase_meter::add_chain(level, 2, t - t2);
        instance_major::phase_meter::CHAIN_CALLS[level]
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    }
    interleaved
}

/// The scratch a chain over `n` samples needs: every level stages its
/// blocks past the level above, `n + n / R_0 + n / (R_0 R_1) + ...`.
pub(super) fn scratch_len<F>(n: usize, levels: &[instance_major::ChainLevel<F>]) -> usize {
    let mut total = 0;
    let mut m = n;
    for level in levels {
        total += m;
        m /= level.radix();
    }
    total
}

/// The radix chain `levels` (outermost first) over `parent` at `depth`
/// below the top, in place where `out` is none and into `out` otherwise,
/// its blocks staged in `spare` ([`scratch_len`] lanes at the top level)
/// — the base blocks where no level remains.
fn chain_from<
    F,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const BLOCK_LANES: usize,
    const TABLE_LANES: usize,
>(
    eight_lanes: bool,
    depth: usize,
    levels: &[instance_major::ChainLevel<F>],
    parent: &mut [F::Complex],
    spare: &mut [F::Complex],
    out: Option<&mut [F::Complex]>,
    plan: &instance_major::BasePlan<F, ROWS, ROW_LEN, TABLE_LANES>,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let [level, below @ ..] = levels else {
        // The base: one block from the slice into its destination.
        let Some(dst) = out else {
            debug_assert!(false, "invariant: the base level runs out of place");
            return false;
        };
        return block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
            dst,
            instance_major::ParentSplit::<F, 1, 0>(lanes::<F>(parent)),
            plan,
            instance_major::DirectSink,
        );
    };
    let n = parent.len();
    let (blocks, spare) = spare.split_at_mut(n);
    step::<F, INVERSE, MEASURE>(
        eight_lanes,
        depth,
        level.radix(),
        parent,
        blocks,
        out,
        level.rows(),
        |slice, dst| {
            chain_from::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES>(
                eight_lanes,
                depth + 1,
                below,
                slice,
                spare,
                Some(dst),
                plan,
            )
        },
    )
}

/// The radix chain of `sinks` over `data` in place, `scratch` of at least
/// [`scratch_len`]. At one level of eight the placement is measured (ADR
/// 0061): against the radix-8 sink over seven spectra it reads 17% under
/// on the performance core and 4 to 5% under on the efficiency core at
/// `f32` 2048, and against the column pass written into scratch with the
/// blocks in place there 4% under on the performance core; at four lanes
/// it took `f64` 2048 9% under the four 512-block sink form. At one level
/// of three it serves 384 at eight lanes in place of the stride-three
/// parent read. Two levels took 8192 to 32768 10 to 24% under the
/// four-step route on the performance core.
pub(super) fn chain<
    F,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const BLOCK_LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [F::Complex],
    scratch: &mut [F::Complex],
    plan: &instance_major::BasePlan<F, ROWS, ROW_LEN, TABLE_LANES>,
    sinks: &instance_major::SplitSinks<F>,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let levels = sinks.chain();
    debug_assert!(!levels.is_empty() && scratch.len() >= scratch_len(data.len(), levels));
    chain_from::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES>(
        plan.native_eight_lanes(),
        0,
        levels,
        data,
        scratch,
        None,
        plan,
    )
}
