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
//! it. That nesting is the radix chain: `n = OUTER x 8 x BASE` runs the
//! outer pass over the parent, each `8 BASE` slice out of place into
//! scratch through the inner step (its base blocks staged in a slice-length
//! temporary), and the outer interleave back — RustFFT's `8xn` chain over
//! its 256 and 512 butterflies, no copy anywhere.
//!
//! For `n = c + B r` and `k = q + R p`, the DFT phase factors as
//! `rq/R + cq/(R B) + cp/B`: the radix across blocks, its twiddle, then
//! one transform of length `B`. Scratch block `q` holds `X[R p + q]` and
//! the interleave restores natural order. The inverse changes the phase
//! sign; normalization belongs to the public plan. Independent direct-sum
//! and exact-permutation tests cover these conventions at every length the
//! plan selects.

use super::{at_plan_width, block, instance_major, lanes, lanes_mut, split_boundary};

/// One column-first step over `parent`: the radix-`BLOCKS` pass in place,
/// its `BLOCKS` slices through `transform` (the slice, then its destination
/// block) into `blocks`, and the interleave into `out` — `parent` itself
/// where `out` is none. `BLOCK_LANES` is one slice in scalar lanes and
/// `rows` its `(BLOCKS - 1) BLOCK_LANES` chunk-major twiddle lanes.
fn step<F, const INVERSE: bool, const BLOCKS: usize, const BLOCK_LANES: usize>(
    eight_lanes: bool,
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
    let dir = if INVERSE { 1.0_f64 } else { -1.0_f64 };
    let eighth = |j: f64| -> [F; 2] {
        let (s, c) = (dir * core::f64::consts::TAU * j / 8.0).sin_cos();
        [F::from_precise(c), F::from_precise(s)]
    };
    let passed = at_plan_width::<F, _>(
        eight_lanes,
        split_boundary::ColumnPass::<F, _, BLOCKS, BLOCK_LANES, INVERSE> {
            source: instance_major::SelfSplit::<1, 0>,
            dst: lanes_mut::<F>(parent),
            twiddles: rows,
            eighth_one: eighth(1.0),
            eighth_three: eighth(3.0),
        },
    );
    if !passed {
        return false;
    }
    let transformed = parent
        .chunks_exact_mut(BLOCK_LANES / 2)
        .zip(blocks.chunks_exact_mut(BLOCK_LANES / 2))
        .all(|(slice, dst)| transform(slice, dst));
    transformed
        && at_plan_width::<F, _>(
            eight_lanes,
            split_boundary::InterleaveBlocks::<F, BLOCKS, BLOCK_LANES> {
                src: lanes::<F>(blocks),
                dst: lanes_mut::<F>(out.unwrap_or(parent)),
            },
        )
}

/// `BLOCKS` base blocks under one column-first step in place: `n = BLOCKS
/// BASE`, `scratch` of `n`. At eight blocks the placement is measured
/// (ADR 0061): against the radix-8 sink over seven spectra it reads 17%
/// under on the performance core and 4 to 5% under on the efficiency core
/// at `f32` 2048, and against the column pass written into scratch with
/// the blocks in place there 4% under on the performance core; at four
/// lanes it took `f64` 2048 9% under the four 512-block sink form. At
/// three it serves 384 at eight lanes in place of the stride-three parent
/// read.
pub(super) fn one_level<
    F,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const BASE: usize,
    const BLOCK_LANES: usize,
    const TABLE_LANES: usize,
    const BLOCKS: usize,
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
    debug_assert!(data.len() == BLOCKS * BASE && scratch.len() >= data.len());
    step::<F, INVERSE, BLOCKS, BLOCK_LANES>(
        plan.native_eight_lanes(),
        data,
        &mut scratch[..BLOCKS * BASE],
        None,
        sinks.rows(),
        |slice, dst| {
            block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
                dst,
                instance_major::ParentSplit::<F, 1, 0>(lanes::<F>(slice)),
                plan,
                instance_major::DirectSink,
            )
        },
    )
}

/// The radix chain: `OUTER` slices of eight base blocks each, `n = OUTER x
/// 8 x BASE`, the outer step in place over `data` and the inner step out
/// of place from each slice into `scratch`, the base blocks staged in the
/// `8 BASE` temporary past it (`scratch` of `n + 8 BASE`). `CHAIN_LANES`
/// is one outer slice in scalar lanes, `8 BLOCK_LANES`; the outer rows
/// carry `(OUTER - 1) CHAIN_LANES` lanes.
pub(super) fn two_level<
    F,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const BASE: usize,
    const BLOCK_LANES: usize,
    const CHAIN_LANES: usize,
    const TABLE_LANES: usize,
    const OUTER: usize,
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
    let n = data.len();
    debug_assert!(
        CHAIN_LANES == 8 * BLOCK_LANES && n == OUTER * 8 * BASE && scratch.len() >= n + 8 * BASE
    );
    let eight_lanes = plan.native_eight_lanes();
    let (blocks, tmp) = scratch.split_at_mut(n);
    let tmp = &mut tmp[..8 * BASE];
    step::<F, INVERSE, OUTER, CHAIN_LANES>(
        eight_lanes,
        data,
        blocks,
        None,
        sinks.outer_rows(),
        |slice, out| {
            step::<F, INVERSE, 8, BLOCK_LANES>(
                eight_lanes,
                slice,
                tmp,
                Some(out),
                sinks.rows(),
                |eighth, dst| {
                    block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
                        dst,
                        instance_major::ParentSplit::<F, 1, 0>(lanes::<F>(eighth)),
                        plan,
                        instance_major::DirectSink,
                    )
                },
            )
        },
    )
}
