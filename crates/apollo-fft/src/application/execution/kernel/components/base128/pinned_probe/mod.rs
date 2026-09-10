//! Pinned measurement for the 8 x 128 construction: first the gate — the
//! inner small-size transforms against the references — then the assembled
//! experiment. Sets no performance threshold; run with `--ignored --nocapture`.
//!
//! This module holds the shared harness — the timed transform call, the
//! phase attribution, and the scalar bridge every probe measures through.
//! Each probe group lives in its own module beside it.

use eunomia::Complex64;

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
// Re-imported here so each probe module reaches them as `super::…` rather
// than climbing two levels; they are the parent module's own items.
use super::{instance_major, split_boundary, transform_via_base_128, BASE};

type BenchTransform =
    fn(&mut [Complex64], &super::instance_major::Plan128<f64>, &[Complex64]) -> bool;

#[inline(never)]
fn run_bench_transform(
    source: &[Complex64],
    work: &mut [Complex64],
    plan: &super::instance_major::Plan128<f64>,
    twiddles: &[Complex64],
    transform: BenchTransform,
) {
    work.copy_from_slice(source);
    std::hint::black_box(transform)(std::hint::black_box(work), plan, twiddles);
}

fn phase_attribution(
    src: &[Complex64],
    work: &mut [Complex64],
    plan: &super::instance_major::Plan128<f64>,
) -> [u64; 3] {
    use std::sync::atomic::Ordering;

    const CALLS: u64 = 8_192;
    for phase in &super::instance_major::phase_meter::PHASES {
        phase.store(0, Ordering::Relaxed);
    }
    super::instance_major::phase_meter::CALLS.store(0, Ordering::Relaxed);
    for _ in 0..CALLS {
        work.copy_from_slice(src);
        assert!(super::instance_major::transform_128::<f64, false, true>(
            std::hint::black_box(work),
            plan,
        ));
    }
    let calls = super::instance_major::phase_meter::CALLS
        .load(Ordering::Relaxed)
        .max(1);
    let mut averages = [0; 3];
    for (average, phase) in averages
        .iter_mut()
        .zip(&super::instance_major::phase_meter::PHASES)
    {
        *average = phase.load(Ordering::Relaxed) / calls;
    }
    averages
}

/// Per-phase cycles of the split construction at `n`: the outer phases per
/// transform (gather, base transforms with their sinks, combine levels)
/// and the inner phases per base transform (the fused load-and-rows pass,
/// the column pass with its sink).
fn split_attribution(
    src: &[Complex64],
    work: &mut [Complex64],
    plan: &super::instance_major::Plan128<f64>,
    twiddles: &[Complex64],
) -> SplitPhases {
    use std::sync::atomic::Ordering;

    let calls: u64 = 4_096;
    for phase in super::instance_major::phase_meter::PHASES
        .iter()
        .chain(&super::instance_major::phase_meter::OUTER)
    {
        phase.store(0, Ordering::Relaxed);
    }
    super::instance_major::phase_meter::CALLS.store(0, Ordering::Relaxed);
    super::instance_major::phase_meter::OUTER_CALLS.store(0, Ordering::Relaxed);
    for _ in 0..calls {
        work.copy_from_slice(src);
        assert!(super::transform_via_base_128::<f64, false, true>(
            std::hint::black_box(work),
            plan,
            twiddles,
        ));
    }
    let inner_calls = super::instance_major::phase_meter::CALLS
        .load(Ordering::Relaxed)
        .max(1);
    let outer_calls = super::instance_major::phase_meter::OUTER_CALLS
        .load(Ordering::Relaxed)
        .max(1);
    let inner = |phase: usize| {
        super::instance_major::phase_meter::PHASES[phase].load(Ordering::Relaxed) / inner_calls
    };
    let outer = |phase: usize| {
        super::instance_major::phase_meter::OUTER[phase].load(Ordering::Relaxed) / outer_calls
    };
    SplitPhases {
        blocks_per_call: inner_calls / outer_calls,
        gather: outer(0),
        blocks: outer(1),
        levels: outer(2),
        rows: inner(0),
        columns: inner(2),
    }
}

/// The split construction's cycles per phase, per transform for the outer
/// three and per base transform for the inner two.
struct SplitPhases {
    blocks_per_call: u64,
    gather: u64,
    blocks: u64,
    levels: u64,
    rows: u64,
    columns: u64,
}

/// Reference-library dispatch for the probe: PhastFT publishes one planner
/// and entry point per scalar, so the generic body selects them by trait.
trait ProbeScalar: MixedRadixScalar + rustfft::FftNum {
    type PhastPlanner;
    fn phast_planner(n: usize) -> Self::PhastPlanner;
    fn phast_forward(re: &mut [Self], im: &mut [Self], planner: &Self::PhastPlanner);
}

impl ProbeScalar for f64 {
    type PhastPlanner = phastft::planner::PlannerDit64;
    fn phast_planner(n: usize) -> Self::PhastPlanner {
        phastft::planner::PlannerDit64::new(n)
    }
    fn phast_forward(re: &mut [Self], im: &mut [Self], planner: &Self::PhastPlanner) {
        phastft::fft_f64_dit_with_planner(re, im, phastft::planner::Direction::Forward, planner);
    }
}

impl ProbeScalar for f32 {
    type PhastPlanner = phastft::planner::PlannerDit32;
    fn phast_planner(n: usize) -> Self::PhastPlanner {
        phastft::planner::PlannerDit32::new(n)
    }
    fn phast_forward(re: &mut [Self], im: &mut [Self], planner: &Self::PhastPlanner) {
        phastft::fft_f32_dit_with_planner(re, im, phastft::planner::Direction::Forward, planner);
    }
}

mod codelet_selection;
mod composite_split_ab;
mod final_store;
mod lane_routes;
mod prime_dispatch;
mod rader_width;
mod radix_order;
mod short_winograd_leaves;
mod small_pot_arms;
mod small_sizes;
mod volume_schedule;

#[cfg(all(test, windows, target_arch = "x86_64"))]
fn transform_via_base_128_incumbent<F, const INVERSE: bool>(
    data: &mut [F::Complex],
    plan: &instance_major::Plan128<F>,
    twiddles: &[F::Complex],
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let n = data.len();
    assert_eq!(n, 4 * BASE, "the incumbent probe covers only N=512");
    const MEASURE: bool = false;
    <F as crate::application::execution::kernel::mixed_radix::MixedRadixScalar>::with_scratch(
        n,
        |scratch| {
            let gathered =
                hermes_simd::vectorize_lanes::<4, F, _>(split_boundary::GatherBlocks::<F, 4> {
                    src: eunomia::layout::cast_slice(&*data),
                    dst: eunomia::layout::cast_slice_mut(&mut scratch[..n]),
                })
                .unwrap_or(false);
            if !gathered {
                for (block_index, block) in scratch.chunks_exact_mut(BASE).enumerate().take(4) {
                    let offset = block_index.reverse_bits() >> (usize::BITS - 2);
                    for (index, slot) in block.iter_mut().enumerate() {
                        *slot = data[4 * index + offset];
                    }
                }
            }
            for block in scratch.chunks_exact_mut(BASE).take(4) {
                if !instance_major::transform_128::<F, INVERSE, MEASURE>(block, plan) {
                    return false;
                }
            }
            combine_final4::<F>(data, scratch, twiddles, BASE);
            true
        },
    )
}

/// Both combine levels of a four-block split in one pass.
///
/// The chained form runs `combine_stage` and then `combine_final` — two
/// full reads and writes of the array. Fusing them applies both butterfly
/// levels per index while every operand is in registers: block values
/// `b0..b3` at index `j` produce the four outputs `j`, `j + len`,
/// `j + 2 * len`, `j + 3 * len` directly, and the array is read and
/// written once (gap_audit.md#split-boundary).
///
/// Level one pairs `(b0, b1)` and `(b2, b3)` with `W_{2 * len}`; level two
/// combines those with `W_{4 * len}` at `j` and `j + len` — the block
/// order is the gather's bit-reversed one, which is exactly what makes the
/// adjacent-pair pairing correct.
#[cfg(all(test, windows, target_arch = "x86_64"))]
fn combine_final4<F>(
    out: &mut [F::Complex],
    scratch: &[F::Complex],
    twiddles: &[F::Complex],
    len: usize,
) where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
{
    let inner = &twiddles[len - 1..2 * len - 1];
    let outer = &twiddles[2 * len - 1..4 * len - 1];
    let (b01, b23) = scratch.split_at(2 * len);
    let (b0, b1) = b01.split_at(len);
    let (b2, b3) = b23.split_at(len);
    let (lo, hi) = out.split_at_mut(2 * len);
    let (out0, out1) = lo.split_at_mut(len);
    let (out2, out3) = hi.split_at_mut(len);
    for j in 0..len {
        let r = b1[j] * inner[j];
        let (e_lo, e_hi) = (b0[j] + r, b0[j] - r);
        let r = b3[j] * inner[j];
        let (o_lo, o_hi) = (b2[j] + r, b2[j] - r);
        let r = o_lo * outer[j];
        out0[j] = e_lo + r;
        out2[j] = e_lo - r;
        let r = o_hi * outer[j + len];
        out1[j] = e_hi + r;
        out3[j] = e_hi - r;
    }
}
