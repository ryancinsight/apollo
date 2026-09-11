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
use super::{instance_major, transform_via_base_256};

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
    run: impl Fn(&mut [Complex64]) -> bool,
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
        assert!(run(std::hint::black_box(work)));
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
mod lane_routes;
mod prime_dispatch;
mod rader_width;
mod radix_order;
mod short_winograd_leaves;
mod small_pot_arms;
mod small_sizes;
mod volume_schedule;
