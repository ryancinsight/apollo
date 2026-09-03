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
use super::{instance_major, transform_via_base_128, transform_via_base_128_incumbent};

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
        assert!(super::instance_major::transform_128_measured::<f64, false>(
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

mod final_store;
mod lane_routes;
mod small_sizes;
