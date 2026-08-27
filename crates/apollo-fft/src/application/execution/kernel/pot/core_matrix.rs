//! Route cost by core type, with the thread pinned so the hybrid scheduler is
//! out of the question.
//!
//! This is the instrument that settled the "process-dependent four-step"
//! anomaly: unpinned benchmark processes get EcoQoS from Windows — efficiency
//! cores at efficiency frequency — and report route costs that say more about
//! scheduling than about routes. Pinned, at N = 4096, four-step beats Stockham
//! on both core types. It asserts nothing; it is a named measurement
//! instrument, run with `--ignored --nocapture` like `crossover` beside it.

use super::route::{FourStep, PotRoute};
use super::strategies::StockhamAutosort;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::kernel::test_utils::pin;
use eunomia::Complex64;
use std::time::Instant;

fn best<R: PotRoute>(src: &[Complex64], work: &mut [Complex64], tw: &[Complex64]) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..200 {
        work.copy_from_slice(src);
        let t = Instant::now();
        R::run::<f64, false, false>(std::hint::black_box(work), tw);
        best = best.min(t.elapsed().as_nanos() as f64);
        std::hint::black_box(&work[0]);
    }
    best
}

#[test]
#[ignore = "measurement probe for the hybrid-core finding"]
fn route_cost_by_core_type() {
    let n = 4096usize;
    let src: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((0.017 * i as f64).sin(), 0.25 * (0.031 * i as f64).cos()))
        .collect();
    let mut work = src.clone();
    let tw = <f64 as MixedRadixScalar>::cached_twiddle_fwd(n);

    // Logical 0..8 are P-cores and 8..24 E-cores on the Core Ultra 9 285K.
    for cpu in [2u32, 12] {
        let landed = pin(cpu);
        let stockham = best::<StockhamAutosort>(&src, &mut work, &tw);
        let four_step = best::<FourStep>(&src, &mut work, &tw);
        println!(
            "CORE cpu={landed:<2} ({}) stockham={stockham:>9.0}ns four_step={four_step:>9.0}ns",
            if landed < 8 { "P" } else { "E" },
        );
    }
}
