//! The lengths past the caches — 65536 to 524288 — against the references
//! at both scalars, the same sweep as the small sizes under the same
//! per-case budget.
//!
//! One run is not a reading here. At 262144 (`f64`, 4 MB of samples, past
//! the performance core's L2) five quiet pinned runs read apollo between
//! 1048 and 1135 us and RustFFT between 1062 and 1252, each run's hundred
//! samples tight within 5% while the runs moved the two arms independently
//! by 4 to 8%: a per-run state (the placement of each arm's buffers against
//! the caches and pages it lands on), not sampling noise, so a longer
//! budget cannot narrow it. The reading is the median of replicated runs
//! (`scripts/pinned_probe.py --runs 5`), each a fresh process, and the
//! sweep is separate from the small sizes so the replication costs seconds
//! rather than minutes.

use super::small_sizes::{sizes_for_scalar, sweep_config, sweep_warm_up_config};
use crate::application::execution::kernel::measurement_cores;
use apollo_bench::BenchmarkSuite;
use hermes_simd::{ProcessorBinding, ProcessorIndex};

/// Powers of two from the first length past the performance core's L2 at
/// `f64` (65536 is 1 MB of samples) to the first past the whole L3 slice
/// budget the probe replicates within.
const LARGE_SIZE_CASES: [usize; 4] = [65_536, 131_072, 262_144, 524_288];

#[test]
#[ignore = "measurement instrument for the lengths past the caches"]
fn large_sizes_against_the_references_by_core_type() {
    let Some(selection) = measurement_cores::selected() else {
        eprintln!("host reports no processor class information; probe not measurable");
        return;
    };
    print!("{}", selection.describe());
    for core in selection.cores() {
        let cpu = core.processor().get();
        let _binding = ProcessorBinding::bind(core.processor())
            .expect("measurement processor must be available");
        std::thread::yield_now();
        let landed = ProcessorIndex::current()
            .expect("Windows supports processor queries")
            .get();
        assert_eq!(landed, cpu, "processor binding must remain exact");
        let core = core.label();
        // Discarded pass: see `half_storage_promotion_cost_by_core_type`.
        let mut warmup = BenchmarkSuite::new(sweep_warm_up_config());
        sizes_for_scalar::<f64>(&mut warmup, core, "f64", &LARGE_SIZE_CASES);
        sizes_for_scalar::<f32>(&mut warmup, core, "f32", &LARGE_SIZE_CASES);
        drop(warmup);
        let mut suite = BenchmarkSuite::new(sweep_config());
        sizes_for_scalar::<f64>(&mut suite, core, "f64", &LARGE_SIZE_CASES);
        sizes_for_scalar::<f32>(&mut suite, core, "f32", &LARGE_SIZE_CASES);
        println!("LRG cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}
