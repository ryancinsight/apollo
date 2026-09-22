//! The composite and prime lengths against the references at both scalars,
//! under the same per-case budget as the power-of-two sweeps beside it.
//!
//! These lengths reach the mixed-radix and Rader routes rather than the base
//! and block forms, and they are the only shard with no PhastFT arm at all:
//! its DIT planner is power-of-two only, so at 100, 180, 384, 1000, 101 and
//! 1009 the comparison is against RustFFT alone. Two families share the
//! shard because they share that property and their per-case cost — the
//! composite splits (100, 180, 384, 1000) and the primes whose Rader
//! transform is itself a composite route (101, 1009).
//!
//! Splitting them out of the small sweep is what keeps every shard inside
//! the runner's 300-second campaign budget with room to replicate: this one
//! is forty-eight cases, so five runs cost well under a minute of
//! measurement, and a change to the composite or Rader routes re-measures
//! here without paying for the power-of-two ladder.

use super::small_sizes::{sizes_for_scalar, sweep_config, sweep_warm_up_config};
use crate::application::execution::kernel::measurement_cores;
use apollo_bench::BenchmarkSuite;
use hermes_simd::{ProcessorBinding, ProcessorIndex};

/// The composite splits and the Rader primes the scoreboard tracks: one
/// smooth length per decade (100, 1000), the two composites whose column
/// routes carry radices 3 and 5 (180, 384), and a prime at each decade
/// (101, 1009).
const MIXED_RADIX_SIZE_CASES: [usize; 6] = [100, 180, 384, 1_000, 101, 1_009];

#[test]
#[ignore = "measurement instrument for the composite and Rader routes"]
fn mixed_radix_sizes_against_the_references_by_core_type() {
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
        sizes_for_scalar::<f64>(&mut warmup, core, "f64", &MIXED_RADIX_SIZE_CASES);
        sizes_for_scalar::<f32>(&mut warmup, core, "f32", &MIXED_RADIX_SIZE_CASES);
        drop(warmup);
        let mut suite = BenchmarkSuite::new(sweep_config());
        sizes_for_scalar::<f64>(&mut suite, core, "f64", &MIXED_RADIX_SIZE_CASES);
        sizes_for_scalar::<f32>(&mut suite, core, "f32", &MIXED_RADIX_SIZE_CASES);
        println!("MIX cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}
