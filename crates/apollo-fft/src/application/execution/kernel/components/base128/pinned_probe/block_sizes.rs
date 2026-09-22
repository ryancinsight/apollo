//! The cache-resident block routes — 2048 to 32768 — against the references
//! at both scalars, under the same per-case budget as the sweeps beside it.
//!
//! These are the lengths the column-first block form covers (ADR 0061): a
//! base of 256 or 512 under one or two column passes, selected by the plan
//! per width. They sit above the base routes the small sweep measures and
//! below the lengths past the caches, so they move for their own reasons —
//! the block count, the twiddle stream per pass, and where the working set
//! falls against L1 and L2 — and a campaign that re-measures a change to the
//! column passes wants these five lengths and nothing else.
//!
//! Splitting them out of the small sweep is what keeps every shard inside
//! the runner's 300-second campaign budget with room to replicate: this one
//! is sixty cases, a quarter of the sweep it came from, so five runs cost
//! under a minute of measurement rather than three.

use super::small_sizes::{sizes_for_scalar, sweep_config, sweep_warm_up_config};
use super::split_attribution;
use crate::application::execution::kernel::measurement_cores;
use apollo_bench::BenchmarkSuite;
use eunomia::Complex64;
use hermes_simd::{ProcessorBinding, ProcessorIndex};

/// Powers of two from the first length the block form covers to the last one
/// resident in the performance core's L2 at `f64` (32768 is 512 KB of
/// samples); 65536 upward runs in the large-sizes sweep.
const BLOCK_SIZE_CASES: [usize; 5] = [2_048, 4_096, 8_192, 16_384, 32_768];

#[test]
#[ignore = "measurement instrument for the column-first block routes"]
fn block_sizes_against_the_references_by_core_type() {
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
        sizes_for_scalar::<f64>(&mut warmup, core, "f64", &BLOCK_SIZE_CASES);
        sizes_for_scalar::<f32>(&mut warmup, core, "f32", &BLOCK_SIZE_CASES);
        drop(warmup);
        let mut suite = BenchmarkSuite::new(sweep_config());
        sizes_for_scalar::<f64>(&mut suite, core, "f64", &BLOCK_SIZE_CASES);
        sizes_for_scalar::<f32>(&mut suite, core, "f32", &BLOCK_SIZE_CASES);
        {
            let n = 2048usize;
            let src: Vec<Complex64> = (0..n)
                .map(|i| {
                    let x = i as f64;
                    Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
                })
                .collect();
            let mut work = src.clone();
            let state = super::instance_major::State512::<f64>::new_if_supported(n)
                .expect("the pinned host must provide a native base capability");
            let split = split_attribution(&src, &mut work, |work| {
                super::transform_via_base_512::<f64, false, true>(work, &state)
            });
            println!(
                "B512 split n={n}: gather={} blocks={} levels={} total={} | per block ({}): load_and_rows={} columns_and_sink={}",
                split.gather,
                split.blocks,
                split.levels,
                split.gather + split.blocks + split.levels,
                split.blocks_per_call,
                split.rows,
                split.columns
            );
        }
        println!("BLK cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}
