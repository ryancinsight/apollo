//! The two recovery routes at fixed sparsity across lengths: the dense
//! `O(N log N)` route against the downsampled `O(K log K)` one (ADR 0064).
//!
//! A local instrument under a committed budget, never a CI job: the scaling
//! claim is read as the ratio's growth with `N` at fixed `K`, on the pinned
//! core class.

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkMode, BenchmarkSuite};
use apollo_sft::SparseFftPlan;
use eunomia::Complex64;
use std::f64::consts::TAU;
use std::hint::black_box;
use std::time::{Duration, Instant};

const BUDGET_SECS: u64 = 60;
const WARM_UP_MS: u64 = 20;
const MEASUREMENT_MS: u64 = 60;
const SPARSITY: usize = 16;
const LENGTHS: [usize; 5] = [1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20];

fn sparse_signal(n: usize, k: usize) -> Vec<Complex64> {
    let tones: Vec<(usize, Complex64)> = (0..k)
        .map(|i| {
            (
                (i * 7919 + 13) % n,
                Complex64::new(1.0 + i as f64 * 0.1, -0.3),
            )
        })
        .collect();
    (0..n)
        .map(|t| {
            tones
                .iter()
                .map(|&(f, a)| {
                    let angle = TAU * ((f * t) % n) as f64 / n as f64;
                    a * Complex64::new(angle.cos(), angle.sin())
                })
                .sum()
        })
        .collect()
}

fn main() -> Result<(), apollo_bench::BenchmarkError> {
    let started = Instant::now();
    let mode = BenchmarkMode::from_environment()?;
    let config = mode.apply(
        BenchmarkConfig::try_with_budgets(
            Duration::from_millis(WARM_UP_MS),
            Duration::from_millis(MEASUREMENT_MS),
        )
        .expect("invariant: benchmark duration constants are non-zero"),
    );
    let mut suite = BenchmarkSuite::new(config);
    for &n in &LENGTHS {
        let signal = sparse_signal(n, SPARSITY);
        let dense = SparseFftPlan::new(n, SPARSITY).expect("plan");
        let routed = SparseFftPlan::downsampled(n, SPARSITY).expect("plan");
        suite.run_with_config(
            config,
            BenchmarkCase::new("sparse_recovery_k16", "dense top-k", n),
            || {
                black_box(dense.forward(black_box(&signal)).expect("dense route"));
            },
        );
        suite.run_with_config(
            config,
            BenchmarkCase::new("sparse_recovery_k16", "downsampled", n),
            || {
                black_box(
                    routed
                        .forward(black_box(&signal))
                        .expect("exactly sparse input resolves"),
                );
            },
        );
    }
    print!("{}", suite.report());
    let elapsed = started.elapsed();
    eprintln!("recovery: completed in {:.2}s", elapsed.as_secs_f64());
    assert!(
        elapsed < Duration::from_secs(BUDGET_SECS),
        "recovery exceeded its {BUDGET_SECS}s budget ({:.2}s)",
        elapsed.as_secs_f64()
    );
    Ok(())
}
