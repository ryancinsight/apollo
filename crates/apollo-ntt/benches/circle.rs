//! The circle FFT against the cyclic NTT across power-of-two lengths (ADR
//! 0065): a local instrument under a committed budget, read as the growth of
//! time per `N log N` with `N`.

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkMode, BenchmarkSuite};
use apollo_ntt::{CircleNttPlan, NttPlan};
use leto::Array1;
use std::hint::black_box;
use std::time::{Duration, Instant};

const BUDGET_SECS: u64 = 60;
const WARM_UP_MS: u64 = 20;
const MEASUREMENT_MS: u64 = 60;
const LENGTHS: [usize; 5] = [1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18];

fn residues(n: usize, modulus: u64) -> Vec<u64> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % modulus
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
        let circle = CircleNttPlan::new(n).expect("Mersenne31 supports the length");
        let cyclic = NttPlan::new(n).expect("the default prime supports the length");
        let circle_input = Array1::from(residues(n, circle.modulus()));
        let cyclic_input = Array1::from(residues(n, cyclic.modulus()));
        let mut work = Array1::from(vec![0u64; n]);
        suite.run_with_config(
            config,
            BenchmarkCase::new("ntt_forward", "circle (Mersenne31)", n),
            || {
                circle
                    .forward_into(black_box(&circle_input), &mut work)
                    .expect("length matches the plan");
                black_box(&work);
            },
        );
        suite.run_with_config(
            config,
            BenchmarkCase::new("ntt_forward", "cyclic (998244353)", n),
            || {
                cyclic
                    .forward_into(black_box(&cyclic_input), &mut work)
                    .expect("length matches the plan");
                black_box(&work);
            },
        );
    }
    print!("{}", suite.report());
    let elapsed = started.elapsed();
    eprintln!("circle: completed in {:.2}s", elapsed.as_secs_f64());
    assert!(
        elapsed < Duration::from_secs(BUDGET_SECS),
        "circle exceeded its {BUDGET_SECS}s budget ({:.2}s)",
        elapsed.as_secs_f64()
    );
    Ok(())
}
