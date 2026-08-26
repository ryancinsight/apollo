//! Derives the one-dimensional four-step crossover by measurement.
//!
//! # Why this lives in the crate rather than in `benches/`
//!
//! It has to call both routes at the same length, and [`PotRoute`] is
//! `pub(crate)`. A bench is a separate crate, so reaching the routes from one
//! would mean making them public solely for measurement — a test-only public
//! item, which this codebase does not permit. So the instrument lives beside
//! the thing it measures and is `#[ignore]`d: it asserts nothing and is a
//! measurement, not a correctness check.
//!
//! # Why it exists at all
//!
//! Two instruments previously disagreed by an order of magnitude about where
//! this crossover sits, and ADR 0039 recorded the crossover without naming the
//! instrument that produced it. The disagreement is explicable: each measured
//! one route in one process against the other route in a *different* process,
//! and the between-process difference — cache state, address layout, turbo
//! residency — was larger than the between-route difference it was trying to
//! resolve.
//!
//! This measures both routes at one length in one process. Two properties make
//! that worth something:
//!
//! - **The cache is flushed before every arm**, so no arm inherits the
//!   previous one's working set. Developing the four-engine census, adding a
//!   fourth arm moved *Apollo's own* figure by a factor of two with its code
//!   untouched, purely through inter-arm cache state.
//! - **Arm order alternates** between repetitions. Flushing removes the shared
//!   working set; alternating removes what it cannot, since whichever arm runs
//!   first still pays for bringing its own input back. A fixed order silently
//!   charges that to one route.
//!
//! Run it with:
//!
//! ```text
//! cargo test --release -p apollo-fft --lib crossover -- --ignored --nocapture
//! ```

use super::route::{FourStep, PotRoute};
use super::strategies::StockhamAutosort;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::Complex64;
use std::hint::black_box;
use std::time::Instant;

/// Even `log2` only: the four-step route admits square splits, so odd lengths
/// have no second arm to compare against.
const LADDER: [u32; 9] = [4, 6, 8, 10, 12, 14, 16, 18, 20];

/// Repetitions per arm. The reported figure is the minimum, which is the
/// least-disturbed observation rather than an average over disturbances.
const REPS: usize = 7;

/// Sized past any plausible last-level cache on a developer machine.
const FLUSH_BYTES: usize = 64 << 20;

fn flush(buffer: &mut [u8]) {
    for chunk in buffer.chunks_mut(64) {
        chunk[0] = chunk[0].wrapping_add(1);
    }
    black_box(&buffer[0]);
}

fn signal(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

/// Minimum wall clock over `REPS` runs of one route at one length.
fn measure<R: PotRoute>(
    source: &[Complex64],
    work: &mut [Complex64],
    twiddles: &[Complex64],
    flush_buffer: &mut [u8],
) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        work.copy_from_slice(source);
        flush(flush_buffer);
        let start = Instant::now();
        R::run::<f64, false, false>(black_box(work), twiddles);
        let elapsed = start.elapsed().as_nanos() as f64;
        black_box(&work[0]);
        best = best.min(elapsed);
    }
    best
}

#[test]
#[ignore = "measurement instrument; derives the crossover ADR 0039 records"]
fn derive_one_dimensional_four_step_crossover() {
    let mut flush_buffer = vec![0u8; FLUSH_BYTES];
    println!(
        "{:>10}  {:>14}  {:>14}  {:>9}  faster",
        "N", "stockham ns", "four-step ns", "four/stock"
    );

    let mut crossover: Option<usize> = None;
    let mut all_above_favour_four_step = true;

    for &log2 in &LADDER {
        let n = 1usize << log2;
        let source = signal(n);
        let mut work = source.clone();
        let twiddles = <f64 as MixedRadixScalar>::cached_twiddle_fwd(n);

        // Alternate which route runs first so neither is charged for being the
        // one that reloads its input after the flush.
        let (stockham, four_step) = if log2 % 4 == 0 {
            let s = measure::<StockhamAutosort>(&source, &mut work, &twiddles, &mut flush_buffer);
            let f = measure::<FourStep>(&source, &mut work, &twiddles, &mut flush_buffer);
            (s, f)
        } else {
            let f = measure::<FourStep>(&source, &mut work, &twiddles, &mut flush_buffer);
            let s = measure::<StockhamAutosort>(&source, &mut work, &twiddles, &mut flush_buffer);
            (s, f)
        };

        let ratio = four_step / stockham;
        let winner = if ratio < 1.0 { "four-step" } else { "stockham" };
        println!("{n:>10}  {stockham:>14.0}  {four_step:>14.0}  {ratio:>9.3}  {winner}");

        if ratio < 1.0 {
            if crossover.is_none() {
                crossover = Some(n);
            }
        } else if crossover.is_some() {
            // Four-step led at a smaller length and lost again here, so the
            // advantage is not monotone and a single threshold cannot express
            // it. Worth knowing rather than averaging away.
            all_above_favour_four_step = false;
        }
    }

    println!();
    match crossover {
        Some(n) if all_above_favour_four_step => println!(
            "\nderived crossover: four-step wins from N = {n} upward, monotonically.\n\
             ONE_DIMENSIONAL_FOUR_STEP_THRESHOLD should equal {n}; it is currently {}.",
            crate::application::execution::kernel::tuning::ONE_DIMENSIONAL_FOUR_STEP_THRESHOLD
        ),
        Some(n) => println!(
            "\nfour-step first wins at N = {n} but loses again above it, so the advantage\n\
             is not monotone and one threshold cannot express it. Record the ladder."
        ),
        None => println!("\nno crossover on this ladder: stockham wins at every measured length."),
    }
}
