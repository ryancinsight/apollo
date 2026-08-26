//! Four-engine census: Apollo against RustFFT, PhastFT, and RealFFT.
//!
//! ## Why this exists separately from the other comparison benches
//!
//! `rustfft_comparison` sweeps arbitrary lengths and feeds
//! `docs/benchmark_results.md`; `phastft_comparison` covers the power-of-two
//! domain against one engine. This one measures the axes that actually decide
//! throughput here — arithmetic rate and **transient allocation per call** —
//! across every reference at once, so a change that trades one for the other is
//! visible rather than inferred.
//!
//! ## Cache flushing between arms
//!
//! Each arm runs against a cache left cold by the previous arm's flush, not warm
//! from the previous arm's data. That is not fastidiousness: while this
//! comparison was being developed, adding a fourth arm to the rotation moved
//! *Apollo's own* timing at 2^14 by a factor of two with its code untouched,
//! because the arms were warming each other's working sets. Two successive runs
//! then reported 2.33x and 0.81x for the same comparison. [`flush_cache`] exists
//! so that result cannot recur.
//!
//! Flushing costs wall-clock and is charged to no arm, since it happens outside
//! every timed region.
//!
//! ## Interpreting the allocation column
//!
//! Allocations are counted by a wrapping global allocator, so the figure is
//! calls and bytes actually requested. Apollo's complex path should report zero
//! — plan, twiddles, and scratch are all cached — and its real path one, the
//! returned spectrum. A rise in either is a regression whatever the timings say.
//!
//! ## Runtime budget
//!
//! ```text
//! per case  = WARM_UP_MS + MEASUREMENT_MS  = 20 ms + 60 ms = 80 ms
//! cases     = 6 per size (4 complex + 2 real)
//! sizes     = 5
//! total     = 5 x 6 x 80 ms                ~ 2.4 s, plus flush overhead
//! ```
//!
//! [`main`] exits non-zero if the sweep exceeds [`BUDGET_SECS`]. A breach is a
//! defect to root-cause, never fixed by raising the bound in the change that
//! caused it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use eunomia::Complex64;
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex as RustComplex;
use rustfft::FftPlanner;

/// Hard wall-clock bound for the sweep.
const BUDGET_SECS: u64 = 60;
const WARM_UP_MS: u64 = 20;
const MEASUREMENT_MS: u64 = 60;

/// Powers of two spanning L1-resident through past last-level cache, which is
/// where the engines' relative standing changes.
const SIZES: [usize; 5] = [1_024, 4_096, 16_384, 65_536, 262_144];

/// Bytes touched to evict the working set between arms. Sized past any
/// plausible last-level cache on a developer machine.
const FLUSH_BYTES: usize = 64 << 20;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicBool = AtomicBool::new(false);

struct Counting;

// SAFETY: every method forwards to `System` unchanged; the counters observe and
// never affect the returned pointer.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(l.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(new, Ordering::Relaxed);
        }
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Evicts the caches so the next arm starts cold, as the previous one did.
///
/// The buffer is written, not just read, so it dirties the lines and they
/// cannot be silently dropped; `black_box` keeps the loop from being elided.
fn flush_cache(buffer: &mut [u8]) {
    for chunk in buffer.chunks_mut(64) {
        chunk[0] = chunk[0].wrapping_add(1);
    }
    black_box(&buffer[0]);
}

/// Counts allocations across one call, after a warm-up call that pays any
/// one-time cache fills so they are not charged to the measurement.
fn count_allocations<F: FnMut()>(mut f: F) -> (usize, usize) {
    f();
    ALLOCS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    f();
    COUNTING.store(false, Ordering::Relaxed);
    (
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}

fn complex_signal(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

fn main() -> Result<(), apollo_bench::BenchmarkConfigError> {
    let started = Instant::now();
    let config = BenchmarkConfig::try_with_budgets(
        Duration::from_millis(WARM_UP_MS),
        Duration::from_millis(MEASUREMENT_MS),
    )?;
    let mut suite = BenchmarkSuite::new(config);
    let mut flush = vec![0u8; FLUSH_BYTES];

    eprintln!(
        "engine_census: {} sizes, warm-up {WARM_UP_MS}ms, measurement {MEASUREMENT_MS}ms, \
         cache flushed between arms, budget {BUDGET_SECS}s (hard)",
        SIZES.len()
    );

    for &n in &SIZES {
        const COMPLEX: &str = "complex_forward_f64";
        const REAL: &str = "real_forward_f64";

        let src = complex_signal(n);
        let rust_src: Vec<RustComplex<f64>> =
            src.iter().map(|v| RustComplex::new(v.re, v.im)).collect();
        let (re_src, im_src): (Vec<f64>, Vec<f64>) = src.iter().map(|v| (v.re, v.im)).unzip();
        let real_src: Vec<f64> = re_src.clone();

        let apollo = apollo_fft::FftPlan1D::<f64>::new(apollo_fft::Shape1D { n });
        let rust = FftPlanner::<f64>::new().plan_fft_forward(n);
        let phast = phastft::planner::PlannerDit64::new(n);
        let mut real_planner = RealFftPlanner::<f64>::new();
        let r2c = real_planner.plan_fft_forward(n);

        let mut work = src.clone();
        flush_cache(&mut flush);
        suite.run_with_config(config, BenchmarkCase::new(COMPLEX, "apollo", n), || {
            work.copy_from_slice(&src);
            apollo.forward_complex_slice_inplace(black_box(&mut work));
            black_box(&work);
        });

        let mut rust_work = rust_src.clone();
        flush_cache(&mut flush);
        suite.run_with_config(config, BenchmarkCase::new(COMPLEX, "rustfft", n), || {
            rust_work.copy_from_slice(&rust_src);
            rust.process(black_box(&mut rust_work));
            black_box(&rust_work);
        });

        let (mut re, mut im) = (re_src.clone(), im_src.clone());
        flush_cache(&mut flush);
        suite.run_with_config(config, BenchmarkCase::new(COMPLEX, "phastft", n), || {
            re.copy_from_slice(&re_src);
            im.copy_from_slice(&im_src);
            phastft::fft_f64_dit_with_planner(
                black_box(&mut re),
                black_box(&mut im),
                phastft::planner::Direction::Forward,
                &phast,
            );
            black_box(&re);
        });

        flush_cache(&mut flush);
        suite.run_with_config(config, BenchmarkCase::new(REAL, "apollo", n), || {
            black_box(apollo_fft::fft_1d_slice::<f64>(black_box(&real_src)));
        });

        let mut rf_in = r2c.make_input_vec();
        let mut rf_out = r2c.make_output_vec();
        flush_cache(&mut flush);
        suite.run_with_config(config, BenchmarkCase::new(REAL, "realfft", n), || {
            rf_in.copy_from_slice(&real_src);
            r2c.process(black_box(&mut rf_in), black_box(&mut rf_out))
                .expect("realfft length agrees with the plan");
            black_box(&rf_out);
        });

        // Allocation is a property of the call, not of the machine, so it is
        // reported once per size rather than timed.
        let (complex_allocs, complex_bytes) = count_allocations(|| {
            work.copy_from_slice(&src);
            apollo.forward_complex_slice_inplace(&mut work);
        });
        let (real_allocs, real_bytes) =
            count_allocations(|| drop(apollo_fft::fft_1d_slice::<f64>(&real_src)));
        eprintln!(
            "engine_census: N={n:<8} apollo allocations/call — complex {complex_allocs} \
             ({complex_bytes} B), real {real_allocs} ({real_bytes} B)"
        );
    }

    suite.emit();
    let elapsed = started.elapsed();
    eprintln!("engine_census: completed in {:.2}s", elapsed.as_secs_f64());
    assert!(
        elapsed < Duration::from_secs(BUDGET_SECS),
        "engine_census exceeded its {BUDGET_SECS}s budget ({:.2}s). Root-cause the \
         slowdown; do not raise the bound in the change that caused it.",
        elapsed.as_secs_f64()
    );
    Ok(())
}
