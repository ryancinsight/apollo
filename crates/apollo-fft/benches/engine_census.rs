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
//! ## The real arms compare like for like
//!
//! `fft_1d_slice` returns all `n` bins; RealFFT returns `n/2 + 1`. A real
//! signal's spectrum is conjugate-symmetric, so the upper half is redundant —
//! which means the original pairing charged Apollo for a mirror pass and twice
//! the output storage that RealFFT never performed, and then reported the
//! difference as a throughput gap. It was measuring a contract difference.
//!
//! So the comparison now runs `real_half_forward_f64`, where both engines
//! produce `n/2 + 1` bins into caller-owned storage. `real_full_forward_f64`
//! keeps Apollo's full-spectrum form beside it, unpaired: the distance between
//! the two Apollo rows *is* the cost of materializing the redundant half, which
//! is worth seeing rather than hiding.
//!
//! ## Interpreting the allocation column
//!
//! Allocations are counted by a wrapping global allocator, so the figure is
//! calls and bytes actually requested. Apollo's 1-D complex path should report
//! zero — plan, twiddles, and scratch are all cached — and its real path one,
//! the returned spectrum. A rise in either is a regression whatever the timings
//! say.
//!
//! The multidimensional paths also report zero warm allocations after the
//! Moirai indexed-scope state became stack-borrowed. Any non-zero result is a
//! regression: plans, twiddles, transpose scratch, and scheduler bookkeeping
//! are all reused.
//!
//! ## Why there is a multidimensional section
//!
//! The first census contained only 1-D arms, while the batched four-step layout
//! was initially reachable only from 2-D and 3-D lane transforms. Standalone
//! 1-D plans now enter the generic four-step route at 65536, but they still do
//! not exercise the lower-size batched layout or multidimensional transpose
//! passes. The 2-D shapes therefore remain necessary to measure those paths,
//! and the 3-D row pins the corresponding multi-plane Leto assignment and
//! allocation contract.
//!
//! Which shape takes which route was established by instrumenting
//! `four_step_fft` and `batched_four_step_applies` and reading the calls, not by
//! inferring it from the gate conditions — inference gave the wrong answer twice
//! while this was being worked out.
//!
//! ## Runtime budget
//!
//! ```text
//! per case  = WARM_UP_MS + MEASUREMENT_MS  = 20 ms + 60 ms = 80 ms
//! 1-D cases = 6 per size x 5 sizes                         = 30
//! 2-D cases = 2 per shape x 4 shapes                       =  8
//! 3-D cases = 1                                             =  1
//! total     = 39 x 80 ms                   ~ 3.1 s, plus flush overhead
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
use leto::{Array2, Array3};
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex as RustComplex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Hard wall-clock bound for the sweep.
const BUDGET_SECS: u64 = 60;
const WARM_UP_MS: u64 = 20;
const MEASUREMENT_MS: u64 = 60;

/// Powers of two spanning L1-resident through past last-level cache, which is
/// where the engines' relative standing changes.
const SIZES: [usize; 5] = [1_024, 4_096, 16_384, 65_536, 262_144];

/// Two-dimensional shapes, chosen so the census actually exercises the batched
/// layout rather than reporting a null result.
///
/// | shape | route taken by the `nx` axis |
/// | --- | --- |
/// | 4096 x 16, 4096 x 64, 16384 x 16 | four-step on the **batched** layout |
/// | 65536 x 4 | four-step with threaded row transforms |
///
/// The batched gate admits square splits with an even `log2` from 4 up to the
/// threading threshold, so 65536 falls past it and gives the batched shapes a
/// same-engine contrast at an equal element count. Three of the four hold
/// 262144 elements for that reason.
const TWO_D_SHAPES: [(usize, usize); 4] = [(4_096, 16), (4_096, 64), (16_384, 16), (65_536, 4)];

/// Three-dimensional shape with a 4096-point X axis and sixteen independent
/// X lanes. It exercises both the volume-wide X transpose and 4096 adjacent
/// 4x4 Y/Z planes while retaining the cache-resident 65,536-element regime.
const THREE_D_SHAPE: [usize; 3] = [4_096, 4, 4];

/// Tile edge for [`transpose_into`], matching the kernel's own choice.
const TRANSPOSE_TILE: usize = 32;

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

/// Transposes a `rows x cols` plane into a `cols x rows` one, in tiles.
///
/// Tiled deliberately. Apollo's 2-D path transposes internally, so a composed
/// reference built on a naive strided copy would measure transpose quality
/// rather than transform throughput, and would flatter Apollo for a reason that
/// has nothing to do with the layout under test.
fn transpose_into<T: Copy>(src: &[T], dst: &mut [T], rows: usize, cols: usize) {
    for row_block in (0..rows).step_by(TRANSPOSE_TILE) {
        let row_end = (row_block + TRANSPOSE_TILE).min(rows);
        for col_block in (0..cols).step_by(TRANSPOSE_TILE) {
            let col_end = (col_block + TRANSPOSE_TILE).min(cols);
            for r in row_block..row_end {
                for c in col_block..col_end {
                    dst[c * rows + r] = src[r * cols + c];
                }
            }
        }
    }
}

/// Composed 2-D forward transform for RustFFT, which has no 2-D planner.
///
/// Rows, transpose, columns, transpose back — the same algorithm shape Apollo's
/// 2-D path runs internally, so the two are comparable. `process_with_scratch`
/// rather than `process` because the latter allocates a scratch buffer per call,
/// which would charge RustFFT for an allocation the comparison is not about.
fn rustfft_2d(
    data: &mut [RustComplex<f64>],
    plane: &mut [RustComplex<f64>],
    scratch: &mut [RustComplex<f64>],
    nx: usize,
    ny: usize,
    along_ny: &Arc<dyn Fft<f64>>,
    along_nx: &Arc<dyn Fft<f64>>,
) {
    along_ny.process_with_scratch(data, scratch);
    transpose_into(data, plane, nx, ny);
    along_nx.process_with_scratch(plane, scratch);
    transpose_into(plane, data, ny, nx);
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
        const REAL_HALF: &str = "real_half_forward_f64";
        const REAL_FULL: &str = "real_full_forward_f64";

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

        // Both engines write n/2 + 1 bins into storage the caller already owns,
        // so neither is charged for an allocation or for redundant output.
        let mut half_out = vec![Complex64::default(); n / 2 + 1];
        flush_cache(&mut flush);
        suite.run_with_config(config, BenchmarkCase::new(REAL_HALF, "apollo", n), || {
            apollo_fft::fft_1d_slice_half_into::<f64>(
                black_box(&real_src),
                black_box(&mut half_out),
            );
            black_box(&half_out);
        });

        let mut rf_in = r2c.make_input_vec();
        let mut rf_out = r2c.make_output_vec();
        flush_cache(&mut flush);
        suite.run_with_config(config, BenchmarkCase::new(REAL_HALF, "realfft", n), || {
            rf_in.copy_from_slice(&real_src);
            r2c.process(black_box(&mut rf_in), black_box(&mut rf_out))
                .expect("realfft length agrees with the plan");
            black_box(&rf_out);
        });

        // Unpaired on purpose: the distance from the Apollo row above is the
        // cost of materializing the redundant half plus its allocation.
        flush_cache(&mut flush);
        suite.run_with_config(config, BenchmarkCase::new(REAL_FULL, "apollo", n), || {
            black_box(apollo_fft::fft_1d_slice::<f64>(black_box(&real_src)));
        });

        // Allocation is a property of the call, not of the machine, so it is
        // reported once per size rather than timed.
        let (complex_allocs, complex_bytes) = count_allocations(|| {
            work.copy_from_slice(&src);
            apollo.forward_complex_slice_inplace(&mut work);
        });
        let (real_allocs, real_bytes) =
            count_allocations(|| drop(apollo_fft::fft_1d_slice::<f64>(&real_src)));
        let (half_allocs, half_bytes) = count_allocations(|| {
            apollo_fft::fft_1d_slice_half_into::<f64>(&real_src, &mut half_out);
        });
        eprintln!(
            "engine_census: N={n:<8} apollo allocations/call — complex {complex_allocs} \
             ({complex_bytes} B), real-full {real_allocs} ({real_bytes} B), real-half {half_allocs} ({half_bytes} B)"
        );
    }

    // ---------------------------------------------------------------------
    // Two-dimensional arms: the only route that reaches the batched layout.
    // ---------------------------------------------------------------------
    for &(nx, ny) in &TWO_D_SHAPES {
        const TWO_D: &str = "complex_forward_2d_f64";
        let total = nx * ny;
        let shape = format!("{nx}x{ny}");

        let src = complex_signal(total);
        let rust_src: Vec<RustComplex<f64>> =
            src.iter().map(|v| RustComplex::new(v.re, v.im)).collect();

        let apollo = apollo_fft::FftPlan2D::<f64>::new(apollo_fft::Shape2D { nx, ny });
        let mut plane =
            Array2::from_shape_vec((nx, ny), src.clone()).expect("shape matches the data");

        flush_cache(&mut flush);
        suite.run_with_config(
            config,
            BenchmarkCase::new(TWO_D, "apollo", shape.clone()),
            || {
                plane
                    .as_slice_mut()
                    .expect("the plane stays contiguous")
                    .copy_from_slice(&src);
                apollo.forward_complex_inplace(black_box(&mut plane));
                black_box(&plane);
            },
        );

        let mut planner = FftPlanner::<f64>::new();
        let along_ny = planner.plan_fft_forward(ny);
        let along_nx = planner.plan_fft_forward(nx);
        let mut rust_work = rust_src.clone();
        let mut rust_plane = rust_src.clone();
        let mut rust_scratch = vec![
            RustComplex::new(0.0, 0.0);
            along_ny
                .get_inplace_scratch_len()
                .max(along_nx.get_inplace_scratch_len())
        ];

        flush_cache(&mut flush);
        suite.run_with_config(
            config,
            BenchmarkCase::new(TWO_D, "rustfft (composed)", shape.clone()),
            || {
                rust_work.copy_from_slice(&rust_src);
                rustfft_2d(
                    black_box(&mut rust_work),
                    &mut rust_plane,
                    &mut rust_scratch,
                    nx,
                    ny,
                    &along_ny,
                    &along_nx,
                );
                black_box(&rust_work);
            },
        );

        let (allocs, bytes) = count_allocations(|| {
            plane
                .as_slice_mut()
                .expect("the plane stays contiguous")
                .copy_from_slice(&src);
            apollo.forward_complex_inplace(&mut plane);
        });
        eprintln!("engine_census: 2-D {shape:<10} apollo allocations/call — {allocs} ({bytes} B)");
    }

    let [nx, ny, nz] = THREE_D_SHAPE;
    let total = nx * ny * nz;
    let shape = format!("{nx}x{ny}x{nz}");
    let src = complex_signal(total);
    let apollo = apollo_fft::FftPlan3D::<f64>::new(apollo_fft::Shape3D { nx, ny, nz });
    let mut volume =
        Array3::from_shape_vec([nx, ny, nz], src.clone()).expect("shape matches the data");

    flush_cache(&mut flush);
    suite.run_with_config(
        config,
        BenchmarkCase::new("complex_forward_3d_f64", "apollo", shape.clone()),
        || {
            volume
                .as_slice_mut()
                .expect("the volume stays contiguous")
                .copy_from_slice(&src);
            apollo.forward_complex_inplace(black_box(&mut volume));
            black_box(&volume);
        },
    );
    let (allocs, bytes) = count_allocations(|| {
        volume
            .as_slice_mut()
            .expect("the volume stays contiguous")
            .copy_from_slice(&src);
        apollo.forward_complex_inplace(&mut volume);
    });
    eprintln!("engine_census: 3-D {shape:<10} apollo allocations/call — {allocs} ({bytes} B)");

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
