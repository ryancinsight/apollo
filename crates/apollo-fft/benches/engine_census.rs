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
//! The peak-working-set section that precedes the timed sweep is allocation
//! accounting, not timing — a plan build and three transforms per engine and
//! size, roughly a second in total, charged before `started` so the budget
//! above is unaffected.
//!
//! [`main`] exits non-zero if the sweep exceeds [`BUDGET_SECS`]. A breach is a
//! defect to root-cause, never fixed by raising the bound in the change that
//! caused it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkMode, BenchmarkSuite};
use eunomia::Complex64;
use leto::{Array2, Array3, ArrayViewMut2, ArrayViewMut3, Layout as ArrayLayout};
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
/// Live-bytes balance inside a counting window. Signed because a window may
/// observe the free of an allocation made before it opened; the peak is only
/// read as a high-water mark, so a transiently negative balance is harmless.
static LIVE: AtomicIsize = AtomicIsize::new(0);
/// High-water mark of [`LIVE`] inside the window.
static PEAK: AtomicIsize = AtomicIsize::new(0);

struct Counting;

fn track_live(delta: isize) {
    let live = LIVE.fetch_add(delta, Ordering::Relaxed) + delta;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

// SAFETY: every method forwards to `System` unchanged; the counters observe and
// never affect the returned pointer.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(l.size(), Ordering::Relaxed);
            track_live(l.size() as isize);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if COUNTING.load(Ordering::Relaxed) {
            track_live(-(l.size() as isize));
        }
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(new, Ordering::Relaxed);
            track_live(new as isize - l.size() as isize);
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

/// Peak and end-of-window live bytes across one closure.
///
/// Opens a fresh counting window (live balance and peak reset to zero), runs
/// `f`, and returns `(peak_bytes, retained_bytes, result)` — the high-water
/// mark of live bytes inside the window and the balance still allocated when
/// it closed. The returned value keeps whatever `f` built (a plan) alive, so
/// `retained` includes it plus any process-global caches it populated.
fn peak_live_bytes<R>(f: impl FnOnce() -> R) -> (isize, isize, R) {
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let result = f();
    COUNTING.store(false, Ordering::Relaxed);
    (
        PEAK.load(Ordering::Relaxed),
        LIVE.load(Ordering::Relaxed),
        result,
    )
}

/// Peak working set per engine across the power-of-two ladder
/// (`ATLAS-APOLLO-PEAK-MEMORY-2026-08-25`).
///
/// Method: the wrapping global allocator tracks a live-bytes balance and its
/// high-water mark inside explicit windows. Per engine and size, the **cold**
/// window covers plan construction plus one forward transform — so it charges
/// twiddle tables, scratch, and any process-global caches at their first
/// touch — and reports its peak and its retained (still-live) balance; the
/// **warm** window covers one further call on the built plan and reports its
/// peak, which is the steady-state transient footprint. Signal buffers are
/// allocated outside the windows: every engine's contract leaves them
/// caller-owned, and they cost the same 16 bytes per element in each layout
/// (interleaved complex vs split re/im planes).
///
/// Allocation counts are exact regardless of host load, so unlike the timing
/// census this section needs no quiet machine. Run it alone with
/// `APOLLO_PEAK_WORKING_SET_ONLY=1`. Ordering matters within a process: each
/// size's cold window must be that size's first touch, so this runs before the
/// timing suites.
fn peak_working_set_census() {
    println!("== peak working set (bytes) ==");
    println!(
        "{:>8}  {:>10} {:>12} {:>12} {:>10}   engine",
        "n", "cold peak", "retained", "warm peak", "16n"
    );
    for &n in &SIZES {
        let src = complex_signal(n);
        let rust_src: Vec<RustComplex<f64>> =
            src.iter().map(|v| RustComplex::new(v.re, v.im)).collect();
        let (re_src, im_src): (Vec<f64>, Vec<f64>) = src.iter().map(|v| (v.re, v.im)).unzip();

        let mut work = src.clone();
        let (peak, retained, apollo) = peak_live_bytes(|| {
            let plan = apollo_fft::FftPlan1D::<f64>::new(
                apollo_fft::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
            );
            plan.forward_complex_slice_inplace(black_box(&mut work));
            plan
        });
        let (warm_peak, _, ()) = peak_live_bytes(|| {
            work.copy_from_slice(&src);
            apollo.forward_complex_slice_inplace(black_box(&mut work));
        });
        drop(apollo);
        println!(
            "{n:>8}  {peak:>10} {retained:>12} {warm_peak:>12} {:>10}   apollo",
            n * 16
        );

        let mut rust_work = rust_src.clone();
        let (peak, retained, rust) = peak_live_bytes(|| {
            let plan = FftPlanner::<f64>::new().plan_fft_forward(n);
            plan.process(black_box(&mut rust_work));
            plan
        });
        let (warm_peak, _, ()) = peak_live_bytes(|| {
            rust_work.copy_from_slice(&rust_src);
            rust.process(black_box(&mut rust_work));
        });
        drop(rust);
        println!(
            "{n:>8}  {peak:>10} {retained:>12} {warm_peak:>12} {:>10}   rustfft",
            n * 16
        );

        let (mut re, mut im) = (re_src.clone(), im_src.clone());
        let (peak, retained, phast) = peak_live_bytes(|| {
            let planner = phastft::planner::PlannerDit64::new(n);
            phastft::fft_f64_dit_with_planner(
                black_box(&mut re),
                black_box(&mut im),
                phastft::planner::Direction::Forward,
                &planner,
            );
            planner
        });
        let (warm_peak, _, ()) = peak_live_bytes(|| {
            re.copy_from_slice(&re_src);
            im.copy_from_slice(&im_src);
            phastft::fft_f64_dit_with_planner(
                black_box(&mut re),
                black_box(&mut im),
                phastft::planner::Direction::Forward,
                &phast,
            );
        });
        drop(phast);
        println!(
            "{n:>8}  {peak:>10} {retained:>12} {warm_peak:>12} {:>10}   phastft",
            n * 16
        );
    }
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

/// Verifies the non-C-dense view boundary reuses its staging role after warm-up.
fn assert_staged_view_allocations() {
    let plan_2d = apollo_fft::FftPlan2D::<f64>::new(
        apollo_fft::Shape2D::new(3, 4).expect("invariant: shape lengths are non-zero"),
    );
    let layout_2d = ArrayLayout::f_contiguous([3, 4]).expect("valid Fortran layout");
    let mut storage_2d = complex_signal(12);
    let (allocs_2d, bytes_2d) = count_allocations(|| {
        let view = ArrayViewMut2::try_new(layout_2d, &mut storage_2d)
            .expect("2-D staging probe layout fits storage");
        plan_2d.forward_complex_leto_inplace(view);
    });
    assert_eq!(
        (allocs_2d, bytes_2d),
        (0, 0),
        "warmed Fortran-order 2-D execution must reuse view staging"
    );

    let plan_3d = apollo_fft::StaticFftPlan3D::<f64, 2, 3, 4>::new();
    let layout_3d =
        ArrayLayout::try_new([2, 3, 4], [40, 10, 2], 1).expect("valid strided 3-D layout");
    let mut storage_3d = vec![Complex64::default(); 68];
    let (allocs_3d, bytes_3d) = count_allocations(|| {
        let view = ArrayViewMut3::try_new(layout_3d, &mut storage_3d)
            .expect("3-D staging probe layout fits storage");
        plan_3d.forward_complex_leto_inplace(view);
    });
    assert_eq!(
        (allocs_3d, bytes_3d),
        (0, 0),
        "warmed strided 3-D execution must reuse view staging"
    );
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

/// Opts this process out of Windows power throttling (EcoQoS).
///
/// Retained as a measured no-op, not as a fix. The claim this once carried —
/// that Windows hands benchmark children EcoQoS and that this call is what
/// made the batched four-step drop from ~45 us to the pinned 13 to 17 us — is
/// withdrawn (ADR 0039, revision 2026-09-01). [`qos_placement_probe`] measured
/// the process's explicit throttling state as unset by default, unpinned
/// calls landing on all 24 processors with a performance-core bias rather
/// than "exclusively on E-cores", and this call changing neither median
/// latency (154.0 to 155.2 us) nor placement under the High performance plan.
/// The process-dependent anomalies in the audit trail are better explained by
/// an unpinned process sampling two core classes at a scheduler-chosen ratio
/// (`ATLAS-APOLLO-CENSUS-UNPINNED-BLEND-2026-09-01`).
///
/// It stays because the probe covered one power plan: EcoQoS heuristics are
/// strongest under Balanced or on battery, which this host cannot exercise.
/// Re-run the probe there before deleting or re-crediting this call.
#[cfg(windows)]
fn opt_out_of_power_throttling() {
    #[repr(C)]
    struct PowerThrottlingState {
        version: u32,
        control_mask: u32,
        state_mask: u32,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn SetProcessInformation(
            process: isize,
            class: i32,
            information: *mut core::ffi::c_void,
            size: u32,
        ) -> i32;
    }
    /// `ProcessPowerThrottling` in the `PROCESS_INFORMATION_CLASS` enumeration.
    const PROCESS_POWER_THROTTLING: i32 = 4;
    /// `PROCESS_POWER_THROTTLING_EXECUTION_SPEED`: naming it in the control
    /// mask while leaving it clear in the state mask disables throttling
    /// outright rather than leaving it to the scheduler.
    const EXECUTION_SPEED: u32 = 0x1;
    let mut state = PowerThrottlingState {
        version: 1,
        control_mask: EXECUTION_SPEED,
        state_mask: 0,
    };
    // SAFETY: documented Win32 call; the struct matches
    // PROCESS_POWER_THROTTLING_STATE and outlives the call.
    let ok = unsafe {
        SetProcessInformation(
            GetCurrentProcess(),
            PROCESS_POWER_THROTTLING,
            (&raw mut state).cast(),
            core::mem::size_of::<PowerThrottlingState>() as u32,
        )
    };
    // Surfaced rather than silent either way: a run that could not opt out is
    // a run whose numbers carry the throttling caveat.
    eprintln!(
        "engine_census: power throttling opt-out {}",
        if ok != 0 { "applied" } else { "REFUSED" }
    );
}

#[cfg(not(windows))]
fn opt_out_of_power_throttling() {}

/// Tests the premise behind [`opt_out_of_power_throttling`] instead of
/// assuming it (`ATLAS-APOLLO-ECOQOS-PREMISE-2026-09-01`).
///
/// ADR 0039's Context attributed a 3x per-call slowdown of the batched
/// four-step kernel in this process to Windows handing benchmark children
/// EcoQoS, on the observation that calls executed "on E-cores (CPUs 8 through
/// 21)". That range holds four of this host's eight performance cores, so it
/// does not establish efficiency-core placement. This records what the
/// narrative asserted but never measured: the explicit power-throttling state
/// the process carries, and the queried efficiency class of the processor each
/// unpinned call lands on, before and after the opt-out, with per-call latency
/// alongside. A probe, not a benchmark: medians over a bounded loop, no
/// harness, and no timing claim beyond the comparison it prints.
///
/// Absence stays typed. Landing comes from `themis::current_processor`, in
/// themis's own processor numbering, so no affinity convention is re-derived
/// here; a topology that reports no efficiency classes yields `unclassified`
/// counts rather than an invented label.
#[cfg(windows)]
fn qos_placement_probe() {
    use themis::CpuTopology;

    #[repr(C)]
    struct PowerThrottlingState {
        version: u32,
        control_mask: u32,
        state_mask: u32,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn GetProcessInformation(
            process: isize,
            class: i32,
            information: *mut core::ffi::c_void,
            size: u32,
        ) -> i32;
    }
    /// `ProcessPowerThrottling` in `PROCESS_INFORMATION_CLASS`; the read twin
    /// of the constant [`opt_out_of_power_throttling`] writes through.
    const PROCESS_POWER_THROTTLING: i32 = 4;
    const EXECUTION_SPEED: u32 = 0x1;
    /// At the ~45 us per call ADR 0039 reports, this is under a tenth of a
    /// second per phase, enough for a placement histogram over the loop.
    const CALLS: usize = 2_000;
    /// The first batched two-dimensional shape: the layout the claim names.
    const SHAPE: (usize, usize) = TWO_D_SHAPES[0];

    /// `(control_mask, state_mask)` of the process's explicit throttling
    /// override, or `None` when the query is refused. A zero control mask
    /// means the process has set nothing and the scheduler decides; that is
    /// evidence of neither throttled nor unthrottled execution.
    fn explicit_throttling_state() -> Option<(u32, u32)> {
        let mut state = PowerThrottlingState {
            version: 1,
            control_mask: 0,
            state_mask: 0,
        };
        // SAFETY: documented Win32 call; the struct matches
        // PROCESS_POWER_THROTTLING_STATE and outlives the call.
        let ok = unsafe {
            GetProcessInformation(
                GetCurrentProcess(),
                PROCESS_POWER_THROTTLING,
                (&raw mut state).cast(),
                core::mem::size_of::<PowerThrottlingState>() as u32,
            )
        };
        (ok != 0).then_some((state.control_mask, state.state_mask))
    }

    fn describe_state(state: Option<(u32, u32)>) -> String {
        match state {
            None => "query REFUSED".to_owned(),
            Some((control, st)) if (control & EXECUTION_SPEED) == 0 => format!(
                "control={control:#x} state={st:#x} (no explicit override: scheduler default)"
            ),
            Some((control, st)) if (st & EXECUTION_SPEED) != 0 => {
                format!("control={control:#x} state={st:#x} (explicitly THROTTLED)")
            }
            Some((control, st)) => {
                format!("control={control:#x} state={st:#x} (explicitly opted out)")
            }
        }
    }

    fn micros(d: Duration) -> f64 {
        d.as_secs_f64() * 1e6
    }

    let (nx, ny) = SHAPE;
    let topology = CpuTopology::detect();
    let class_count = topology
        .as_ref()
        .and_then(CpuTopology::efficiency_class_count);
    let src = complex_signal(nx * ny);
    let plan = apollo_fft::FftPlan2D::<f64>::new(
        apollo_fft::Shape2D::new(nx, ny).expect("invariant: shape lengths are non-zero"),
    );
    let mut plane = Array2::from_shape_vec((nx, ny), src.clone()).expect("shape matches the data");

    let mut phase = |label: &str| {
        let mut latencies = Vec::with_capacity(CALLS);
        let mut landed: Vec<Option<u32>> = Vec::with_capacity(CALLS);
        for _ in 0..CALLS {
            plane
                .as_slice_mut()
                .expect("the plane stays contiguous")
                .copy_from_slice(&src);
            let started = Instant::now();
            plan.forward_complex_inplace(std::hint::black_box(&mut plane));
            latencies.push(started.elapsed());
            std::hint::black_box(&plane);
            landed.push(themis::current_processor());
        }
        latencies.sort_unstable();
        let median = latencies[CALLS / 2];
        let p90 = latencies[CALLS * 9 / 10];

        let mut per_rank: Vec<usize> = vec![0; class_count.unwrap_or(0)];
        let mut unclassified = 0usize;
        let mut distinct = std::collections::BTreeSet::new();
        for processor in &landed {
            let class = processor.and_then(|p| topology.as_ref()?.processor_efficiency_class(p));
            match class.and_then(|c| per_rank.get_mut(usize::from(c.rank()))) {
                Some(count) => *count += 1,
                None => unclassified += 1,
            }
            if let Some(p) = processor {
                distinct.insert(*p);
            }
        }

        println!(
            "qos probe [{label}]: {CALLS} unpinned forward_complex_inplace calls, {nx}x{ny} f64"
        );
        println!(
            "  latency   median {:.1} us   p90 {:.1} us",
            micros(median),
            micros(p90)
        );
        println!(
            "  landed on {} distinct processors: {distinct:?}",
            distinct.len()
        );
        for (rank, count) in per_rank.iter().enumerate() {
            let marker = if Some(rank + 1) == class_count {
                " (highest class)"
            } else if rank == 0 && class_count.is_some_and(|c| c > 1) {
                " (lowest class)"
            } else {
                ""
            };
            println!(
                "  class rank {rank}{marker}: {count} calls ({}%)",
                count * 100 / CALLS
            );
        }
        if class_count.is_none() {
            println!(
                "  topology reports no efficiency classes: placement by class is unmeasurable here"
            );
        }
        if unclassified > 0 {
            println!("  unclassified landings: {unclassified}");
        }
    };

    println!(
        "qos probe: explicit throttling state before opt-out: {}",
        describe_state(explicit_throttling_state())
    );
    phase("before opt-out");
    opt_out_of_power_throttling();
    println!(
        "qos probe: explicit throttling state after opt-out:  {}",
        describe_state(explicit_throttling_state())
    );
    phase("after opt-out");
}

#[cfg(not(windows))]
fn qos_placement_probe() {
    println!(
        "qos probe: requires the Windows power-throttling API; nothing measured on this target"
    );
}

fn main() -> Result<(), apollo_bench::BenchmarkError> {
    // The probe observes the process's default state, so it runs before the
    // opt-out below and performs that opt-out itself midway.
    if std::env::var_os("APOLLO_QOS_PLACEMENT_PROBE").is_some() {
        qos_placement_probe();
        return Ok(());
    }
    opt_out_of_power_throttling();
    // Pin after the opt-out so the whole census runs on one processor of a
    // queried class; unpinned, its numbers are a scheduler blend of two
    // classes (ATLAS-APOLLO-CENSUS-UNPINNED-BLEND-2026-09-01).
    let processor = apollo_bench::bind_measurement_processor()?;
    eprintln!("engine_census: {}", processor.describe());
    // Before the timing suites, so each size's cold window is its first touch
    // of the process-global caches (see the function's method note).
    peak_working_set_census();
    if std::env::var_os("APOLLO_PEAK_WORKING_SET_ONLY").is_some() {
        return Ok(());
    }
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
    let mut flush = vec![0u8; FLUSH_BYTES];

    eprintln!(
        "engine_census: {mode:?} mode, {} sizes, measurement configuration warm-up {WARM_UP_MS}ms and measurement {MEASUREMENT_MS}ms, \
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

        let apollo = apollo_fft::FftPlan1D::<f64>::new(
            apollo_fft::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
        );
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

        let apollo = apollo_fft::FftPlan2D::<f64>::new(
            apollo_fft::Shape2D::new(nx, ny).expect("invariant: shape lengths are non-zero"),
        );
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
    let apollo = apollo_fft::FftPlan3D::<f64>::new(
        apollo_fft::Shape3D::new(nx, ny, nz).expect("invariant: shape lengths are non-zero"),
    );
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

    assert_staged_view_allocations();
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
