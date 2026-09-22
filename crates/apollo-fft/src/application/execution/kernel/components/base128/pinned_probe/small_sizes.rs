//! The inner gate: the base-route transforms this construction is built
//! from, measured against the reference implementations at both scalars.
//!
//! One shard of the reference sweep. The lengths are split by the routes
//! they reach and the reasons they move, so a campaign re-measures the one
//! it changed and every shard fits the runner's 300-second budget with room
//! to replicate: the base routes here (8 to 1024), the column-first block
//! routes in `block_sizes` (2048 to 32768), the composite and Rader routes
//! in `mixed_radix_sizes`, and the lengths past the caches in
//! `large_sizes`. All four share `sizes_for_scalar` and the budgets below,
//! so their arms stay comparable across shards.

use super::{phase_attribution, split_attribution, ProbeScalar};
use crate::application::execution::kernel::measurement_cores;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use core::mem::size_of;
use eunomia::Complex64;
use hermes_simd::{ProcessorBinding, ProcessorIndex};
use rustfft::num_complex::Complex as RustComplex;
use std::time::Duration;

/// The per-case budget every reference shard measures under, derived from
/// the shards' size.
///
/// `BenchmarkConfig::regression()` spends 100 ms of warm-up and 400 ms of
/// measurement, which is the right budget for *one* case. A shard runs far
/// more than one. Counting arms rather than lengths: apollo and RustFFT at
/// every length, PhastFT at the powers of two, plus the two base-128 shapes
/// and the base-256 arm — 27 cases per scalar here, doubled for the second
/// scalar and again for the second core class, so this shard is 108, the
/// block shard 60, the mixed-radix shard 48 and the large shard 48. The
/// discarded warm-up pass repeats the set at half the budget, so the cost of
/// a shard is `cases × 1.5 ×` the reported budget.
///
/// At half a second per case that is 81 s for this shard alone, against a
/// committed nextest bound of 60 s. At a hundred milliseconds it is 16 s,
/// and the runner's five replicated processes cost 81 s of the 300-second
/// campaign budget — which is what leaves room for the replication a reading
/// past L1 requires. The estimator is unchanged at 100 samples; each one
/// simply calibrates to fewer iterations, which at these lengths still
/// leaves thousands per sample. Sizing the instrument to a committed bound
/// is instrument design; the alternative — raising the bound — would be
/// hiding a breach.
pub(super) fn sweep_config() -> BenchmarkConfig {
    BenchmarkConfig::try_with_budgets(Duration::from_millis(20), Duration::from_millis(80))
        .expect("invariant: both budgets above are non-zero")
}

/// The discarded pass exists to warm the freshly linked binary, not to produce
/// numbers, so it runs at half the reported pass's budget.
///
/// Half rather than less: at a fifth, the first sweep after a build read
/// N = 32 at 32.4 ns where the two runs after it read 21.4 and 21.2, and
/// RustFFT at the same length moved the opposite way. The reported pass is
/// reproducible to about 1% once the machine is warm, so what the discarded
/// pass buys is not precision but the absence of a cold first run — and it
/// buys that only if it is long enough to do the warming.
pub(super) fn sweep_warm_up_config() -> BenchmarkConfig {
    BenchmarkConfig::try_with_budgets(Duration::from_millis(10), Duration::from_millis(40))
        .expect("invariant: both budgets above are non-zero")
}

/// The base routes: the register and L1-resident powers of two, up to the
/// last length the 256 base covers without a column pass. 2048 upward runs
/// in `block_sizes`, and the non-power-of-two lengths in
/// `mixed_radix_sizes`.
const SMALL_SIZE_CASES: [usize; 8] = [8, 16, 32, 64, 128, 256, 512, 1024];
const LIVENESS_CASES: [usize; 3] = [16, 32, 64];

/// `n` copies of `fill` in a buffer whose returned range starts on a
/// 64-byte boundary. The allocator places a `Vec` at 16 bytes, so whether a
/// working buffer's 32-byte vector accesses split cache lines is luck of
/// the heap state before it — a bimodal swing that moved `f64` at 1024 by
/// 10% between builds whose kernels the meter read as identical. Every arm
/// works in such a buffer, so the instrument measures the kernels, not the
/// heap.
pub(super) fn aligned_work<T: Copy>(fill: T, n: usize) -> (Vec<T>, core::ops::Range<usize>) {
    let slack = 64 / size_of::<T>();
    let buffer = vec![fill; n + slack];
    let misalignment = buffer.as_ptr().align_offset(64);
    (buffer, misalignment..misalignment + n)
}

pub(super) fn sizes_for_scalar<T>(
    suite: &mut BenchmarkSuite,
    core: &str,
    scalar: &str,
    sizes: &[usize],
) where
    T: ProbeScalar + MixedRadixScalar<Complex = eunomia::Complex<T>>,
    eunomia::Complex<T>: eunomia::layout::Pod,
{
    // Powers of two, then the classes the bar "at all sizes" also covers:
    // smooth composites, a 2/3/5-smooth length with an odd leading factor, and
    // primes, which reach Rader or Bluestein. A sweep of powers of two alone
    // cannot see a result that depends on the length class
    // (`gap_audit.md#length-class-split`).
    for &n in sizes {
        let src: Vec<eunomia::Complex<T>> = (0..n)
            .map(|i| {
                let x = i as f64;
                eunomia::Complex::new(
                    T::from_precise((0.017 * x).sin()),
                    T::from_precise(0.25 * (0.031 * x).cos()),
                )
            })
            .collect();
        let rust_src: Vec<RustComplex<T>> =
            src.iter().map(|v| RustComplex::new(v.re, v.im)).collect();
        let (re_src, im_src): (Vec<T>, Vec<T>) = src.iter().map(|v| (v.re, v.im)).unzip();

        let plan = crate::FftPlan1D::<T>::new(
            crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
        );
        let rust = rustfft::FftPlanner::<T>::new().plan_fft_forward(n);
        let mut rust_scratch = vec![
            RustComplex::new(T::from_precise(0.0), T::from_precise(0.0));
            rust.get_inplace_scratch_len()
        ];
        let phast = n.is_power_of_two().then(|| T::phast_planner(n));

        let (mut work_buffer, work_range) = aligned_work(src[0], n);
        let work = &mut work_buffer[work_range];
        suite.run(
            BenchmarkCase::new(core, format!("apollo-{scalar}"), n),
            || {
                work.copy_from_slice(&src);
                plan.forward_complex_slice_inplace(std::hint::black_box(&mut *work));
            },
        );
        if n == 128 {
            // Both shapes of the 128 base as bare arms in one run: the
            // plan's form and the other, so their comparison is within the
            // run and not across the probe's run-to-run drift.
            let eight_rows = super::instance_major::BasePlan::<
                T,
                8,
                16,
                { super::instance_major::table_lanes(8, 16) },
            >::new_if_supported::<false>()
            .expect("the pinned host must provide a native base capability");
            let four_rows = super::instance_major::BasePlan::<
                T,
                4,
                32,
                { super::instance_major::table_lanes(4, 32) },
            >::new_if_supported::<false>()
            .expect("the pinned host must provide a native base capability");
            work.copy_from_slice(&src);
            assert!(
                super::instance_major::transform_block::<
                    T,
                    false,
                    false,
                    8,
                    16,
                    256,
                    { super::instance_major::table_lanes(8, 16) },
                >(work, &eight_rows),
                "the pinned host must provide a native base capability"
            );
            suite.run(
                BenchmarkCase::new(core, format!("base-128-8x16-{scalar}"), n),
                || {
                    work.copy_from_slice(&src);
                    std::hint::black_box(super::instance_major::transform_block::<
                        T,
                        false,
                        false,
                        8,
                        16,
                        256,
                        { super::instance_major::table_lanes(8, 16) },
                    >(
                        std::hint::black_box(&mut *work), &eight_rows
                    ));
                },
            );
            work.copy_from_slice(&src);
            assert!(
                super::instance_major::transform_block::<
                    T,
                    false,
                    false,
                    4,
                    32,
                    256,
                    { super::instance_major::table_lanes(4, 32) },
                >(work, &four_rows),
                "the pinned host must provide a native base capability"
            );
            suite.run(
                BenchmarkCase::new(core, format!("base-128-4x32-{scalar}"), n),
                || {
                    work.copy_from_slice(&src);
                    std::hint::black_box(super::instance_major::transform_block::<
                        T,
                        false,
                        false,
                        4,
                        32,
                        256,
                        { super::instance_major::table_lanes(4, 32) },
                    >(
                        std::hint::black_box(&mut *work), &four_rows
                    ));
                },
            );
        }
        if let Some(base_plan) = (n == 256)
            .then(super::instance_major::Plan256::<T>::new_if_supported::<false>)
            .flatten()
        {
            work.copy_from_slice(&src);
            assert!(
                super::instance_major::transform_256::<T, false, false>(work, &base_plan),
                "the pinned host must provide a native base capability"
            );
            suite.run(
                BenchmarkCase::new(core, format!("base-256-{scalar}"), n),
                || {
                    work.copy_from_slice(&src);
                    std::hint::black_box(super::instance_major::transform_256::<T, false, false>(
                        std::hint::black_box(&mut *work),
                        &base_plan,
                    ));
                },
            );
        }
        let (mut rust_buffer, rust_range) = aligned_work(rust_src[0], n);
        let rust_work = &mut rust_buffer[rust_range];
        suite.run(
            BenchmarkCase::new(core, format!("rustfft-{scalar}"), n),
            || {
                rust_work.copy_from_slice(&rust_src);
                rust.process_with_scratch(std::hint::black_box(&mut *rust_work), &mut rust_scratch);
            },
        );
        // PhastFT's DIT planner is power-of-two only, so it has no arm at the
        // other lengths rather than a slow one.
        if let Some(phast) = &phast {
            let (mut re_buffer, re_range) = aligned_work(re_src[0], n);
            let (mut im_buffer, im_range) = aligned_work(im_src[0], n);
            let re = &mut re_buffer[re_range];
            let im = &mut im_buffer[im_range];
            suite.run(
                BenchmarkCase::new(core, format!("phastft-{scalar}"), n),
                || {
                    re.copy_from_slice(&re_src);
                    im.copy_from_slice(&im_src);
                    T::phast_forward(
                        std::hint::black_box(&mut *re),
                        std::hint::black_box(&mut *im),
                        phast,
                    );
                },
            );
        }
    }
}

#[test]
#[ignore = "measurement instrument for the 8x128 construction's inner gate"]
fn small_sizes_against_the_references_by_core_type() {
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
        sizes_for_scalar::<f64>(&mut warmup, core, "f64", &SMALL_SIZE_CASES);
        sizes_for_scalar::<f32>(&mut warmup, core, "f32", &SMALL_SIZE_CASES);
        drop(warmup);
        let mut suite = BenchmarkSuite::new(sweep_config());
        sizes_for_scalar::<f64>(&mut suite, core, "f64", &SMALL_SIZE_CASES);
        sizes_for_scalar::<f32>(&mut suite, core, "f32", &SMALL_SIZE_CASES);
        {
            let src: Vec<Complex64> = (0..128)
                .map(|i| {
                    let x = i as f64;
                    Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
                })
                .collect();
            let mut work = src.clone();
            let state = super::instance_major::State128::<f64>::new_if_supported(128)
                .expect("the pinned host must provide the four-lane base capability");
            let phases = phase_attribution(&src, &mut work, &state);
            println!(
                "B128 phases: load_and_rows={} retired={} columns_and_sink={}",
                phases[0], phases[1], phases[2]
            );
        }
        {
            let n = 1024usize;
            let src: Vec<Complex64> = (0..n)
                .map(|i| {
                    let x = i as f64;
                    Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
                })
                .collect();
            let mut work = src.clone();
            let state = super::instance_major::State256::<f64>::new_if_supported(n)
                .expect("the pinned host must provide a native base capability");
            {
                let split = split_attribution(&src, &mut work, |work| {
                    super::transform_via_base_256::<f64, false, true>(work, &state)
                });
                println!(
                    "B256 split n={n}: gather={} blocks={} levels={} total={} | per block ({}): load_and_rows={} columns_and_sink={}",
                    split.gather,
                    split.blocks,
                    split.levels,
                    split.gather + split.blocks + split.levels,
                    split.blocks_per_call,
                    split.rows,
                    split.columns
                );
            }
        }
        println!("SML cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}

#[test]
#[ignore = "focused liveness instrument for the n=16/32/64 codelets"]
fn small_codelet_liveness_against_the_references_by_core_type() {
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
        let mut warmup = BenchmarkSuite::new(BenchmarkConfig::regression());
        sizes_for_scalar::<f64>(&mut warmup, core, "f64", &LIVENESS_CASES);
        sizes_for_scalar::<f32>(&mut warmup, core, "f32", &LIVENESS_CASES);
        drop(warmup);

        let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
        sizes_for_scalar::<f64>(&mut suite, core, "f64", &LIVENESS_CASES);
        sizes_for_scalar::<f32>(&mut suite, core, "f32", &LIVENESS_CASES);
        println!("SML liveness cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}
