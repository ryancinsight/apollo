//! The inner gate: the small-size transforms this construction is built
//! from, measured against the reference implementations at both scalars.

use super::{phase_attribution, split_attribution, ProbeScalar};
use crate::application::execution::kernel::measurement_cores;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use core::mem::size_of;
use eunomia::Complex64;
use hermes_simd::{ProcessorBinding, ProcessorIndex};
use rustfft::num_complex::Complex as RustComplex;
use std::time::Duration;

/// The per-case budget this sweep measures under, derived from its own size.
///
/// `BenchmarkConfig::regression()` spends 100 ms of warm-up and 400 ms of
/// measurement, which is the right budget for *one* case. This sweep runs
/// about ninety: seventeen lengths against apollo and RustFFT at both scalars,
/// PhastFT at the eleven powers of two, plus the base-128 and half-storage
/// cases. The discarded warm-up pass repeats the set, and both core types run
/// the whole thing, so half a second per case is 184 s against a committed
/// nextest bound of 60 s — the sweep has been terminated rather than reported.
///
/// A hundred milliseconds per case brings the reported pass to about 17 s. The
/// estimator is unchanged at 100 samples; each one simply calibrates to fewer
/// iterations, which at these lengths still leaves thousands per sample below
/// N = 1024 and a handful at N = 32768. Sizing the instrument to a committed
/// bound is instrument design; the alternative — raising the bound — would be
/// hiding a breach.
fn sweep_config() -> BenchmarkConfig {
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
fn sweep_warm_up_config() -> BenchmarkConfig {
    BenchmarkConfig::try_with_budgets(Duration::from_millis(10), Duration::from_millis(40))
        .expect("invariant: both budgets above are non-zero")
}

const SMALL_SIZE_CASES: [usize; 17] = [
    8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 32768, 100, 180, 384, 1000, 101, 1009,
];
const LIVENESS_CASES: [usize; 3] = [16, 32, 64];

/// `n` copies of `fill` in a buffer whose returned range starts on a
/// 64-byte boundary. The allocator places a `Vec` at 16 bytes, so whether a
/// working buffer's 32-byte vector accesses split cache lines is luck of
/// the heap state before it — a bimodal swing that moved `f64` at 1024 by
/// 10% between builds whose kernels the meter read as identical. Every arm
/// works in such a buffer, so the instrument measures the kernels, not the
/// heap.
fn aligned_work<T: Copy>(fill: T, n: usize) -> (Vec<T>, core::ops::Range<usize>) {
    let slack = 64 / size_of::<T>();
    let buffer = vec![fill; n + slack];
    let misalignment = buffer.as_ptr().align_offset(64);
    (buffer, misalignment..misalignment + n)
}

fn small_sizes_for_scalar<T>(suite: &mut BenchmarkSuite, core: &str, scalar: &str, sizes: &[usize])
where
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
        small_sizes_for_scalar::<f64>(&mut warmup, core, "f64", &SMALL_SIZE_CASES);
        small_sizes_for_scalar::<f32>(&mut warmup, core, "f32", &SMALL_SIZE_CASES);
        drop(warmup);
        let mut suite = BenchmarkSuite::new(sweep_config());
        small_sizes_for_scalar::<f64>(&mut suite, core, "f64", &SMALL_SIZE_CASES);
        small_sizes_for_scalar::<f32>(&mut suite, core, "f32", &SMALL_SIZE_CASES);
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
#[ignore = "focused liveness instrument for the n=16/32/64 f64 codelets"]
fn n32_f64_liveness_against_rustfft() {
    let Some(selection) = measurement_cores::selected() else {
        eprintln!("host reports no processor class information; probe not measurable");
        return;
    };
    let core = selection
        .cores()
        .first()
        .expect("measurement selection must contain one processor class");
    let cpu = core.processor().get();
    let _binding =
        ProcessorBinding::bind(core.processor()).expect("measurement processor must be available");
    std::thread::yield_now();
    let landed = ProcessorIndex::current()
        .expect("Windows supports processor queries")
        .get();
    assert_eq!(landed, cpu, "processor binding must remain exact");

    let core = core.label();
    let mut warmup = BenchmarkSuite::new(BenchmarkConfig::regression());
    small_sizes_for_scalar::<f64>(&mut warmup, core, "f64", &LIVENESS_CASES);
    drop(warmup);

    let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
    small_sizes_for_scalar::<f64>(&mut suite, core, "f64", &LIVENESS_CASES);
    println!("SML liveness cpu={landed} ({core})");
    print!("{}", suite.report());
}
