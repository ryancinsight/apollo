//! The inner gate: the small-size transforms this construction is built
//! from, measured against the reference implementations at both scalars.

use super::{phase_attribution, ProbeScalar};
use crate::application::execution::kernel::measurement_cores;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
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

fn small_sizes_for_scalar<T>(suite: &mut BenchmarkSuite, core: &str, scalar: &str)
where
    T: ProbeScalar + MixedRadixScalar<Complex = eunomia::Complex<T>>,
    eunomia::Complex<T>: eunomia::layout::Pod,
{
    // Powers of two, then the classes the bar "at all sizes" also covers:
    // smooth composites, a 2/3/5-smooth length with an odd leading factor, and
    // primes, which reach Rader or Bluestein. A sweep of powers of two alone
    // cannot see a result that depends on the length class
    // (`gap_audit.md#length-class-split`).
    for n in [
        8usize, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 32768, 100, 180, 384, 1000, 101, 1009,
    ] {
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

        let mut work = src.clone();
        suite.run(
            BenchmarkCase::new(core, format!("apollo-{scalar}"), n),
            || {
                work.copy_from_slice(&src);
                plan.forward_complex_slice_inplace(std::hint::black_box(&mut work));
            },
        );
        if n == 128 {
            let base_plan = super::instance_major::Plan128::<T>::new_if_supported::<false>()
                .expect("the pinned host must provide a native base capability");
            work.copy_from_slice(&src);
            assert!(
                super::instance_major::transform_128::<T, false>(&mut work, &base_plan),
                "the pinned host must provide a native base capability"
            );
            suite.run(
                BenchmarkCase::new(core, format!("base-128-{scalar}"), n),
                || {
                    work.copy_from_slice(&src);
                    std::hint::black_box(super::instance_major::transform_128::<T, false>(
                        std::hint::black_box(&mut work),
                        &base_plan,
                    ));
                },
            );
        }
        let mut rust_work = rust_src.clone();
        suite.run(
            BenchmarkCase::new(core, format!("rustfft-{scalar}"), n),
            || {
                rust_work.copy_from_slice(&rust_src);
                rust.process_with_scratch(std::hint::black_box(&mut rust_work), &mut rust_scratch);
            },
        );
        // PhastFT's DIT planner is power-of-two only, so it has no arm at the
        // other lengths rather than a slow one.
        if let Some(phast) = &phast {
            let (mut re, mut im) = (re_src.clone(), im_src.clone());
            suite.run(
                BenchmarkCase::new(core, format!("phastft-{scalar}"), n),
                || {
                    re.copy_from_slice(&re_src);
                    im.copy_from_slice(&im_src);
                    T::phast_forward(
                        std::hint::black_box(&mut re),
                        std::hint::black_box(&mut im),
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
        small_sizes_for_scalar::<f64>(&mut warmup, core, "f64");
        small_sizes_for_scalar::<f32>(&mut warmup, core, "f32");
        drop(warmup);
        let mut suite = BenchmarkSuite::new(sweep_config());
        small_sizes_for_scalar::<f64>(&mut suite, core, "f64");
        small_sizes_for_scalar::<f32>(&mut suite, core, "f32");
        {
            let src: Vec<Complex64> = (0..128)
                .map(|i| {
                    let x = i as f64;
                    Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
                })
                .collect();
            let mut work = src.clone();
            let base_plan = super::instance_major::Plan128::<f64>::new_if_supported::<false>()
                .expect("the pinned host must provide the four-lane base capability");
            let phases = phase_attribution(&src, &mut work, &base_plan);
            println!(
                "B128 phases: redistribute={} rows16={} columns8={}",
                phases[0], phases[1], phases[2]
            );
        }
        println!("SML cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}
