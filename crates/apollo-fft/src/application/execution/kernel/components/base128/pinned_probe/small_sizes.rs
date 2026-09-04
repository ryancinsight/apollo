//! The inner gate: the small-size transforms this construction is built
//! from, measured against the reference implementations at both scalars.

use super::{phase_attribution, ProbeScalar};
use crate::application::execution::kernel::measurement_cores;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkMode, BenchmarkSuite};
use eunomia::Complex64;
use hermes_simd::{ProcessorBinding, ProcessorIndex};
use rustfft::num_complex::Complex as RustComplex;

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
        // Smoke mode: the pass exists only to absorb first-touch costs
        // (planner caches, page faults, branch promotion), so one observation
        // per case is the entire contract — its figures are never read.
        let mut warmup =
            BenchmarkSuite::new(BenchmarkMode::Smoke.apply(BenchmarkConfig::regression()));
        small_sizes_for_scalar::<f64>(&mut warmup, core, "f64");
        small_sizes_for_scalar::<f32>(&mut warmup, core, "f32");
        drop(warmup);
        // Reported pass: the sweep contract. The per-case cost is analytic
        // (warm-up + measurement), so the sweep's wall time is the case count
        // times that constant; at the regression budgets it exceeds the
        // committed nextest timeout regardless of host speed, which terminated
        // the sweep under harness (`APOLLO-SWEEP-EXCEEDS-BUDGET-2026-09-04`).
        // The sweep contract keeps the 100-sample estimator with a 90
        // millisecond case budget, completing inside the timeout with margin.
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::sweep());
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
