//! Pinned measurement for the 8 x 128 construction: first the gate — the
//! inner small-size transforms against the references — then the assembled
//! experiment. Sets no performance threshold; run with `--ignored --nocapture`.

use crate::application::execution::kernel::test_utils::pin;
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use eunomia::Complex64;
use rustfft::num_complex::Complex as RustComplex;

fn phase_attribution(
    src: &[Complex64],
    work: &mut [Complex64],
    plan: &super::instance_major::Plan128<f64>,
) -> [u64; 3] {
    use std::sync::atomic::Ordering;

    const CALLS: u64 = 8_192;
    for phase in &super::instance_major::phase_meter::PHASES {
        phase.store(0, Ordering::Relaxed);
    }
    super::instance_major::phase_meter::CALLS.store(0, Ordering::Relaxed);
    for _ in 0..CALLS {
        work.copy_from_slice(src);
        assert!(super::instance_major::transform_128_measured::<f64, false>(
            std::hint::black_box(work),
            plan,
        ));
    }
    let calls = super::instance_major::phase_meter::CALLS
        .load(Ordering::Relaxed)
        .max(1);
    let mut averages = [0; 3];
    for (average, phase) in averages
        .iter_mut()
        .zip(&super::instance_major::phase_meter::PHASES)
    {
        *average = phase.load(Ordering::Relaxed) / calls;
    }
    averages
}

#[test]
#[ignore = "measurement instrument for the 8x128 construction's inner gate"]
fn small_sizes_against_the_references_by_core_type() {
    for cpu in [2u32, 12] {
        let landed = pin(cpu);
        let core = if landed < 8 { "p-core" } else { "e-core" };
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
        for n in [64usize, 128, 256, 512] {
            let src: Vec<Complex64> = (0..n)
                .map(|i| {
                    let x = i as f64;
                    Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
                })
                .collect();
            let rust_src: Vec<RustComplex<f64>> =
                src.iter().map(|v| RustComplex::new(v.re, v.im)).collect();
            let (re_src, im_src): (Vec<f64>, Vec<f64>) = src.iter().map(|v| (v.re, v.im)).unzip();

            let plan = crate::FftPlan1D::<f64>::new(crate::Shape1D { n });
            let rust = rustfft::FftPlanner::<f64>::new().plan_fft_forward(n);
            let mut rust_scratch = vec![RustComplex::new(0.0, 0.0); rust.get_inplace_scratch_len()];
            let phast = phastft::planner::PlannerDit64::new(n);

            let mut work = src.clone();
            suite.run(BenchmarkCase::new(core, "apollo", n), || {
                work.copy_from_slice(&src);
                plan.forward_complex_slice_inplace(std::hint::black_box(&mut work));
            });
            if n == 128 {
                let base_plan = super::instance_major::Plan128::<f64>::new_if_supported::<false>()
                    .expect("the pinned host must provide the four-lane base capability");
                work.copy_from_slice(&src);
                assert!(
                    super::instance_major::transform_128::<f64, false>(&mut work, &base_plan),
                    "the pinned host must provide the four-lane base capability"
                );
                suite.run(BenchmarkCase::new(core, "base-128", n), || {
                    work.copy_from_slice(&src);
                    std::hint::black_box(super::instance_major::transform_128::<f64, false>(
                        std::hint::black_box(&mut work),
                        &base_plan,
                    ));
                });
                let phases = phase_attribution(&src, &mut work, &base_plan);
                println!(
                    "B128 phases: redistribute={} rows16={} columns8={}",
                    phases[0], phases[1], phases[2]
                );
            }
            let mut rust_work = rust_src.clone();
            suite.run(BenchmarkCase::new(core, "rustfft", n), || {
                rust_work.copy_from_slice(&rust_src);
                rust.process_with_scratch(std::hint::black_box(&mut rust_work), &mut rust_scratch);
            });
            let (mut re, mut im) = (re_src.clone(), im_src.clone());
            suite.run(BenchmarkCase::new(core, "phastft", n), || {
                re.copy_from_slice(&re_src);
                im.copy_from_slice(&im_src);
                phastft::fft_f64_dit_with_planner(
                    std::hint::black_box(&mut re),
                    std::hint::black_box(&mut im),
                    phastft::planner::Direction::Forward,
                    &phast,
                );
            });
        }
        println!("SML cpu={landed} ({core})");
        suite.emit();
    }
}
