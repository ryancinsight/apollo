//! A Bluestein plan, which owns its chirps and kernel spectrum, against the
//! free route, which builds them every call (`APOLLO-MEM-BLUESTEIN-PER-CALL`):
//! the plan runs two transforms at `p` where the free route runs three and
//! evaluates `n` sines and cosines. Both arms in one pinned process per core
//! class, alternating.

use super::small_sizes::sweep_config;
use crate::application::execution::kernel::components::bluestein::bluestein_fft;
use crate::application::execution::kernel::measurement_cores;
use apollo_bench::{BenchmarkCase, BenchmarkSuite};
use eunomia::Complex64;
use hermes_simd::{ProcessorBinding, ProcessorIndex};

#[test]
#[ignore = "measurement instrument for the Bluestein plan tables"]
fn bluestein_plan_against_the_free_route() {
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
        let mut suite = BenchmarkSuite::new(sweep_config());
        for n in [361_usize, 961, 1681, 2209] {
            let plan = crate::FftPlan1D::<f64>::new(
                crate::Shape1D::new(n).expect("invariant: probe lengths are non-zero"),
            );
            let src: Vec<Complex64> = (0..n)
                .map(|i| {
                    let x = i as f64;
                    Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
                })
                .collect();
            let mut work = src.clone();
            for _ in 0..2 {
                suite.run(BenchmarkCase::new(core.label(), "plan", n), || {
                    work.copy_from_slice(&src);
                    plan.forward_complex_slice_inplace(std::hint::black_box(&mut work));
                });
                suite.run(BenchmarkCase::new(core.label(), "free", n), || {
                    work.copy_from_slice(&src);
                    bluestein_fft::<f64, false, false>(std::hint::black_box(&mut work));
                });
            }
        }
        print!("{}", suite.report());
    }
}
