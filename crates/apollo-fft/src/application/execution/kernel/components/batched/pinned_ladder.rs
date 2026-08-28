//! Pinned measurement: the batched route against the reference engines
//! across the power-of-two ladder, by core type. Asserts nothing; run with
//! `--ignored --nocapture`.

use super::{four_step_batched, scratch_len};
use crate::application::execution::kernel::test_utils::pin;
use eunomia::Complex64;
use rustfft::num_complex::Complex as RustComplex;
use std::time::Instant;

fn best_block<F: FnMut()>(calls: u32, mut f: F) -> f64 {
    const BLOCKS: usize = 12;
    let mut best = f64::INFINITY;
    for _ in 0..BLOCKS {
        let t = Instant::now();
        for _ in 0..calls {
            f();
        }
        best = best.min(t.elapsed().as_nanos() as f64 / f64::from(calls));
    }
    best
}

#[test]
#[ignore = "measurement instrument for the mid-size acceptance bar"]
fn batched_against_the_references_across_the_ladder() {
    for cpu in [2u32, 12] {
        let landed = pin(cpu);
        for exp in [8u32, 9, 10, 11, 12, 13, 14] {
            let n = 1usize << exp;
            // Keep each block near one millisecond so twelve blocks resist
            // scheduler noise without inflating the suite budget.
            let calls = (1_000_000 / (n as u32)).max(16);
            let src: Vec<Complex64> = (0..n)
                .map(|i| {
                    let x = i as f64;
                    Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
                })
                .collect();
            let rust_src: Vec<RustComplex<f64>> =
                src.iter().map(|v| RustComplex::new(v.re, v.im)).collect();
            let (re_src, im_src): (Vec<f64>, Vec<f64>) = src.iter().map(|v| (v.re, v.im)).unzip();

            let rust = rustfft::FftPlanner::<f64>::new().plan_fft_forward(n);
            let mut rust_scratch = vec![RustComplex::new(0.0, 0.0); rust.get_inplace_scratch_len()];
            let phast = phastft::planner::PlannerDit64::new(n);

            let mut work = src.clone();
            let plan = crate::FftPlan1D::<f64>::new(crate::Shape1D { n });
            let batched_ns = best_block(calls, || {
                work.copy_from_slice(&src);
                plan.forward_complex_slice_inplace(std::hint::black_box(&mut work));
            });
            let mut rust_work = rust_src.clone();
            let rust_ns = best_block(calls, || {
                rust_work.copy_from_slice(&rust_src);
                rust.process_with_scratch(std::hint::black_box(&mut rust_work), &mut rust_scratch);
            });
            let (mut re, mut im) = (re_src.clone(), im_src.clone());
            let phast_ns = best_block(calls, || {
                re.copy_from_slice(&re_src);
                im.copy_from_slice(&im_src);
                phastft::fft_f64_dit_with_planner(
                    std::hint::black_box(&mut re),
                    std::hint::black_box(&mut im),
                    phastft::planner::Direction::Forward,
                    &phast,
                );
            });
            println!(
                "LAD cpu={landed:<2} ({}) n={n:<5} batched={batched_ns:>9.1} rustfft={rust_ns:>9.1} phastft={phast_ns:>9.1} vs_rust={:>5.2} vs_phast={:>5.2}",
                if landed < 8 { "P" } else { "E" },
                batched_ns / rust_ns,
                batched_ns / phast_ns,
            );
        }
    }
}
