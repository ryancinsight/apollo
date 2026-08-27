//! Pinned measurement for the 8 x 128 construction: first the gate — the
//! inner small-size transforms against the references — then the assembled
//! experiment. Asserts nothing; run with `--ignored --nocapture`.

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
#[ignore = "measurement instrument for the 8x128 construction's inner gate"]
fn small_sizes_against_the_references_by_core_type() {
    for cpu in [2u32, 12] {
        let landed = pin(cpu);
        for n in [64usize, 128, 256, 512] {
            let calls = 8192u32;
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
            let apollo_ns = best_block(calls, || {
                work.copy_from_slice(&src);
                plan.forward_complex_slice_inplace(std::hint::black_box(&mut work));
            });
            let base_ns = if n == 128 {
                use std::sync::atomic::Ordering;
                let m = &super::butterfly::phase_meter::ACTIVE;
                for p in &super::butterfly::phase_meter::PHASES {
                    p.store(0, Ordering::Relaxed);
                }
                super::butterfly::phase_meter::CALLS.store(0, Ordering::Relaxed);
                m.store(true, Ordering::Relaxed);
                let b = best_block(calls, || {
                    work.copy_from_slice(&src);
                    assert!(super::butterfly::transform_128::<f64, false>(
                        std::hint::black_box(&mut work)
                    ));
                });
                m.store(false, Ordering::Relaxed);
                let calls_seen = super::butterfly::phase_meter::CALLS
                    .load(Ordering::Relaxed)
                    .max(1);
                let ph: Vec<u64> = super::butterfly::phase_meter::PHASES
                    .iter()
                    .map(|p| p.load(Ordering::Relaxed) / calls_seen)
                    .collect();
                println!(
                    "B128 phases: redistribute={} rows16={} columns8={}",
                    ph[0], ph[1], ph[2]
                );
                format!(" base128={b:>7.1}")
            } else {
                String::new()
            };
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
                "SML cpu={landed:<2} ({}) n={n:<4} apollo={apollo_ns:>7.1}{base_ns} rustfft={rust_ns:>7.1} phastft={phast_ns:>7.1} vs_rust={:>5.2}",
                if landed < 8 { "P" } else { "E" },
                apollo_ns / rust_ns,
            );
        }
    }
}
