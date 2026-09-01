//! Pinned measurement: the resident driver against the batched route and the
//! reference engines, at N = 1024, by core type. Asserts nothing; run with
//! `--ignored --nocapture`.

use super::four_step_resident;
use super::planar::four_step_planar;
use crate::application::execution::kernel::components::batched;
use crate::application::execution::kernel::core_class;
use eunomia::Complex64;
use hermes_simd::{ProcessorBinding, ProcessorIndex};
use rustfft::num_complex::Complex as RustComplex;
use std::time::Instant;

fn best_block<F: FnMut()>(mut f: F) -> f64 {
    const CALLS: u32 = 2048;
    const BLOCKS: usize = 14;
    let mut best = f64::INFINITY;
    for _ in 0..BLOCKS {
        let t = Instant::now();
        for _ in 0..CALLS {
            f();
        }
        best = best.min(t.elapsed().as_nanos() as f64 / f64::from(CALLS));
    }
    best
}

#[test]
#[ignore = "measurement instrument for the resident driver's acceptance oracle"]
fn resident_against_batched_and_the_references_by_core_type() {
    let n = 1024usize;
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

    let Some(selection) = core_class::selected() else {
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
        let mut work = src.clone();
        let mut scratch = vec![Complex64::default(); batched::scratch_len(n)];

        let batched_ns = best_block(|| {
            work.copy_from_slice(&src);
            batched::four_step_batched::<f64, false>(std::hint::black_box(&mut work), &mut scratch);
        });
        let resident_ns = best_block(|| {
            work.copy_from_slice(&src);
            assert!(four_step_resident::<f64, false>(std::hint::black_box(
                &mut work
            )));
        });
        let planar_ns = best_block(|| {
            work.copy_from_slice(&src);
            assert!(four_step_planar::<f64, false>(std::hint::black_box(
                &mut work
            )));
        });
        let mut rust_work = rust_src.clone();
        let rust_ns = best_block(|| {
            rust_work.copy_from_slice(&rust_src);
            rust.process_with_scratch(std::hint::black_box(&mut rust_work), &mut rust_scratch);
        });
        let (mut re, mut im) = (re_src.clone(), im_src.clone());
        let phast_ns = best_block(|| {
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
            "RES cpu={landed:<2} ({}) batched={batched_ns:>7.1} resident={resident_ns:>7.1} planar={planar_ns:>7.1} rustfft={rust_ns:>7.1} phastft={phast_ns:>7.1}",
            core.class().label(),
        );
    }
}
