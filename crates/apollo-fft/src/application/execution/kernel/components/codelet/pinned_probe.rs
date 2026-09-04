//! Pinned measurement: the N = 16 codelet against the incumbent sized route,
//! by core type. Asserts nothing; run with `--ignored --nocapture`.
//!
//! Block-timed — thousands of calls per timing — because one N = 16 transform
//! sits near the timer's resolution. The buffer is refilled once per block,
//! not per call, so both arms measure transform cost, not memcpy.

use super::try_transform_16;
use crate::application::execution::kernel::measurement_cores;
use eunomia::Complex64;
use hermes_simd::{ProcessorBinding, ProcessorIndex};
use std::time::Instant;

const CALLS_PER_BLOCK: u32 = 4096;
const BLOCKS: usize = 40;

fn best_block<F: FnMut()>(mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..BLOCKS {
        let start = Instant::now();
        for _ in 0..CALLS_PER_BLOCK {
            f();
        }
        best = best.min(start.elapsed().as_nanos() as f64 / f64::from(CALLS_PER_BLOCK));
    }
    best
}

#[test]
#[ignore = "measurement instrument: interleaved against planar batched kernels"]
fn interleaved_against_planar_by_core_type() {
    use crate::application::execution::kernel::components::batched;
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
        for k in [8u32, 10, 12, 14, 16] {
            let n = 1usize << k;
            let src: Vec<Complex64> = (0..n)
                .map(|i| {
                    let x = i as f64;
                    Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
                })
                .collect();
            let mut work = src.clone();
            let mut scratch = vec![Complex64::default(); batched::scratch_len(n)];
            let blocks = if n <= 4096 { 4096u32 } else { 256 };
            let planar = {
                let mut best = f64::INFINITY;
                for _ in 0..12 {
                    let t = std::time::Instant::now();
                    for _ in 0..blocks {
                        work.copy_from_slice(&src);
                        batched::four_step_batched::<f64, false>(
                            std::hint::black_box(&mut work),
                            &mut scratch,
                        );
                    }
                    best = best.min(t.elapsed().as_nanos() as f64 / f64::from(blocks));
                }
                best
            };
            let interleaved = {
                let mut best = f64::INFINITY;
                for _ in 0..12 {
                    let t = std::time::Instant::now();
                    for _ in 0..blocks {
                        work.copy_from_slice(&src);
                        batched::interleaved::four_step_interleaved::<f64, false>(
                            std::hint::black_box(&mut work),
                        );
                    }
                    best = best.min(t.elapsed().as_nanos() as f64 / f64::from(blocks));
                }
                best
            };
            println!(
                "IVP cpu={landed:<2} ({}) n={n:<6} planar={planar:>10.1}ns interleaved={interleaved:>10.1}ns ratio={:.2}",
                core.label(),
                interleaved / planar,
            );
        }
    }
}

#[test]
#[ignore = "measurement instrument: is the batched four-step the better 128..1024 route"]
fn mid_sizes_against_the_batched_four_step_by_core_type() {
    use crate::application::execution::kernel::components::batched;
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
        for k in [8u32, 10] {
            let n = 1usize << k;
            let src: Vec<Complex64> = (0..n)
                .map(|i| {
                    let x = i as f64;
                    Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
                })
                .collect();
            let plan = crate::FftPlan1D::<f64>::new(
                crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
            );
            let mut work = src.clone();
            let route = best_block(|| {
                work.copy_from_slice(&src);
                plan.forward_complex_slice_inplace(std::hint::black_box(&mut work));
            });
            let mut scratch = vec![Complex64::default(); batched::scratch_len(n)];
            let four_step = best_block(|| {
                work.copy_from_slice(&src);
                batched::four_step_batched::<f64, false>(
                    std::hint::black_box(&mut work),
                    &mut scratch,
                );
            });
            println!(
                "MID cpu={landed:<2} ({}) n={n:<5} route={route:>9.1}ns batched={four_step:>9.1}ns ratio={:.2}",
                core.label(),
                four_step / route,
            );
        }
    }
}

#[test]
#[ignore = "measurement instrument for the codelet acceptance oracle"]
fn codelet_against_the_incumbent_by_core_type() {
    let src: Vec<Complex64> = (0..16)
        .map(|i| {
            let x = i as f64;
            Complex64::new((0.4 * x + 0.3).sin(), (0.7 * x - 0.1).cos())
        })
        .collect();
    let plan = crate::FftPlan1D::<f64>::new(
        crate::Shape1D::new(16).expect("invariant: shape lengths are non-zero"),
    );

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
        let mut work = src.clone();

        // Three control arms, so the instrument reports its own validity
        // alongside its verdict.
        //
        // `floor` is the timing loop with no work in it. `calibration` is 256
        // dependent f64 multiply-adds, written as separate operations because
        // without `+fma` at compile time `f64::mul_add` lowers to a libm call
        // and would measure the software routine instead of the machine: the
        // multiply and add latencies are about four cycles each and the chain
        // cannot overlap, so the true cost is near 2048 cycles, about 550 ns at
        // the 3.7 GHz nominal clock. A reading far above that means the core is
        // downclocked and every absolute figure below is scaled with it.
        // `dispatch` is the codelet's own entry with a kernel that does
        // nothing, so the per-call cost of `vectorize_lanes` is separated from
        // the transform it wraps.
        let floor = best_block(|| {
            std::hint::black_box(&mut work);
        });
        let calibration = best_block(|| {
            let mut x = std::hint::black_box(1.000_000_1f64);
            for _ in 0..256 {
                x = x * 1.000_000_1 + 1.0;
            }
            std::hint::black_box(x);
        });
        let dispatch = {
            use hermes_simd::{LaneKernel, Simd, SimdArch, SimdKernel};
            struct Nothing;
            impl<T: hermes_simd::LaneScalar> LaneKernel<T> for Nothing {
                type Output = bool;
                fn call<A: SimdArch + SimdKernel<T>>(self, _s: Simd<T, A>) -> bool {
                    true
                }
            }
            best_block(|| {
                std::hint::black_box(hermes_simd::vectorize_lanes::<4, f64, _>(
                    std::hint::black_box(Nothing),
                ));
            })
        };

        // The bare codelet the route ends in, with no plan, no dispatch and no
        // length check: everything between it and `incumbent` is overhead.
        let mut work = src.clone();
        let bare = best_block(|| {
            work.copy_from_slice(&src);
            let data: &mut [Complex64; 16] = std::hint::black_box(&mut work)
                .as_mut_slice()
                .try_into()
                .expect("invariant: sixteen samples");
            crate::application::execution::kernel::components::winograd::dft16_impl::<f64, false>(
                data,
            );
        });
        // RustFFT at the same length through the same harness. It is not a
        // competitor here; it is the external control that says whether this
        // probe's absolute scale agrees with the crate's other instruments.
        let rust = rustfft::FftPlanner::<f64>::new().plan_fft_forward(16);
        let rust_src: Vec<rustfft::num_complex::Complex<f64>> = src
            .iter()
            .map(|v| rustfft::num_complex::Complex::new(v.re, v.im))
            .collect();
        let mut rust_scratch =
            vec![rustfft::num_complex::Complex::new(0.0, 0.0); rust.get_inplace_scratch_len()];
        let mut rust_work = rust_src.clone();
        let reference = best_block(|| {
            rust_work.copy_from_slice(&rust_src);
            rust.process_with_scratch(std::hint::black_box(&mut rust_work), &mut rust_scratch);
        });

        // Both measured arms restore the fixture per call. Restoring per block
        // instead measured the same to within noise, so the growth of repeated
        // forward transforms is not what either arm is reading.
        let mut work = src.clone();
        let incumbent = best_block(|| {
            work.copy_from_slice(&src);
            plan.forward_complex_slice_inplace(std::hint::black_box(&mut work));
        });
        let mut work = src.clone();
        let codelet = best_block(|| {
            work.copy_from_slice(&src);
            assert!(try_transform_16::<f64, false, false>(std::hint::black_box(
                &mut work
            )));
        });
        println!(
            "CORE cpu={landed:<2} ({}) incumbent={incumbent:>7.1}ns codelet={codelet:>7.1}ns ratio={:.2}",
            core.label(),
            codelet / incumbent,
        );
        println!(
            "CTRL cpu={landed:<2} ({}) bare={bare:>5.1}ns floor={floor:>5.1}ns dispatch={dispatch:>6.1}ns calibration={calibration:>7.1}ns rustfft={reference:>7.1}ns",
            core.label(),
        );
    }
}
