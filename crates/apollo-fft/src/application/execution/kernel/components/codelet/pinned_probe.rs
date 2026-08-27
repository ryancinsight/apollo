//! Pinned measurement: the N = 16 codelet against the incumbent sized route,
//! by core type. Asserts nothing; run with `--ignored --nocapture`.
//!
//! Block-timed — thousands of calls per timing — because one N = 16 transform
//! sits near the timer's resolution. The buffer is refilled once per block,
//! not per call, so both arms measure transform cost, not memcpy.

use super::try_transform_16;
use eunomia::Complex64;
use std::time::Instant;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn SetThreadAffinityMask(thread: isize, mask: usize) -> usize;
    fn GetCurrentThread() -> isize;
    fn GetCurrentProcessorNumber() -> u32;
}

fn pin(cpu: u32) -> u32 {
    // SAFETY: pins the current thread; both calls are documented Win32.
    unsafe {
        SetThreadAffinityMask(GetCurrentThread(), 1usize << cpu);
    }
    std::thread::yield_now();
    // SAFETY: no arguments, no state.
    unsafe { GetCurrentProcessorNumber() }
}

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
#[ignore = "measurement instrument: is the batched four-step the better 128..1024 route"]
fn mid_sizes_against_the_batched_four_step_by_core_type() {
    use crate::application::execution::kernel::components::batched;
    for cpu in [2u32, 12] {
        let landed = pin(cpu);
        for k in [8u32, 10] {
            let n = 1usize << k;
            let src: Vec<Complex64> = (0..n)
                .map(|i| {
                    let x = i as f64;
                    Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
                })
                .collect();
            let plan = crate::FftPlan1D::<f64>::new(crate::Shape1D { n });
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
                if landed < 8 { "P" } else { "E" },
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
    let plan = crate::FftPlan1D::<f64>::new(crate::Shape1D { n: 16 });

    // Logical 0..8 are P-cores and 8..24 E-cores on the Core Ultra 9 285K.
    for cpu in [2u32, 12] {
        let landed = pin(cpu);
        let mut work = src.clone();
        let incumbent = best_block(|| {
            plan.forward_complex_slice_inplace(std::hint::black_box(&mut work));
        });
        let codelet = best_block(|| {
            assert!(try_transform_16::<f64, false, false>(std::hint::black_box(
                &mut work
            )));
        });
        println!(
            "CORE cpu={landed:<2} ({}) incumbent={incumbent:>7.1}ns codelet={codelet:>7.1}ns ratio={:.2}",
            if landed < 8 { "P" } else { "E" },
            codelet / incumbent,
        );
    }
}
