//! Pinned per-pass attribution for the planar route. Asserts nothing; run
//! with `--ignored --nocapture`.
//!
//! [`super::pinned_ladder`] says where a size stands against the references;
//! this says where its time goes. The two answer different questions and the
//! second is the one that decides what to work on next, because a route can
//! be behind for want of arithmetic or for want of movement and the totals
//! separate them.

use crate::application::execution::kernel::measurement_cores;
use eunomia::{Complex, Complex32, Complex64};
use hermes_simd::{ProcessorBinding, ProcessorIndex};

use crate::application::execution::kernel::mixed_radix::scalar::MixedRadixScalar;

/// Sizes worth attributing: the even powers, which take the square route
/// whole, and the odd powers, which decimate and run it twice.
const SIZES: [usize; 6] = [1024, 2048, 4096, 8192, 16384, 32768];

/// Calls per size. Enough that per-pass totals are stable, few enough that
/// the whole probe stays inside the suite's runtime budget.
const CALLS: u32 = 200;

#[test]
#[ignore = "measurement instrument for the planar route's pass attribution"]
fn planar_passes_by_size() {
    let Some(selection) = measurement_cores::selected() else {
        eprintln!("host reports no processor class information; probe not measurable");
        return;
    };
    let Some(core) = selection.performance() else {
        eprintln!("host reports no performance-core information; probe not measurable");
        return;
    };
    print!("{}", selection.describe());
    let cpu = core.processor().get();
    let _binding =
        ProcessorBinding::bind(core.processor()).expect("measurement processor must be available");
    std::thread::yield_now();
    let landed = ProcessorIndex::current()
        .expect("Windows supports processor queries")
        .get();
    assert_eq!(landed, cpu, "processor binding must remain exact");

    measure_precision("f64", landed, |n| {
        (0..n)
            .map(|i| {
                let x = i as f64;
                Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
            })
            .collect()
    });
    measure_precision("f32", landed, |n| {
        (0..n)
            .map(|i| {
                let x = i as f32;
                Complex32::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
            })
            .collect()
    });
}

fn measure_precision<F>(
    precision: &str,
    landed: u32,
    mut source: impl FnMut(usize) -> Vec<Complex<F>>,
) where
    F: MixedRadixScalar<Complex = Complex<F>>,
{
    for n in SIZES {
        let src = source(n);
        let plan = crate::FftPlan1D::<F>::new(
            crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
        );
        let mut work = src.clone();

        // One untimed call so plan and twiddle caches are warm, then drain
        // whatever it recorded so the totals below are steady-state.
        plan.forward_complex_slice_inplace(&mut work);
        let _ = super::sections::take();

        for _ in 0..CALLS {
            work.copy_from_slice(&src);
            plan.forward_complex_slice_inplace(std::hint::black_box(&mut work));
        }

        let totals = super::sections::take();
        let all: u64 = totals.iter().map(|&(_, cycles, _)| cycles).sum();
        for (label, cycles, passes) in totals {
            let per_call = cycles as f64 / f64::from(CALLS);
            let share = 100.0 * cycles as f64 / all as f64;
            let per_call_passes = passes as f64 / f64::from(CALLS);
            println!(
                "BPASS cpu={landed:<2} type={precision:<3} n={n:<5} {label:<9} tsc={per_call:>9.1} share={share:>5.1}% passes={per_call_passes:>3.1}"
            );
        }
        println!(
            "BPASS cpu={landed:<2} type={precision:<3} n={n:<5} {:<9} tsc={:>9.1}",
            "total",
            all as f64 / f64::from(CALLS)
        );
    }
}
