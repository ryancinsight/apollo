//! Per-level attribution of the column-first chain on each core type: the
//! column pass, the slice transforms below it and the interleave, in
//! cycles per step, for the lengths whose selection differs by width
//! (`APOLLO-FOUR-STEP-E-CORE-F32-131072`). The separately instantiated
//! `MEASURE` variant stamps the meter; the comparison kernel carries no
//! stamps.

use super::{instance_major, transform_via_base_256, transform_via_base_512};
use crate::application::execution::kernel::measurement_cores;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::Complex;
use hermes_simd::{ProcessorBinding, ProcessorIndex};
use std::sync::atomic::Ordering;

const CALLS: u64 = 64;

/// One level's reading: steps per transform and cycles per step for the
/// column pass, the slices below it and the interleave.
struct Level {
    steps: u64,
    pass: u64,
    slices: u64,
    interleave: u64,
}

/// The levels, outermost first, after `CALLS` transforms of `n` on `run`.
fn chain_attribution<T>(n: usize, run: impl Fn(&mut [Complex<T>]) -> bool) -> Vec<Level>
where
    T: MixedRadixScalar<Complex = Complex<T>>,
{
    let src: Vec<Complex<T>> = (0..n)
        .map(|i| {
            let x = i as f64;
            Complex::new(
                T::from_precise((0.017 * x).sin()),
                T::from_precise(0.25 * (0.031 * x).cos()),
            )
        })
        .collect();
    let mut work = src.clone();
    for level in &instance_major::phase_meter::CHAIN {
        for phase in level {
            phase.store(0, Ordering::Relaxed);
        }
    }
    for calls in &instance_major::phase_meter::CHAIN_CALLS {
        calls.store(0, Ordering::Relaxed);
    }
    for _ in 0..CALLS {
        work.copy_from_slice(&src);
        assert!(run(std::hint::black_box(&mut work)));
    }
    instance_major::phase_meter::CHAIN
        .iter()
        .zip(&instance_major::phase_meter::CHAIN_CALLS)
        .map_while(|(level, calls)| {
            let calls = calls.load(Ordering::Relaxed);
            (calls > 0).then(|| Level {
                steps: calls / CALLS,
                pass: level[0].load(Ordering::Relaxed) / calls,
                slices: level[1].load(Ordering::Relaxed) / calls,
                interleave: level[2].load(Ordering::Relaxed) / calls,
            })
        })
        .collect()
}

fn report(core: &str, scalar: &str, n: usize, levels: &[Level]) {
    let mut passes = 0;
    for (depth, level) in levels.iter().enumerate() {
        println!(
            "CHN {core} {scalar} n={n} level={depth} steps={} pass={} slices={} interleave={} (cycles a step)",
            level.steps, level.pass, level.slices, level.interleave
        );
        passes += (level.pass + level.interleave) * level.steps;
    }
    let base = levels.last().map_or(0, |level| level.slices * level.steps);
    println!("CHN {core} {scalar} n={n} passes_and_interleaves={passes} innermost_slices={base} cycles a transform");
}

#[test]
#[ignore = "measurement instrument for the chain's per-level cost by core type"]
fn chain_phases_by_core_type() {
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
        for n in [131_072usize, 16_384] {
            let state = instance_major::State256::<f32>::new_if_supported(n)
                .expect("the pinned host must provide a native base capability");
            let levels = chain_attribution::<f32>(n, |work| {
                transform_via_base_256::<f32, false, true>(work, &state)
            });
            report(core, "f32", n, &levels);
            let state = instance_major::State256::<f64>::new_if_supported(n)
                .expect("the pinned host must provide a native base capability");
            let levels = chain_attribution::<f64>(n, |work| {
                transform_via_base_256::<f64, false, true>(work, &state)
            });
            report(core, "f64", n, &levels);
        }
        for n in [262_144usize, 32_768] {
            let state = instance_major::State512::<f32>::new_if_supported(n)
                .expect("the pinned host must provide a native base capability");
            let levels = chain_attribution::<f32>(n, |work| {
                transform_via_base_512::<f32, false, true>(work, &state)
            });
            report(core, "f32", n, &levels);
            let state = instance_major::State512::<f64>::new_if_supported(n)
                .expect("the pinned host must provide a native base capability");
            let levels = chain_attribution::<f64>(n, |work| {
                transform_via_base_512::<f64, false, true>(work, &state)
            });
            report(core, "f64", n, &levels);
        }
    }
}
