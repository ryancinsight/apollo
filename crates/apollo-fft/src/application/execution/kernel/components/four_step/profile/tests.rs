//! Phase observations over the existing generic FourStep operation.

use super::{capture, capture_buffers, observe_buffers, Buffers, Phase, Total};
use crate::application::execution::kernel::measurement_cores;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use eunomia::{Complex, Complex64};
use hermes_simd::{ProcessorBinding, ProcessorIndex};
use std::time::Duration;
use themis::CpuTopology;

const SIZES: [usize; 2] = [65_536, 262_144];
// Three complete observations reveal between-block spread. Sixty-four calls
// amortize phase timestamps while the two sizes, two scalar widths and two
// processor classes stay below the committed 60-second Nextest bound: even a
// 10-ms large transform consumes 3*64*2*2*2*10ms = 15.36 seconds.
const REPETITIONS: u32 = 3;
const CALLS: u32 = 64;

#[test]
#[ignore = "phase attribution instrument; run alone with the committed Nextest budget"]
fn generic_four_step_phase_attribution() {
    let selection = measurement_cores::selected()
        .expect("phase attribution requires queried processor classes");
    // Topology discovery and formatting stay outside every timed region.
    // Leto reports process-wide policy; the sharing sets below preserve the
    // distinct physical caches behind each selected processor class.
    let topology = CpuTopology::detect().expect("cache attribution requires CPU topology");
    let caches = topology
        .cache_levels()
        .expect("cache attribution requires reported cache levels");
    let geometry = leto_ops::cached_cache_geometry();
    print!("{}", selection.describe());
    println!(
        "FCACHE policy=process-wide l1_bytes={} l2_bytes={} l3_bytes={} line_bytes={} reported_caches={}",
        geometry.l1_bytes(),
        geometry.l2_bytes(),
        geometry.l3_bytes(),
        geometry.cache_line_bytes(),
        caches.len(),
    );
    for core in selection.cores() {
        for cache in caches
            .iter()
            .filter(|cache| cache.shared_processors.contains(&core.processor().get()))
        {
            println!(
                "FCACHE cpu={} class={} level={} size_bytes={} line_bytes={:?} shared_processors={:?}",
                core.processor().get(),
                core.label(),
                cache.level,
                cache.size_bytes,
                cache.line_bytes,
                cache.shared_processors,
            );
        }
        let _binding = ProcessorBinding::bind(core.processor())
            .expect("measurement processor must be available");
        let landed = ProcessorIndex::current().expect("Windows supports processor queries");
        assert_eq!(
            landed.get(),
            core.processor().get(),
            "binding must be exact"
        );
        measure::<f64>("f64", landed.get(), core.label(), f64::EPSILON / 2.0);
        measure::<f32>(
            "f32",
            landed.get(),
            core.label(),
            f64::from(f32::EPSILON) / 2.0,
        );
    }
}

fn measure<F>(precision: &str, processor: u32, core_class: &str, unit_roundoff: f64)
where
    F: MixedRadixScalar<Complex = Complex<F>> + From<f32> + Into<f64>,
{
    let mut row_suite = BenchmarkSuite::new(
        BenchmarkConfig::try_with_budgets(Duration::from_millis(20), Duration::from_millis(60))
            .expect("measurement budgets are positive"),
    );
    for n in SIZES {
        let source: Vec<Complex<F>> = (0..n)
            .map(|index| {
                F::complex(
                    (0.017 * index as f64).sin(),
                    0.25 * (0.031 * index as f64).cos(),
                )
            })
            .collect();
        let plan =
            crate::FftPlan1D::<F>::new(crate::Shape1D::new(n).expect("valid profile length"));
        let mut work = source.clone();
        let ((), buffers) = capture_buffers(|| plan.forward_complex_slice_inplace(&mut work));
        let side = 1usize << (n.trailing_zeros() / 2);
        let matrix_bytes = n * size_of::<Complex<F>>();
        assert_eq!(buffers.data_address, work.as_ptr().addr());
        assert_eq!(buffers.data_bytes, matrix_bytes);
        assert_eq!(buffers.scratch_bytes, matrix_bytes);
        assert_eq!((buffers.rows, buffers.columns), (side, side));
        assert_eq!(buffers.data_row_bytes, side * size_of::<Complex<F>>());
        assert_eq!(buffers.scratch_row_bytes, buffers.data_row_bytes);
        assert!(buffers.data_address.abs_diff(buffers.scratch_address) >= matrix_bytes);
        println!(
            "FBUFFER cpu={processor} class={core_class} type={precision} n={n} data_address={:#x} scratch_address={:#x} data_bytes={} scratch_bytes={} rows={} columns={} data_row_bytes={} scratch_row_bytes={}",
            buffers.data_address, buffers.scratch_address, buffers.data_bytes,
            buffers.scratch_bytes, buffers.rows, buffers.columns,
            buffers.data_row_bytes, buffers.scratch_row_bytes,
        );
        plan.inverse_complex_slice_inplace(&mut work);
        // A forward and normalized inverse each contribute gamma_k ||x||_1,
        // with at most sixteen rounded real operations per radix stage.
        // Propagating the first error gives (2 gamma_k + gamma_k^2)||x||_1.
        let ku = 16.0 * f64::from(n.ilog2()) * unit_roundoff;
        let gamma = ku / (1.0 - ku);
        let norm = source
            .iter()
            .map(|value| Complex64::new(value.re.into(), value.im.into()).norm())
            .sum::<f64>();
        let bound = (2.0 * gamma + gamma * gamma) * norm;
        for (actual, original) in work.iter().zip(&source) {
            let error = (Complex64::new(actual.re.into(), actual.im.into())
                - Complex64::new(original.re.into(), original.im.into()))
            .norm();
            assert!(
                error <= bound,
                "roundtrip error {error:e} exceeds {bound:e}"
            );
        }

        for repetition in 0..REPETITIONS {
            let mut totals = [Total::default(); Phase::ALL.len()];
            let mut elapsed = Duration::ZERO;
            for _ in 0..CALLS {
                work.copy_from_slice(&source);
                let ((), observed, call_elapsed) = capture(|| {
                    plan.forward_complex_slice_inplace(std::hint::black_box(&mut work));
                    std::hint::black_box(&work);
                });
                elapsed += call_elapsed;
                for (total, observation) in totals.iter_mut().zip(observed) {
                    assert_eq!(
                        observation.calls, 1,
                        "square route executes each phase once"
                    );
                    total.elapsed += observation.elapsed;
                    total.calls += observation.calls;
                }
            }
            for (phase, total) in Phase::ALL.into_iter().zip(totals) {
                println!(
                    "FPHASE cpu={processor} class={core_class} type={precision} n={n} block={repetition} phase={phase:?} ns_per_call={:.1} share={:.6} calls={}",
                    total.elapsed.as_secs_f64() * 1e9 / f64::from(CALLS),
                    total.elapsed.as_secs_f64() / elapsed.as_secs_f64(),
                    total.calls,
                );
            }
            println!(
                "FPHASE cpu={processor} class={core_class} type={precision} n={n} block={repetition} phase=Complete ns_per_call={:.1}",
                elapsed.as_secs_f64() * 1e9 / f64::from(CALLS),
            );
        }

        let row_len = 1usize << (n.trailing_zeros() / 2);
        let twiddles = F::cached_twiddle_fwd(row_len);
        let mut scratch = vec![F::complex(0.0, 0.0); n];
        row_suite.run_batched(
            BenchmarkCase::new(
                "direct_stockham_rows",
                precision,
                format!("cpu{processor}-{n}"),
            ),
            || source.clone(),
            |rows| {
                for (row, row_scratch) in rows
                    .chunks_exact_mut(row_len)
                    .zip(scratch.chunks_exact_mut(row_len))
                {
                    F::stockham_forward(std::hint::black_box(row), row_scratch, &twiddles);
                }
                std::hint::black_box(rows);
            },
        );
    }
    println!("{}", row_suite.report());
}

#[test]
fn buffer_observation_preserves_borrowed_extents() {
    let data = [0u32; 10];
    let scratch = [0u32; 12];
    let (result, observed) = capture_buffers(|| {
        observe_buffers(&data[1..7], &scratch[2..8], 2, 3);
        7
    });
    assert_eq!(result, 7);
    assert_eq!(
        observed,
        Buffers {
            data_address: data[1..].as_ptr().addr(),
            scratch_address: scratch[2..].as_ptr().addr(),
            data_bytes: 24,
            scratch_bytes: 24,
            rows: 2,
            columns: 3,
            data_row_bytes: 12,
            scratch_row_bytes: 8,
        }
    );
    let ((), totals, _) = capture(|| observe_buffers(&data, &scratch, 2, 3));
    assert_eq!(totals.map(|total| total.calls), [0; Phase::ALL.len()]);
    assert_eq!(super::BUFFERS.get(), None);
}
