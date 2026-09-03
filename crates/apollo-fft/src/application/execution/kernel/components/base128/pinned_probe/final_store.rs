//! The assembled experiment: the N=512 final-store sink against the
//! incumbent, and the cost of each piece the split produces.

use super::{phase_attribution, run_bench_transform, BenchTransform, ProbeScalar};
use crate::application::execution::kernel::measurement_cores;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use eunomia::Complex64;
use hermes_simd::{ProcessorBinding, ProcessorIndex};
use rustfft::num_complex::Complex as RustComplex;

#[test]
#[ignore = "paired measurement of the N=512 final-store sink"]
fn final_store_sink_against_incumbent_by_core_type() {
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
        let plan = super::instance_major::Plan128::<f64>::new_if_supported::<false>()
            .expect("the pinned host must provide the base capability");
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
        for n in [256usize, 512] {
            let source: Vec<Complex64> = (0..n)
                .map(|index| {
                    let x = index as f64;
                    Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
                })
                .collect();
            let mut candidate = source.clone();
            let mut incumbent = source.clone();
            let mut work = source.clone();
            let twiddles = <f64 as MixedRadixScalar>::cached_twiddle_fwd(n);
            assert!(super::transform_via_base_128::<f64, false>(
                &mut candidate,
                &plan,
                &twiddles,
            ));
            if n == 512 {
                assert!(super::transform_via_base_128_incumbent::<f64, false>(
                    &mut incumbent,
                    &plan,
                    &twiddles,
                ));
            } else {
                assert!(super::transform_via_base_128::<f64, false>(
                    &mut incumbent,
                    &plan,
                    &twiddles,
                ));
            }
            let error = candidate
                .iter()
                .zip(&incumbent)
                .map(|(actual, reference)| {
                    (actual.re - reference.re).hypot(actual.im - reference.im)
                })
                .fold(0.0_f64, f64::max);
            let scale: f64 = source.iter().map(|value| value.re.hypot(value.im)).sum();
            let operations = 24.0 * n as f64;
            let scaled_epsilon = operations * (f64::EPSILON / 2.0);
            let bound = scaled_epsilon / (1.0 - scaled_epsilon) * scale;
            assert!(
                error <= bound,
                "N={n} final-store sink differs by {error:.3e} > {bound:.3e}"
            );

            let candidate_transform: BenchTransform = super::transform_via_base_128::<f64, false>;
            let incumbent_transform: BenchTransform = if n == 512 {
                super::transform_via_base_128_incumbent::<f64, false>
            } else {
                candidate_transform
            };
            suite.run(BenchmarkCase::new(core, "final-store-a", n), || {
                run_bench_transform(&source, &mut work, &plan, &twiddles, candidate_transform);
            });
            suite.run(BenchmarkCase::new(core, "incumbent-a", n), || {
                run_bench_transform(&source, &mut work, &plan, &twiddles, incumbent_transform);
            });
            suite.run(BenchmarkCase::new(core, "incumbent-b", n), || {
                run_bench_transform(&source, &mut work, &plan, &twiddles, incumbent_transform);
            });
            suite.run(BenchmarkCase::new(core, "final-store-b", n), || {
                run_bench_transform(&source, &mut work, &plan, &twiddles, candidate_transform);
            });
        }
        println!("SINK cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}

/// Piece attribution for the small-size split: gather, base blocks, combine.
#[test]
#[ignore = "measurement instrument for the split's piece costs"]
fn split_pieces_by_size() {
    use super::instance_major::{transform_128, Plan128};

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
    let plan = Plan128::<f64>::new_if_supported::<false>().expect("four-lane host");
    for n in [256usize, 512] {
        let blocks = n / 128;
        let bits = blocks.trailing_zeros();
        let src: Vec<Complex64> = (0..n)
            .map(|i| {
                let x = i as f64;
                Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
            })
            .collect();
        let mut scratch = src.clone();
        let mut out = src.clone();
        let calls = 8000u32;
        let (mut t_gather, mut t_base, mut t_route) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let production = crate::FftPlan1D::<f64>::new(
            crate::Shape1D::new(n).expect("invariant: shape lengths are non-zero"),
        );
        for _ in 0..12 {
            let t = std::time::Instant::now();
            for _ in 0..calls {
                for (b, block) in scratch.chunks_exact_mut(128).enumerate().take(blocks) {
                    let offset = b.reverse_bits() >> (usize::BITS - bits);
                    for (j, slot) in block.iter_mut().enumerate() {
                        *slot = src[j * blocks + offset];
                    }
                }
                std::hint::black_box(&mut scratch);
            }
            t_gather = t_gather.min(t.elapsed().as_nanos() as f64 / f64::from(calls));

            let t = std::time::Instant::now();
            for _ in 0..calls {
                for block in scratch.chunks_exact_mut(128).take(blocks) {
                    assert!(transform_128::<f64, false>(
                        std::hint::black_box(block),
                        &plan
                    ));
                }
            }
            t_base = t_base.min(t.elapsed().as_nanos() as f64 / f64::from(calls));

            let t = std::time::Instant::now();
            for _ in 0..calls {
                out.copy_from_slice(&src);
                production.forward_complex_slice_inplace(std::hint::black_box(&mut out));
            }
            t_route = t_route.min(t.elapsed().as_nanos() as f64 / f64::from(calls));
        }
        let combine = t_route - t_gather - t_base;
        println!(
            "SPLIT cpu={landed:<2} n={n:<4} route={t_route:>7.1} gather={t_gather:>6.1} bases={t_base:>6.1} combine+copy~={combine:>6.1}"
        );
    }
}
