//! Pinned measurement for the 8 x 128 construction: first the gate — the
//! inner small-size transforms against the references — then the assembled
//! experiment. Sets no performance threshold; run with `--ignored --nocapture`.

use crate::application::execution::kernel::core_class;
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use eunomia::Complex64;
use hermes_simd::{ProcessorBinding, ProcessorIndex};
use rustfft::num_complex::Complex as RustComplex;

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;

type BenchTransform =
    fn(&mut [Complex64], &super::instance_major::Plan128<f64>, &[Complex64]) -> bool;

#[inline(never)]
fn run_bench_transform(
    source: &[Complex64],
    work: &mut [Complex64],
    plan: &super::instance_major::Plan128<f64>,
    twiddles: &[Complex64],
    transform: BenchTransform,
) {
    work.copy_from_slice(source);
    std::hint::black_box(transform)(std::hint::black_box(work), plan, twiddles);
}

fn phase_attribution(
    src: &[Complex64],
    work: &mut [Complex64],
    plan: &super::instance_major::Plan128<f64>,
) -> [u64; 3] {
    use std::sync::atomic::Ordering;

    const CALLS: u64 = 8_192;
    for phase in &super::instance_major::phase_meter::PHASES {
        phase.store(0, Ordering::Relaxed);
    }
    super::instance_major::phase_meter::CALLS.store(0, Ordering::Relaxed);
    for _ in 0..CALLS {
        work.copy_from_slice(src);
        assert!(super::instance_major::transform_128_measured::<f64, false>(
            std::hint::black_box(work),
            plan,
        ));
    }
    let calls = super::instance_major::phase_meter::CALLS
        .load(Ordering::Relaxed)
        .max(1);
    let mut averages = [0; 3];
    for (average, phase) in averages
        .iter_mut()
        .zip(&super::instance_major::phase_meter::PHASES)
    {
        *average = phase.load(Ordering::Relaxed) / calls;
    }
    averages
}

/// Reference-library dispatch for the probe: PhastFT publishes one planner
/// and entry point per scalar, so the generic body selects them by trait.
trait ProbeScalar: MixedRadixScalar + rustfft::FftNum {
    type PhastPlanner;
    fn phast_planner(n: usize) -> Self::PhastPlanner;
    fn phast_forward(re: &mut [Self], im: &mut [Self], planner: &Self::PhastPlanner);
}

impl ProbeScalar for f64 {
    type PhastPlanner = phastft::planner::PlannerDit64;
    fn phast_planner(n: usize) -> Self::PhastPlanner {
        phastft::planner::PlannerDit64::new(n)
    }
    fn phast_forward(re: &mut [Self], im: &mut [Self], planner: &Self::PhastPlanner) {
        phastft::fft_f64_dit_with_planner(re, im, phastft::planner::Direction::Forward, planner);
    }
}

impl ProbeScalar for f32 {
    type PhastPlanner = phastft::planner::PlannerDit32;
    fn phast_planner(n: usize) -> Self::PhastPlanner {
        phastft::planner::PlannerDit32::new(n)
    }
    fn phast_forward(re: &mut [Self], im: &mut [Self], planner: &Self::PhastPlanner) {
        phastft::fft_f32_dit_with_planner(re, im, phastft::planner::Direction::Forward, planner);
    }
}

fn small_sizes_for_scalar<T>(suite: &mut BenchmarkSuite, core: &str, scalar: &str)
where
    T: ProbeScalar + MixedRadixScalar<Complex = eunomia::Complex<T>>,
    eunomia::Complex<T>: bytemuck::Pod,
{
    for n in [64usize, 128, 256, 512] {
        let src: Vec<eunomia::Complex<T>> = (0..n)
            .map(|i| {
                let x = i as f64;
                eunomia::Complex::new(
                    T::from_precise((0.017 * x).sin()),
                    T::from_precise(0.25 * (0.031 * x).cos()),
                )
            })
            .collect();
        let rust_src: Vec<RustComplex<T>> =
            src.iter().map(|v| RustComplex::new(v.re, v.im)).collect();
        let (re_src, im_src): (Vec<T>, Vec<T>) = src.iter().map(|v| (v.re, v.im)).unzip();

        let plan = crate::FftPlan1D::<T>::new(crate::Shape1D { n });
        let rust = rustfft::FftPlanner::<T>::new().plan_fft_forward(n);
        let mut rust_scratch = vec![
            RustComplex::new(T::from_precise(0.0), T::from_precise(0.0));
            rust.get_inplace_scratch_len()
        ];
        let phast = T::phast_planner(n);

        let mut work = src.clone();
        suite.run(
            BenchmarkCase::new(core, format!("apollo-{scalar}"), n),
            || {
                work.copy_from_slice(&src);
                plan.forward_complex_slice_inplace(std::hint::black_box(&mut work));
            },
        );
        if n == 128 {
            let base_plan = super::instance_major::Plan128::<T>::new_if_supported::<false>()
                .expect("the pinned host must provide a native base capability");
            work.copy_from_slice(&src);
            assert!(
                super::instance_major::transform_128::<T, false>(&mut work, &base_plan),
                "the pinned host must provide a native base capability"
            );
            suite.run(
                BenchmarkCase::new(core, format!("base-128-{scalar}"), n),
                || {
                    work.copy_from_slice(&src);
                    std::hint::black_box(super::instance_major::transform_128::<T, false>(
                        std::hint::black_box(&mut work),
                        &base_plan,
                    ));
                },
            );
        }
        let mut rust_work = rust_src.clone();
        suite.run(
            BenchmarkCase::new(core, format!("rustfft-{scalar}"), n),
            || {
                rust_work.copy_from_slice(&rust_src);
                rust.process_with_scratch(std::hint::black_box(&mut rust_work), &mut rust_scratch);
            },
        );
        let (mut re, mut im) = (re_src.clone(), im_src.clone());
        suite.run(
            BenchmarkCase::new(core, format!("phastft-{scalar}"), n),
            || {
                re.copy_from_slice(&re_src);
                im.copy_from_slice(&im_src);
                T::phast_forward(
                    std::hint::black_box(&mut re),
                    std::hint::black_box(&mut im),
                    &phast,
                );
            },
        );
    }
}

#[test]
#[ignore = "measurement instrument for the 8x128 construction's inner gate"]
fn small_sizes_against_the_references_by_core_type() {
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
        let core = core.class().label();
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
        small_sizes_for_scalar::<f64>(&mut suite, core, "f64");
        small_sizes_for_scalar::<f32>(&mut suite, core, "f32");
        {
            let src: Vec<Complex64> = (0..128)
                .map(|i| {
                    let x = i as f64;
                    Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
                })
                .collect();
            let mut work = src.clone();
            let base_plan = super::instance_major::Plan128::<f64>::new_if_supported::<false>()
                .expect("the pinned host must provide the four-lane base capability");
            let phases = phase_attribution(&src, &mut work, &base_plan);
            println!(
                "B128 phases: redistribute={} rows16={} columns8={}",
                phases[0], phases[1], phases[2]
            );
        }
        println!("SML cpu={landed} ({core})");
        suite.emit();
    }
}

#[test]
#[ignore = "paired measurement of the N=512 final-store sink"]
fn final_store_sink_against_incumbent_by_core_type() {
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
        let core = core.class().label();
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
        suite.emit();
    }
}

/// Piece attribution for the small-size split: gather, base blocks, combine.
#[test]
#[ignore = "measurement instrument for the split's piece costs"]
fn split_pieces_by_size() {
    use super::instance_major::{transform_128, Plan128};

    let Some(core) = core_class::selected().and_then(core_class::Selection::performance) else {
        eprintln!("host reports no performance-core information; probe not measurable");
        return;
    };
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
        let production = crate::FftPlan1D::<f64>::new(crate::Shape1D { n });
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
