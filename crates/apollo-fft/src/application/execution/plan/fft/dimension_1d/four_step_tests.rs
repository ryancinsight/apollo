//! Generic plan routing against a sparse, exact Fourier-series oracle.

use super::strategy::PlanStrategy;
use super::{FftPlan1D, StaticFftPlan1D};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::Shape1D;
use eunomia::{Complex, Complex64};

#[derive(Clone, Copy, Debug)]
enum Direction {
    Forward,
    Inverse,
    UnnormalizedInverse,
}

fn assert_spectrum<F: Copy + Into<f64>, const N: usize>(
    actual: &[Complex<F>],
    direction: Direction,
    unit_roundoff: f64,
) {
    let n = f64::from(u32::try_from(N).expect("test length fits u32"));
    let scale = match direction {
        Direction::Inverse => n.recip(),
        Direction::Forward | Direction::UnnormalizedInverse => 1.0,
    };
    // Each FFT stage has at most sixteen rounded real operations along an
    // output dependency path. Higham's gamma_k bounds that path's error
    // against the input L1 norm. The two nonzero inputs have norms sqrt(5/4)
    // and sqrt(5/64); the reference uses only exact dyadic operations and
    // fourth roots of unity. Power-of-two normalization is exact here.
    let ku = 16.0 * f64::from(N.ilog2()) * unit_roundoff;
    let bound = ku / (1.0 - ku) * (1.25_f64.sqrt() + 0.078_125_f64.sqrt()) * scale;
    let origin = Complex64::new(1.0, 0.5);
    let quarter = Complex64::new(-0.25, 0.125);
    for (bin, value) in actual.iter().enumerate() {
        let phase = match direction {
            Direction::Forward => (4 - bin % 4) % 4,
            Direction::Inverse | Direction::UnnormalizedInverse => bin % 4,
        };
        let rotated = match phase {
            0 => quarter,
            1 => Complex64::new(-quarter.im, quarter.re),
            2 => -quarter,
            3 => Complex64::new(quarter.im, -quarter.re),
            _ => unreachable!("phase is reduced modulo four"),
        };
        let expected = (origin + rotated) * scale;
        let error = (Complex64::new(value.re.into(), value.im.into()) - expected).norm();
        assert!(
            error <= bound,
            "N={N} {direction:?} bin={bin}: error {error:e} exceeds {bound:e}"
        );
    }
}

fn check_length<F, const N: usize>(unit_roundoff: f64)
where
    F: MixedRadixScalar<Complex = Complex<F>> + From<f32> + Into<f64>,
{
    let dynamic = FftPlan1D::<F>::new(Shape1D::new(N).expect("nonzero test length"));
    let static_plan = StaticFftPlan1D::<F, N>::new();
    let mut input = vec![Complex::new(F::from(0.0), F::from(0.0)); N];
    input[0] = Complex::new(F::from(1.0), F::from(0.5));
    input[N / 4] = Complex::new(F::from(-0.25), F::from(0.125));

    for direction in [
        Direction::Forward,
        Direction::Inverse,
        Direction::UnnormalizedInverse,
    ] {
        let mut dynamic_output = input.clone();
        let mut static_output = input.clone();
        match direction {
            Direction::Forward => {
                dynamic.forward_complex_slice_inplace(&mut dynamic_output);
                static_plan.forward_complex_slice_inplace(&mut static_output);
            }
            Direction::Inverse => {
                dynamic.inverse_complex_slice_inplace(&mut dynamic_output);
                static_plan.inverse_complex_slice_inplace(&mut static_output);
            }
            Direction::UnnormalizedInverse => {
                dynamic.inverse_complex_slice_unnorm_inplace(&mut dynamic_output);
                static_plan.inverse_complex_slice_unnorm_inplace(&mut static_output);
            }
        }
        assert_spectrum::<F, N>(&dynamic_output, direction, unit_roundoff);
        assert_spectrum::<F, N>(&static_output, direction, unit_roundoff);
    }

    // 2048 to 32768 are column-first chains over the width's base where it
    // builds, and 131072 at four lanes (ADR 0061); the four-step route
    // serves the rest past 4096.
    if N == 1024
        || ((N == 2048 || N == 8192 || N == 16384 || N == 131_072) && dynamic.base256.is_some())
        || ((N == 4096 || N == 32_768 || N == 262_144) && dynamic.base512.is_some())
    {
        assert!(matches!(dynamic.strategy, PlanStrategy::PowerOfTwo { .. }));
        assert_eq!(dynamic.twiddle_fwd.as_deref().map_or(0, <[_]>::len), 0);
    } else {
        assert!(matches!(dynamic.strategy, PlanStrategy::FourStep));
        // Execution must not acquire a plan-owned stage table, including on
        // odd powers whose combining pass acquires its own global table.
        assert_eq!(dynamic.twiddle_fwd.as_deref().map_or(0, <[_]>::len), 0);
        assert_eq!(dynamic.twiddle_inv.get().map_or(0, |table| table.len()), 0);
        let cloned = dynamic.clone();
        assert!(matches!(cloned.strategy, PlanStrategy::FourStep));
        assert_eq!(cloned.twiddle_fwd.as_deref().map_or(0, <[_]>::len), 0);
        assert_eq!(cloned.twiddle_inv.get().map_or(0, |table| table.len()), 0);
    }
}

fn check_routes<F>(unit_roundoff: f64)
where
    F: MixedRadixScalar<Complex = Complex<F>> + From<f32> + Into<f64>,
{
    // The sized/base boundary, first odd/even generic routes, and both sides
    // of the planar-to-parallel boundary, including the nested odd split.
    check_length::<F, 1024>(unit_roundoff);
    check_length::<F, 2048>(unit_roundoff);
    check_length::<F, 4096>(unit_roundoff);
    check_length::<F, 8192>(unit_roundoff);
    check_length::<F, 16_384>(unit_roundoff);
    check_length::<F, 32_768>(unit_roundoff);
    check_length::<F, 65_536>(unit_roundoff);
    check_length::<F, 131_072>(unit_roundoff);
    check_length::<F, 262_144>(unit_roundoff);
    check_length::<F, 524_288>(unit_roundoff);
}

#[test]
fn generic_four_step_plans_match_sparse_fourier_series() {
    check_routes::<f32>(f64::from(f32::EPSILON) / 2.0);
    check_routes::<f64>(f64::EPSILON / 2.0);
}

#[test]
#[ignore = "local instrument for the bounded owner used by static plans"]
fn static_plan_owner_profile() {
    use crate::application::execution::kernel::measurement_cores;
    use crate::PlanCacheProvider;
    use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
    use hermes_simd::{ProcessorBinding, ProcessorIndex};

    const N: usize = 65_536;
    let source: Vec<Complex64> = (0..N)
        .map(|index| {
            let x = index as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect();
    let static_plan = StaticFftPlan1D::<f64, N>::new();
    let cached_plan = <f64 as PlanCacheProvider>::get_1d_plan(
        Shape1D::new(N).expect("invariant: the profile length is non-zero"),
    );

    let measure = |label: &str| {
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
        for round in 0..2 {
            suite.run_batched(
                BenchmarkCase::new(label, "static", round),
                || source.clone(),
                |work| static_plan.forward_complex_slice_inplace(std::hint::black_box(work)),
            );
            suite.run_batched(
                BenchmarkCase::new(label, "bounded-owner", round),
                || source.clone(),
                |work| cached_plan.forward_complex_slice_inplace(std::hint::black_box(work)),
            );
        }
        print!("{}", suite.report());
    };

    let Some(selection) = measurement_cores::selected() else {
        println!(
            "processor class unavailable; unpinned run on {} logical processors",
            std::thread::available_parallelism().map_or(0, std::num::NonZero::get)
        );
        measure("unpinned");
        return;
    };
    print!("{}", selection.describe());
    for core in selection.cores() {
        let _binding = ProcessorBinding::bind(core.processor())
            .expect("measurement processor must be available");
        std::thread::yield_now();
        let landed = ProcessorIndex::current().expect("Windows supports processor queries");
        assert_eq!(landed.get(), core.processor().get());
        measure(core.label());
    }
}
