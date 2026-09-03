//! The generated-codelet arm against the decomposition it displaces.
//!
//! `use_generated_codelet_plan` routes 25 lengths to the short-Winograd
//! codelets ahead of any composite decomposition, and nothing has ever checked
//! that the codelet is the faster of the two. It is not always: n = 180
//! measures 3.6x against `Composite([4, 3, 3, 5])`, while n = 96 measures
//! faster than its composite order. This runs both routes over the same data
//! for every length the predicate accepts, so the boundary is measured rather
//! than assumed.

use crate::application::execution::kernel::benchmark_kernels::composite_forward_with_radices;
use crate::application::execution::kernel::measurement_cores;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use hermes_simd::{ProcessorBinding, ProcessorIndex};

/// Every length `use_generated_codelet_plan` accepts, in both scalar arms.
const ACCEPTED: &[usize] = &[
    72, 81, 96, 99, 108, 112, 120, 121, 126, 128, 144, 154, 168, 180, 189, 222, 242, 246, 259, 275,
    280, 296, 363, 400, 484,
];

/// Largest componentwise gap between two spectra, relative to the larger
/// magnitude present, so the two routes are known to compute the same
/// transform before either timing is believed.
#[cfg(test)]
fn relative_gap<F: Copy + Into<f64>>(
    left: &[eunomia::Complex<F>],
    right: &[eunomia::Complex<F>],
) -> f64 {
    let magnitude = |value: &eunomia::Complex<F>| {
        let (re, im): (f64, f64) = (value.re.into(), value.im.into());
        re.abs().max(im.abs())
    };
    let scale = left
        .iter()
        .chain(right.iter())
        .map(magnitude)
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let (ar, ai): (f64, f64) = (a.re.into(), a.im.into());
            let (br, bi): (f64, f64) = (b.re.into(), b.im.into());
            (ar - br).abs().max((ai - bi).abs())
        })
        .fold(0.0_f64, f64::max)
        / scale
}

#[cfg(test)]
fn codelet_against_composite<F>(suite: &mut BenchmarkSuite, core: &str, arm: &str)
where
    F: MixedRadixScalar<Complex = eunomia::Complex<F>>
        + crate::application::execution::kernel::components::winograd::ShortWinogradScalar
        + Copy
        + Into<f64>,
{
    use crate::application::execution::kernel::mixed_radix::caches::cached_coprime_factors;
    use crate::application::execution::kernel::mixed_radix::caches::cached_prime23_radices;
    use crate::application::execution::kernel::mixed_radix::dispatch::{
        static_coprime_factors, static_prime23_radices,
    };

    let arm_codelet: &str = &format!("codelet-{arm}");
    let arm_composite: &str = &format!("composite-{arm}");
    let arm_pfa: &str = &format!("pfa-{arm}");

    for &n in ACCEPTED {
        // The alternative is whatever `dispatch_inplace` reaches once the
        // codelet declines, and it consults the static table before the cached
        // one. Measuring the cached order for a length the static table
        // answers would time a route the dispatcher never takes.
        let cached = cached_prime23_radices(n);
        let radices: Option<&[usize]> = match (static_prime23_radices(n), cached.as_deref()) {
            (Some(statik), _) => Some(statik),
            (None, Some(cached)) => Some(cached),
            (None, None) => None,
        };
        // A length with a prime above 23 has no prime-2/3 composite; the
        // dispatcher answers it from the coprime tables instead, so that is the
        // arm to compare against for 222, 246, 259 and 296.
        let coprime = radices.and(None).or_else(|| {
            static_coprime_factors(n)
                .or_else(|| (n > 64).then(|| cached_coprime_factors(n)).flatten())
        });
        let alternative = |data: &mut [eunomia::Complex<F>]| {
            if let Some(radices) = radices {
                composite_forward_with_radices(data, radices);
            } else if let Some((n1, n2)) = coprime {
                crate::application::execution::kernel::components::good_thomas::pfa_fft::<F, false>(
                    data, n1, n2,
                );
            }
        };
        if radices.is_none() && coprime.is_none() {
            eprintln!("n={n}: neither a prime-2/3 composite nor a coprime split; skipping");
            continue;
        }
        let alt_label = if radices.is_some() {
            arm_composite
        } else {
            arm_pfa
        };
        let source: Vec<eunomia::Complex<F>> = (0..n)
            .map(|index| {
                let x = index as f64;
                F::complex((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
            })
            .collect();

        // Both routes must agree before either time means anything.
        let mut codelet_out = source.clone();
        if !F::short_winograd::<false, false>(&mut codelet_out) {
            eprintln!("n={n}: the codelet arm declined the length; skipping");
            continue;
        }
        let mut composite_out = source.clone();
        alternative(&mut composite_out);
        let gap = relative_gap(&codelet_out, &composite_out);
        assert!(
            gap < 1e-4,
            "n={n}: the two routes disagree by {gap:e} relative, so they are \
             not the same transform and no timing between them is meaningful"
        );

        let mut work = source.clone();
        suite.run(BenchmarkCase::new(core, arm_codelet, n), || {
            work.copy_from_slice(&source);
            let _ = F::short_winograd::<false, false>(std::hint::black_box(&mut work));
        });

        let mut work_composite = source.clone();
        suite.run(BenchmarkCase::new(core, alt_label, n), || {
            work_composite.copy_from_slice(&source);
            alternative(std::hint::black_box(&mut work_composite));
        });
    }
}

#[test]
#[ignore = "measurement instrument for the codelet-selection boundary"]
fn codelet_selection_by_core_type() {
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
        // No separate warm-up suite: the equivalence check inside
        // `codelet_against_composite` already runs each arm once per length
        // before either is timed, so the twiddle and plan caches are warm. A
        // second full pass doubled the probe's cost and pushed it past the
        // 60-second runner bound.
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
        codelet_against_composite::<f32>(&mut suite, core, "f32");
        println!("CODELET cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}

#[test]
#[ignore = "measurement instrument for the codelet-selection boundary (f64)"]
fn codelet_selection_f64_by_core_type() {
    // The f64 arm carries its own copy of the accepted list, so its boundary is
    // its own question: the codelet's cost relative to a composite pass differs
    // with the scalar width. Separate test, separate runner budget.
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
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
        codelet_against_composite::<f64>(&mut suite, core, "f64");
        println!("CODELET64 cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}
