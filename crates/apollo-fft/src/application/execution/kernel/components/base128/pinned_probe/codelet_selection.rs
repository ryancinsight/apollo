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
fn relative_gap(left: &[eunomia::Complex32], right: &[eunomia::Complex32]) -> f32 {
    let scale = left
        .iter()
        .chain(right.iter())
        .map(|value| value.re.abs().max(value.im.abs()))
        .fold(0.0_f32, f32::max)
        .max(f32::MIN_POSITIVE);
    left.iter()
        .zip(right)
        .map(|(a, b)| (a.re - b.re).abs().max((a.im - b.im).abs()))
        .fold(0.0_f32, f32::max)
        / scale
}

#[cfg(test)]
fn codelet_against_composite(suite: &mut BenchmarkSuite, core: &str) {
    use crate::application::execution::kernel::mixed_radix::caches::cached_prime23_radices;
    use crate::application::execution::kernel::mixed_radix::dispatch::static_prime23_radices;
    use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
    use eunomia::Complex32;

    for &n in ACCEPTED {
        // The alternative is whatever `dispatch_inplace` reaches once the
        // codelet declines, and it consults the static table before the cached
        // one. Measuring the cached order for a length the static table
        // answers would time a route the dispatcher never takes.
        let cached = cached_prime23_radices(n);
        let radices: &[usize] = match (static_prime23_radices(n), cached.as_deref()) {
            (Some(statik), _) => statik,
            (None, Some(cached)) => cached,
            (None, None) => continue,
        };
        let source: Vec<Complex32> = (0..n)
            .map(|index| {
                let x = index as f32;
                Complex32::new((0.017_f32 * x).sin(), 0.25 * (0.031_f32 * x).cos())
            })
            .collect();

        // Both routes must agree before either time means anything.
        let mut codelet_out = source.clone();
        if !<f32 as MixedRadixScalar>::short_winograd::<false, false>(&mut codelet_out) {
            eprintln!("n={n}: the codelet arm declined the length; skipping");
            continue;
        }
        let mut composite_out = source.clone();
        composite_forward_with_radices(&mut composite_out, radices);
        let gap = relative_gap(&codelet_out, &composite_out);
        assert!(
            gap < 1e-4,
            "n={n}: the two routes disagree by {gap:e} relative, so they are \
             not the same transform and no timing between them is meaningful"
        );

        let mut work = source.clone();
        suite.run(BenchmarkCase::new(core, "codelet", n), || {
            work.copy_from_slice(&source);
            let _ = <f32 as MixedRadixScalar>::short_winograd::<false, false>(
                std::hint::black_box(&mut work),
            );
        });

        let mut work_composite = source.clone();
        suite.run(BenchmarkCase::new(core, "composite", n), || {
            work_composite.copy_from_slice(&source);
            composite_forward_with_radices(std::hint::black_box(&mut work_composite), radices);
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
        codelet_against_composite(&mut suite, core);
        println!("CODELET cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}
