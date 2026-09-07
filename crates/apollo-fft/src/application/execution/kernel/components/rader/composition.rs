//! Same-run attribution for the half-cyclic prime path.
//!
//! The ignored probe is deliberately kept beside the Rader composition code:
//! it measures the plan/Rader entry, the two half-convolution dispatch arms,
//! and the split/recombine phases on the queried processor classes from the
//! same process and suite. It is an attribution instrument, not a release
//! acceptance test; the native transform tests remain the correctness gate.

use super::convolution::rader_negacyclic_convolve_inplace;
use super::generator;
use super::{rader_fft_with_convolution_backend, HalfCyclicWinograd};
use crate::application::execution::kernel::components::butterflies;
use crate::application::execution::kernel::measurement_cores;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use eunomia::Complex32;
use hermes_simd::{ProcessorBinding, ProcessorIndex};
use std::sync::Arc;

const TARGET: usize = 101;
const CONTROLS: &[usize] = &[97, 113, 151];

#[derive(Clone)]
struct CompositionInput {
    padded: Vec<Complex32>,
    cyclic_spectrum: Arc<[Complex32]>,
    negacyclic_spectrum: Arc<[Complex32]>,
    twiddles: Arc<[Complex32]>,
}

#[derive(Clone)]
struct PhaseInput {
    first: Vec<Complex32>,
    second: Vec<Complex32>,
    twiddles: Arc<[Complex32]>,
}

fn signal(len: usize) -> Vec<Complex32> {
    (0..len)
        .map(|index| {
            let x = index as f32;
            Complex32::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

fn composition_input(n: usize) -> CompositionInput {
    let m = (n - 1) / 2;
    let (_, generator_inverse) = generator::primitive_root_and_inverse(n);
    let (cyclic_spectrum, negacyclic_spectrum) =
        <f32 as MixedRadixScalar>::cached_rader_negacyclic_spectra::<false>(n, generator_inverse);
    CompositionInput {
        padded: signal(2 * m),
        cyclic_spectrum,
        negacyclic_spectrum,
        twiddles: <f32 as MixedRadixScalar>::cached_rader_neg_twiddles(m),
    }
}

fn phase_input(n: usize) -> PhaseInput {
    let m = (n - 1) / 2;
    let source = signal(2 * m);
    let (first, second) = source.split_at(m);
    PhaseInput {
        first: first.to_vec(),
        second: second.to_vec(),
        twiddles: <f32 as MixedRadixScalar>::cached_rader_neg_twiddles(m),
    }
}

fn run_composition(input: &mut CompositionInput) {
    {
        let padded = std::hint::black_box(input.padded.as_mut_slice());
        let cyclic_spectrum = std::hint::black_box(input.cyclic_spectrum.as_ref());
        let negacyclic_spectrum = std::hint::black_box(input.negacyclic_spectrum.as_ref());
        let twiddles = std::hint::black_box(input.twiddles.as_ref());
        rader_negacyclic_convolve_inplace::<f32>(
            padded,
            cyclic_spectrum,
            negacyclic_spectrum,
            twiddles,
        );
    }
    std::hint::black_box(input.padded.as_slice());
}

fn phase_split(
    first: &mut [Complex32],
    second: &mut [Complex32],
    twiddles: &[Complex32],
    twiddle: bool,
) {
    assert_eq!(first.len(), second.len());
    assert_eq!(first.len(), twiddles.len());
    let mut j = 0usize;
    let len4 = (first.len() / 4) * 4;
    while j < len4 {
        let a0 = first[j];
        let b0 = second[j];
        let a1 = first[j + 1];
        let b1 = second[j + 1];
        let a2 = first[j + 2];
        let b2 = second[j + 2];
        let a3 = first[j + 3];
        let b3 = second[j + 3];
        first[j] = a0 + b0;
        first[j + 1] = a1 + b1;
        first[j + 2] = a2 + b2;
        first[j + 3] = a3 + b3;
        let d0 = a0 - b0;
        let d1 = a1 - b1;
        let d2 = a2 - b2;
        let d3 = a3 - b3;
        if twiddle {
            second[j] = d0 * twiddles[j];
            second[j + 1] = d1 * twiddles[j + 1];
            second[j + 2] = d2 * twiddles[j + 2];
            second[j + 3] = d3 * twiddles[j + 3];
        } else {
            second[j] = d0;
            second[j + 1] = d1;
            second[j + 2] = d2;
            second[j + 3] = d3;
        }
        j += 4;
    }
    while j < first.len() {
        let a = first[j];
        let b = second[j];
        first[j] = a + b;
        let difference = a - b;
        second[j] = if twiddle {
            difference * twiddles[j]
        } else {
            difference
        };
        j += 1;
    }
}

fn phase_recombine(
    first: &mut [Complex32],
    second: &mut [Complex32],
    twiddles: &[Complex32],
    twiddle: bool,
) {
    assert_eq!(first.len(), second.len());
    assert_eq!(first.len(), twiddles.len());
    let half = Complex32::new(0.5, 0.0);
    let mut j = 0usize;
    let len4 = (first.len() / 4) * 4;
    while j < len4 {
        let c0 = first[j];
        let c1 = first[j + 1];
        let c2 = first[j + 2];
        let c3 = first[j + 3];
        let n0 = if twiddle {
            butterflies::mul_conj::<f32>(second[j], twiddles[j])
        } else {
            second[j]
        };
        let n1 = if twiddle {
            butterflies::mul_conj::<f32>(second[j + 1], twiddles[j + 1])
        } else {
            second[j + 1]
        };
        let n2 = if twiddle {
            butterflies::mul_conj::<f32>(second[j + 2], twiddles[j + 2])
        } else {
            second[j + 2]
        };
        let n3 = if twiddle {
            butterflies::mul_conj::<f32>(second[j + 3], twiddles[j + 3])
        } else {
            second[j + 3]
        };
        first[j] = (c0 + n0) * half;
        second[j] = (c0 - n0) * half;
        first[j + 1] = (c1 + n1) * half;
        second[j + 1] = (c1 - n1) * half;
        first[j + 2] = (c2 + n2) * half;
        second[j + 2] = (c2 - n2) * half;
        first[j + 3] = (c3 + n3) * half;
        second[j + 3] = (c3 - n3) * half;
        j += 4;
    }
    while j < first.len() {
        let c = first[j];
        let n = if twiddle {
            butterflies::mul_conj::<f32>(second[j], twiddles[j])
        } else {
            second[j]
        };
        first[j] = (c + n) * half;
        second[j] = (c - n) * half;
        j += 1;
    }
}

#[test]
#[ignore = "measurement instrument for N=101 f32 prime-path attribution"]
fn half_cyclic_composition_attribution_by_core_type() {
    if cfg!(debug_assertions) {
        eprintln!(
            "rader composition: built without optimization; re-run with --cargo-profile \
             bench-quick. No timings reported."
        );
        return;
    }
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
        let label = core.label();
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());

        let target_shape = crate::Shape1D::new(TARGET)
            .expect("invariant: the prime attribution target is non-zero");
        let target_plan = crate::FftPlan1D::<f32>::new(target_shape);
        let target_source = signal(TARGET);
        suite.run_batched(
            BenchmarkCase::new(label, "dispatch/plan", TARGET),
            || target_source.clone(),
            |work| {
                target_plan.forward_complex_slice_inplace(std::hint::black_box(work));
                std::hint::black_box(work[0]);
            },
        );
        suite.run_batched(
            BenchmarkCase::new(label, "dispatch/rader-half", TARGET),
            || target_source.clone(),
            |work| {
                rader_fft_with_convolution_backend::<f32, false, HalfCyclicWinograd>(
                    std::hint::black_box(work),
                );
                std::hint::black_box(work[0]);
            },
        );

        for &n in [TARGET].iter().chain(CONTROLS) {
            let input = composition_input(n);
            suite.run_batched(
                BenchmarkCase::new(label, "dispatch/composition", n),
                || input.clone(),
                run_composition,
            );
        }

        let phase = phase_input(TARGET);
        suite.run_batched(
            BenchmarkCase::new(label, "phase/split-no-twiddle", TARGET),
            || phase.clone(),
            |work| {
                {
                    let first = std::hint::black_box(work.first.as_mut_slice());
                    let second = std::hint::black_box(work.second.as_mut_slice());
                    let twiddles = std::hint::black_box(work.twiddles.as_ref());
                    phase_split(first, second, twiddles, false);
                }
                std::hint::black_box(work.first.as_slice());
                std::hint::black_box(work.second.as_slice());
            },
        );
        suite.run_batched(
            BenchmarkCase::new(label, "phase/split-with-twiddle", TARGET),
            || phase.clone(),
            |work| {
                {
                    let first = std::hint::black_box(work.first.as_mut_slice());
                    let second = std::hint::black_box(work.second.as_mut_slice());
                    let twiddles = std::hint::black_box(work.twiddles.as_ref());
                    phase_split(first, second, twiddles, true);
                }
                std::hint::black_box(work.first.as_slice());
                std::hint::black_box(work.second.as_slice());
            },
        );
        suite.run_batched(
            BenchmarkCase::new(label, "phase/recombine-no-twiddle", TARGET),
            || phase.clone(),
            |work| {
                {
                    let first = std::hint::black_box(work.first.as_mut_slice());
                    let second = std::hint::black_box(work.second.as_mut_slice());
                    let twiddles = std::hint::black_box(work.twiddles.as_ref());
                    phase_recombine(first, second, twiddles, false);
                }
                std::hint::black_box(work.first.as_slice());
                std::hint::black_box(work.second.as_slice());
            },
        );
        suite.run_batched(
            BenchmarkCase::new(label, "phase/recombine-with-twiddle", TARGET),
            || phase.clone(),
            |work| {
                {
                    let first = std::hint::black_box(work.first.as_mut_slice());
                    let second = std::hint::black_box(work.second.as_mut_slice());
                    let twiddles = std::hint::black_box(work.twiddles.as_ref());
                    phase_recombine(first, second, twiddles, true);
                }
                std::hint::black_box(work.first.as_slice());
                std::hint::black_box(work.second.as_slice());
            },
        );

        println!("RADER COMPOSITION cpu={landed} ({label})");
        print!("{}", suite.report());
    }
}
