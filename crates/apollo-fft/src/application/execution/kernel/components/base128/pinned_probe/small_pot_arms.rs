//! The small power-of-two codelet arms, vector against scalar, per core type.
//!
//! Each `small_pot_inplace_sized_precise` arm carries a note recording why it
//! is scalar or vector. Two of those notes — N = 16 and N = 8 — attributed a
//! measured loss to "call plus probe", which dispatch measurements since put
//! at 0.3 ns, and both predate the probe's profile guard. This probe is the
//! instrument that re-decides them: it times the dispatched arm against the
//! scalar Winograd codelet it would replace, at both core types, in one pinned
//! run.
//!
//! **The measured operation is a forward followed by a normalized inverse**,
//! not a forward alone, and that choice is what keeps the instrument sound at
//! these sizes. A codelet this small is a few nanoseconds, so anything the
//! harness does per iteration competes with the body:
//!
//! - Re-seeding the buffer inside the timed closure charges a `copy_from_slice`
//!   to the operation. At N = 8 that is 128 bytes against a body of a few
//!   nanoseconds, and because it is a constant added to both arms it
//!   compresses their *ratio* — which is the quantity the arm decision is
//!   drawn on.
//! - `run_batched` moves the seeding out of the reading, but it does so by
//!   pre-building one input per iteration: calibration asks for tens of
//!   thousands, so the batch is 8-26 MB of separately allocated buffers and
//!   every operand arrives from DRAM. An earlier revision of this probe did
//!   exactly that and read N = 16 at 42 ns against the 6.7 ns the same arm
//!   measures resident — it was timing the memory system, not the codelet.
//!
//! A round trip needs no reset at all: the inverse returns the buffer to its
//! input, so one L1-resident buffer is reused for every iteration with nothing
//! copied. Rounding drift over a sample accumulates as the square root of the
//! iteration count against `f64::EPSILON`, and
//! `round_trip_is_stable_over_a_sample` pins that.
//!
//! What it costs is that the reading is the *sum* of a forward and an inverse,
//! so a per-direction asymmetry would be averaged rather than shown. Both
//! directions here run one body under a const generic, so that is a limit on
//! the quantity's name rather than on the comparison.
//!
//! # The three arms
//!
//! - `dispatched` is what `small_pot_inplace_sized` actually routes to, so it
//!   is the shipped behaviour at each size.
//! - `scalar-winograd` is the scalar codelet, which is what `dispatched`
//!   already is at N = 8 — the two agreeing there is the check that the
//!   declined arm is not wired in.
//! - `vector-direct` is the register codelet reached without the `OnceLock`
//!   capability check. At N = 16 and 32 that is the shipped arm's body; at
//!   N = 8 it is [`super::super::super::super::mixed_radix::scalar`]'s
//!   declined `n8`, which exists only so this comparison stays runnable.
//!   It does **not** remove the `#[target_feature]` call boundary — the
//!   unchecked entries do not carry the attribute either — so what the gap
//!   between it and `dispatched` bounds is the capability check alone.
//!
//! It reports rather than asserts. The value oracles live beside each codelet;
//! what this answers is only which arm is faster, and a loss on either core
//! type closes the corresponding item as falsified.

use crate::application::execution::kernel::components::winograd::{
    dft16_impl, dft32_impl, dft8_array_impl,
};
use crate::application::execution::kernel::measurement_cores;
use crate::application::execution::kernel::mixed_radix::scalar::{
    n16_fused_round_trip, n16_vector_arm_unchecked, n8_fused_round_trip, n8_vector_arm_unchecked,
};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use eunomia::Complex64;
use hermes_simd::{ProcessorBinding, ProcessorIndex};

/// Builds the same deterministic input every arm and every core type sees.
fn source(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|index| {
            let x = index as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

/// One round trip through the dispatched arm — the codelet under test.
fn dispatched_round_trip<const N: usize>(work: &mut [Complex64]) {
    // SAFETY: `small_pot_inplace_sized` requires `data.len() == N`, which every
    // caller establishes by sizing the buffer with `source(N)`.
    unsafe {
        <f64 as MixedRadixScalar>::small_pot_inplace_sized::<N, false, false>(work);
        <f64 as MixedRadixScalar>::small_pot_inplace_sized::<N, true, true>(work);
    }
}

/// One round trip through the scalar Winograd codelet the dispatch replaces.
fn scalar_round_trip<const N: usize>(work: &mut [Complex64]) {
    debug_assert_eq!(
        work.len(),
        N,
        "invariant: the probe sizes every buffer to N"
    );
    let ptr = work.as_mut_ptr();
    // SAFETY: the caller passes a slice of exactly `N` samples, and each arm
    // below casts to the array length its own match guard fixes. The inverse
    // arms normalize, matching the dispatched round trip.
    unsafe {
        match N {
            8 => {
                let data = &mut *ptr.cast::<[Complex64; 8]>();
                dft8_array_impl::<f64, false, false>(data);
                dft8_array_impl::<f64, true, true>(data);
            }
            16 => {
                let data = &mut *ptr.cast::<[Complex64; 16]>();
                dft16_impl::<f64, false>(data);
                dft16_impl::<f64, true>(data);
                let scale = Complex64::new(1.0 / 16.0, 0.0);
                for value in data.iter_mut() {
                    *value *= scale;
                }
            }
            32 => {
                let data = &mut *ptr.cast::<[Complex64; 32]>();
                dft32_impl::<f64, false>(data);
                dft32_impl::<f64, true>(data);
                let scale = Complex64::new(1.0 / 32.0, 0.0);
                for value in data.iter_mut() {
                    *value *= scale;
                }
            }
            _ => unreachable!("invariant: the probe measures N in 8, 16 or 32"),
        }
    }
}

/// One round trip through the vector body reached without the capability check.
///
/// Not a shippable configuration — the capability is established once before
/// the loop rather than per call. It isolates the `OnceLock` probe, not the
/// `#[target_feature]` call boundary, which both arms still pay.
fn direct_round_trip<const N: usize>(work: &mut [Complex64]) {
    // SAFETY: the caller establishes AVX and FMA once before the loop, and
    // sizes the buffer to `N`.
    unsafe {
        match N {
            8 => {
                n8_vector_arm_unchecked::<false, false>(work);
                n8_vector_arm_unchecked::<true, true>(work);
            }
            16 => {
                n16_vector_arm_unchecked::<false, false>(work);
                n16_vector_arm_unchecked::<true, true>(work);
            }
            _ => unreachable!("invariant: the direct arm covers N in 8 or 16"),
        }
    }
}

/// The same two transforms, crossing into a target-feature frame once.
///
/// Paired against [`direct_round_trip`], which crosses twice, this prices the
/// `#[target_feature]` boundary that every vector arm pays and the scalar
/// codelet does not — the scalar codelet inlines into its dispatcher, the
/// vector arm cannot, because a target-feature function is not inlinable into
/// a caller without the feature. The gap is an upper bound on what hoisting
/// the frame would buy; the entries it calls carry the reason.
fn fused_round_trip<const N: usize>(work: &mut [Complex64]) {
    // SAFETY: the caller establishes AVX and FMA once before the loop, and
    // sizes the buffer to `N`.
    unsafe {
        match N {
            8 => n8_fused_round_trip(work),
            16 => n16_fused_round_trip(work),
            _ => unreachable!("invariant: the fused arm covers N in 8 or 16"),
        }
    }
}

/// Confirms both arms compute the same transform before either is timed.
///
/// A timing comparison between arms that disagree measures nothing, so this
/// runs first and panics rather than reporting.
fn assert_arms_agree<const N: usize>() {
    let input = source(N);
    let mut dispatched = input.clone();
    let mut scalar = input.clone();
    dispatched_round_trip::<N>(&mut dispatched);
    scalar_round_trip::<N>(&mut scalar);

    // Every arm's round trip must return its input, which also proves each
    // arm's inverse is the inverse of its own forward. Only arms that ran are
    // checked: an untouched buffer would pass this vacuously.
    let mut arms = vec![("dispatched", dispatched), ("scalar", scalar)];
    // N = 32's arm has no unchecked entry, so it is measured only through
    // the dispatch.
    if N == 8 || N == 16 {
        let mut direct = input.clone();
        direct_round_trip::<N>(&mut direct);
        arms.push(("direct", direct));
        let mut fused = input.clone();
        fused_round_trip::<N>(&mut fused);
        arms.push(("fused", fused));
    }

    for (label, result) in &arms {
        let error = result
            .iter()
            .zip(&input)
            .map(|(actual, start)| (actual.re - start.re).hypot(actual.im - start.im))
            .fold(0.0_f64, f64::max);
        // Two transforms of log2(N) pairwise-addition stages each, over
        // unit-scale inputs; a routing error would instead be O(1).
        let bound = 64.0 * (N as f64) * f64::EPSILON;
        assert!(
            error <= bound,
            "N={N}: the {label} round trip departs from its input by {error:e}              against a derived bound of {bound:e}"
        );
    }
}

fn arms_for_size<const N: usize>(suite: &mut BenchmarkSuite, core: &str) {
    let input = source(N);
    let mut work = input.clone();
    suite.run(BenchmarkCase::new(core, "dispatched", N), || {
        dispatched_round_trip::<N>(std::hint::black_box(&mut work));
    });
    let mut work = input.clone();
    suite.run(BenchmarkCase::new(core, "scalar-winograd", N), || {
        scalar_round_trip::<N>(std::hint::black_box(&mut work));
    });
    if N == 8 || N == 16 {
        let mut work = input.clone();
        suite.run(BenchmarkCase::new(core, "vector-direct", N), || {
            direct_round_trip::<N>(std::hint::black_box(&mut work));
        });
        let mut work = input.clone();
        suite.run(BenchmarkCase::new(core, "vector-fused", N), || {
            fused_round_trip::<N>(std::hint::black_box(&mut work));
        });
    }
}

#[test]
#[ignore = "measurement instrument for the small power-of-two codelet arms"]
fn small_pot_arms_by_core_type() {
    // A debug build measures unoptimized code, which cannot answer a question
    // about vector width. Refuse rather than report a misleading number.
    if cfg!(debug_assertions) {
        eprintln!(
            "small_pot_arms: built without optimization; re-run with \
             --cargo-profile bench-quick. No timings reported."
        );
        return;
    }
    assert_arms_agree::<8>();
    assert_arms_agree::<16>();
    assert_arms_agree::<32>();

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

        // A discarded pass warms the freshly linked binary; only the second
        // one is reported.
        let mut warmup = BenchmarkSuite::new(BenchmarkConfig::regression());
        arms_for_size::<8>(&mut warmup, label);
        arms_for_size::<16>(&mut warmup, label);
        arms_for_size::<32>(&mut warmup, label);
        drop(warmup);

        let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
        arms_for_size::<8>(&mut suite, label);
        arms_for_size::<16>(&mut suite, label);
        arms_for_size::<32>(&mut suite, label);
        println!("ARMS cpu={landed} ({label}) forward+normalized-inverse round trip");
        // `median_ps` is already per-iteration; it is the number as printed.
        print!("{}", suite.report());
    }
}

#[cfg(test)]
mod tests {
    /// The round trip is the instrument's reset, so its drift over a sample's
    /// worth of iterations is part of the instrument's validity.
    #[test]
    fn round_trip_is_stable_over_a_sample() {
        let input = super::source(8);
        let mut work = input.clone();
        for _ in 0..100_000 {
            super::dispatched_round_trip::<8>(&mut work);
        }
        let drift = work
            .iter()
            .zip(&input)
            .map(|(actual, start)| (actual.re - start.re).hypot(actual.im - start.im))
            .fold(0.0_f64, f64::max);
        // Rounding accumulates as a random walk, so 1e5 round trips of six
        // pairwise-addition stages sit near `sqrt(1e5) * 48 * f64::EPSILON`.
        assert!(
            drift < 1.0e-11,
            "round-trip drift {drift:e} over 100k iterations would contaminate the reading"
        );
    }
}
