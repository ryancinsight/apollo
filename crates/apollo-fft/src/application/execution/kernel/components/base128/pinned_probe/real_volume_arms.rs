//! The 64³ real half-spectrum pair, timed as the consumer runs it.
//!
//! kwavers' split-field PSTD step spends 75 to 86% of each of its two
//! spectral phases inside these two entries
//! (`backlog.md#apollo-pstd-consumer-transform-weight`). Its own probe timed
//! each phase against the transforms that phase runs — one forward and three
//! inverses for the velocity update, three of each for the density update —
//! and recorded 1369 and 1402 microseconds for those two arms on a quiet
//! host, 1483 and 1582 under peer builds.
//!
//! Two arms of a known composition are two equations in the forward cost `f`
//! and the inverse cost `i`: `f + 3i` and `3f + 3i` differ by `2f` alone. The
//! consumer's numbers solve to a forward under 50 microseconds against an
//! inverse near 460, a ratio no pair of same-size transforms should show, and
//! `2f` there is a small difference between two large means. This probe runs
//! the same two arms on apollo's own buffers, beside the two transforms timed
//! singly, so the solve is checked against direct measurements rather than
//! trusted: the single arms are the oracle for the pair arms.
//!
//! Every buffer is C-contiguous, so no arm takes the consumer's staging
//! branch; that is the point of comparing against the consumer's figure
//! rather than reproducing its call.
//!
//! The two compositions run over one spectrum and one real output, which is
//! the smallest working set the sequence admits: 2 MiB of field, 1 MiB of
//! spectrum, 2 MiB of output. The consumer's velocity update instead reads
//! one field, writes a spectrum the inverses never read, reads a second
//! spectrum, and writes three distinct real outputs -- 10 MiB across six
//! buffers. `velocity-consumer-shape` runs the same four transforms over that
//! buffer set, so the difference between the two velocity arms is the
//! caller's layout and nothing else.
//!
//! # Why this probe binds no processor
//!
//! A 64³ transform hands its lanes to moirai, whose workers span the machine
//! (`super::volume_schedule`). Binding the calling thread constrains the
//! caller and nothing that does the work, so a per-core table here reports
//! one machine-wide number twice under two labels — and when the host
//! carries peer load, two different ones, which reads as a core effect that
//! is not there. A first pass of this probe did exactly that and put an
//! efficiency-core four-transform arm below its own single inverse. The
//! consumer binds nothing either, so machine-wide is also the comparable
//! measurement. Constraining the pool needs process affinity applied before
//! the pool exists, which is `volume_schedule`'s mechanism and its question,
//! not this one's.

use crate::application::execution::kernel::measurement_cores;
use crate::{FftPlan3D, PlanCacheProvider, RealFftData, Shape3D};
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use eunomia::Complex64;
use leto::Array3;

/// The consumer's grid: 64³ f64, a 2 MiB real field and a 1 MiB half
/// spectrum, neither cache-resident.
const N: usize = 64;

/// Inverse transforms per arm; both of the consumer's spectral phases run
/// three spectral derivatives, so three inverses each.
const INVERSES: usize = 3;

/// A deterministic field with energy across the band, so no arm runs on a
/// spectrum the kernels could short-circuit.
fn field() -> Array3<f64> {
    Array3::from_shape_fn([N, N, N], |[i, j, k]| {
        let x = ((i * N + j) * N + k) as f64;
        (0.017 * x).sin() + 0.25 * (0.031 * x).cos()
    })
}

#[test]
#[ignore = "measurement instrument for the consumer's 64 cubed transform weight"]
fn real_volume_arms_against_the_consumer() {
    let Some(selection) = measurement_cores::selected() else {
        eprintln!("host reports no processor class information; probe not measurable");
        return;
    };
    print!("{}", selection.describe());

    let plan = <f64 as PlanCacheProvider>::get_3d_plan(
        Shape3D::new(N, N, N).expect("invariant: the probe's extents are non-zero"),
    );
    let real = field();
    let mut half = Array3::from_elem([N, N, N / 2 + 1], Complex64::default());
    let mut back = Array3::from_elem([N, N, N], 0.0_f64);

    // The round trip once, checked, before anything is timed: an arm that
    // does not transform its input is not an arm.
    <f64 as RealFftData>::forward_3d_half_into(&plan, &real, &mut half);
    <f64 as RealFftData>::inverse_3d_half_into(&plan, &mut half, &mut back);
    let error = back
        .iter()
        .zip(&real)
        .map(|(actual, start)| (actual - start).abs())
        .fold(0.0_f64, f64::max);
    let l1: f64 = real.iter().map(|value| value.abs()).sum();
    // The bound `tests/real_half_api` derives for this pair: the forward and
    // the normalized inverse each within 16 log2(N³) u ||x||₁.
    let bound = 2.0 * 16.0 * ((N * N * N) as f64).log2() * f64::EPSILON / 2.0 * l1;
    assert!(
        error <= bound,
        "the 64 cubed round trip departs from its input by {error:e} against {bound:e}"
    );

    // A discarded pass warms the plan's twiddles and scratch, the page tables
    // of every buffer, and the freshly linked binary.
    let mut forward_out = Array3::from_elem([N, N, N / 2 + 1], Complex64::default());
    let mut gradient = Array3::from_elem([N, N, N / 2 + 1], Complex64::default());
    <f64 as RealFftData>::forward_3d_half_into(&plan, &real, &mut gradient);
    let mut outputs: [Array3<f64>; INVERSES] =
        core::array::from_fn(|_| Array3::from_elem([N, N, N], 0.0_f64));

    let mut eviction_scratch = vec![0_u64; EVICTION_BYTES / size_of::<u64>()];

    let mut warmup = BenchmarkSuite::new(BenchmarkConfig::regression());
    arms(&mut warmup, &plan, &real, &mut half, &mut back);
    cold_arms(
        &mut warmup,
        &plan,
        &real,
        &mut half,
        &mut back,
        &mut eviction_scratch,
    );
    consumer_shape_arm(
        &mut warmup,
        &plan,
        &real,
        &mut forward_out,
        &mut gradient,
        &mut outputs,
    );
    drop(warmup);

    let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
    arms(&mut suite, &plan, &real, &mut half, &mut back);
    consumer_shape_arm(
        &mut suite,
        &plan,
        &real,
        &mut forward_out,
        &mut gradient,
        &mut outputs,
    );
    cold_arms(
        &mut suite,
        &plan,
        &real,
        &mut half,
        &mut back,
        &mut eviction_scratch,
    );
    println!("RVA machine-wide");
    print!("{}", suite.report());
}

/// The four arms: each transform singly, then the consumer's two phase
/// compositions over the same buffers.
fn arms(
    suite: &mut BenchmarkSuite,
    plan: &FftPlan3D<f64>,
    real: &Array3<f64>,
    half: &mut Array3<Complex64>,
    back: &mut Array3<f64>,
) {
    let total = N * N * N;
    suite.run(BenchmarkCase::new("forward", "machine", total), || {
        <f64 as RealFftData>::forward_3d_half_into(plan, std::hint::black_box(real), half);
    });
    suite.run(BenchmarkCase::new("inverse", "machine", total), || {
        <f64 as RealFftData>::inverse_3d_half_into(plan, half, std::hint::black_box(back));
    });
    suite.run(BenchmarkCase::new("velocity-arm", "machine", total), || {
        <f64 as RealFftData>::forward_3d_half_into(plan, std::hint::black_box(real), half);
        for _ in 0..INVERSES {
            <f64 as RealFftData>::inverse_3d_half_into(plan, half, std::hint::black_box(back));
        }
    });
    suite.run(BenchmarkCase::new("density-arm", "machine", total), || {
        for _ in 0..INVERSES {
            <f64 as RealFftData>::forward_3d_half_into(plan, std::hint::black_box(real), half);
        }
        for _ in 0..INVERSES {
            <f64 as RealFftData>::inverse_3d_half_into(plan, half, std::hint::black_box(back));
        }
    });
}

/// Bytes the consumer's velocity update writes beside its transforms: three
/// velocity components and their lane scratch over a 64³ grid, enough to
/// evict every transform buffer from L2 between one arm and the next.
const EVICTION_BYTES: usize = 12 << 20;

/// The velocity arm run on cold buffers, beside the eviction alone.
///
/// The consumer times its transform arm alternately with the update arm
/// inside one loop, so every transform arm starts on buffers the update has
/// just evicted; this probe's other arms repeat back to back on hot ones.
/// The difference between these two cases is the transforms' cost from a
/// cold cache.
fn cold_arms(
    suite: &mut BenchmarkSuite,
    plan: &FftPlan3D<f64>,
    real: &Array3<f64>,
    half: &mut Array3<Complex64>,
    back: &mut Array3<f64>,
    scratch: &mut [u64],
) {
    let total = N * N * N;
    suite.run(
        BenchmarkCase::new("eviction-only", "machine", total),
        || {
            for (index, slot) in scratch.iter_mut().enumerate() {
                *slot = index as u64;
            }
            std::hint::black_box(&scratch);
        },
    );
    suite.run(
        BenchmarkCase::new("velocity-cold", "machine", total),
        || {
            for (index, slot) in scratch.iter_mut().enumerate() {
                *slot = index as u64;
            }
            std::hint::black_box(&scratch);
            <f64 as RealFftData>::forward_3d_half_into(plan, std::hint::black_box(real), half);
            for _ in 0..INVERSES {
                <f64 as RealFftData>::inverse_3d_half_into(plan, half, std::hint::black_box(back));
            }
        },
    );
}

/// The velocity arm's four transforms over the consumer's buffer set: the
/// forward's spectrum is not the inverses' spectrum, and each inverse writes
/// its own real output.
fn consumer_shape_arm(
    suite: &mut BenchmarkSuite,
    plan: &FftPlan3D<f64>,
    real: &Array3<f64>,
    forward_out: &mut Array3<Complex64>,
    gradient: &mut Array3<Complex64>,
    outputs: &mut [Array3<f64>; INVERSES],
) {
    let total = N * N * N;
    suite.run(
        BenchmarkCase::new("velocity-consumer-shape", "machine", total),
        || {
            <f64 as RealFftData>::forward_3d_half_into(
                plan,
                std::hint::black_box(real),
                forward_out,
            );
            for out in outputs.iter_mut() {
                <f64 as RealFftData>::inverse_3d_half_into(
                    plan,
                    gradient,
                    std::hint::black_box(out),
                );
            }
        },
    );
}
