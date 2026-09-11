//! Where a 3-D forward spends its time: the lane loop, the transposes, or
//! neither.
//!
//! A `FftPlan3D` forward is three axis passes. The C-order chain uses three
//! full-volume moves; the rotated order uses two and passes its layout to
//! the inverse. The axis-isolation arms below also retain the gather/run/
//! scatter form to separate lane and movement costs. The kwavers baseline
//! (`#kw-fft3d-baseline`) measured the whole
//! forward and *inferred* the split from a codelet extrapolation, then had to
//! correct it when the codelet was measured directly. This probe measures the
//! pieces themselves, on the same volume, so the split is read rather than
//! derived:
//!
//! - `full` — the plan's forward followed by its normalized inverse.
//! - `lanes-z` — one forward lane pass over the contiguous axis, then the
//!   inverse pass, through `lanes::contiguous` with the plan's own twiddle
//!   table: exactly the call axis 2 makes, and the same call axes 0 and 1 make
//!   on their transposed scratch.
//! - `transpose-y` / `transpose-x` — the two transposes a separate axis-1 or
//!   axis-0 pass pays, there and back, as `transpose_matrices` with each
//!   axis's real geometry.
//! - `transpose-chain` — the three single-matrix moves the full transform
//!   runs instead, `(x, y, z)` to `(y, z, x)` to `(z, x, y)` and back.
//! - `rotated-pair` — the same round trip through the entry points that leave
//!   and accept `(z, x, y)`: four moves against `full`'s six.
//!
//! Every arm is a round trip so its buffer returns to its input and no
//! per-iteration reseed is charged to the reading. A forward is then
//! `3 x lanes + transpose-chain` in per-pass units, and the report
//! prints that sum beside the measured forward: the gap is what the pieces do
//! not account for (scratch acquisition, the C-order view, cache state between
//! phases the isolated arms do not reproduce).
//!
//! # Not pinned
//!
//! Unlike the codelet probes this runs on whatever cores the scheduler gives
//! it, because that is what the solver does: at 64³ the lane loop is above
//! moirai's parallel threshold and is spread across core types. Which types is
//! part of the question, so an untimed pass counts the lanes each processor
//! ran, outside the timed arms — an atomic per lane inside the timed closure
//! would contend on one cache line and charge that to the codelet.
//!
//! Reports; asserts nothing beyond agreement of each round trip with its
//! input. Run with `--run-ignored all --no-capture` under `bench-quick`.

use std::sync::atomic::{AtomicU32, Ordering};

use apollo_bench::{BenchmarkCase, BenchmarkSuite};
use eunomia::Complex64;
use hermes_simd::ProcessorIndex;
use leto::Array3;

use super::super::lanes;
use super::super::layout::transpose_matrices;
use crate::application::execution::kernel::mixed_radix::dispatch_inplace;
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::with_3d_y_scratch;
use crate::{FftPlan3D, Shape3D};

/// Extents the consumers plan; the same two the kwavers baseline reads.
const EXTENTS: [usize; 2] = [32, 64];

/// Deterministic, non-degenerate volume; a zero field would let the arithmetic
/// collapse.
fn volume(n: usize) -> Vec<Complex64> {
    (0..n * n * n)
        .map(|index| {
            let x = index as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

/// Largest processor index this histogram can hold; the host reports 24.
const PROCESSOR_SLOTS: usize = 256;

/// Lanes each processor ran during one counted pass.
static LANES_PER_PROCESSOR: [AtomicU32; PROCESSOR_SLOTS] =
    [const { AtomicU32::new(0) }; PROCESSOR_SLOTS];

/// One lane pass in the given direction, exactly as an axis pass issues it.
fn lane_pass<const FORWARD: bool>(
    plan: &FftPlan3D<f64>,
    data: &mut [Complex64],
    count_processors: bool,
) {
    let lane_len = plan.nz;
    let (forward, inverse) = (plan.twiddle_z_fwd.as_deref(), plan.twiddle_z_inv.as_deref());
    let lane_fn = |lane: &mut [Complex64]| {
        if count_processors {
            let processor = ProcessorIndex::current()
                .expect("Windows supports processor queries")
                .get();
            let slot = usize::try_from(processor).expect("invariant: processor index fits usize");
            LANES_PER_PROCESSOR[slot].fetch_add(1, Ordering::Relaxed);
        }
        match (FORWARD, forward, inverse) {
            (true, Some(tw), _) => dispatch_inplace::<f64, false, false>(lane, Some(tw)),
            (false, _, Some(tw)) => dispatch_inplace::<f64, true, true>(lane, Some(tw)),
            _ => {
                if FORWARD {
                    crate::application::execution::kernel::mixed_radix::forward_inplace::<f64>(
                        lane,
                    );
                } else {
                    crate::application::execution::kernel::mixed_radix::inverse_inplace::<f64>(
                        lane,
                    );
                }
            }
        }
    };
    lanes::contiguous::<f64, FORWARD, 3>(data, lane_len, lane_fn);
}

/// The same lane pass with the scheduling chosen here rather than by
/// `lanes::execute`, which hands moirai one lane — one kilobyte at N = 64 —
/// per task.
///
/// `lanes_per_task == 0` runs every lane on the calling thread with no
/// scheduler at all; otherwise moirai receives `lanes_per_task` lanes per task
/// and the closure walks them. The codelet call is identical in every case.
fn lane_pass_scheduled<const FORWARD: bool>(
    plan: &FftPlan3D<f64>,
    data: &mut [Complex64],
    lanes_per_task: usize,
) {
    let lane_len = plan.nz;
    let (forward, inverse) = (plan.twiddle_z_fwd.as_deref(), plan.twiddle_z_inv.as_deref());
    let one_lane = |lane: &mut [Complex64]| match (FORWARD, forward, inverse) {
        (true, Some(tw), _) => dispatch_inplace::<f64, false, false>(lane, Some(tw)),
        (false, _, Some(tw)) => dispatch_inplace::<f64, true, true>(lane, Some(tw)),
        _ => unreachable!("invariant: the extents this probe runs carry cached twiddles"),
    };
    if lanes_per_task == 0 {
        for lane in data.chunks_exact_mut(lane_len) {
            one_lane(lane);
        }
        return;
    }
    // Threshold 1: parallelise regardless of size, so the task width is the
    // only thing this arm varies against the per-lane default.
    moirai::for_each_chunk_mut_with::<moirai::AdaptiveWithThreshold<1>, _, _>(
        data,
        lane_len * lanes_per_task,
        |task| {
            for lane in task.chunks_exact_mut(lane_len) {
                one_lane(lane);
            }
        },
    );
}

/// Task widths, in lanes, the coarse arms try: 64 lanes is 64 KB at N = 64,
/// inside one core's L2 with room; 256 is a quarter megabyte, the coarsest
/// that still gives every processor several tasks at 4,096 lanes.
const TASK_WIDTHS: [usize; 2] = [64, 256];

/// Axis 1's transpose with its `n` matrices as independent tasks.
///
/// Each destination matrix is a contiguous chunk, so moirai hands out disjoint
/// destination chunks by index and the closure reads the matching source
/// matrix from the shared slice; the per-matrix kernel is leto's own.
fn transpose_pair_by_matrix(source: &[Complex64], destination: &mut [Complex64], n: usize) {
    let matrix_len = n * n;
    moirai::for_each_chunk_mut_enumerated_with::<moirai::AdaptiveWithThreshold<1>, _, _>(
        destination,
        matrix_len,
        |index, matrix| {
            let start = index * matrix_len;
            leto::transpose_copy(&source[start..start + matrix_len], matrix, n, n)
                .expect("invariant: every matrix is n x n");
        },
    );
}

/// Axis 1 exactly as the plan runs it: transpose into the thread-local scratch,
/// the lane pass over that scratch, transpose back — in one direction.
///
/// The isolated arms each run hot on their own buffer; here the lanes read
/// scratch that the transpose just wrote from other cores, and the pass pays
/// the scratch acquisition and moirai's join. The gap between this and
/// `2 x transpose-y + lanes` is that interaction, which no isolated arm shows.
fn axis1_pass<const FORWARD: bool>(plan: &FftPlan3D<f64>, data: &mut [Complex64], n: usize) {
    with_3d_y_scratch::<Complex64, _>(n * n * n, |scratch| {
        transpose_matrices(data, scratch, n, n, n);
        lane_pass::<FORWARD>(plan, scratch, false);
        transpose_matrices(scratch, data, n, n, n);
    });
}

fn assert_returns_to_input(label: &str, n: usize, result: &[Complex64], input: &[Complex64]) {
    assert!(matches!(n, 32 | 64));
    assert!(is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma"));
    assert_eq!(result.len(), input.len());
    let error = result
        .iter()
        .zip(input)
        .map(|(actual, start)| {
            let error = (actual.re - start.re).abs() + (actual.im - start.im).abs();
            assert!(error.is_finite(), "{label}: non-finite round-trip error");
            error
        })
        .fold(0.0_f64, f64::max);
    // The AVX/FMA FFT64 in small_pot/precise.rs has six addition layers and
    // three complex-twiddle layers. With u=eps/2 and coefficient error <=8u,
    // each latter layer is bounded by 8u + sqrt(2)*gamma_2*(1+8u) <= gamma_12.
    // Six axis transforms therefore contribute gamma_[6*(6+3*12)]=gamma_252.
    // FFT32 needs only five addition and two twiddle layers. The combine
    // literals were interval-checked at 200 bits against half-angle roots:
    // maximum coefficient error <4.627u (64) and <3.077u (32).
    // volume() has norm_2 <= sqrt(64^3)*(1+1/4)=640. Factor two bounds the
    // component 1-norm by the complex 2-norm; two more rounding steps cover
    // subtraction and addition. This is ONE pair, with normal-range IEEE
    // rounding; inverse power-of-two normalization and permutations are exact.
    let unit_roundoff = f64::EPSILON / 2.0;
    let gamma = 254.0 * unit_roundoff / (1.0 - 254.0 * unit_roundoff);
    let bound = (1280.0 * gamma).next_up();
    assert!(
        error <= bound,
        "{label} at {n}: round trip departs from its input by {error:e} against {bound:e}"
    );
}

/// A column of the named case, read from the suite's CSV report.
///
/// The report is the suite's stable surface (`case,min_ps,median_ps,...`); the
/// record type's fields are private to `apollo-bench`, and this probe needs
/// only the columns it already prints.
fn column_ps(report: &str, name: &str, column: usize) -> f64 {
    report
        .lines()
        .find(|line| line.starts_with(name) && line[name.len()..].starts_with(','))
        .and_then(|line| line.split(',').nth(column))
        .and_then(|value| value.parse::<f64>().ok())
        .expect("invariant: every arm this probe runs is reported under its name")
}

/// Median picoseconds of the named case.
fn median_ps(report: &str, name: &str) -> f64 {
    column_ps(report, name, 2)
}

/// Fastest sample of the named case — under a loaded host the least-contended
/// reading, and so the lower bound the attribution is also printed against.
fn min_ps(report: &str, name: &str) -> f64 {
    column_ps(report, name, 1)
}

fn arms_for_extent(suite: &mut BenchmarkSuite, n: usize) {
    let shape = Shape3D::new(n, n, n).expect("invariant: extents are non-zero");
    let plan = FftPlan3D::<f64>::new(shape);
    let input = volume(n);

    // Plan construction and first-call scratch acquisition are warm-plan costs
    // paid once, not per step; one untimed round trip retires them.
    let mut array = Array3::from_shape_vec([n, n, n], input.clone())
        .expect("invariant: the volume has n^3 elements");
    plan.forward_complex_inplace(&mut array);
    plan.inverse_complex_inplace(&mut array);
    assert_returns_to_input("full", n, array.as_slice().expect("C order"), &input);
    suite.run(BenchmarkCase::new("unpinned", "full", n), || {
        plan.forward_complex_inplace(std::hint::black_box(&mut array));
        plan.inverse_complex_inplace(std::hint::black_box(&mut array));
    });

    let mut data = input.clone();
    lane_pass::<true>(&plan, &mut data, false);
    lane_pass::<false>(&plan, &mut data, false);
    assert_returns_to_input("lanes-z", n, &data, &input);
    suite.run(BenchmarkCase::new("unpinned", "lanes-z", n), || {
        lane_pass::<true>(&plan, std::hint::black_box(&mut data), false);
        lane_pass::<false>(&plan, std::hint::black_box(&mut data), false);
    });

    let mut data = input.clone();
    lane_pass_scheduled::<true>(&plan, &mut data, 0);
    lane_pass_scheduled::<false>(&plan, &mut data, 0);
    assert_returns_to_input("lanes-serial", n, &data, &input);
    suite.run(BenchmarkCase::new("unpinned", "lanes-serial", n), || {
        lane_pass_scheduled::<true>(&plan, std::hint::black_box(&mut data), 0);
        lane_pass_scheduled::<false>(&plan, std::hint::black_box(&mut data), 0);
    });
    for width in TASK_WIDTHS {
        let mut data = input.clone();
        lane_pass_scheduled::<true>(&plan, &mut data, width);
        lane_pass_scheduled::<false>(&plan, &mut data, width);
        let label = format!("lanes-task{width}");
        assert_returns_to_input(&label, n, &data, &input);
        suite.run(BenchmarkCase::new("unpinned", label, n), || {
            lane_pass_scheduled::<true>(&plan, std::hint::black_box(&mut data), width);
            lane_pass_scheduled::<false>(&plan, std::hint::black_box(&mut data), width);
        });
    }

    let mut data = input.clone();
    axis1_pass::<true>(&plan, &mut data, n);
    axis1_pass::<false>(&plan, &mut data, n);
    assert_returns_to_input("axis1-pass", n, &data, &input);
    suite.run(BenchmarkCase::new("unpinned", "axis1-pass", n), || {
        axis1_pass::<true>(&plan, std::hint::black_box(&mut data), n);
        axis1_pass::<false>(&plan, std::hint::black_box(&mut data), n);
    });

    // Axis 1's pair: `nx` matrices of `[ny, nz]` there, `[nz, ny]` back.
    let mut data = input.clone();
    let mut scratch = vec![Complex64::default(); n * n * n];
    transpose_matrices(&data, &mut scratch, n, n, n);
    transpose_matrices(&scratch, &mut data, n, n, n);
    assert_returns_to_input("transpose-y", n, &data, &input);
    suite.run(BenchmarkCase::new("unpinned", "transpose-y", n), || {
        transpose_matrices(std::hint::black_box(&data), &mut scratch, n, n, n);
        transpose_matrices(std::hint::black_box(&scratch), &mut data, n, n, n);
    });

    // Axis 1's pair again, with the `nx` independent matrices spread over
    // moirai tasks — one matrix (64 KiB at N = 64) per task — each through
    // leto's serial tile kernel. The shipped path runs all `nx` on one thread.
    transpose_pair_by_matrix(&data, &mut scratch, n);
    transpose_pair_by_matrix(&scratch, &mut data, n);
    assert_returns_to_input("transpose-y-parallel", n, &data, &input);
    suite.run(
        BenchmarkCase::new("unpinned", "transpose-y-parallel", n),
        || {
            transpose_pair_by_matrix(std::hint::black_box(&data), &mut scratch, n);
            transpose_pair_by_matrix(std::hint::black_box(&scratch), &mut data, n);
        },
    );

    // Axis 0's pair: one `[nx, ny*nz]` matrix there, `[ny*nz, nx]` back.
    transpose_matrices(&data, &mut scratch, 1, n, n * n);
    transpose_matrices(&scratch, &mut data, 1, n * n, n);
    assert_returns_to_input("transpose-x", n, &data, &input);
    suite.run(BenchmarkCase::new("unpinned", "transpose-x", n), || {
        transpose_matrices(std::hint::black_box(&data), &mut scratch, 1, n, n * n);
        transpose_matrices(std::hint::black_box(&scratch), &mut data, 1, n * n, n);
    });

    // The rotated pair: a forward that leaves `(z, x, y)` and an inverse that
    // takes it, two moves each against the C-order pair's three.
    let mut array = Array3::from_shape_vec([n, n, n], input.clone())
        .expect("invariant: the volume has n^3 elements");
    {
        let rotated = plan.forward_complex_rotated(&mut array);
        plan.inverse_complex_rotated(rotated);
    }
    assert_returns_to_input(
        "rotated-pair",
        n,
        array.as_slice().expect("C order"),
        &input,
    );
    suite.run(BenchmarkCase::new("unpinned", "rotated-pair", n), || {
        let rotated = plan.forward_complex_rotated(std::hint::black_box(&mut array));
        plan.inverse_complex_rotated(rotated);
    });

    // The chain the full transform runs: `(x, y, z)` to `(y, z, x)` to
    // `(z, x, y)` and back, three single-matrix moves through two scratches —
    // a cycle, so the arm is its own round trip.
    let mut staged = vec![Complex64::default(); n * n * n];
    transpose_matrices(&data, &mut scratch, 1, n, n * n);
    transpose_matrices(&scratch, &mut staged, 1, n, n * n);
    transpose_matrices(&staged, &mut data, 1, n, n * n);
    assert_returns_to_input("transpose-chain", n, &data, &input);
    suite.run(BenchmarkCase::new("unpinned", "transpose-chain", n), || {
        transpose_matrices(std::hint::black_box(&data), &mut scratch, 1, n, n * n);
        transpose_matrices(std::hint::black_box(&scratch), &mut staged, 1, n, n * n);
        transpose_matrices(std::hint::black_box(&staged), &mut data, 1, n, n * n);
    });
}

#[cfg(test)]
mod report;

#[cfg(test)]
mod move_geometry;
