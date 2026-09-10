//! Where a 3-D forward spends its time: the lane loop, the transposes, or
//! neither.
//!
//! A `FftPlan3D` forward is three axis passes. Axis 2's lanes are contiguous
//! and run in place; axes 1 and 0 each transpose the volume into scratch, run
//! their lanes there, and transpose back — four full-volume transposes per
//! forward. The kwavers baseline (`#kw-fft3d-baseline`) measured the whole
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
//! - `transpose-y` / `transpose-x` — the two transposes axis 1 and axis 0 pay,
//!   there and back, as `transpose_matrices` with each axis's real geometry.
//!
//! Every arm is a round trip so its buffer returns to its input and no
//! per-iteration reseed is charged to the reading. A forward is then
//! `3 x lanes + transpose-y + transpose-x` in per-pass units, and the report
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

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use eunomia::Complex64;
use hermes_simd::ProcessorIndex;
use leto::Array3;

use super::super::lanes;
use super::super::layout::transpose_matrices;
use crate::application::execution::kernel::measurement_cores;
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
    let error = result
        .iter()
        .zip(input)
        .map(|(actual, start)| (actual.re - start.re).hypot(actual.im - start.im))
        .fold(0.0_f64, f64::max);
    // A transpose pair is exact; a transform pair carries 2 log2(n) stages of
    // pairwise addition over unit-scale inputs. 64 n eps sits far above either
    // and far below the O(1) a routing error would show.
    let bound = 64.0 * (n as f64) * f64::EPSILON;
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
}

fn processor_histogram(n: usize) {
    for slot in &LANES_PER_PROCESSOR {
        slot.store(0, Ordering::Relaxed);
    }
    let shape = Shape3D::new(n, n, n).expect("invariant: extents are non-zero");
    let plan = FftPlan3D::<f64>::new(shape);
    let mut data = volume(n);
    lane_pass::<true>(&plan, &mut data, true);

    let total: u32 = LANES_PER_PROCESSOR
        .iter()
        .map(|slot| slot.load(Ordering::Relaxed))
        .sum();
    println!("LANES n={n}: {total} lanes in one forward pass");
    let Some(selection) = measurement_cores::selected() else {
        println!("  host reports no processor class information");
        return;
    };
    for (class_label, class) in [
        ("performance", selection.performance_class()),
        ("efficiency", selection.efficiency_class()),
    ] {
        let Some(class) = class else { continue };
        let members: Vec<u32> = selection.processors_in_class(class).collect();
        let lanes: u32 = members
            .iter()
            .map(|&processor| {
                let slot = usize::try_from(processor).expect("invariant: processor index fits");
                LANES_PER_PROCESSOR[slot].load(Ordering::Relaxed)
            })
            .sum();
        let share = if total == 0 {
            0.0
        } else {
            100.0 * f64::from(lanes) / f64::from(total)
        };
        println!(
            "  {class_label}: {lanes} lanes ({share:.1}%) over {} processors",
            members.len()
        );
    }
    let busy: Vec<String> = LANES_PER_PROCESSOR
        .iter()
        .enumerate()
        .filter(|(_, slot)| slot.load(Ordering::Relaxed) > 0)
        .map(|(processor, slot)| format!("{processor}:{}", slot.load(Ordering::Relaxed)))
        .collect();
    println!("  per processor: {}", busy.join(" "));
}

#[test]
#[ignore = "measurement instrument for the 3-D axis-pass attribution"]
fn axis_pass_attribution() {
    if cfg!(debug_assertions) {
        eprintln!(
            "pass_attribution: built without optimization; re-run with  --cargo-profile bench-quick. No timings reported."
        );
        return;
    }

    // A discarded pass warms the freshly linked binary; only the second is
    // reported.
    let mut warmup = BenchmarkSuite::new(BenchmarkConfig::regression());
    for n in EXTENTS {
        arms_for_extent(&mut warmup, n);
    }
    drop(warmup);

    let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
    for n in EXTENTS {
        arms_for_extent(&mut suite, n);
    }
    println!("PASS unpinned; every row is a round trip (two of the named operation)");
    let report = suite.report();
    print!("{report}");

    for n in EXTENTS {
        // Per forward: three lane passes, axis 1's transpose pair and axis
        // 0's. Each arm is a round trip, so halve the lane and full rows.
        // Printed at the median and at the fastest sample: on a loaded host
        // the median of the full arm carries whatever else ran during it, and
        // the residual it leaves is then contention, not the plan.
        for (statistic, read) in [
            ("median", median_ps as fn(&str, &str) -> f64),
            ("min", min_ps),
        ] {
            let full = read(&report, &format!("unpinned/full/{n}")) / 2.0;
            let lanes = read(&report, &format!("unpinned/lanes-z/{n}")) / 2.0;
            let ty = read(&report, &format!("unpinned/transpose-y/{n}"));
            let tx = read(&report, &format!("unpinned/transpose-x/{n}"));
            let pieces = 3.0 * lanes + ty + tx;
            println!(
                "ATTRIBUTION n={n} {statistic}: forward {:.1} us; lanes 3 x {:.1} = {:.1} us ({:.0}%), transposes {:.1} + {:.1} = {:.1} us ({:.0}%); pieces sum {:.1} us, unaccounted {:.1} us ({:.0}%)",
                full / 1e6,
                lanes / 1e6,
                3.0 * lanes / 1e6,
                100.0 * 3.0 * lanes / full,
                ty / 1e6,
                tx / 1e6,
                (ty + tx) / 1e6,
                100.0 * (ty + tx) / full,
                pieces / 1e6,
                (full - pieces) / 1e6,
                100.0 * (full - pieces) / full,
            );
        }
        let lanes = median_ps(&report, &format!("unpinned/lanes-z/{n}")) / 2.0;
        for (statistic, read) in [
            ("median", median_ps as fn(&str, &str) -> f64),
            ("min", min_ps),
        ] {
            let axis = read(&report, &format!("unpinned/axis1-pass/{n}")) / 2.0;
            let pieces = read(&report, &format!("unpinned/transpose-y/{n}"))
                + read(&report, &format!("unpinned/lanes-z/{n}")) / 2.0;
            println!(
                "INTERACTION n={n} {statistic}: axis-1 pass {:.1} us against its pieces {:.1} us; gap {:.1} us ({:.0}%)",
                axis / 1e6, pieces / 1e6, (axis - pieces) / 1e6, 100.0 * (axis - pieces) / axis
            );
        }
        let ty_serial = median_ps(&report, &format!("unpinned/transpose-y/{n}"));
        let ty_parallel = median_ps(&report, &format!("unpinned/transpose-y-parallel/{n}"));
        println!(
            "TRANSPOSE n={n}: axis-1 pair serial {:.1} us; one matrix per task {:.1} us",
            ty_serial / 1e6,
            ty_parallel / 1e6
        );
        let serial = median_ps(&report, &format!("unpinned/lanes-serial/{n}")) / 2.0;
        let widths: Vec<String> = TASK_WIDTHS
            .iter()
            .map(|width| {
                let pass = median_ps(&report, &format!("unpinned/lanes-task{width}/{n}")) / 2.0;
                format!("{width} lanes/task {:.1} us", pass / 1e6)
            })
            .collect();
        println!(
            "SCHEDULING n={n}: shipped lanes::execute {:.1} us; serial on one thread {:.1} us; {}",
            lanes / 1e6,
            serial / 1e6,
            widths.join("; ")
        );
        processor_histogram(n);
    }
}
