//! Directional attribution at 64³. Each move reads an unchanged source;
//! unlike a transpose pair, its timing cannot hide directional asymmetry.
//! Four blocks reverse case order to expose drift. The 13 cases take about
//! 26 seconds at the existing regression budget, within Nextest's 60 seconds.

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use eunomia::Complex64;
use leto::Array3;

use super::{assert_returns_to_input, transpose_matrices, volume};
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::with_3d_x_scratch;
use crate::{FftPlan3D, Shape3D};

const N: usize = 64;

/// Give this bounded measurement process priority over concurrent compiler
/// work. This changes no production worker policy or other process. The
/// normal-priority run remains available as an explicit control.
fn measurement_priority() {
    if std::env::var_os("APOLLO_MEASUREMENT_HIGH_PRIORITY").is_none() {
        println!("MOVE-GEOMETRY priority=inherited");
        return;
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
        fn SetPriorityClass(process: *mut core::ffi::c_void, priority: u32) -> i32;
    }
    // Windows HIGH_PRIORITY_CLASS, as declared by windows-sys 0.61.2.
    const HIGH_PRIORITY_CLASS: u32 = 0x80;
    // SAFETY: GetCurrentProcess returns a valid pseudo-handle to this process;
    // SetPriorityClass borrows it and accepts the documented priority class.
    let success = unsafe { SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS) };
    assert_ne!(
        success,
        0,
        "cannot set measurement priority: {}",
        std::io::Error::last_os_error()
    );
    println!("MOVE-GEOMETRY priority=high; affinity=inherited");
}

fn round_trip(plan: &FftPlan3D<f64>, array: &mut Array3<Complex64>, case: usize) {
    if case == 10 {
        plan.forward_complex_inplace(array);
        plan.inverse_complex_inplace(array);
        return;
    }
    let mut rotated = plan.forward_complex_rotated(array);
    if case == 11 {
        plan.inverse_complex_rotated(rotated);
        return;
    }
    // All three lengths and twiddle tables coincide at this cubic shape.
    // Only the two inverse moves change task width; the same lane primitive
    // runs around them, and the same thread-local scratch role is borrowed.
    let data = rotated.as_mut_slice();
    super::lane_pass::<false>(plan, data, false);
    with_3d_x_scratch::<Complex64, _>(data.len(), |staged| {
        scheduled_move(data, staged, N * N, N, 256 * 1024);
        super::lane_pass::<false>(plan, staged, false);
        scheduled_move(staged, data, N * N, N, 256 * 1024);
        super::lane_pass::<false>(plan, data, false);
    });
}

/// Same destination-row partition as the provider, with task width controlled
/// by the instrument. The serial kernel and source pitch remain unchanged.
fn scheduled_move(
    source: &[Complex64],
    destination: &mut [Complex64],
    rows: usize,
    columns: usize,
    task_bytes: usize,
) {
    let rows_per_task = (task_bytes / (rows * size_of::<Complex64>())).max(1);
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Parallel, _, _>(
        destination,
        rows_per_task * rows,
        |index, window| {
            let column = index * rows_per_task;
            let width = window.len() / rows;
            let span = (rows - 1) * columns + width;
            leto::transpose_copy_strided(
                &source[column..column + span],
                columns,
                window,
                rows,
                width,
            )
            .expect("invariant: each task spans complete destination rows");
        },
    );
}

fn check_move(source: &[Complex64], destination: &[Complex64], rows: usize, columns: usize) {
    for (row, source_row) in source.chunks_exact(columns).enumerate() {
        for (column, value) in source_row.iter().enumerate() {
            assert_eq!(destination[column * rows + row], *value);
        }
    }
}

fn measure(suite: &mut BenchmarkSuite, case: usize) {
    let source = volume(N);
    let mut destination = vec![Complex64::default(); source.len()];
    // The three chain geometries coincide at a cubic shape, as do the two
    // inverse geometries. Keep their axis labels to make the prescribed
    // attribution explicit, rather than inferring one from a round trip.
    let (label, rows, columns, task_bytes) = match case {
        0 => ("chain-x", N, N * N, None),
        1 => ("chain-y", N, N * N, None),
        2 => ("chain-z", N, N * N, None),
        3 => ("inverse-y", N * N, N, None),
        4 => ("inverse-x", N * N, N, None),
        5 => ("wide-serial", N, N * N, Some(0)),
        6 => ("tall-serial", N * N, N, Some(0)),
        7 => ("tall-task-quarter-mib", N * N, N, Some(256 * 1024)),
        8 => ("tall-task-half-mib", N * N, N, Some(512 * 1024)),
        9 => ("wide-task-quarter-mib", N, N * N, Some(256 * 1024)),
        10..=12 => {
            let plan =
                FftPlan3D::<f64>::new(Shape3D::new(N, N, N).expect("invariant: nonzero extents"));
            let mut array = Array3::from_shape_vec([N, N, N], source.clone())
                .expect("invariant: volume has N cubed elements");
            round_trip(&plan, &mut array, case);
            assert_returns_to_input(
                "transform pair",
                N,
                array.as_slice().expect("invariant: C order"),
                &source,
            );
            let mut run = || {
                round_trip(&plan, std::hint::black_box(&mut array), case);
            };
            suite.run(
                BenchmarkCase::new(
                    "geometry",
                    match case {
                        10 => "full-pair",
                        11 => "rotated-pair",
                        _ => "rotated-task-quarter-mib",
                    },
                    N,
                ),
                &mut run,
            );
            return;
        }
        _ => unreachable!("invariant: the instrument enumerates thirteen cases"),
    };
    let mut run = || match task_bytes {
        None => transpose_matrices(
            std::hint::black_box(&source),
            std::hint::black_box(&mut destination),
            1,
            rows,
            columns,
        ),
        Some(0) => leto::transpose_copy(
            std::hint::black_box(&source),
            std::hint::black_box(&mut destination),
            rows,
            columns,
        )
        .expect("invariant: exact matrix storage"),
        Some(bytes) => scheduled_move(
            std::hint::black_box(&source),
            std::hint::black_box(&mut destination),
            rows,
            columns,
            bytes,
        ),
    };
    run();
    suite.run(BenchmarkCase::new("geometry", label, N), &mut run);
    check_move(&source, &destination, rows, columns);
}

#[test]
#[ignore = "64 cubed directional movement measurement; bench-quick profile"]
fn move_geometry_attribution() {
    #[expect(
        clippy::manual_assert,
        reason = "a constant profile assertion fails assertions_on_constants in dev builds"
    )]
    if cfg!(debug_assertions) {
        panic!("measurement requires bench-quick");
    }
    measurement_priority();
    for block in 0..4 {
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
        for index in 0..13 {
            measure(&mut suite, if block % 2 == 0 { index } else { 12 - index });
        }
        println!("MOVE-GEOMETRY block={block}; moves are single transposes; pairs contain both transforms");
        print!("{}", suite.report());
    }
}
