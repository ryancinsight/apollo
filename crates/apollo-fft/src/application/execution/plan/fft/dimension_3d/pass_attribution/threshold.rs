//! Where a lane pass starts to gain from moirai, read at the shapes that
//! straddle `lanes::PARALLEL_THRESHOLD`.
//!
//! The threshold is one element count for every lane pass: serial below it,
//! moirai-parallel at or above it. The complex pair runs its passes on the
//! whole volume and the real pair on the `(nx, ny, nz/2 + 1)` half volume, so
//! the shapes here put one or both of them on either side of each candidate
//! value. The constant is compile-time, so a sweep rebuilds this probe once
//! per candidate and reads the rows side by side; each row prints the element
//! counts its passes run on, so it is read against the candidate in force.
//!
//! Reports; asserts only that each round trip returns to its input. Run with
//! `--run-ignored all --no-capture` under `bench-quick`.

use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use eunomia::Complex64;
use leto::Array3;

use super::{median_ps, min_ps};
use crate::{FftPlan3D, RealFftData, Shape3D};

/// Shapes whose whole or half volume falls on either side of 16,384 and
/// 32,768 elements: 16³ (4,096 / 2,304), 24³ (13,824 / 7,488), 32×32×16
/// (16,384 / 9,216), 32³ (32,768 / 17,408), 48³ (110,592 / 57,600).
const SHAPES: [[usize; 3]; 5] = [
    [16, 16, 16],
    [24, 24, 24],
    [32, 32, 16],
    [32, 32, 32],
    [48, 48, 48],
];

fn label([nx, ny, nz]: [usize; 3]) -> String {
    format!("{nx}x{ny}x{nz}")
}

fn arms(suite: &mut BenchmarkSuite, shape @ [nx, ny, nz]: [usize; 3]) {
    let plan = FftPlan3D::<f64>::new(
        Shape3D::new(nx, ny, nz).expect("invariant: the probe's extents are non-zero"),
    );
    let total = nx * ny * nz;
    let real = Array3::from_shape_fn(shape, |[i, j, k]| {
        let x = ((i * ny + j) * nz + k) as f64;
        (0.017 * x).sin() + 0.25 * (0.031 * x).cos()
    });
    let reference: Vec<f64> = real.iter().copied().collect();
    let l1: f64 = reference.iter().map(|value| value.abs()).sum();
    // The forward and the normalized inverse each within 16 log2(N) u ||x||_1
    // per sample, the bound `tests/real_half_api` derives for both pairs.
    let bound = 2.0 * 16.0 * (total as f64).log2() * (f64::EPSILON / 2.0) * l1;
    let check = |pair: &str, values: &[f64]| {
        let error = values
            .iter()
            .zip(&reference)
            .map(|(actual, start)| (actual - start).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            error <= bound,
            "{pair} {}: round trip departs from its input by {error:e} against {bound:e}",
            label(shape)
        );
    };

    let mut complex = real.mapv(|value| Complex64::new(value, 0.0));
    plan.forward_complex_inplace(&mut complex);
    plan.inverse_complex_inplace(&mut complex);
    check("complex", &complex.iter().map(|z| z.re).collect::<Vec<_>>());
    suite.run(BenchmarkCase::new("complex", label(shape), total), || {
        plan.forward_complex_inplace(std::hint::black_box(&mut complex));
        plan.inverse_complex_inplace(std::hint::black_box(&mut complex));
    });

    let mut half = Array3::from_elem([nx, ny, nz / 2 + 1], Complex64::default());
    let mut back = Array3::from_elem(shape, 0.0_f64);
    f64::forward_3d_half_into(&plan, &real, &mut half);
    f64::inverse_3d_half_into(&plan, &mut half, &mut back);
    check("real", back.as_slice().expect("C order"));
    suite.run(BenchmarkCase::new("real", label(shape), total), || {
        f64::forward_3d_half_into(&plan, std::hint::black_box(&real), &mut half);
        f64::inverse_3d_half_into(&plan, &mut half, std::hint::black_box(&mut back));
    });
}

#[test]
#[ignore = "measurement instrument for the lane-pass parallel threshold"]
fn lane_threshold_crossover() {
    if cfg!(debug_assertions) {
        eprintln!(
            "lane_threshold_crossover: built without optimization; re-run with --cargo-profile bench-quick. No timings reported."
        );
        return;
    }

    // A discarded pass warms plans, scratch, and the freshly linked binary.
    let mut warmup = BenchmarkSuite::new(BenchmarkConfig::regression());
    for shape in SHAPES {
        arms(&mut warmup, shape);
    }
    drop(warmup);

    let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
    for shape in SHAPES {
        arms(&mut suite, shape);
    }
    let report = suite.report();
    // Each arm is a round trip, so a transform is half a row.
    for shape @ [nx, ny, nz] in SHAPES {
        let total = nx * ny * nz;
        let key = |pair: &str| format!("{pair}/{}/{total}", label(shape));
        let read = |pair: &str| {
            (
                min_ps(&report, &key(pair)) / 2e6,
                median_ps(&report, &key(pair)) / 2e6,
            )
        };
        let ((complex_min, complex_median), (real_min, real_median)) =
            (read("complex"), read("real"));
        println!(
            "CROSSOVER {}: passes on {total} and {} elements; complex min {complex_min:.1} median {complex_median:.1} us; real min {real_min:.1} median {real_median:.1} us",
            label(shape),
            nx * ny * (nz / 2 + 1),
        );
    }
}
