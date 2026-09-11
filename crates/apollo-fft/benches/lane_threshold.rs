//! Lane-parallel crossover at whole and half-volume task geometries.
//!
//! Compare immutable executables of each candidate with the same instrument.
//! The CSV preserves all observations and median intervals; minima alone
//! cannot establish a crossover or exclude control regressions.
//!
//! Checks an analytical spectrum and dense round trips before reporting. Run with
//! `cargo bench -p apollo-fft --bench lane_threshold`. Rows are round trips.

use apollo_bench::{
    bind_measurement_processor, BenchmarkCase, BenchmarkConfig, BenchmarkError, BenchmarkMode,
    BenchmarkSuite,
};
use eunomia::Complex;
use leto::Array3;

use apollo_fft::{PlanCacheProvider, PlanScratch, RealFftData, Shape3D};

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

/// Two impulses at z=0 and z=nz/4 have an exact quarter-turn spectrum.
/// Their L1 norm is 1.5, so the existing forward error bound remains
/// informative even for the largest single-precision case. This preflight
/// does not change the dense input or operations in the timing region.
fn impulses<T>(shape @ [nx, ny, nz]: [usize; 3], unit: f64)
where
    T: RealFftData + PlanCacheProvider + From<f32> + Into<f64>,
    T::PlanScalar: PlanCacheProvider + Into<f64>,
    Complex<T::PlanScalar>: PlanScratch,
{
    assert_eq!(nz % 4, 0);
    let plan = T::get_3d_plan(Shape3D::new(nx, ny, nz).expect("nonzero probe shape"));
    let input = Array3::from_shape_fn(shape, |index| {
        T::from(if index == [0, 0, 0] {
            1.0
        } else if index == [0, 0, nz / 4] {
            0.5
        } else {
            0.0
        })
    });
    let one = 16.0 * ((nx * ny * nz) as f64).log2() * unit * 1.5;
    let check_spectrum = |spectrum: &Array3<Complex<T::PlanScalar>>, depth: usize| {
        for (index, value) in spectrum.iter().enumerate() {
            let (real, imaginary) = match (index % depth) % 4 {
                0 => (1.5, 0.0),
                1 => (1.0, -0.5),
                2 => (0.5, 0.0),
                _ => (1.0, 0.5),
            };
            let actual: (f64, f64) = (value.re.into(), value.im.into());
            let error = (actual.0 - real).hypot(actual.1 - imaginary);
            assert!(
                error.is_finite() && error <= one,
                "{} {shape:?} bin {index}: error {error} exceeds {one}",
                std::any::type_name::<T>()
            );
        }
    };
    let mut complex = input.mapv(RealFftData::to_spectrum);
    plan.forward_complex_inplace(&mut complex);
    check_spectrum(&complex, nz);
    plan.inverse_complex_inplace(&mut complex);
    for (value, expected) in complex.iter().zip(input.iter()) {
        let actual: (f64, f64) = (value.re.into(), value.im.into());
        let error = (actual.0 - (*expected).into()).hypot(actual.1);
        assert!(error.is_finite() && error <= 2.0 * one);
    }
    let depth = nz / 2 + 1;
    let mut spectrum = Array3::from_elem([nx, ny, depth], Complex::<T::PlanScalar>::default());
    T::forward_3d_half_into(&plan, &input, &mut spectrum);
    check_spectrum(&spectrum, depth);
    let mut back = Array3::from_elem(shape, T::from(0.0));
    T::inverse_3d_half_into(&plan, &mut spectrum, &mut back);
    for (value, expected) in back.iter().zip(input.iter()) {
        let actual: f64 = (*value).into();
        let error = (actual - (*expected).into()).abs();
        assert!(error.is_finite() && error <= 2.0 * one);
    }
}

fn arms<T>(suite: &mut BenchmarkSuite, shape @ [nx, ny, nz]: [usize; 3], unit: f64)
where
    T: RealFftData + PlanCacheProvider + From<f32> + Into<f64>,
    T::PlanScalar: PlanCacheProvider + Into<f64>,
    Complex<T::PlanScalar>: PlanScratch,
{
    impulses::<T>(shape, unit);
    let plan = T::get_3d_plan(
        Shape3D::new(nx, ny, nz).expect("invariant: the probe's extents are non-zero"),
    );
    let total = nx * ny * nz;
    let real = Array3::from_shape_fn(shape, |[i, j, k]| {
        let x = ((i * ny + j) * nz + k) as f32;
        T::from((0.017 * x).sin() + 0.25 * (0.031 * x).cos())
    });
    let reference: Vec<f64> = real.iter().map(|&value| value.into()).collect();
    let l1: f64 = reference.iter().map(|value| value.abs()).sum();
    // The forward and the normalized inverse each within 16 log2(N) u ||x||_1
    // per sample, the bound `tests/real_half_api` derives for both pairs.
    let bound = 2.0 * 16.0 * (total as f64).log2() * unit * l1;
    let check = |pair: &str, values: &[f64]| {
        assert_eq!(values.len(), reference.len());
        let error = values
            .iter()
            .zip(&reference)
            .map(|(actual, start)| {
                let error = (actual - start).abs();
                assert!(error.is_finite(), "{pair}: non-finite error");
                error
            })
            .fold(0.0_f64, f64::max);
        assert!(
            error <= bound,
            "{pair} {}: round trip departs from its input by {error:e} against {bound:e}",
            label(shape)
        );
    };

    let case = format!("{}/{}", std::any::type_name::<T>(), label(shape));
    let mut complex = real.mapv(RealFftData::to_spectrum);
    plan.forward_complex_inplace(&mut complex);
    plan.inverse_complex_inplace(&mut complex);
    check(
        "complex",
        &complex.iter().map(|z| z.re.into()).collect::<Vec<_>>(),
    );
    for value in &complex {
        let imaginary: f64 = value.im.into();
        assert!(imaginary.is_finite() && imaginary.abs() <= bound);
    }
    suite.run(BenchmarkCase::new("complex", &case, total), || {
        plan.forward_complex_inplace(std::hint::black_box(&mut complex));
        plan.inverse_complex_inplace(std::hint::black_box(&mut complex));
    });

    let mut half = Array3::from_elem([nx, ny, nz / 2 + 1], Complex::<T::PlanScalar>::default());
    let mut back = Array3::from_elem(shape, T::from(0.0));
    T::forward_3d_half_into(&plan, &real, &mut half);
    T::inverse_3d_half_into(&plan, &mut half, &mut back);
    check(
        "real",
        &back.iter().map(|&value| value.into()).collect::<Vec<_>>(),
    );
    suite.run(BenchmarkCase::new("real", &case, total), || {
        T::forward_3d_half_into(&plan, std::hint::black_box(&real), &mut half);
        T::inverse_3d_half_into(&plan, &mut half, std::hint::black_box(&mut back));
    });
}

fn main() -> Result<(), BenchmarkError> {
    let processor = bind_measurement_processor()?;
    eprintln!("lane_threshold: {}", processor.describe());
    let mode = BenchmarkMode::from_environment()?;
    assert!(
        !cfg!(debug_assertions) || mode == BenchmarkMode::Smoke,
        "measurement requires an optimized build"
    );

    // A discarded pass warms plans, scratch, and the freshly linked binary.
    let config = mode.apply(BenchmarkConfig::regression());
    let mut warmup = BenchmarkSuite::new(config);
    for shape in SHAPES {
        arms::<f64>(&mut warmup, shape, f64::EPSILON / 2.0);
        arms::<f32>(&mut warmup, shape, f64::from(f32::EPSILON) / 2.0);
    }
    drop(warmup);

    let mut suite = BenchmarkSuite::new(config);
    for shape in SHAPES {
        arms::<f64>(&mut suite, shape, f64::EPSILON / 2.0);
        arms::<f32>(&mut suite, shape, f64::from(f32::EPSILON) / 2.0);
    }
    let report = suite.report();
    print!("{report}");
    Ok(())
}
