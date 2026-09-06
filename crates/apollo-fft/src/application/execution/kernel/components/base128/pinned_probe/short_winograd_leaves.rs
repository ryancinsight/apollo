//! Leaf-codelet width audit for the short-Winograd family.
//!
//! `codelet_selection` times whole generated codelets (Good-Thomas and
//! Cooley-Tukey pairs), and the board item
//! `#atlas-apollo-f32-nonpot-width` narrowed the f32 excess to that family —
//! fifteen of twenty lengths between 1.03x and 1.83x against f64 — without
//! being able to say which leaf inside each pair carries it. This probe
//! times the leaves directly, both scalars inside one pinned run, so the
//! per-leaf f32/f64 ratio is the reported quantity and the defective leaf is
//! named rather than inferred.
//!
//! The leaf set covers every distinct kind of leaf the family reaches:
//! the scalar `ShortWinogradScalar` methods (3, 5, 7, 8, 11, 13, 16, 17, 19,
//! 23, 29, 31, 32), the macro-generated Good-Thomas leaves (6, 10, 14) and
//! the Cooley-Tukey / prime-power leaves (9, 12, 20, 24, 25, 27). Longer
//! generated codelets compose these same leaves, so a defect at a leaf is
//! visible here without reproducing every pair.

use crate::application::execution::kernel::measurement_cores;
use crate::application::execution::kernel::mixed_radix::traits::ShortDft;
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use eunomia::Complex;
use hermes_simd::{ProcessorBinding, ProcessorIndex};

/// Distinct leaves the short-Winograd family reaches, one arm per scalar.
const LEAVES: &[usize] = &[
    3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 19, 20, 23, 24, 25, 27, 29, 31, 32,
];

/// Componentwise gap against the f64 reference, relative to the largest
/// magnitude present. f32 leaves share the exact operation order with their
/// f64 generics, so agreement to f32 epsilon is expected; the check exists to
/// prove both arms compute the same transform before either timing is read.
#[cfg(test)]
fn relative_gap_f32(left: &[Complex<f32>], right: &[Complex<f64>]) -> f64 {
    let mag32 = |v: &Complex<f32>| {
        let (re, im): (f64, f64) = (v.re.into(), v.im.into());
        re.abs().max(im.abs())
    };
    let mag64 = |v: &Complex<f64>| v.re.abs().max(v.im.abs());
    let scale = left
        .iter()
        .map(mag32)
        .fold(0.0_f64, f64::max)
        .max(right.iter().map(mag64).fold(0.0_f64, f64::max))
        .max(f64::MIN_POSITIVE);
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let (ar, ai): (f64, f64) = (a.re.into(), a.im.into());
            (ar - b.re).abs().max((ai - b.im).abs())
        })
        .fold(0.0_f64, f64::max)
        / scale
}

#[cfg(test)]
fn source_f32(n: usize) -> Vec<Complex<f32>> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            Complex::new((0.017 * x).sin() as f32, (0.031 * x).cos() as f32)
        })
        .collect()
}

#[cfg(test)]
fn source_f64(n: usize) -> Vec<Complex<f64>> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            Complex::new((0.017 * x).sin(), (0.031 * x).cos())
        })
        .collect()
}

macro_rules! leaf_stub {
    ($name:ident, $F:ty, $n:literal) => {
        #[cfg(test)]
        #[inline(never)]
        pub(crate) fn $name(data: &mut [Complex<$F>; $n]) {
            <$F as ShortDft<$n>>::dft::<false>(data);
        }
    };
}

leaf_stub!(leaf_f32_3, f32, 3);
leaf_stub!(leaf_f32_5, f32, 5);
leaf_stub!(leaf_f32_6, f32, 6);
leaf_stub!(leaf_f32_7, f32, 7);
leaf_stub!(leaf_f32_8, f32, 8);
leaf_stub!(leaf_f32_9, f32, 9);
leaf_stub!(leaf_f32_10, f32, 10);
leaf_stub!(leaf_f32_11, f32, 11);
leaf_stub!(leaf_f32_12, f32, 12);
leaf_stub!(leaf_f32_13, f32, 13);
leaf_stub!(leaf_f32_16, f32, 16);
leaf_stub!(leaf_f32_17, f32, 17);
leaf_stub!(leaf_f32_19, f32, 19);
leaf_stub!(leaf_f32_20, f32, 20);
leaf_stub!(leaf_f32_23, f32, 23);
leaf_stub!(leaf_f32_24, f32, 24);
leaf_stub!(leaf_f32_25, f32, 25);
leaf_stub!(leaf_f32_27, f32, 27);
leaf_stub!(leaf_f32_29, f32, 29);
leaf_stub!(leaf_f32_31, f32, 31);
leaf_stub!(leaf_f32_32, f32, 32);
leaf_stub!(leaf_f64_3, f64, 3);
leaf_stub!(leaf_f64_5, f64, 5);
leaf_stub!(leaf_f64_6, f64, 6);
leaf_stub!(leaf_f64_7, f64, 7);
leaf_stub!(leaf_f64_8, f64, 8);
leaf_stub!(leaf_f64_9, f64, 9);
leaf_stub!(leaf_f64_10, f64, 10);
leaf_stub!(leaf_f64_11, f64, 11);
leaf_stub!(leaf_f64_12, f64, 12);
leaf_stub!(leaf_f64_13, f64, 13);
leaf_stub!(leaf_f64_16, f64, 16);
leaf_stub!(leaf_f64_17, f64, 17);
leaf_stub!(leaf_f64_19, f64, 19);
leaf_stub!(leaf_f64_20, f64, 20);
leaf_stub!(leaf_f64_23, f64, 23);
leaf_stub!(leaf_f64_24, f64, 24);
leaf_stub!(leaf_f64_25, f64, 25);
leaf_stub!(leaf_f64_27, f64, 27);
leaf_stub!(leaf_f64_29, f64, 29);
leaf_stub!(leaf_f64_31, f64, 31);
leaf_stub!(leaf_f64_32, f64, 32);

#[cfg(test)]
fn run_leaf_f64(n: usize, data: &mut [Complex<f64>]) {
    match n {
        3 => leaf_f64_3(data.try_into().unwrap()),
        5 => leaf_f64_5(data.try_into().unwrap()),
        6 => leaf_f64_6(data.try_into().unwrap()),
        7 => leaf_f64_7(data.try_into().unwrap()),
        8 => leaf_f64_8(data.try_into().unwrap()),
        9 => leaf_f64_9(data.try_into().unwrap()),
        10 => leaf_f64_10(data.try_into().unwrap()),
        11 => leaf_f64_11(data.try_into().unwrap()),
        12 => leaf_f64_12(data.try_into().unwrap()),
        13 => leaf_f64_13(data.try_into().unwrap()),
        16 => leaf_f64_16(data.try_into().unwrap()),
        17 => leaf_f64_17(data.try_into().unwrap()),
        19 => leaf_f64_19(data.try_into().unwrap()),
        20 => leaf_f64_20(data.try_into().unwrap()),
        23 => leaf_f64_23(data.try_into().unwrap()),
        24 => leaf_f64_24(data.try_into().unwrap()),
        25 => leaf_f64_25(data.try_into().unwrap()),
        27 => leaf_f64_27(data.try_into().unwrap()),
        29 => leaf_f64_29(data.try_into().unwrap()),
        31 => leaf_f64_31(data.try_into().unwrap()),
        32 => leaf_f64_32(data.try_into().unwrap()),
        _ => unreachable!("leaf set is fixed"),
    }
}

#[cfg(test)]
fn run_leaf_f32(n: usize, data: &mut [Complex<f32>]) {
    match n {
        3 => leaf_f32_3(data.try_into().unwrap()),
        5 => leaf_f32_5(data.try_into().unwrap()),
        6 => leaf_f32_6(data.try_into().unwrap()),
        7 => leaf_f32_7(data.try_into().unwrap()),
        8 => leaf_f32_8(data.try_into().unwrap()),
        9 => leaf_f32_9(data.try_into().unwrap()),
        10 => leaf_f32_10(data.try_into().unwrap()),
        11 => leaf_f32_11(data.try_into().unwrap()),
        12 => leaf_f32_12(data.try_into().unwrap()),
        13 => leaf_f32_13(data.try_into().unwrap()),
        16 => leaf_f32_16(data.try_into().unwrap()),
        17 => leaf_f32_17(data.try_into().unwrap()),
        19 => leaf_f32_19(data.try_into().unwrap()),
        20 => leaf_f32_20(data.try_into().unwrap()),
        23 => leaf_f32_23(data.try_into().unwrap()),
        24 => leaf_f32_24(data.try_into().unwrap()),
        25 => leaf_f32_25(data.try_into().unwrap()),
        27 => leaf_f32_27(data.try_into().unwrap()),
        29 => leaf_f32_29(data.try_into().unwrap()),
        31 => leaf_f32_31(data.try_into().unwrap()),
        32 => leaf_f32_32(data.try_into().unwrap()),
        _ => unreachable!("leaf set is fixed"),
    }
}

#[test]
#[ignore = "measurement instrument for the f32 width defect inside the codelet family"]
fn short_winograd_leaves_by_core_type() {
    // A debug build measures unoptimized code, which cannot answer a width
    // question. Refuse rather than report a misleading number: apollo defines
    // no `[profile.test]`, so a plain `cargo nextest run` builds this at
    // opt-level 0. Run with `--cargo-profile bench-quick`.
    if cfg!(debug_assertions) {
        eprintln!(
            "short_winograd_leaves: built without optimization; re-run with        --cargo-profile bench-quick. No timings reported."
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
        let core = core.label();

        let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
        for &n in LEAVES {
            // Equivalence first: the f32 leaf against the f64 leaf on the
            // same input. Both arms share the generic operation order, so
            // they must agree to f32 precision.
            let reference = {
                let mut data = source_f64(n);
                run_leaf_f64(n, &mut data);
                data
            };
            let mut candidate = source_f32(n);
            run_leaf_f32(n, &mut candidate);
            let gap = relative_gap_f32(&candidate, &reference);
            assert!(
                gap < 1e-3,
                "n={n}: the f32 leaf disagrees with the f64 leaf by {gap:e} \
                 relative, so no timing between them is meaningful"
            );

            // The reset is setup, not operation: one input per iteration is
            // built before the timer starts (`run_batched`).
            suite.run_batched(
                BenchmarkCase::new(core, "leaf-f64", n),
                || source_f64(n),
                |work| run_leaf_f64(n, std::hint::black_box(work)),
            );
            suite.run_batched(
                BenchmarkCase::new(core, "leaf-f32", n),
                || source_f32(n),
                |work| run_leaf_f32(n, std::hint::black_box(work)),
            );
        }
        println!("LEAVES cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}
