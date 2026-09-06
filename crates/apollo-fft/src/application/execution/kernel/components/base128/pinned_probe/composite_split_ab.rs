//! Same-run A/B: fused composite codelet body vs its split variant, per
//! scalar, per length.
//!
//! The split variant (transform phases in `#[inline(never)]` helpers) exists
//! so a scalar whose fused monomorphization explodes into spills — the
//! documented f32 LLVM codegen asymmetry — can take the split instead. A
//! cross-run ratio cannot decide the routing: f64's absolute level drifts
//! tens of percent between runs on this machine, which once drew the
//! `use_generated_codelet_plan` boundary on flattened numbers. This probe
//! times all four arms (f64/fused, f64/split, f32/fused, f32/split) inside
//! one pinned run per core, so the per-length routing decision is a
//! within-run quantity.
//!
//! Each split arm replicates the exact phase order and buffer roles of the
//! equivalence test in `composite::split_equivalence_tests`: Good–Thomas
//! (n=50) runs rows → cols through scratch, Cooley–Tukey (n=144) runs
//! cols → rows. A divergence there would time a different computation than
//! the one the dispatch would route to.

use crate::application::execution::kernel::measurement_cores;
use crate::application::execution::kernel::components::winograd::composite::split::{
    dft144_cols, dft144_rows, dft50_cols, dft50_rows,
};
use crate::application::execution::kernel::components::winograd::composite::{dft144_impl, dft50_impl};
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use eunomia::{Complex, Complex32, Complex64};
use hermes_simd::{ProcessorBinding, ProcessorIndex};

#[cfg(test)]
fn source_f64(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            Complex::new((0.017 * x).sin(), (0.031 * x).cos())
        })
        .collect()
}

#[cfg(test)]
fn source_f32(n: usize) -> Vec<Complex32> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            Complex::new((0.017 * x).sin() as f32, (0.031 * x).cos() as f32)
        })
        .collect()
}

#[cfg(test)]
macro_rules! ab_arm {
    ($fused:ident, $rows:ident, $cols:ident, $F:ty, $n:literal, rows_cols) => {
        #[inline(never)]
        pub(crate) fn fused(data: &mut [Complex<$F>]) {
            $fused::<$F, false>(data.try_into().expect("A/B arm length"));
        }
        #[inline(never)]
        pub(crate) fn split(data: &mut [Complex<$F>]) {
            let data: &mut [Complex<$F>; $n] = data.try_into().expect("A/B arm length");
            let mut scratch = [Complex::<$F>::new(0.0, 0.0); $n];
            $rows::<$F, false>(data, &mut scratch);
            $cols::<$F, false>(&mut scratch, data);
        }
    };
    ($fused:ident, $rows:ident, $cols:ident, $F:ty, $n:literal, cols_rows) => {
        #[inline(never)]
        pub(crate) fn fused(data: &mut [Complex<$F>]) {
            $fused::<$F, false>(data.try_into().expect("A/B arm length"));
        }
        #[inline(never)]
        pub(crate) fn split(data: &mut [Complex<$F>]) {
            let data: &mut [Complex<$F>; $n] = data.try_into().expect("A/B arm length");
            let mut scratch = [Complex::<$F>::new(0.0, 0.0); $n];
            $cols::<$F, false>(data, &mut scratch);
            $rows::<$F, false>(&mut scratch, data);
        }
    };
}

// One arm module per (length, family); each exposes `fused` and `split` for
// that scalar. Good–Thomas (n=50) runs rows → cols, Cooley–Tukey (n=144)
// runs cols → rows, matching the equivalence test.
#[cfg(test)]
mod ab50_f64 {
    use super::*;
    ab_arm!(dft50_impl, dft50_rows, dft50_cols, f64, 50, rows_cols);
}
#[cfg(test)]
mod ab50_f32 {
    use super::*;
    ab_arm!(dft50_impl, dft50_rows, dft50_cols, f32, 50, rows_cols);
}
#[cfg(test)]
mod ab144_f64 {
    use super::*;
    ab_arm!(dft144_impl, dft144_rows, dft144_cols, f64, 144, cols_rows);
}
#[cfg(test)]
mod ab144_f32 {
    use super::*;
    ab_arm!(dft144_impl, dft144_rows, dft144_cols, f32, 144, cols_rows);
}

#[test]
#[ignore = "routing instrument: same-run fused-vs-split A/B per scalar"]
fn composite_split_ab_by_core_type() {
    if cfg!(debug_assertions) {
        eprintln!("composite_split_ab: built without optimization; re-run with --cargo-profile bench-quick. No timings reported.");
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

        // n=50: fused vs split, both scalars, one run.
        suite.run_batched(
            BenchmarkCase::new(core, "fused-f64", 50),
            || source_f64(50),
            |work| ab50_f64::fused(std::hint::black_box(work)),
        );
        suite.run_batched(
            BenchmarkCase::new(core, "split-f64", 50),
            || source_f64(50),
            |work| ab50_f64::split(std::hint::black_box(work)),
        );
        suite.run_batched(
            BenchmarkCase::new(core, "fused-f32", 50),
            || source_f32(50),
            |work| ab50_f32::fused(std::hint::black_box(work)),
        );
        suite.run_batched(
            BenchmarkCase::new(core, "split-f32", 50),
            || source_f32(50),
            |work| ab50_f32::split(std::hint::black_box(work)),
        );

        // n=144 likewise.
        suite.run_batched(
            BenchmarkCase::new(core, "fused-f64", 144),
            || source_f64(144),
            |work| ab144_f64::fused(std::hint::black_box(work)),
        );
        suite.run_batched(
            BenchmarkCase::new(core, "split-f64", 144),
            || source_f64(144),
            |work| ab144_f64::split(std::hint::black_box(work)),
        );
        suite.run_batched(
            BenchmarkCase::new(core, "fused-f32", 144),
            || source_f32(144),
            |work| ab144_f32::fused(std::hint::black_box(work)),
        );
        suite.run_batched(
            BenchmarkCase::new(core, "split-f32", 144),
            || source_f32(144),
            |work| ab144_f32::split(std::hint::black_box(work)),
        );

        println!("SPLIT-AB cpu={landed} ({core})");
        print!("{}", suite.report());
    }
}
