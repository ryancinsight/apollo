//! Same-run comparison of fused and split composite schedules per scalar and length.
//!
//! Both controls call the same generated arithmetic with a compile-time
//! schedule. Production selects the fused schedule; the split schedule
//! exists only in tests. The generic composite test compares both controls
//! and production dispatch for lengths 50 and 144 in both directions.
//!
//! This forward-only probe measures all four arms inside one pinned run per
//! core. Its results apply to that machine and workload, not other machines
//! or inverse transforms.
#![cfg(test)]

use crate::application::execution::kernel::components::winograd::composite::schedule::{
    Fused, Split,
};
use crate::application::execution::kernel::components::winograd::composite::{
    dft144_impl, dft50_impl,
};
use crate::application::execution::kernel::measurement_cores;
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
    ($codelet:ident, $F:ty, $n:literal) => {
        #[inline(never)]
        pub(crate) fn fused(data: &mut [Complex<$F>]) {
            let data: &mut [Complex<$F>; $n] = data.try_into().expect("A/B arm length");
            $codelet::<$F, false, Fused>(data);
        }
        #[inline(never)]
        pub(crate) fn split(data: &mut [Complex<$F>]) {
            let data: &mut [Complex<$F>; $n] = data.try_into().expect("A/B arm length");
            $codelet::<$F, false, Split>(data);
        }
    };
}

#[cfg(test)]
mod ab50_f64 {
    use super::*;
    ab_arm!(dft50_impl, f64, 50);
}
#[cfg(test)]
mod ab50_f32 {
    use super::*;
    ab_arm!(dft50_impl, f32, 50);
}
#[cfg(test)]
mod ab144_f64 {
    use super::*;
    ab_arm!(dft144_impl, f64, 144);
}
#[cfg(test)]
mod ab144_f32 {
    use super::*;
    ab_arm!(dft144_impl, f32, 144);
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
