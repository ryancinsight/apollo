//! Scalar-vs-AVX Stockham backend cost, per size and precision, pinned by
//! core type.
//!
//! The instrument `ATLAS-APOLLO-AVX-STOCKHAM-AUDIT-2026-08-25` requires: both
//! backends instantiated in one binary (the scalar `PreciseStockham` /
//! `ReducedStockham` impls compile under `cfg(test)`), interleaved against the
//! same cache state, thread pinned so the hybrid scheduler is excluded. A
//! same-binary table avoids the cross-build codegen coupling that invalidated
//! the reverted per-size routing experiment: here neither backend's
//! instantiation set changes between the two arms being compared.
//!
//! Asserts nothing about speed; it is a named measurement instrument, run with
//! `--ignored --nocapture` like `pot::core_matrix`. It does panic if the two
//! backends disagree numerically beyond the reduction-order bound, because a
//! wrong-answer backend would make the timing table meaningless.

use super::precision::{
    PreciseStockham, PreciseStockhamAvxFma, ReducedStockham, ReducedStockhamAvxFma,
    StockhamPrecision,
};
use super::transform::transform_sized;
use crate::application::execution::kernel::measurement_cores;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::{Complex32, Complex64};
use hermes_simd::{ProcessorBinding, ProcessorIndex};
use std::time::Instant;

/// Best-of-blocks per-call cost in nanoseconds, clone-inclusive.
///
/// The input restore sits inside the timed region — the comparison contract
/// `rustfft_comparison` documents. An identical additive cost does not cancel
/// in a ratio; it attenuates the ratio toward 1, so orderings survive and the
/// reported advantage of the faster arm is understated, never overstated.
/// Inner-call count scales inversely with `n` so every size times a block
/// long enough for the 100 ns Windows timer granularity to vanish.
fn per_call_ns<P: StockhamPrecision>(
    src: &[P::Complex],
    work: &mut [P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
    log2: u32,
) -> f64 {
    let n = src.len();
    let calls = ((1usize << 18) / n).clamp(8, 2048);
    let mut best = f64::INFINITY;
    for _ in 0..24 {
        let t = Instant::now();
        for _ in 0..calls {
            work.copy_from_slice(src);
            transform_sized::<P>(std::hint::black_box(work), scratch, twiddles, None, log2);
        }
        best = best.min(t.elapsed().as_nanos() as f64 / calls as f64);
        std::hint::black_box(&work[0]);
    }
    best
}

/// One forward run of `P` at `log2`, into a fresh copy of `src`.
fn run_once<P: StockhamPrecision>(
    src: &[P::Complex],
    scratch: &mut [P::Complex],
    twiddles: &[P::Complex],
    log2: u32,
) -> Vec<P::Complex> {
    let mut work = src.to_vec();
    transform_sized::<P>(&mut work, scratch, twiddles, None, log2);
    work
}

#[test]
#[ignore = "measurement probe for the AVX Stockham backend audit"]
fn stockham_backend_cost_matrix() {
    // The AVX arm is invoked directly, without the production route's runtime
    // detection, so the probe itself must gate on the features.
    if !(std::arch::is_x86_feature_detected!("avx") && std::arch::is_x86_feature_detected!("fma")) {
        eprintln!("host lacks avx+fma; backend matrix not measurable");
        return;
    }
    let Some(selection) = measurement_cores::selected() else {
        eprintln!("host reports no processor class information; backend matrix not measurable");
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
        println!("== {} core (cpu {landed}) ==", core.label());
        println!(
            "{:>6}  {:>12} {:>12} {:>7}   {:>12} {:>12} {:>7}",
            "n", "f64 scalar", "f64 avx", "s/a", "f32 scalar", "f32 avx", "s/a"
        );
        for log2 in 7u32..=15 {
            let n = 1usize << log2;

            // f64 pair.
            let src64: Vec<Complex64> = (0..n)
                .map(|i| Complex64::new((0.017 * i as f64).sin(), 0.25 * (0.031 * i as f64).cos()))
                .collect();
            let tw64 = <f64 as MixedRadixScalar>::cached_twiddle_fwd(n);
            let mut work64 = src64.clone();
            let mut scratch64 = vec![Complex64::new(0.0, 0.0); n];
            // Both backends fuse the same stage schedule but the AVX arm uses
            // fmaddsub, so agreement is bounded by rounding-difference growth
            // over log2 n stages, not bitwise: |Δ| ≤ c·log2(n)·ε·‖x‖∞ with a
            // generous c. Divergence past this is a broken backend, which
            // would make the table meaningless.
            let ref64 = run_once::<PreciseStockham>(&src64, &mut scratch64, tw64.as_ref(), log2);
            let avx64 =
                run_once::<PreciseStockhamAvxFma>(&src64, &mut scratch64, tw64.as_ref(), log2);
            let peak64 = ref64.iter().map(|c| c.norm()).fold(0.0, f64::max);
            let err64 = ref64
                .iter()
                .zip(avx64.iter())
                .map(|(a, b)| (*a - *b).norm())
                .fold(0.0, f64::max);
            assert!(
                err64 <= 64.0 * f64::from(log2) * f64::EPSILON * peak64,
                "f64 backend divergence at n={n}: {err64:e} (peak {peak64:e})"
            );
            let s64 = per_call_ns::<PreciseStockham>(
                &src64,
                &mut work64,
                &mut scratch64,
                tw64.as_ref(),
                log2,
            );
            let a64 = per_call_ns::<PreciseStockhamAvxFma>(
                &src64,
                &mut work64,
                &mut scratch64,
                tw64.as_ref(),
                log2,
            );

            // f32 pair.
            let src32: Vec<Complex32> = (0..n)
                .map(|i| Complex32::new((0.017 * i as f32).sin(), 0.25 * (0.031 * i as f32).cos()))
                .collect();
            let tw32 = <f32 as MixedRadixScalar>::cached_twiddle_fwd(n);
            let mut work32 = src32.clone();
            let mut scratch32 = vec![Complex32::new(0.0, 0.0); n];
            let ref32 = run_once::<ReducedStockham>(&src32, &mut scratch32, tw32.as_ref(), log2);
            let avx32 =
                run_once::<ReducedStockhamAvxFma>(&src32, &mut scratch32, tw32.as_ref(), log2);
            let peak32 = ref32.iter().map(|c| c.norm()).fold(0.0f32, f32::max);
            let err32 = ref32
                .iter()
                .zip(avx32.iter())
                .map(|(a, b)| (*a - *b).norm())
                .fold(0.0f32, f32::max);
            assert!(
                err32 <= 64.0 * log2 as f32 * f32::EPSILON * peak32,
                "f32 backend divergence at n={n}: {err32:e} (peak {peak32:e})"
            );
            let s32 = per_call_ns::<ReducedStockham>(
                &src32,
                &mut work32,
                &mut scratch32,
                tw32.as_ref(),
                log2,
            );
            let a32 = per_call_ns::<ReducedStockhamAvxFma>(
                &src32,
                &mut work32,
                &mut scratch32,
                tw32.as_ref(),
                log2,
            );

            println!(
                "{n:>6}  {s64:>12.1} {a64:>12.1} {:>7.3}   {s32:>12.1} {a32:>12.1} {:>7.3}",
                s64 / a64,
                s32 / a32,
            );
        }
    }
}
