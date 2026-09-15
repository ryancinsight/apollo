//! The downsampled recovery route (ADR 0064) against the dense oracle.

use crate::{RecoveryRoute, SparseFftPlan};
use eunomia::Complex64;
use proptest::prelude::*;
use std::f64::consts::TAU;

/// A signal whose spectrum is exactly the given tones: `X[f] = n · a_f`.
fn sparse_signal(n: usize, tones: &[(usize, Complex64)]) -> Vec<Complex64> {
    (0..n)
        .map(|t| {
            tones
                .iter()
                .map(|&(f, a)| {
                    let angle = TAU * ((f * t) % n) as f64 / n as f64;
                    a * Complex64::new(angle.cos(), angle.sin())
                })
                .sum()
        })
        .collect()
}

/// Per-coefficient bound: the forward error of an `n`-point transform of a
/// signal of 1-norm `l1` in f64, with the oracles' constant.
fn bound(n: usize, l1: f64) -> f64 {
    16.0 * (n as f64).log2().max(1.0) * (f64::EPSILON / 2.0) * l1
}

fn l1(signal: &[Complex64]) -> f64 {
    signal.iter().map(|v| v.norm()).sum()
}

/// The downsampled route finds exactly the tones of an exactly `k`-sparse
/// signal, with the dense route's coefficients within the derived bound.
fn recovers_exactly(n: usize, tones: &[(usize, Complex64)]) {
    let k = tones.len();
    let signal = sparse_signal(n, tones);
    let plan = SparseFftPlan::downsampled(n, k).expect("length admits the route");
    assert_eq!(plan.route(), RecoveryRoute::Downsampled);
    let spectrum = plan
        .forward(&signal)
        .expect("an exactly k-sparse signal resolves");
    let dense = SparseFftPlan::new(n, k)
        .expect("plan")
        .forward(&signal)
        .expect("dense route");
    let mut want: Vec<usize> = tones.iter().map(|&(f, _)| f).collect();
    want.sort_unstable();
    assert_eq!(spectrum.frequencies, want, "n={n} k={k}: support");
    assert_eq!(dense.frequencies, want, "n={n} k={k}: the oracle's support");
    // Both routes err by at most one transform's bound from the exact value.
    let tolerance = 2.0 * bound(n, l1(&signal));
    for ((f, got), oracle) in spectrum
        .frequencies
        .iter()
        .zip(&spectrum.values)
        .zip(&dense.values)
    {
        assert!(
            (got - oracle).norm() <= tolerance,
            "n={n} k={k} bin {f}: {got} against the dense route's {oracle} (bound {tolerance:.3e})"
        );
    }
}

#[test]
fn a_single_tone_is_recovered_at_every_admitted_length() {
    for n in [4usize, 8, 64, 256, 1024, 4096] {
        recovers_exactly(n, &[(n / 3, Complex64::new(0.7, -1.1))]);
    }
}

#[test]
fn tones_forced_into_one_bucket_resolve_by_doubling() {
    // Ten tones in one residue class modulo the starting bucket count
    // (next_power_of_two(40) = 64): more than the decoder's four, so the
    // first rounds fail and the count doubles until they separate.
    let n = 4096;
    let tones: Vec<(usize, Complex64)> = (0..10)
        .map(|i| (5 + 64 * i, Complex64::new(1.0 + i as f64, -0.5 * i as f64)))
        .collect();
    recovers_exactly(n, &tones);
}

#[test]
fn a_length_the_doubling_cannot_reach_is_rejected() {
    let err = SparseFftPlan::downsampled(96, 2).expect_err("96 is 8 times 12, not a power of two");
    assert!(
        matches!(err, apollo_fft::ApolloError::Validation { ref field, .. } if field == "n"),
        "{err:?}"
    );
}

#[test]
fn a_dense_signal_still_yields_the_top_k() {
    // Nothing sparse about it: every round collides until the count reaches
    // n, where each bucket is one bin and the top-k of the full spectrum is
    // what the dense route returns.
    let n = 256;
    let signal: Vec<Complex64> = (0..n)
        .map(|t| Complex64::new((0.37 * t as f64).sin(), (0.11 * t as f64).cos()))
        .collect();
    let routed = SparseFftPlan::downsampled(n, 3)
        .expect("plan")
        .forward(&signal)
        .expect("resolves at n");
    let dense = SparseFftPlan::new(n, 3)
        .expect("plan")
        .forward(&signal)
        .expect("dense");
    assert_eq!(routed.frequencies, dense.frequencies);
    let tolerance = 2.0 * bound(n, l1(&signal));
    for (got, oracle) in routed.values.iter().zip(&dense.values) {
        assert!((got - oracle).norm() <= tolerance);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Random exactly k-sparse supports at power-of-two lengths.
    #[test]
    fn random_sparse_supports_are_recovered(
        log_n in 6u32..=12,
        k in 1usize..=12,
        seed in any::<u64>(),
    ) {
        let n = 1usize << log_n;
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut frequencies: Vec<usize> = Vec::new();
        while frequencies.len() < k.min(n) {
            let f = (next() % n as u64) as usize;
            if !frequencies.contains(&f) {
                frequencies.push(f);
            }
        }
        let tones: Vec<(usize, Complex64)> = frequencies
            .iter()
            .map(|&f| {
                let re = (next() % 2000) as f64 / 1000.0 - 1.0;
                let im = (next() % 2000) as f64 / 1000.0 - 1.0;
                (f, Complex64::new(re + 0.25_f64.copysign(re), im))
            })
            .collect();
        recovers_exactly(n, &tones);
    }
}
