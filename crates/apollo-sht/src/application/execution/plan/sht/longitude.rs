//! Longitude factorization: the `e^{imφ}` factor of a spherical harmonic is a
//! DFT kernel, so the sum over longitude is a DFT bin.
//!
//! # Derivation
//!
//! [`spherical_harmonic`] returns `N_lm P_l^m(cos θ) e^{imφ}` for `m ≥ 0`, and
//! `(-1)^{|m|} conj(·)` of the `|m|` harmonic for `m < 0`. Both branches
//! factor the same way: only `e^{imφ}` depends on φ, and the remaining factor
//! is real. So for every degree and order,
//!
//! ```text
//! Y_lm(θ, φ) = Y_lm(θ, 0) · e^{imφ}
//! ```
//!
//! which is [`harmonic_amplitude`] times a DFT kernel. Check the negative
//! branch: with `M = |m|`, `Y_{l,-M}(θ, φ) = (-1)^M N_lM P_l^M e^{-iMφ}`, whose
//! φ = 0 value is the real `(-1)^M N_lM P_l^M`, and `e^{imφ} = e^{-iMφ}` — the
//! identity holds on both branches, with no separate sign convention to pin.
//!
//! ## Forward
//!
//! The plan's longitude nodes are `φ_j = 2πj/n_lon`, so
//!
//! ```text
//! Σ_j f_j conj(Y_lm(θ, φ_j)) = conj(Y_lm(θ, 0)) · Σ_j f_j e^{-2πi m j / n_lon}
//!                            = conj(Y_lm(θ, 0)) · F[m mod n_lon]
//! ```
//!
//! where `F` is the unnormalized forward DFT of the latitude row. The residue
//! is not an approximation: `e^{-2πi m j/n_lon}` is exactly periodic in `m`
//! with period `n_lon`, so the direct sum aliases identically for `|m| ≥
//! n_lon`. One [`spherical_harmonic`] evaluation per (latitude, mode) replaces
//! `n_lon` of them.
//!
//! ## Inverse
//!
//! ```text
//! f(θ, φ_j) = Σ_lm a_lm Y_lm(θ, 0) e^{imφ_j} = Σ_k g_k e^{+2πi k j / n_lon}
//! ```
//!
//! with `g_k = Σ_{m ≡ k (mod n_lon)} a_lm Y_lm(θ, 0)`. That is the
//! *unnormalized* inverse DFT of `g`, which is why this uses
//! `inverse_complex_slice_unnorm_inplace`: the normalized inverse would divide
//! by `n_lon`, and the synthesis carries no such factor.
//!
//! # Routability
//!
//! `n_lon` is chosen by the caller, so this path is only usable if
//! `apollo-fft` transforms every width. It once did not — see
//! `ATLAS-APOLLO-COMPOSITE-RADIX-WRONG-ANSWERS-2026-08-28`, which is why this
//! module carried a direct-sum fallback and a routability probe. Bluestein now
//! serves the lengths no shaped strategy accepts, so the factored path is
//! unconditional and the fallback is gone; the direct sums remain as the
//! differential oracle the tests check against.

use crate::infrastructure::kernel::spherical_harmonic::spherical_harmonic;
use apollo_fft::{FftPlan1D, PlanCacheProvider, Shape1D};
use eunomia::Complex64;
use mnemosyne::scratch::ScratchPool;
use std::sync::Arc;

thread_local! {
    /// Per-row longitude spectrum. One buffer per worker thread, reused across
    /// every latitude row that worker takes.
    pub(super) static SHT_LONGITUDE_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
}

/// Retrieve the cached longitude plan.
///
/// The cache is thread-local inside `apollo-fft`, so each worker in the
/// latitude loop builds at most one plan and reuses it for every row it takes.
///
/// # Panics
///
/// Panics if `n_lon` is zero, which the grid specification forbids.
pub(super) fn longitude_plan(n_lon: usize) -> Arc<FftPlan1D<f64>> {
    <f64 as PlanCacheProvider>::get_1d_plan(
        Shape1D::new(n_lon).expect("grid specification rejects a zero longitude count"),
    )
}

/// The φ-independent factor of `Y_lm`, which is `Y_lm(θ, 0)`.
///
/// Real for every degree and order — the module derivation shows why — but
/// carried as a complex value so the transform reads as the same expression
/// the direct sum evaluates, rather than one that assumes the derivation.
pub(super) fn harmonic_amplitude(degree: usize, order: isize, theta: f64) -> Complex64 {
    spherical_harmonic(degree, order, theta, 0.0)
}

/// The DFT bin carrying order `m`, aliased into `0..n_lon` exactly as the
/// direct sum aliases it.
pub(super) fn order_bin(order: isize, n_lon: usize) -> usize {
    let modulus = isize::try_from(n_lon).expect("longitude count fits an isize");
    usize::try_from(order.rem_euclid(modulus)).expect("a euclidean residue is non-negative")
}

/// Forward DFT of one latitude row into `spectrum`.
pub(super) fn longitude_spectrum(
    plan: &FftPlan1D<f64>,
    row: &[Complex64],
    spectrum: &mut [Complex64],
) {
    spectrum.copy_from_slice(row);
    plan.forward_complex_slice_inplace(spectrum);
}

/// Unnormalized inverse DFT turning per-order sums into samples along φ.
pub(super) fn longitude_synthesis(plan: &FftPlan1D<f64>, orders: &mut [Complex64]) {
    plan.inverse_complex_slice_unnorm_inplace(orders);
}

#[cfg(test)]
mod tests;
