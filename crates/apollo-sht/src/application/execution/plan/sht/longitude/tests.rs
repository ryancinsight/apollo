//! Evidence for the longitude factorization.
//!
//! Two independent lines. The differential tests compare against the direct
//! per-longitude sums the factorization replaces, built here from the same
//! `pub(super)` primitives the fallback path still uses in production. The
//! analytic tests consult no reference implementation at all: they assert
//! orthonormality of the basis and the closed form of `Y_0^0`, which fix the
//! sign and normalization conventions independently of any code.

use super::super::quadrature::{sht_forward_mode_sum, sht_inverse_sample};
use super::super::ShtPlan;
use super::longitude_route_available;
use crate::domain::spectrum::coefficients::SphericalHarmonicCoefficients;
use crate::infrastructure::kernel::spherical_harmonic::spherical_harmonic;
use eunomia::Complex64;
use leto::Array2;

/// A length `apollo-fft` cannot route, so the plan must fall back to the
/// direct sums. `361 = 19²` asserts during plan construction.
const UNROUTABLE_LONGITUDES: usize = 361;

fn all_modes(max_degree: usize) -> Vec<(usize, isize)> {
    (0..=max_degree)
        .flat_map(|l| (-(l as isize)..=(l as isize)).map(move |m| (l, m)))
        .collect()
}

/// A reproducible band-unlimited field; nothing about it is aligned to the
/// basis, so every mode carries signal.
fn sample_field(plan: &ShtPlan) -> Array2<Complex64> {
    let n_lat = plan.grid().latitudes();
    let n_lon = plan.grid().longitudes();
    let mut samples = Array2::<Complex64>::zeros([n_lat, n_lon]);
    for lat in 0..n_lat {
        let theta = plan.theta(lat);
        for lon in 0..n_lon {
            let phi = plan.phi(lon);
            samples[[lat, lon]] = Complex64::new(
                (2.0 * theta).cos() * (3.0 * phi).sin() + 0.25,
                (theta).sin() * (2.0 * phi).cos() - 0.125,
            );
        }
    }
    samples
}

/// The transform this replaces: `a_lm = Σ_lat w_lat Δφ Σ_lon f conj(Y_lm)`,
/// with the inner sum evaluated per longitude.
fn direct_forward(plan: &ShtPlan, samples: &Array2<Complex64>) -> SphericalHarmonicCoefficients {
    let max_degree = plan.grid().max_degree();
    let n_lat = plan.grid().latitudes();
    let n_lon = plan.grid().longitudes();
    let longitude_weight = std::f64::consts::TAU / n_lon as f64;
    let mut coefficients = SphericalHarmonicCoefficients::zeros(max_degree);
    let row_buffer: Vec<Complex64> = samples.iter().copied().collect();
    for lat in 0..n_lat {
        let theta = plan.theta(lat);
        let weight = plan.theta_weights[lat] * longitude_weight;
        let row = &row_buffer[lat * n_lon..lat * n_lon + n_lon];
        for &(degree, order) in &all_modes(max_degree) {
            let contribution = sht_forward_mode_sum(row, degree, order, theta, n_lon) * weight;
            let existing = coefficients.get(degree, order);
            coefficients.set(degree, order, existing + contribution);
        }
    }
    coefficients
}

/// The synthesis this replaces: every mode summed at every sample point.
fn direct_inverse(
    plan: &ShtPlan,
    coefficients: &SphericalHarmonicCoefficients,
) -> Array2<Complex64> {
    let n_lat = plan.grid().latitudes();
    let n_lon = plan.grid().longitudes();
    let modes = all_modes(plan.grid().max_degree());
    let mut samples = Array2::<Complex64>::zeros([n_lat, n_lon]);
    for lat in 0..n_lat {
        let theta = plan.theta(lat);
        for lon in 0..n_lon {
            samples[[lat, lon]] = sht_inverse_sample(coefficients, &modes, theta, plan.phi(lon));
        }
    }
    samples
}

/// Bound on the disagreement between the two routes.
///
/// The associated Legendre recurrence and the normalization run identically on
/// both paths — the factored path calls them once where the direct path calls
/// them `n_lon` times with the same arguments and gets the same answer — so
/// their error is common-mode and cancels in the difference. It does not enter
/// this bound, and the recurrence's conditioning at high degree is therefore
/// not what this tolerance has to cover.
///
/// What differs is the accumulation and the phase factors:
///
/// - the direct longitude sum accumulates `n_lon` terms sequentially,
///   contributing `O(n_lon · ε)` relative to `Σ|f|`;
/// - the FFT contributes `O(log₂(n_lon) · ε)`, which the previous term
///   dominates;
/// - the direct path forms each phase as `cos(m·φ_j)` where the product
///   `m·φ_j` is rounded before the trigonometric call, giving a phase error of
///   order `ε·|m|·τ`; the FFT's twiddles come from a different construction.
///   Over `|m| ≤ lmax` this contributes `O(lmax · ε)`;
/// - the latitude quadrature adds `n_lat` sequential terms, `O(n_lat · ε)`.
///
/// Summing, the difference is bounded by `C·ε·(n_lat + n_lon + lmax)·scale`
/// for a small constant absorbing the per-operation counts. `C = 32` is used
/// below; the observed margin is several orders of magnitude, so the verdict
/// does not rest on the constant.
fn differential_tolerance(plan: &ShtPlan, scale: f64) -> f64 {
    let terms = plan.grid().latitudes() + plan.grid().longitudes() + plan.grid().max_degree();
    32.0 * f64::EPSILON * terms as f64 * scale
}

fn max_abs(values: impl Iterator<Item = Complex64>) -> f64 {
    values.map(|v| v.norm()).fold(0.0f64, f64::max)
}

/// Grids spanning the routing cases: powers of two, non-powers of two, a prime
/// longitude count, widths on both sides of the Hermes dot threshold, and one
/// width with no route at all.
const GRIDS: [(usize, usize, usize); 7] = [
    (8, 16, 4),
    (10, 12, 5),
    (6, 24, 3),
    (12, 17, 5),
    (9, 320, 4),
    (6, 512, 4),
    (5, UNROUTABLE_LONGITUDES, 3),
];

#[test]
fn forward_matches_the_direct_longitude_sums() {
    for (n_lat, n_lon, max_degree) in GRIDS {
        let plan = ShtPlan::new(n_lat, n_lon, max_degree).expect("valid grid");
        let samples = sample_field(&plan);
        let produced = plan.forward_complex(&samples).expect("forward");
        let reference = direct_forward(&plan, &samples);
        let scale = max_abs(reference.values().iter().copied()).max(f64::MIN_POSITIVE);
        let tolerance = differential_tolerance(&plan, scale);
        let error = max_abs(
            produced
                .values()
                .iter()
                .zip(reference.values().iter())
                .map(|(a, b)| a - b),
        );
        assert!(
            error <= tolerance,
            "forward at {n_lat}x{n_lon} lmax={max_degree}: error {error:.3e} exceeds {tolerance:.3e}"
        );
    }
}

#[test]
fn inverse_matches_the_direct_mode_sums() {
    for (n_lat, n_lon, max_degree) in GRIDS {
        let plan = ShtPlan::new(n_lat, n_lon, max_degree).expect("valid grid");
        let coefficients = plan
            .forward_complex(&sample_field(&plan))
            .expect("forward for coefficients");
        let produced = plan.inverse_complex(&coefficients).expect("inverse");
        let reference = direct_inverse(&plan, &coefficients);
        let scale = max_abs(reference.iter().copied()).max(f64::MIN_POSITIVE);
        let tolerance = differential_tolerance(&plan, scale);
        let error = max_abs(produced.iter().zip(reference.iter()).map(|(a, b)| a - b));
        assert!(
            error <= tolerance,
            "inverse at {n_lat}x{n_lon} lmax={max_degree}: error {error:.3e} exceeds {tolerance:.3e}"
        );
    }
}

#[test]
fn unroutable_longitude_count_falls_back_instead_of_panicking() {
    assert!(
        !longitude_route_available(UNROUTABLE_LONGITUDES),
        "this test is only meaningful while {UNROUTABLE_LONGITUDES} has no route; \
         if apollo-fft gained one, pick another width or delete the fallback"
    );
    let plan = ShtPlan::new(5, UNROUTABLE_LONGITUDES, 3).expect("valid grid");
    let samples = sample_field(&plan);
    let coefficients = plan.forward_complex(&samples).expect("forward");
    let reference = direct_forward(&plan, &samples);
    let scale = max_abs(reference.values().iter().copied());
    let error = max_abs(
        coefficients
            .values()
            .iter()
            .zip(reference.values().iter())
            .map(|(a, b)| a - b),
    );
    // Both sides evaluate the same harmonics per longitude, so no phase or
    // transform term enters. What remains is the summation order: this width
    // is above `SHT_HERMES_DOT_LEN_THRESHOLD`, so the plan reduces the
    // longitude sum through the Hermes lanes while the reference accumulates
    // sequentially. That is the accumulation term of the shared bound and
    // nothing else.
    let tolerance = differential_tolerance(&plan, scale);
    assert!(
        error <= tolerance,
        "fallback path diverged from the direct sums by {error:.3e}, \
         bound {tolerance:.3e} (scale {scale:.3e})"
    );
}

/// Orthonormality: `∫ Y_lm conj(Y_l'm') dΩ = δ`. Sampling one basis function
/// and transforming must return a unit coefficient at its own mode and zero
/// everywhere else. No reference implementation participates — the expected
/// values come from the theorem.
#[test]
fn basis_functions_transform_to_unit_coefficients() {
    let max_degree = 4;
    // Gauss-Legendre with n_lat nodes integrates polynomials of degree
    // 2·n_lat−1 exactly; the product of two harmonics of degree ≤ lmax needs
    // degree 2·lmax, so n_lat > lmax suffices. Longitude needs n_lon > 2·lmax
    // for the orders to stay unaliased.
    let plan = ShtPlan::new(max_degree + 2, 4 * max_degree + 2, max_degree).expect("valid grid");
    let n_lat = plan.grid().latitudes();
    let n_lon = plan.grid().longitudes();
    for &(degree, order) in &all_modes(max_degree) {
        let mut samples = Array2::<Complex64>::zeros([n_lat, n_lon]);
        for lat in 0..n_lat {
            let theta = plan.theta(lat);
            for lon in 0..n_lon {
                samples[[lat, lon]] = spherical_harmonic(degree, order, theta, plan.phi(lon));
            }
        }
        let coefficients = plan.forward_complex(&samples).expect("forward");
        // Quadrature error for an exactly-integrable integrand is the
        // accumulation error alone: O((n_lat + n_lon)·ε) on an integrand of
        // unit scale.
        let tolerance = differential_tolerance(&plan, 1.0);
        for &(other_degree, other_order) in &all_modes(max_degree) {
            let expected = f64::from(u8::from(other_degree == degree && other_order == order));
            let got = coefficients.get(other_degree, other_order);
            assert!(
                (got - Complex64::new(expected, 0.0)).norm() <= tolerance,
                "Y_{degree}^{order} produced {got:?} at mode ({other_degree}, {other_order}), \
                 expected {expected}"
            );
        }
    }
}

/// `Y_0^0 = 1/sqrt(4π)` exactly, so a constant field `c` has the single
/// coefficient `a_00 = c·sqrt(4π)` and synthesis returns the constant.
#[test]
fn constant_field_carries_only_the_monopole() {
    let plan = ShtPlan::new(6, 15, 3).expect("valid grid");
    let value = Complex64::new(0.375, -0.125);
    let samples = Array2::from_elem([plan.grid().latitudes(), plan.grid().longitudes()], value);
    let coefficients = plan.forward_complex(&samples).expect("forward");
    let tolerance = differential_tolerance(&plan, value.norm() * 4.0);
    let monopole = value * (4.0 * std::f64::consts::PI).sqrt();
    assert!(
        (coefficients.get(0, 0) - monopole).norm() <= tolerance,
        "monopole {:?} differs from the closed form {monopole:?}",
        coefficients.get(0, 0)
    );
    for &(degree, order) in &all_modes(plan.grid().max_degree()) {
        if (degree, order) == (0, 0) {
            continue;
        }
        assert!(
            coefficients.get(degree, order).norm() <= tolerance,
            "mode ({degree}, {order}) is non-zero for a constant field"
        );
    }
    let synthesized = plan.inverse_complex(&coefficients).expect("inverse");
    for &produced in synthesized.iter() {
        assert!(
            (produced - value).norm() <= tolerance,
            "synthesis returned {produced:?} for the constant field {value:?}"
        );
    }
}

/// Analysis followed by synthesis reproduces a band-limited field, which pins
/// the relative normalization and the sign of the two directions against each
/// other without either reference implementation.
#[test]
fn round_trip_reproduces_a_band_limited_field() {
    let max_degree = 3;
    let plan = ShtPlan::new(max_degree + 3, 4 * max_degree + 4, max_degree).expect("valid grid");
    let n_lat = plan.grid().latitudes();
    let n_lon = plan.grid().longitudes();
    let mut field = Array2::<Complex64>::zeros([n_lat, n_lon]);
    for (index, &(degree, order)) in all_modes(max_degree).iter().enumerate() {
        let amplitude = Complex64::new(0.5 - 0.03 * index as f64, 0.125 * (index as f64).cos());
        for lat in 0..n_lat {
            let theta = plan.theta(lat);
            for lon in 0..n_lon {
                field[[lat, lon]] +=
                    amplitude * spherical_harmonic(degree, order, theta, plan.phi(lon));
            }
        }
    }
    let coefficients = plan.forward_complex(&field).expect("forward");
    let recovered = plan.inverse_complex(&coefficients).expect("inverse");
    let scale = max_abs(field.iter().copied());
    let tolerance = differential_tolerance(&plan, scale);
    let error = max_abs(recovered.iter().zip(field.iter()).map(|(a, b)| a - b));
    assert!(
        error <= tolerance,
        "round trip error {error:.3e} exceeds {tolerance:.3e} at scale {scale:.3e}"
    );
}
