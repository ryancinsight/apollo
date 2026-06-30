//! Spherical harmonic basis and Gauss-Legendre quadrature kernels.
//!
//! The complex spherical harmonic is
//! `Y_l^m(theta, phi) = N_lm P_l^m(cos(theta)) exp(i m phi)` for `m >= 0`,
//! with `Y_l^-m = (-1)^m conj(Y_l^m)`. `P_l^m` includes the
//! Condon-Shortley phase. The normalization constant is
//! `N_lm = sqrt((2l + 1)/(4pi) * (l - m)!/(l + m)!)`.
//!
//! # Theorem 1 — Bonnet's Recurrence for Associated Legendre Polynomials
//!
//! **Statement.** For fixed order `m >= 0` and degree `l > m + 1`:
//!
//! ```text
//! P_l^m(x) = ((2l - 1) x P_{l-1}^m(x) - (l + m - 1) P_{l-2}^m(x)) / (l - m)
//! ```
//!
//! **Seeds.**
//!
//! ```text
//! P_m^m(x)   = (-1)^m (2m - 1)!! (1 - x^2)^(m/2)   (Condon-Shortley seed)
//! P_{m+1}^m(x) = x (2m + 1) P_m^m(x)               (one-step bootstrap)
//! ```
//!
//! **Proof sketch.** The three-term recurrence follows directly from Bonnet's formula
//! `(l - m) P_l^m = (2l - 1) x P_{l-1}^m - (l + m - 1) P_{l-2}^m` applied at
//! fixed order `m`. The seed `P_m^m` arises by applying the Rodrigues formula
//! `P_l(x) = (1/(2^l l!)) d^l/dx^l (x^2 - 1)^l` and differentiating `m` times with
//! respect to the generating function, yielding the factor `(2m - 1)!! = 1 * 3 * 5 * ... * (2m-1)`.
//! The factor `(-1)^m` is the Condon-Shortley phase convention adopted throughout this
//! crate. Upward recurrence in `l` at fixed `m` is numerically stable: the physically
//! desired solution grows faster than any parasitic complement for increasing `l`.
//!
//! **References.** Abramowitz & Stegun §8.5.3; DLMF §14.10.3.
//!
//! # Theorem 2 — Gauss-Legendre Polynomial Exactness
//!
//! **Statement.** The `n`-point Gauss-Legendre rule integrates every polynomial `p` of
//! degree `<= 2n - 1` exactly on `[-1, 1]`:
//!
//! ```text
//! ∫_{-1}^{1} p(x) dx = Σ_{i=1}^{n} w_i p(x_i),   deg(p) <= 2n - 1
//! ```
//!
//! **Proof sketch.** The GL nodes `x_i` are the `n` distinct roots of the degree-`n`
//! Legendre polynomial `P_n(x)`. For any polynomial `p` with `deg(p) <= 2n - 1`,
//! write `p(x) = q(x) P_n(x) + r(x)` where `deg(q) <= n - 1` and `deg(r) <= n - 1`
//! (Euclidean division). Since `P_n` is `L^2`-orthogonal to all polynomials of degree
//! `< n`, one has `∫_{-1}^{1} q(x) P_n(x) dx = 0`. The interpolatory quadrature rule
//! based on `n` nodes integrates `r` (degree `<= n - 1`) exactly, and `q(x_i) P_n(x_i) = 0`
//! at every node because `x_i` is a root of `P_n`. Combining, the rule produces the
//! exact value for all `p` of degree `<= 2n - 1`. The weights are given by
//! `w_i = 2 / ((1 - x_i^2)(P_n'(x_i))^2) > 0` and can be computed from the Golub-Welsch
//! eigenproblem for the symmetric tridiagonal Jacobi matrix of the Legendre three-term
//! recurrence.
//!
//! **Reference.** Golub & Welsch (1969), "Calculation of Gauss quadrature rules," Theorem 1.
//!
//! # Theorem 3 — Spherical Harmonic Orthonormality
//!
//! **Statement.** With solid-angle measure `dΩ = sin θ dθ dφ`:
//!
//! ```text
//! ∫_{S²} Y_l^m(Ω) conj(Y_{l'}^{m'}(Ω)) dΩ = δ_{l,l'} δ_{m,m'}
//! ```
//!
//! **Proof sketch.**
//!
//! 1. *Longitude integral.* The azimuthal factors contribute
//!    `∫_0^{2π} exp(i(m - m')φ) dφ = 2π δ_{m,m'}`. On a uniform longitude grid of
//!    `N_lon` points with spacing `Δφ = 2π / N_lon`, the discrete sum equals `2π δ_{m,m'}`
//!    exactly for all `|m - m'| < N_lon` (DFT orthogonality identity).
//!
//! 2. *Latitude integral.* After substituting `x = cos θ`, the latitude integral
//!    reduces to `N_lm N_{l'm} ∫_{-1}^{1} P_l^m(x) P_{l'}^m(x) dx`. The Legendre
//!    polynomial orthogonality theorem states
//!    `∫_{-1}^{1} P_l^m(x) P_{l'}^m(x) dx = 2(l+m)! / ((2l+1)(l-m)!) δ_{l,l'}`.
//!    With `N_lm^2 = (2l+1)(l-m)! / (4π(l+m)!)` (Theorem 5), the product is
//!    `N_lm^2 * 2(l+m)! / ((2l+1)(l-m)!) = 1/(2π)` times `δ_{l,l'}`.
//!
//! 3. *Combined.* Multiplying the longitude result `2π δ_{m,m'}` by the latitude
//!    result `(1/(2π)) δ_{l,l'}` yields `δ_{l,l'} δ_{m,m'}`. ∎
//!
//! **Reference.** Driscoll & Healy (1994), "Computing Fourier transforms and convolutions
//! on the 2-sphere," Theorem 1.
//!
//! # Theorem 4 — SHT Forward-Inverse Exactness
//!
//! **Statement.** Let `f` be band-limited to degree `<= L` (i.e., `f = Σ_{l<=L} Σ_m a_lm Y_l^m`
//! for finitely many coefficients). If the grid satisfies `L < N_lat` and `2L + 1 <= N_lon`,
//! then in exact arithmetic:
//!
//! ```text
//! inverse(forward(f)) = f
//! ```
//!
//! **Proof sketch.**
//!
//! - *Longitude exactness.* The uniform DFT on `N_lon >= 2L + 1` points recovers all
//!   Fourier modes `|m| <= L` without aliasing (Shannon-Nyquist criterion for integer
//!   frequencies; the maximum mode product `|m| + |m'| <= 2L < N_lon`).
//!
//! - *Latitude exactness.* For fixed `(l, m)`, the forward quadrature integrand
//!   `f(θ) conj(Y_l^m(θ))` is a polynomial of degree `<= 2L` in `cos θ` (product of
//!   two associated Legendre polynomials, each of degree `<= L` in `cos θ`). The
//!   `N_lat`-point GL rule is exact for polynomials of degree `<= 2 N_lat - 1 >= 2L`
//!   because `N_lat > L`, so `forward` recovers the true spectral coefficients `a_lm`.
//!
//! - *Reconstruction.* By Theorem 3, orthonormality holds exactly under the GL
//!   quadrature-based inner product (Theorem 2 guarantees the GL rule is exact for the
//!   degree-`2L` integrand). Linearity and the band-limitation `l <= L` complete the
//!   argument: `inverse(forward(f)) = Σ a_lm Y_l^m = f`.
//!
//! # Theorem 5 — Normalization Constant Derivation
//!
//! **Statement.** The constant
//!
//! ```text
//! N_lm = sqrt((2l + 1)/(4π) * (l - m)! / (l + m)!)
//! ```
//!
//! satisfies `∫_0^π (N_lm P_l^m(cos θ))^2 sin θ dθ = 1/(2π)`, so that the full
//! sphere integral `∫_{S²} |Y_l^m(Ω)|^2 dΩ = 1`.
//!
//! **Proof sketch.** Substituting `x = cos θ`, `dx = -sin θ dθ` transforms the integral to
//! `N_lm^2 ∫_{-1}^{1} (P_l^m(x))^2 dx`. The Legendre orthogonality integral evaluates to
//! `∫_{-1}^{1} (P_l^m(x))^2 dx = 2(l+m)! / ((2l+1)(l-m)!)` (standard result, e.g. A&S 8.14.11).
//! Multiplying:
//!
//! ```text
//! N_lm^2 * 2(l+m)!/((2l+1)(l-m)!)
//!   = (2l+1)(l-m)!/(4π(l+m)!) * 2(l+m)!/((2l+1)(l-m)!)
//!   = 2/(4π)
//!   = 1/(2π).
//! ```
//!
//! Including the azimuthal integral `∫_0^{2π} |exp(imφ)|^2 dφ = 2π` gives
//! `∫_{S²} |Y_l^m|^2 dΩ = 2π * (1/(2π)) = 1`. ∎
//!
//! ## References
//! - Associated Legendre polynomials: DLMF Section 14.10, Abramowitz & Stegun Section 8.5
//! - Condon-Shortley phase convention: Messiah, "Quantum Mechanics" Vol. 1, App. C
//! - Gauss-Legendre quadrature: Golub & Welsch (1969), "Calculation of Gauss quadrature rules"
//! - Spherical harmonic transform: Driscoll & Healy (1994), "Computing Fourier transforms and
//!   convolutions on the 2-sphere"

const MAX_GL_ITERATIONS: usize = 50;

use eunomia::Complex64;

/// Return Gauss-Legendre nodes and weights on `[-1, 1]`.
///
/// Computes the `n` quadrature nodes `x_i ∈ (-1, 1)` and positive weights `w_i` such that
/// `Σ_{i=0}^{n-1} w_i p(x_i) = ∫_{-1}^{1} p(x) dx` holds exactly for every polynomial
/// `p` of degree `<= 2n - 1` (Theorem 2 in the module-level documentation).
///
/// # Algorithm
///
/// Nodes are the `n` roots of the Legendre polynomial `P_n(x)`, located by Newton
/// iteration. Each iteration refines `x ← x - P_n(x)/P_n'(x)`. The initial estimate
/// `x_i ≈ cos(π(i + 3/4) / (n + 1/2))` is the asymptotic approximation of the `i`-th
/// zero of `P_n`. The symmetry `x_{n-1-i} = -x_i`, `w_{n-1-i} = w_i` is exploited:
/// only `ceil(n/2)` Newton solves are performed and each result is mirrored. Weights are
/// computed from `w_i = 2 / ((1 - x_i^2)(P_n'(x_i))^2)`.
///
/// Convergence is declared when `|Δx| < ε * |x| + ε` where `ε = f64::EPSILON`.
/// A `debug_assert` fires if the loop reaches `MAX_GL_ITERATIONS` without convergence.
///
/// # Invariants
///
/// - `Σ w_i = 2` (exact integral of the constant polynomial 1 on `[-1, 1]`).
/// - All weights are strictly positive.
/// - Nodes satisfy `x_i ∈ (-1, 1)` and are returned in ascending order.
///
/// # Panics
///
/// Panics if `n == 0`.
#[must_use]
pub fn gauss_legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n > 0, "Gauss-Legendre order must be non-zero");
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let half = n.div_ceil(2);
    for i in 0..half {
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for iter in 0..MAX_GL_ITERATIONS {
            let (p, dp) = legendre_and_derivative(n, x);
            let dx = p / dp;
            x -= dx;
            if dx.abs() < f64::EPSILON * x.abs() + f64::EPSILON {
                break;
            }
            if iter == MAX_GL_ITERATIONS - 1 {
                debug_assert!(
                    false,
                    "GL Newton iteration did not converge for n={n}, node={i}"
                );
            }
        }
        let (_, derivative) = legendre_and_derivative(n, x);
        let weight = 2.0 / ((1.0 - x * x) * derivative * derivative);
        nodes[i] = -x;
        nodes[n - 1 - i] = x;
        weights[i] = weight;
        weights[n - 1 - i] = weight;
    }

    (nodes, weights)
}

/// Evaluate the orthonormal complex spherical harmonic `Y_l^m(θ, φ)`.
///
/// Returns `N_lm P_l^m(cos θ) exp(i m φ)` for `m >= 0`, where:
///
/// - `N_lm = sqrt((2l + 1)/(4π) * (l - m)!/(l + m)!)` is the normalization constant
///   that gives unit `L^2(S²)` norm (Theorem 5 in the module-level documentation).
/// - `P_l^m` is the associated Legendre polynomial with the Condon-Shortley phase
///   `(-1)^m` in the seed (Theorem 1 in the module-level documentation).
///
/// For negative order `m < 0`, the relation `Y_l^{-|m|} = (-1)^|m| conj(Y_l^|m|)`
/// is applied, consistent with the Condon-Shortley convention.
///
/// # Orthonormality
///
/// These harmonics satisfy
/// `∫_{S²} Y_l^m(Ω) conj(Y_{l'}^{m'}(Ω)) dΩ = δ_{l,l'} δ_{m,m'}`
/// (Theorem 3 in the module-level documentation). Combined with Gauss-Legendre
/// quadrature of sufficient order, the forward SHT recovers exact spectral
/// coefficients for band-limited fields (Theorem 4).
///
/// # Panics
///
/// Panics if `|m| > l`.
#[must_use]
pub fn spherical_harmonic(degree: usize, order: isize, theta: f64, phi: f64) -> Complex64 {
    let abs_order = order.unsigned_abs();
    assert!(abs_order <= degree, "spherical harmonic requires |m| <= l");

    let positive = spherical_harmonic_nonnegative_order(degree, abs_order, theta, phi);
    if order >= 0 {
        positive
    } else if abs_order % 2 == 0 {
        positive.conj()
    } else {
        -positive.conj()
    }
}

fn spherical_harmonic_nonnegative_order(
    degree: usize,
    order: usize,
    theta: f64,
    phi: f64,
) -> Complex64 {
    let x = theta.cos();
    let associated = associated_legendre(degree, order, x);
    let normalization = normalization_constant(degree, order);
    let angle = order as f64 * phi;
    Complex64::new(angle.cos(), angle.sin()) * (normalization * associated)
}

/// Evaluate the associated Legendre polynomial `P_l^m(x)` with Condon-Shortley phase.
///
/// Implements Theorem 1 (module-level documentation): upward recurrence in degree `l`
/// at fixed order `m`, seeded by `P_m^m(x) = (-1)^m (2m-1)!! (1-x^2)^{m/2}`.
///
/// The argument `x` must lie in `[-1, 1]`; values slightly outside due to floating-point
/// rounding are clamped via `(1 - x^2).max(0.0)` before taking the square root.
pub(crate) fn associated_legendre(degree: usize, order: usize, x: f64) -> f64 {
    debug_assert!(order <= degree);
    let one_minus_x2 = (1.0 - x * x).max(0.0);
    let mut p_mm = 1.0;
    if order > 0 {
        let sqrt_term = one_minus_x2.sqrt();
        let mut factor = 1.0;
        for _ in 1..=order {
            p_mm *= -factor * sqrt_term;
            factor += 2.0;
        }
    }
    if degree == order {
        return p_mm;
    }

    let mut p_mmp1 = x * (2 * order + 1) as f64 * p_mm;
    if degree == order + 1 {
        return p_mmp1;
    }

    let mut p_lm_minus_two = p_mm;
    let mut p_lm_minus_one = p_mmp1;
    for ell in (order + 2)..=degree {
        let numerator =
            (2 * ell - 1) as f64 * x * p_lm_minus_one - (ell + order - 1) as f64 * p_lm_minus_two;
        let p_lm = numerator / (ell - order) as f64;
        p_lm_minus_two = p_lm_minus_one;
        p_lm_minus_one = p_lm;
    }
    p_mmp1 = p_lm_minus_one;
    p_mmp1
}

/// Compute `N_lm = sqrt((2l+1)/(4π) * (l-m)!/(l+m)!)`.
///
/// This is the normalization constant from Theorem 5 (module-level documentation).
/// The factorial ratio `(l-m)!/(l+m)!` is evaluated as `1 / product_{k=l-m+1}^{l+m} k`
/// to avoid overflow for large `l` and to preserve numerical precision.
fn normalization_constant(degree: usize, order: usize) -> f64 {
    let ratio = factorial_ratio(degree - order, degree + order);
    (((2 * degree + 1) as f64 / (4.0 * std::f64::consts::PI)) * ratio).sqrt()
}

/// Return `numerator! / denominator!` as a floating-point value.
///
/// Computes `1 / product_{k=numerator+1}^{denominator} k` directly, without
/// materialising either factorial. Returns `1.0` when `numerator == denominator`.
fn factorial_ratio(numerator: usize, denominator: usize) -> f64 {
    if numerator == denominator {
        return 1.0;
    }
    let mut product = 1.0;
    for value in (numerator + 1)..=denominator {
        product *= value as f64;
    }
    1.0 / product
}

/// Evaluate the Legendre polynomial `P_n(x)` and its derivative `P_n'(x)`.
///
/// Uses the Bonnet recurrence `(n) P_n = (2n-1) x P_{n-1} - (n-1) P_{n-2}` (the `m=0`
/// specialisation of Theorem 1) and the derivative identity
/// `P_n'(x) = n (x P_n(x) - P_{n-1}(x)) / (x^2 - 1)` for `|x| != 1`.
/// Called by `gauss_legendre_nodes_weights` during Newton iteration (Theorem 2).
fn legendre_and_derivative(degree: usize, x: f64) -> (f64, f64) {
    if degree == 0 {
        return (1.0, 0.0);
    }
    let mut p_nm2 = 1.0;
    let mut p_nm1 = x;
    for ell in 2..=degree {
        let ell_f = ell as f64;
        let p_n = ((2.0 * ell_f - 1.0) * x * p_nm1 - (ell_f - 1.0) * p_nm2) / ell_f;
        p_nm2 = p_nm1;
        p_nm1 = p_n;
    }
    let derivative = degree as f64 * (x * p_nm1 - p_nm2) / (x * x - 1.0);
    (p_nm1, derivative)
}
