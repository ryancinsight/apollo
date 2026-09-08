//! Compensated direct DFT oracle, accurate to a derived multiple of `u` in
//! the 2-norm, for verifying an `O(log N · u)` kernel at either precision.
//!
//! Each bin is two real dot products of length `2N` evaluated with Dot2
//! (Ogita, Rump and Oishi, "Accurate sum and dot product", SIAM J. Sci.
//! Comput. 26(6), 2005, Algorithm 5.3): products and partial sums are split
//! with error-free transformations, so the result is as if computed in twice
//! the working precision and rounded once. Its error per bin is
//! `u·|X_k| + γ²_{2N} · Σ|x_n||w_kn|`, and the second term is below `u` for
//! every length exercised here.
//!
//! The twiddles `(cos, sin)` of `−2π·k·n / N` are evaluated directly in `f64`
//! with a relative error `μ ≤ 4u` per unit; that perturbs the DFT matrix
//! entrywise, so its effect is bounded by `‖E‖_F·‖x‖₂ ≤ μN·‖x‖₂ = μ√N·‖X‖₂`
//! (Parseval). [`oracle_bound`] is the sum of both contributions.

use eunomia::Complex64;
use std::f64::consts::TAU;

/// Unit roundoff of `f64`.
pub(super) const U64: f64 = f64::EPSILON / 2.0;

/// Error-free product: `a·b = p + e` exactly.
#[inline]
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

/// Error-free sum: `a + b = s + e` exactly.
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let z = s - a;
    (s, (a - (s - z)) + (b - z))
}

/// Dot2 over paired iterators.
fn dot2(terms: impl Iterator<Item = (f64, f64)>) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for (a, b) in terms {
        let (product, product_error) = two_product(a, b);
        let (next, sum_error) = two_sum(sum, product);
        sum = next;
        compensation += sum_error + product_error;
    }
    sum + compensation
}

/// Direct forward DFT `X_k = Σ x_n · exp(−2πi·k·n / N)` with compensated
/// accumulation and directly evaluated twiddles.
pub(super) fn direct_dft(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    (0..n)
        .map(|k| {
            let twiddles: Vec<(f64, f64)> = (0..n)
                .map(|m| {
                    let reduced = (k * m) % n;
                    let angle = -TAU * reduced as f64 / n as f64;
                    let (sin, cos) = angle.sin_cos();
                    (cos, sin)
                })
                .collect();
            // (xr + i·xi)(c + i·s) = (xr·c − xi·s) + i(xr·s + xi·c)
            let re = dot2(
                input
                    .iter()
                    .zip(&twiddles)
                    .flat_map(|(x, (c, s))| [(x.re, *c), (x.im, -*s)]),
            );
            let im = dot2(
                input
                    .iter()
                    .zip(&twiddles)
                    .flat_map(|(x, (c, s))| [(x.re, *s), (x.im, *c)]),
            );
            Complex64::new(re, im)
        })
        .collect()
}

/// Relative 2-norm bound on the oracle's own error for length `n`:
/// twiddle perturbation `4u√N` plus the compensated rounding `u` and one
/// guard unit.
pub(super) fn oracle_bound(n: usize) -> f64 {
    (4.0 * (n as f64).sqrt() + 2.0) * U64
}

/// Compensated squared 2-norm.
pub(super) fn norm_sqr(values: &[Complex64]) -> f64 {
    dot2(values.iter().flat_map(|z| [(z.re, z.re), (z.im, z.im)]))
}

/// 2-norm of the elementwise difference.
pub(super) fn distance(a: &[Complex64], b: &[Complex64]) -> f64 {
    let diff: Vec<Complex64> = a.iter().zip(b).map(|(x, y)| *x - *y).collect();
    norm_sqr(&diff).sqrt()
}
