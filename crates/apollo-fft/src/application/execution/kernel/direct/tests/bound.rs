//! Error bounds derived from the unit roundoffs of the accumulator and of the element.

use core::f64::consts::{SQRT_2, TAU};
use eunomia::{Complex, FloatElement};

/// Unit roundoffs of one instantiation: `accumulator` for every product
/// and sum, `storage` for the single rounding of each output into `T`.
#[derive(Clone, Copy)]
pub(super) struct Roundoff {
    pub(super) accumulator: f64,
    pub(super) storage: f64,
}

/// Unit roundoff `u = ε / 2` of round-to-nearest in `T`, where the epsilon
/// `ε` is the spacing above 1: `1 + ε/2` is the tie that rounds back to 1.
///
/// The bounds below model rounding as a relative error `u`, which holds
/// in a format's normal range. The tests instantiate every format that
/// satisfies it on their values; the 4-bit formats `F4` and `Bf4` do not
/// (the length-3 inverse component `−1/6` stores as `−1/4`) and are
/// excluded.
pub(super) fn unit_roundoff<T: FloatElement>() -> f64 {
    let mut epsilon = 1.0;
    while T::from_f64(1.0 + epsilon / 2.0).to_f64() != 1.0 {
        epsilon /= 2.0;
    }
    epsilon / 2.0
}

pub(super) fn roundoff<T: FloatElement>() -> Roundoff {
    Roundoff {
        accumulator: unit_roundoff::<T::Accumulator>(),
        storage: unit_roundoff::<T>(),
    }
}

/// `γ_k = k·u / (1 − k·u)` (Higham, 2nd ed., §3.1, Lemma 3.1).
pub(super) fn gamma(k: usize, unit: f64) -> f64 {
    let ku = f64::from_count(k) * unit;
    ku / (1.0 - ku)
}

/// Bound on the error of a twiddle component generated in `f64`.
///
/// The `f64` angle `fl(fl(τ₆₄·j)/N)` carries the rounding of the `2π`
/// constant, the product, and the quotient (`j`, `N` convert exactly
/// below `2^53`), so it is within `γ₃(u₆₄)·|angle| < 2π·γ₃(u₆₄)` of the
/// exact angle, and sine and cosine are 1-Lipschitz. The libm sine and
/// cosine are held to 1 ulp of the MPFR result by the libm test harness
/// (rust-lang/compiler-builtins, `libm-test/src/precision.rs`,
/// `default_ulp`), at most `2u₆₄` for a value in `[−1, 1]`.
pub(super) fn generation_error() -> f64 {
    let reference = f64::EPSILON / 2.0;
    TAU * gamma(3, reference) + 2.0 * reference
}

/// Per-component bound `τ` on the twiddle error `|ŵ − w|`: the generation
/// error plus the rounding into the accumulator, `u_A` relative to a
/// value in `[−1, 1]`.
pub(super) fn twiddle_error(roundoff: Roundoff) -> f64 {
    generation_error() * (1.0 + roundoff.accumulator) + roundoff.accumulator
}

/// Which pass a bound covers: the inverse adds one rounding for `/N`.
#[derive(Clone, Copy)]
pub(super) enum Pass {
    Forward,
    Inverse,
}

/// Per-component accumulator error of one pass over input of modulus mass
/// `mass` (`Σ |x_j|` forward, `Σ |X_k| / N` inverse).
///
/// Each term is `x_j·ŵ_j` with `|ŵ_j − w_j| ≤ √2·τ` and `|ŵ_j| ≤ ω =
/// 1 + √2·τ`; the complex product errs by `√2·γ₂·|x_j||ŵ_j|` (Higham,
/// 2nd ed., §3.6, Lemma 3.5) and sequential summation of the `N` terms
/// by `γ_{N−1}·Σ|terms|` (§4.2), which combine to `γ_{N+2}·ω·mass`
/// through `√2·γ₂ ≤ γ₃` and `γ_j + γ_k + γ_j·γ_k ≤ γ_{j+k}` (§3.1,
/// Lemma 3.3). The inverse's division by `N` adds one rounding, giving
/// `γ_{N+3}` and a `(1 + u_A)` factor on the twiddle term.
pub(super) fn pass_error(n: usize, roundoff: Roundoff, mass: f64, pass: Pass) -> f64 {
    let tau = twiddle_error(roundoff);
    let twiddle_modulus = 1.0 + SQRT_2 * tau;
    let (terms, twiddle_term) = match pass {
        Pass::Forward => (n + 2, SQRT_2 * tau),
        Pass::Inverse => (n + 3, SQRT_2 * tau * (1.0 + roundoff.accumulator)),
    };
    (twiddle_term + gamma(terms, roundoff.accumulator) * twiddle_modulus) * mass
}

/// Forward bound: the accumulator error plus one storage rounding of a
/// component no larger than `mass + error`.
pub(super) fn forward_bound(n: usize, roundoff: Roundoff, mass: f64) -> f64 {
    let error = pass_error(n, roundoff, mass, Pass::Forward);
    error + roundoff.storage * (mass + error)
}

/// Inverse bound: the accumulator error plus one storage rounding.
/// `mass` is `Σ |X_k| / N`.
pub(super) fn inverse_bound(n: usize, roundoff: Roundoff, mass: f64) -> f64 {
    let error = pass_error(n, roundoff, mass, Pass::Inverse);
    error + roundoff.storage * (mass + error)
}

/// Round-trip bound for `inverse(forward(x))` against `x`.
///
/// The stored spectrum `X̂` differs from `X` by at most `b_f` per
/// component, `√2·b_f` in modulus, so the exact inverse of `X̂` lies within
/// `√2·b_f` of `x` and `Σ|X̂_k| / N ≤ ‖x‖₁ + √2·b_f` (`|X_k| ≤ ‖x‖₁`). The
/// inverse pass adds its accumulator error over that mass, and the final
/// storage rounding acts on a component no larger than `‖x‖₁` plus both.
pub(super) fn round_trip_bound(n: usize, roundoff: Roundoff, mass: f64) -> f64 {
    let spectrum = SQRT_2 * forward_bound(n, roundoff, mass);
    let error = spectrum + pass_error(n, roundoff, mass + spectrum, Pass::Inverse);
    error + roundoff.storage * (mass + error)
}

/// Bound on a component of the transform of a unit impulse, against the
/// `f64` reference `r / divisor` where `r` is a unit-circle coordinate.
///
/// The product `1·ŵ` and every sum with an exact zero are exact, so the
/// accumulator holds the rounded twiddle `ŵ = fl_A(c)`, `c` being the `f64`
/// value generated for `r`'s angle: `|c − w| ≤ E` and `|r − w| ≤ E`
/// (`generation_error`), hence
/// `|ŵ − r| ≤ u_A·|c| + 2E ≤ u_A·|r| + 2E·(1 + u_A) =: t`. The inverse
/// divides by `N` in the accumulator, one more rounding of the quotient:
/// `a = (t + u_A·(|r| + t)) / N`. The output rounds into `T`, and the
/// reference quotient `r / N` rounds once in `f64`:
/// `a + u_T·(|r| / N + a) + u₆₄·|r| / N`. The error is relative to the
/// component rather than to a unit-modulus scale, so the bound is also
/// valid for components that are exactly zero in the exact transform.
pub(super) fn impulse_bound(roundoff: Roundoff, reference: f64, pass: Pass, n: usize) -> f64 {
    let magnitude = reference.abs();
    let twiddle =
        roundoff.accumulator * magnitude + 2.0 * generation_error() * (1.0 + roundoff.accumulator);
    let (divisor, accumulated) = match pass {
        Pass::Forward => (1.0, twiddle),
        Pass::Inverse => (
            f64::from_count(n),
            twiddle + roundoff.accumulator * (magnitude + twiddle),
        ),
    };
    let accumulated = accumulated / divisor;
    accumulated
        + roundoff.storage * (magnitude / divisor + accumulated)
        + (f64::EPSILON / 2.0) * magnitude / divisor
}

/// `Σ |x_j|` evaluated in `f64` from the exactly widened components.
pub(super) fn modulus_mass<T: FloatElement>(input: &[Complex<T>]) -> f64 {
    input
        .iter()
        .map(|z| z.re.to_f64().hypot(z.im.to_f64()))
        .sum()
}
