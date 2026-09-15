//! The circle group `C(F_p) = {(x, y) : x² + y² = 1}` of a prime `p ≡ 3 (mod 4)`
//! (Haböck, Levit, Papini, "Circle STARKs", ePrint 2024/278, Section 3.1).
//!
//! The `p + 1` points form a cyclic group under
//! `(x₀, y₀) · (x₁, y₁) = (x₀x₁ − y₀y₁, x₀y₁ + y₀x₁)`, with neutral element
//! `(1, 0)`, inverse `J(x, y) = (x, −y)`, and squaring map
//! `π(x, y) = (2x² − 1, 2xy)`. A prime is CFFT-friendly supporting the order
//! `m` when `2^{m+1} | p + 1`; Mersenne `2³¹ − 1` supports every order up to
//! `30`.

use super::math::{mod_add, mod_inv, mod_mul, mod_pow, mod_sub};

/// A point of the circle group over `F_p`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CirclePoint {
    /// The `x` coordinate.
    pub x: u64,
    /// The `y` coordinate.
    pub y: u64,
}

impl CirclePoint {
    /// The neutral element `(1, 0)`.
    #[must_use]
    pub const fn identity() -> Self {
        Self { x: 1, y: 0 }
    }

    /// The group product.
    #[must_use]
    pub fn mul(self, other: Self, modulus: u64) -> Self {
        Self {
            x: mod_sub(
                mod_mul(self.x, other.x, modulus),
                mod_mul(self.y, other.y, modulus),
                modulus,
            ),
            y: mod_add(
                mod_mul(self.x, other.y, modulus),
                mod_mul(self.y, other.x, modulus),
                modulus,
            ),
        }
    }

    /// The squaring map `π(x, y) = (2x² − 1, 2xy)`.
    #[must_use]
    pub fn square(self, modulus: u64) -> Self {
        self.mul(self, modulus)
    }

    /// The group inverse `J(x, y) = (x, −y)`.
    #[must_use]
    pub fn conj(self, modulus: u64) -> Self {
        Self {
            x: self.x,
            y: mod_sub(0, self.y, modulus),
        }
    }

    /// `self^exponent` by repeated squaring.
    #[must_use]
    pub fn pow(self, mut exponent: u64, modulus: u64) -> Self {
        let mut base = self;
        let mut result = Self::identity();
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = result.mul(base, modulus);
            }
            base = base.square(modulus);
            exponent >>= 1;
        }
        result
    }

    /// Whether the point lies on the circle.
    #[must_use]
    pub fn is_on_circle(self, modulus: u64) -> bool {
        mod_add(
            mod_mul(self.x, self.x, modulus),
            mod_mul(self.y, self.y, modulus),
            modulus,
        ) == 1
    }
}

/// The 2-adicity of `p + 1`: the largest `m` with `2^m | p + 1`.
#[must_use]
pub fn two_adicity(modulus: u64) -> u32 {
    (modulus + 1).trailing_zeros()
}

/// A square root of `value` modulo a prime `p ≡ 3 (mod 4)`, or `None` when
/// `value` is not a square: `value^{(p+1)/4}` squares to `value^{(p+1)/2} =
/// value · value^{(p−1)/2}`, which is `value` exactly for squares.
#[must_use]
pub fn sqrt_mod(value: u64, modulus: u64) -> Option<u64> {
    let root = mod_pow(value, (modulus + 1) / 4, modulus);
    (mod_mul(root, root, modulus) == value % modulus).then_some(root)
}

/// A generator of the 2-Sylow subgroup of the circle group: a point of order
/// exactly `2^{two_adicity(p)}`, found by walking `x = 2, 3, …`, lifting each
/// `x` with a square `1 − x²` to the circle and raising it to the odd part of
/// `p + 1`.
///
/// Returns `None` for a modulus that is not `≡ 3 (mod 4)` or has no point of
/// full two-adic order among the first candidates, which no CFFT-friendly
/// prime does.
#[must_use]
pub fn two_adic_generator(modulus: u64) -> Option<CirclePoint> {
    if modulus % 4 != 3 {
        return None;
    }
    let adicity = two_adicity(modulus);
    let odd_part = (modulus + 1) >> adicity;
    let half_order = 1u64 << (adicity - 1);
    (2..modulus.min(1 << 16)).find_map(|x| {
        let y_squared = mod_sub(1, mod_mul(x, x, modulus), modulus);
        let y = sqrt_mod(y_squared, modulus)?;
        let candidate = CirclePoint { x, y }.pow(odd_part, modulus);
        (candidate.pow(half_order, modulus) != CirclePoint::identity()).then_some(candidate)
    })
}

/// `1 / value` modulo the prime.
#[must_use]
pub fn inverse(value: u64, modulus: u64) -> u64 {
    mod_inv(value, modulus)
}

#[cfg(test)]
mod tests {
    use super::{two_adic_generator, two_adicity, CirclePoint};
    use crate::domain::contracts::config::MERSENNE31;

    #[test]
    fn mersenne31_supports_order_thirty() {
        assert_eq!(two_adicity(MERSENNE31), 31);
        let generator = two_adic_generator(MERSENNE31).expect("Mersenne31 is CFFT-friendly");
        assert!(generator.is_on_circle(MERSENNE31));
        assert_eq!(generator.pow(1 << 31, MERSENNE31), CirclePoint::identity());
        assert_ne!(generator.pow(1 << 30, MERSENNE31), CirclePoint::identity());
    }

    #[test]
    fn the_group_law_squares_and_inverts() {
        let p = MERSENNE31;
        let g = two_adic_generator(p).expect("generator");
        assert_eq!(g.mul(g, p), g.square(p));
        assert_eq!(g.mul(g.conj(p), p), CirclePoint::identity());
        let (x, y) = (g.x as u128, g.y as u128);
        let pi = g.square(p);
        assert_eq!(
            pi.x as u128,
            (2 * x * x % p as u128 + p as u128 - 1) % p as u128
        );
        assert_eq!(pi.y as u128, 2 * x * y % p as u128);
    }

    #[test]
    fn a_prime_not_three_mod_four_has_no_generator() {
        assert!(two_adic_generator(998_244_353).is_none());
    }
}
