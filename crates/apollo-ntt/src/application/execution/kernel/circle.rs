//! The circle FFT over a twin-coset domain (Haböck, Levit, Papini, "Circle
//! STARKs", ePrint 2024/278, Section 4).
//!
//! # Mathematical contract
//!
//! Let `p ≡ 3 (mod 4)` support the order `n` (`2^{n+1} | p + 1`), `Q` a
//! circle-group element of order `2^{n+1}`, and `G_{n−1} = ⟨Q⁴⟩`. The domain
//! is the standard position coset of size `N = 2^n`,
//!
//! ```text
//! D = Q · G_{n−1} ∪ Q⁻¹ · G_{n−1},
//! ```
//!
//! and the FFT basis of order `n` is, for `k = k₀ + 2k₁ + … + 2^{n−1}k_{n−1}`,
//!
//! ```text
//! b_k(x, y) = y^{k₀} · v₁(x)^{k₁} · … · v_{n−1}(x)^{k_{n−1}},   v₁(x) = x,  v_{j+1}(x) = v_j(2x² − 1).
//! ```
//!
//! The inverse transform evaluates `Σ_k c_k b_k` over `D`; the forward
//! transform recovers the `c_k` from the values (Theorems 2 and 3): the first
//! split separates the `J`-even and `J`-odd parts along `y`,
//! `f₀(x) = (f(x, y) + f(x, −y)) / 2`, `f₁(x) = (f(x, y) − f(x, −y)) / 2y`,
//! and every later split separates along `x` under the squaring map
//! `π(x) = 2x² − 1`, `f₀(π(x)) = (f(x) + f(−x)) / 2`,
//! `f₁(π(x)) = (f(x) − f(−x)) / 2x`. Both directions cost `N · n` additions
//! and at most `N · n` multiplications by precomputed twiddles.
//!
//! The basis spans `L′_N`, the `N`-dimensional subspace of the bivariate
//! polynomials of total degree at most `N/2` modulo `x² + y² − 1` whose
//! canonical form `p₀(x) + y·p₁(x)` has `deg p_i ≤ N/2 − 1` (Lemma 6). The
//! full space `L_N` has one more dimension, spanned by the vanishing
//! polynomial `v_n(x)` of `D`, so the pointwise product of two functions on
//! `D` interpolates to their polynomial product modulo `v_n(x)`, the circle
//! analogue of the cyclic convolution's reduction modulo `x^N − 1`.
//!
//! # Layout
//!
//! Point `i < N/2` of the domain is `(x_i, y_i)` and point `i + N/2` its
//! conjugate `(x_i, −y_i)`; the `x_i` are ordered so that within every block
//! of the recursion the two preimages `±x` of one `π(x)` sit half a block
//! apart. The forward butterflies therefore run in place and leave the
//! coefficients in bit-reversed order, undone by one permutation.

use crate::domain::contracts::circle::{inverse, two_adic_generator, two_adicity, CirclePoint};
use crate::domain::contracts::error::NttError;
use crate::domain::contracts::math::{bit_reverse_permute, mod_add, mod_mul, mod_sub};
use std::collections::hash_map::Entry;
use std::collections::HashMap;

/// A twin-coset domain of size `2^n` with the twiddles of both directions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircleDomain {
    modulus: u64,
    log_size: u32,
    points: Vec<CirclePoint>,
    /// `1 / 2`.
    half: u64,
    /// The forward `y` twiddles `1 / (2 y_i)`, `N/2` of them.
    forward_y: Vec<u64>,
    /// The forward `x` twiddles `1 / (2 x)` per level, `N/4, N/8, …, 1` long.
    forward_x: Vec<Vec<u64>>,
    /// The inverse `y` twiddles `y_i`.
    inverse_y: Vec<u64>,
    /// The inverse `x` twiddles `x` per level.
    inverse_x: Vec<Vec<u64>>,
}

impl CircleDomain {
    /// The standard position coset of size `2^log_size` over `modulus`.
    ///
    /// # Errors
    ///
    /// [`NttError::EmptyLength`] for `log_size = 0` (no twin-coset of size
    /// one exists), [`NttError::InvalidModulus`] when `modulus < 3` or
    /// `modulus ≢ 3 (mod 4)`, [`NttError::CompositeModulus`] for a composite
    /// modulus, [`NttError::UnsupportedLength`] when the prime
    /// does not support the order (`2^{log_size + 1} ∤ modulus + 1`).
    pub fn new(log_size: u32, modulus: u64) -> Result<Self, NttError> {
        if log_size == 0 {
            return Err(NttError::EmptyLength);
        }
        if modulus < 3 || modulus % 4 != 3 {
            return Err(NttError::InvalidModulus);
        }
        if !crate::domain::contracts::math::is_prime(modulus) {
            return Err(NttError::CompositeModulus { modulus });
        }
        if log_size + 1 > two_adicity(modulus) {
            return Err(NttError::UnsupportedLength);
        }
        let generator = two_adic_generator(modulus).ok_or(NttError::InvalidModulus)?;
        // `Q` of order `2^{log_size + 1}`.
        let q = generator.pow(1u64 << (two_adicity(modulus) - log_size - 1), modulus);
        let half_size = 1usize << (log_size - 1);
        // The `J`-orbit representatives `Q^{4k+1}`, `k < N/2`, and their `y`
        // by `x`.
        let q_four = q.pow(4, modulus);
        let mut representatives = Vec::with_capacity(half_size);
        let mut point = q;
        for _ in 0..half_size {
            representatives.push(point);
            point = point.mul(q_four, modulus);
        }
        let x_values: Vec<u64> = representatives.iter().map(|point| point.x).collect();
        let ordered = order_for_splitting(&x_values, modulus);
        // Lookup table, not a search: `ordered` holds every representative `x`
        // exactly once (the distinctness test below pins this), so indexing
        // replaces the former linear scan per entry.
        let y_by_x: HashMap<u64, u64> = representatives
            .iter()
            .map(|point| (point.x, point.y))
            .collect();
        let y_of = |x: u64| {
            *y_by_x
                .get(&x)
                .expect("invariant: every ordered x is a representative's x")
        };
        let mut points: Vec<CirclePoint> = ordered
            .iter()
            .map(|&x| CirclePoint { x, y: y_of(x) })
            .collect();
        let conjugates: Vec<CirclePoint> = points.iter().map(|point| point.conj(modulus)).collect();
        points.extend(conjugates);

        let half = inverse(2, modulus);
        let inverse_y: Vec<u64> = points[..half_size].iter().map(|point| point.y).collect();
        let forward_y: Vec<u64> = inverse_y
            .iter()
            .map(|&y| inverse(mod_mul(2, y, modulus), modulus))
            .collect();
        // The x levels: the ordered `S_n`, then `π` of its first half, and so on
        // down to a single pair.
        let mut inverse_x: Vec<Vec<u64>> = Vec::new();
        let mut level: Vec<u64> = ordered;
        while level.len() >= 2 {
            let twiddles: Vec<u64> = level[..level.len() / 2].to_vec();
            let next: Vec<u64> = twiddles.iter().map(|&x| square_x(x, modulus)).collect();
            inverse_x.push(twiddles);
            level = next;
        }
        let forward_x: Vec<Vec<u64>> = inverse_x
            .iter()
            .map(|level| {
                level
                    .iter()
                    .map(|&x| inverse(mod_mul(2, x, modulus), modulus))
                    .collect()
            })
            .collect();
        Ok(Self {
            modulus,
            log_size,
            points,
            half,
            forward_y,
            forward_x,
            inverse_y,
            inverse_x,
        })
    }

    /// The domain size `N = 2^n`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the domain is empty, which no constructed domain is.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// `n` with `N = 2^n`.
    #[must_use]
    pub const fn log_size(&self) -> u32 {
        self.log_size
    }

    /// The modulus.
    #[must_use]
    pub const fn modulus(&self) -> u64 {
        self.modulus
    }

    /// The domain points in the layout the transforms use: `i < N/2` is
    /// `(x_i, y_i)` and `i + N/2` its conjugate.
    #[must_use]
    pub fn points(&self) -> &[CirclePoint] {
        &self.points
    }

    /// Values over the domain to coefficients in the FFT basis, in place.
    ///
    /// # Panics
    ///
    /// Panics when `data.len()` is not the domain size.
    pub fn forward(&self, data: &mut [u64]) {
        assert_eq!(data.len(), self.len(), "circle FFT length mismatch");
        let p = self.modulus;
        let half_len = data.len() / 2;
        let (even, odd) = data.split_at_mut(half_len);
        for ((a, b), &twiddle) in even.iter_mut().zip(odd.iter_mut()).zip(&self.forward_y) {
            let sum = mod_add(*a, *b, p);
            let difference = mod_sub(*a, *b, p);
            *a = mod_mul(sum, self.half, p);
            *b = mod_mul(difference, twiddle, p);
        }
        let mut block = half_len;
        for twiddles in &self.forward_x {
            for chunk in data.chunks_exact_mut(block) {
                let (even, odd) = chunk.split_at_mut(block / 2);
                for ((a, b), &twiddle) in even.iter_mut().zip(odd.iter_mut()).zip(twiddles) {
                    let sum = mod_add(*a, *b, p);
                    let difference = mod_sub(*a, *b, p);
                    *a = mod_mul(sum, self.half, p);
                    *b = mod_mul(difference, twiddle, p);
                }
            }
            block /= 2;
        }
        bit_reverse_permute(data);
    }

    /// Coefficients in the FFT basis to values over the domain, in place.
    ///
    /// # Panics
    ///
    /// Panics when `data.len()` is not the domain size.
    pub fn inverse(&self, data: &mut [u64]) {
        assert_eq!(data.len(), self.len(), "circle FFT length mismatch");
        let p = self.modulus;
        bit_reverse_permute(data);
        let mut block = 2;
        for twiddles in self.inverse_x.iter().rev() {
            for chunk in data.chunks_exact_mut(block) {
                let (even, odd) = chunk.split_at_mut(block / 2);
                for ((a, b), &twiddle) in even.iter_mut().zip(odd.iter_mut()).zip(twiddles) {
                    let scaled = mod_mul(*b, twiddle, p);
                    let value = *a;
                    *a = mod_add(value, scaled, p);
                    *b = mod_sub(value, scaled, p);
                }
            }
            block *= 2;
        }
        let half_len = data.len() / 2;
        let (even, odd) = data.split_at_mut(half_len);
        for ((a, b), &twiddle) in even.iter_mut().zip(odd.iter_mut()).zip(&self.inverse_y) {
            let scaled = mod_mul(*b, twiddle, p);
            let value = *a;
            *a = mod_add(value, scaled, p);
            *b = mod_sub(value, scaled, p);
        }
    }
}

/// `π` on the `x` axis: `2x² − 1`.
fn square_x(x: u64, modulus: u64) -> u64 {
    mod_sub(mod_mul(2, mod_mul(x, x, modulus), modulus), 1, modulus)
}

/// Orders the `x` values of a `J`-quotient `S_j` so that the two preimages
/// `±x` of every `π(x)` sit half the list apart, recursively down the chain
/// `S_j → π(S_j) → …`: the first half is one preimage per element of the
/// recursively ordered `π(S_j)`, the second half their negatives.
fn order_for_splitting(values: &[u64], modulus: u64) -> Vec<u64> {
    if values.len() <= 1 {
        return values.to_vec();
    }
    // One representative per `{x, −x}` pair, keyed by `π(x)`. The map keeps
    // first-seen order in `images` alongside it: `HashMap` iteration order is
    // unspecified, and the recursion below must see images in input order so
    // the domain layout is bit-identical to the former linear scan.
    let mut representative_of: HashMap<u64, u64> = HashMap::with_capacity(values.len() / 2);
    let mut images: Vec<u64> = Vec::with_capacity(values.len() / 2);
    for &x in values {
        let image = square_x(x, modulus);
        if let Entry::Vacant(slot) = representative_of.entry(image) {
            slot.insert(x);
            images.push(image);
        }
    }
    debug_assert_eq!(representative_of.len(), values.len() / 2);
    let ordered_images = order_for_splitting(&images, modulus);
    let preimage_of = |image: u64| {
        *representative_of
            .get(&image)
            .expect("invariant: every ordered image came from a representative")
    };
    let firsts: Vec<u64> = ordered_images
        .iter()
        .map(|&image| preimage_of(image))
        .collect();
    let mut ordered = firsts.clone();
    ordered.extend(firsts.iter().map(|&x| mod_sub(0, x, modulus)));
    ordered
}

#[cfg(test)]
mod tests {
    use super::{square_x, CircleDomain};
    use crate::domain::contracts::config::MERSENNE31;
    use crate::domain::contracts::math::mod_mul;

    fn values(n: usize, seed: u64) -> Vec<u64> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state % MERSENNE31
            })
            .collect()
    }

    #[test]
    fn the_domain_lies_on_the_circle_and_pairs_conjugates() {
        for log_size in 1..=6 {
            let domain = CircleDomain::new(log_size, MERSENNE31).expect("domain");
            let half = domain.len() / 2;
            let points = domain.points();
            for i in 0..half {
                assert!(points[i].is_on_circle(MERSENNE31));
                assert_eq!(points[i + half], points[i].conj(MERSENNE31));
            }
            let mut xs: Vec<u64> = points[..half].iter().map(|p| p.x).collect();
            xs.sort_unstable();
            xs.dedup();
            assert_eq!(xs.len(), half, "distinct x per J-orbit");
        }
    }

    /// The inverse transform of a unit coefficient vector is the basis
    /// polynomial `b_k = y^{k₀} Π v_j(x)^{k_j}` evaluated over the domain.
    #[test]
    fn unit_coefficients_evaluate_the_basis() {
        let p = MERSENNE31;
        for log_size in 1..=5u32 {
            let domain = CircleDomain::new(log_size, p).expect("domain");
            let n = domain.len();
            for k in 0..n {
                let mut data = vec![0u64; n];
                data[k] = 1;
                domain.inverse(&mut data);
                for (i, point) in domain.points().iter().enumerate() {
                    let mut want = if k & 1 == 1 { point.y } else { 1 };
                    let mut v = point.x;
                    for j in 1..log_size {
                        if (k >> j) & 1 == 1 {
                            want = mod_mul(want, v, p);
                        }
                        v = square_x(v, p);
                    }
                    assert_eq!(data[i], want, "n=2^{log_size} b_{k} at point {i}");
                }
            }
        }
    }

    #[test]
    fn forward_and_inverse_round_trip_exactly() {
        for log_size in [1u32, 2, 3, 6, 10, 16] {
            let domain = CircleDomain::new(log_size, MERSENNE31).expect("domain");
            let original = values(domain.len(), 0x9E37_79B9 + u64::from(log_size));
            let mut data = original.clone();
            domain.forward(&mut data);
            domain.inverse(&mut data);
            assert_eq!(data, original, "n=2^{log_size}");
            let mut data = original.clone();
            domain.inverse(&mut data);
            domain.forward(&mut data);
            assert_eq!(data, original, "n=2^{log_size} the other way");
        }
    }

    #[test]
    fn unsupported_configurations_are_rejected() {
        use crate::domain::contracts::error::NttError;
        assert_eq!(
            CircleDomain::new(0, MERSENNE31).unwrap_err(),
            NttError::EmptyLength
        );
        assert_eq!(
            CircleDomain::new(31, MERSENNE31).unwrap_err(),
            NttError::UnsupportedLength
        );
        assert_eq!(
            CircleDomain::new(3, 998_244_353).unwrap_err(),
            NttError::InvalidModulus
        );
    }
}
