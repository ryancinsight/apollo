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

#![cfg_attr(
    test,
    expect(
        clippy::unwrap_used,
        reason = "ratchet APOLLO-UNWRAP-1: pre-existing debt"
    )
)]

use crate::domain::contracts::circle::{inverse, two_adic_generator, two_adicity, CirclePoint};
use crate::domain::contracts::error::NttError;
use crate::domain::contracts::math::{bit_reverse_permute, mod_add, mod_mul, mod_sub};
use std::collections::hash_map::Entry;
use std::collections::HashMap;

mod twiddles;
use twiddles::LevelTwiddles;

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
    forward_x: LevelTwiddles,
    /// The inverse `y` twiddles `y_i`.
    inverse_y: Vec<u64>,
    /// The inverse `x` twiddles `x` per level.
    inverse_x: LevelTwiddles,
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
        let half_points = points.len();
        points.reserve(half_points);
        for index in 0..half_points {
            points.push(points[index].conj(modulus));
        }

        let half = inverse(2, modulus);
        let inverse_y: Vec<u64> = points[..half_size].iter().map(|point| point.y).collect();
        let forward_y: Vec<u64> = inverse_y
            .iter()
            .map(|&y| inverse(mod_mul(2, y, modulus), modulus))
            .collect();
        // The x levels: the ordered `S_n`, then `π` of its first half, and so
        // on down to a single pair. They are written back to back. The
        // geometric lengths derive their boundaries, and the shrinking work
        // buffer is updated in place so plan construction never allocates one
        // temporary per level.
        let mut inverse_values: Vec<u64> = Vec::with_capacity(half_size.saturating_sub(1));
        let mut level = ordered;
        while level.len() >= 2 {
            let twiddle_count = level.len() / 2;
            inverse_values.extend_from_slice(&level[..twiddle_count]);
            for x in &mut level[..twiddle_count] {
                *x = square_x(*x, modulus);
            }
            level.truncate(twiddle_count);
        }
        let forward_values: Vec<u64> = inverse_values
            .iter()
            .map(|&x| inverse(mod_mul(2, x, modulus), modulus))
            .collect();
        let inverse_x = LevelTwiddles::from_values(inverse_values);
        let forward_x = LevelTwiddles::from_values(forward_values);
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
        apply_forward_pair_stage(data, &self.forward_y, self.half, p);
        let mut block = half_len;
        for twiddles in self.forward_x.levels() {
            apply_forward_x_stage(data, block, twiddles, self.half, p);
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
        for twiddles in self.inverse_x.levels_reversed() {
            apply_inverse_x_stage(data, block, twiddles, p);
            block *= 2;
        }
        apply_inverse_pair_stage(data, &self.inverse_y, p);
    }
}

/// Apply the `y`-split stage for the conjugate pair layout: every first-half
/// element is paired with its conjugate in the second half and updated with the
/// same `1 / (2y)` twiddle used by the basis decomposition.
fn apply_forward_pair_stage(data: &mut [u64], twiddles: &[u64], half: u64, modulus: u64) {
    let (left, right) = data.split_at_mut(data.len() / 2);
    for ((a, b), &twiddle) in left.iter_mut().zip(right.iter_mut()).zip(twiddles) {
        let (sum, difference) = forward_butterfly_pair(*a, *b, twiddle, half, modulus);
        *a = sum;
        *b = difference;
    }
}

/// Inverse of [`apply_forward_pair_stage`]: unwind the conjugate-split using the
/// `y` twiddle values without moving data from the pair layout.
fn apply_inverse_pair_stage(data: &mut [u64], twiddles: &[u64], modulus: u64) {
    let (left, right) = data.split_at_mut(data.len() / 2);
    for ((a, b), &twiddle) in left.iter_mut().zip(right.iter_mut()).zip(twiddles) {
        let (value, scaled) = inverse_butterfly_pair(*a, *b, twiddle, modulus);
        *a = value;
        *b = scaled;
    }
}

/// Apply one `x`-split stage to each block: each block is split in half, then
/// the `±x` pair is combined with the matching twiddle and the block is kept in
/// place so the bit-reversal permutation is still the single final reorder.
fn apply_forward_x_stage(data: &mut [u64], block: usize, twiddles: &[u64], half: u64, modulus: u64) {
    for chunk in data.chunks_exact_mut(block) {
        let (even, odd) = chunk.split_at_mut(block / 2);
        for ((a, b), &twiddle) in even.iter_mut().zip(odd.iter_mut()).zip(twiddles) {
            let (sum, difference) = forward_butterfly_pair(*a, *b, twiddle, half, modulus);
            *a = sum;
            *b = difference;
        }
    }
}

fn apply_inverse_x_stage(data: &mut [u64], block: usize, twiddles: &[u64], modulus: u64) {
    for chunk in data.chunks_exact_mut(block) {
        let (even, odd) = chunk.split_at_mut(block / 2);
        for ((a, b), &twiddle) in even.iter_mut().zip(odd.iter_mut()).zip(twiddles) {
            let (value, scaled) = inverse_butterfly_pair(*a, *b, twiddle, modulus);
            *a = value;
            *b = scaled;
        }
    }
}

fn forward_butterfly_pair(a: u64, b: u64, twiddle: u64, half: u64, modulus: u64) -> (u64, u64) {
    let sum = mod_add(a, b, modulus);
    let difference = mod_sub(a, b, modulus);
    (mod_mul(sum, half, modulus), mod_mul(difference, twiddle, modulus))
}

fn inverse_butterfly_pair(a: u64, b: u64, twiddle: u64, modulus: u64) -> (u64, u64) {
    let scaled = mod_mul(b, twiddle, modulus);
    let value = a;
    (mod_add(value, scaled, modulus), mod_sub(value, scaled, modulus))
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
    use super::{apply_forward_pair_stage, forward_butterfly_pair, square_x, CircleDomain};
    use crate::domain::contracts::circle::inverse;
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

    /// Both directions hold their `x` levels in one contiguous run: the level
    /// count, every level's length and the total value count follow from the
    /// domain size, so the layout cannot silently drift back to per-level
    /// buffers.
    #[test]
    fn x_twiddle_levels_are_one_contiguous_run() {
        for power in 1..=8usize {
            let log_size = u32::try_from(power).expect("invariant: fixture power fits in u32");
            let domain = CircleDomain::new(log_size, MERSENNE31).expect("domain");
            let n = domain.len();
            for twiddles in [&domain.forward_x, &domain.inverse_x] {
                let expected_levels = power - 1;
                assert_eq!(
                    twiddles.level_count(),
                    expected_levels,
                    "level count at n=2^{log_size}"
                );
                let mut total = 0usize;
                for (index, level) in twiddles.levels().enumerate() {
                    let expected = (n / 4) >> index;
                    assert_eq!(
                        level.len(),
                        expected,
                        "level {index} length at n=2^{log_size}"
                    );
                    total += level.len();
                }
                assert_eq!(total, n / 2 - 1, "total twiddle values at n=2^{log_size}");
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
    fn pair_stages_match_the_manual_butterfly_formula() {
        let modulus = MERSENNE31;
        let half = inverse(2, modulus);
        let twiddle = 7u64;
        let mut data = vec![17, 23, 31, 41];
        let original = data.clone();

        apply_forward_pair_stage(&mut data, &[twiddle; 2], half, modulus);
        let expected = [
            forward_butterfly_pair(original[0], original[2], twiddle, half, modulus),
            forward_butterfly_pair(original[1], original[3], twiddle, half, modulus),
        ];
        assert_eq!(data[0], expected[0].0, "forward-left pair 0");
        assert_eq!(data[2], expected[0].1, "forward-right pair 0");
        assert_eq!(data[1], expected[1].0, "forward-left pair 1");
        assert_eq!(data[3], expected[1].1, "forward-right pair 1");
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
