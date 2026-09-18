//! The circle-group transform plan: lengths `2^n` over primes whose `p + 1`
//! is two-adic enough, such as Mersenne `2³¹ − 1` (ADR 0065).
//!
//! # Mathematical contract
//!
//! The plan is a multipoint evaluation, not the cyclic DFT: its forward
//! transform maps the values of a function over the size-`N` twin-coset
//! domain `D` of the circle group to its coefficients in the FFT basis
//! `b_k(x, y) = y^{k₀} v₁(x)^{k₁} ⋯ v_{n−1}(x)^{k_{n−1}}` (`v₁ = x`,
//! `v_{j+1} = v_j ∘ π`, `π(x) = 2x² − 1`), and the inverse evaluates
//! `Σ_k c_k b_k` over `D`. See
//! [`CircleDomain`](crate::application::execution::kernel::circle::CircleDomain)
//! for the algorithm and the layout of `D`.
//!
//! The basis spans `L′_N`, the polynomials `p₀(x) + y p₁(x)` with
//! `deg p_i ≤ N/2 − 1`; the pointwise product of two functions over `D`
//! interpolates to their polynomial product reduced modulo the vanishing
//! polynomial `v_n(x)` of `D`, the analogue of the cyclic NTT's reduction
//! modulo `x^N − 1`.

use crate::application::execution::kernel::circle::CircleDomain;
use crate::domain::contracts::circle::CirclePoint;
use crate::domain::contracts::config::MERSENNE31;
use crate::domain::contracts::error::NttError;
use leto::Array1;

/// Reusable circle FFT plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircleNttPlan {
    domain: CircleDomain,
}

impl CircleNttPlan {
    /// Build a plan of length `n` over Mersenne `2³¹ − 1`.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::with_modulus`].
    pub fn new(n: usize) -> Result<Self, NttError> {
        Self::with_modulus(n, MERSENNE31)
    }

    /// Build a plan of length `n` over an explicit prime `modulus ≡ 3 (mod 4)`.
    ///
    /// # Errors
    ///
    /// [`NttError::EmptyLength`] for `n < 2` (no twin-coset of size one),
    /// [`NttError::NonPowerOfTwo`] for a length that is not a power of two,
    /// [`NttError::InvalidModulus`] for a modulus below three or not
    /// `≡ 3 (mod 4)`, [`NttError::CompositeModulus`] for a composite
    /// modulus, [`NttError::UnsupportedLength`] when `2n ∤ modulus + 1`.
    pub fn with_modulus(n: usize, modulus: u64) -> Result<Self, NttError> {
        if n == 0 {
            return Err(NttError::EmptyLength);
        }
        if !n.is_power_of_two() {
            return Err(NttError::NonPowerOfTwo);
        }
        let domain = CircleDomain::new(n.trailing_zeros(), modulus)?;
        Ok(Self { domain })
    }

    /// Return the transform length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.domain.len()
    }

    /// Return whether the plan length is zero, which it never is.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.domain.is_empty()
    }

    /// Return the modulus.
    #[must_use]
    pub const fn modulus(&self) -> u64 {
        self.domain.modulus()
    }

    /// The evaluation domain, in the transforms' layout.
    #[must_use]
    pub fn domain(&self) -> &[CirclePoint] {
        self.domain.points()
    }

    /// Allocate and execute the forward transform: values over the domain to
    /// coefficients in the FFT basis.
    ///
    /// # Errors
    ///
    /// [`NttError::LengthMismatch`] when `input` is not the plan length.
    pub fn forward(&self, input: &Array1<u64>) -> Result<Array1<u64>, NttError> {
        let mut output = Array1::zeros([self.len()]);
        self.forward_into(input, &mut output)?;
        Ok(output)
    }

    /// Allocate and execute the inverse transform: coefficients to values.
    ///
    /// # Errors
    ///
    /// [`NttError::LengthMismatch`] when `input` is not the plan length.
    pub fn inverse(&self, input: &Array1<u64>) -> Result<Array1<u64>, NttError> {
        let mut output = Array1::zeros([self.len()]);
        self.inverse_into(input, &mut output)?;
        Ok(output)
    }

    /// Execute the forward transform into caller-owned output storage.
    ///
    /// # Errors
    ///
    /// [`NttError::LengthMismatch`] when either array is not the plan length.
    pub fn forward_into(
        &self,
        input: &Array1<u64>,
        output: &mut Array1<u64>,
    ) -> Result<(), NttError> {
        self.forward_slice_into(
            input
                .as_slice()
                .expect("owned Array1 storage is contiguous"),
            output
                .as_slice_mut()
                .expect("owned Array1 storage is contiguous"),
        )
    }

    /// Execute the inverse transform into caller-owned output storage.
    ///
    /// # Errors
    ///
    /// [`NttError::LengthMismatch`] when either array is not the plan length.
    pub fn inverse_into(
        &self,
        input: &Array1<u64>,
        output: &mut Array1<u64>,
    ) -> Result<(), NttError> {
        self.inverse_slice_into(
            input
                .as_slice()
                .expect("owned Array1 storage is contiguous"),
            output
                .as_slice_mut()
                .expect("owned Array1 storage is contiguous"),
        )
    }

    /// Execute the forward transform into caller-owned contiguous storage.
    ///
    /// # Errors
    ///
    /// [`NttError::LengthMismatch`] when either slice is not the plan length.
    pub fn forward_slice_into(&self, input: &[u64], output: &mut [u64]) -> Result<(), NttError> {
        self.check_len(input.len())?;
        self.check_len(output.len())?;
        output.copy_from_slice(input);
        self.forward_slice_inplace(output)
    }

    /// Execute the inverse transform into caller-owned contiguous storage.
    ///
    /// # Errors
    ///
    /// [`NttError::LengthMismatch`] when either slice is not the plan length.
    pub fn inverse_slice_into(&self, input: &[u64], output: &mut [u64]) -> Result<(), NttError> {
        self.check_len(input.len())?;
        self.check_len(output.len())?;
        output.copy_from_slice(input);
        self.inverse_slice_inplace(output)
    }

    /// Execute the forward transform in place.
    ///
    /// # Errors
    ///
    /// [`NttError::LengthMismatch`] when `data` is not the plan length.
    pub fn forward_slice_inplace(&self, data: &mut [u64]) -> Result<(), NttError> {
        self.check_len(data.len())?;
        self.domain.forward(data);
        Ok(())
    }

    /// Execute the inverse transform in place.
    ///
    /// # Errors
    ///
    /// [`NttError::LengthMismatch`] when `data` is not the plan length.
    pub fn inverse_slice_inplace(&self, data: &mut [u64]) -> Result<(), NttError> {
        self.check_len(data.len())?;
        self.domain.inverse(data);
        Ok(())
    }

    fn check_len(&self, len: usize) -> Result<(), NttError> {
        if len == self.len() {
            Ok(())
        } else {
            Err(NttError::LengthMismatch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CircleNttPlan;
    use crate::domain::contracts::circle::CirclePoint;
    use crate::domain::contracts::config::MERSENNE31;
    use crate::domain::contracts::error::NttError;
    use crate::domain::contracts::math::{mod_add, mod_mul, mod_sub};
    use leto::Array1;

    const P: u64 = MERSENNE31;

    /// A polynomial over the circle in canonical form `p₀(x) + y p₁(x)`.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct CirclePolynomial {
        even: Vec<u64>,
        odd: Vec<u64>,
    }

    fn poly_mul(a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut out = vec![0u64; a.len() + b.len() - 1];
        for (i, &x) in a.iter().enumerate() {
            for (j, &y) in b.iter().enumerate() {
                out[i + j] = mod_add(out[i + j], mod_mul(x, y, P), P);
            }
        }
        out
    }

    fn poly_add(a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut out = vec![0u64; a.len().max(b.len())];
        for (i, slot) in out.iter_mut().enumerate() {
            let x = a.get(i).copied().unwrap_or(0);
            let y = b.get(i).copied().unwrap_or(0);
            *slot = mod_add(x, y, P);
        }
        out
    }

    fn poly_sub(a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut out = vec![0u64; a.len().max(b.len())];
        for (i, slot) in out.iter_mut().enumerate() {
            let x = a.get(i).copied().unwrap_or(0);
            let y = b.get(i).copied().unwrap_or(0);
            *slot = mod_sub(x, y, P);
        }
        out
    }

    /// `p(q(x))` for univariate `p`, `q`.
    fn poly_compose(p: &[u64], q: &[u64]) -> Vec<u64> {
        let mut result = vec![0u64];
        for &coefficient in p.iter().rev() {
            result = poly_add(&poly_mul(&result, q), &[coefficient]);
        }
        result
    }

    fn eval(p: &[u64], x: u64) -> u64 {
        p.iter()
            .rev()
            .fold(0, |acc, &c| mod_add(mod_mul(acc, x, P), c, P))
    }

    impl CirclePolynomial {
        /// `(p₀ + y p₁)(q₀ + y q₁) = p₀q₀ + (1 − x²) p₁q₁ + y (p₀q₁ + p₁q₀)`.
        fn mul(&self, other: &Self) -> Self {
            let one_minus_x_squared = [1, 0, mod_sub(0, 1, P)];
            let even = poly_add(
                &poly_mul(&self.even, &other.even),
                &poly_mul(&poly_mul(&self.odd, &other.odd), &one_minus_x_squared),
            );
            let odd = poly_add(
                &poly_mul(&self.even, &other.odd),
                &poly_mul(&self.odd, &other.even),
            );
            Self { even, odd }
        }

        fn sub(&self, other: &Self) -> Self {
            Self {
                even: poly_sub(&self.even, &other.even),
                odd: poly_sub(&self.odd, &other.odd),
            }
        }

        fn eval(&self, point: CirclePoint) -> u64 {
            mod_add(
                eval(&self.even, point.x),
                mod_mul(point.y, eval(&self.odd, point.x), P),
                P,
            )
        }
    }

    /// The basis polynomial `b_k` of order `log_size` in canonical form.
    fn basis(k: usize, log_size: u32) -> CirclePolynomial {
        let pi = [mod_sub(0, 1, P), 0, 2];
        let mut v = vec![0u64, 1];
        let mut product = vec![1u64];
        for j in 1..log_size {
            if (k >> j) & 1 == 1 {
                product = poly_mul(&product, &v);
            }
            v = poly_compose(&v, &pi);
        }
        if k & 1 == 1 {
            CirclePolynomial {
                even: vec![0],
                odd: product,
            }
        } else {
            CirclePolynomial {
                even: product,
                odd: vec![0],
            }
        }
    }

    /// `Σ c_k b_k` in canonical form.
    fn interpolant(coefficients: &[u64], log_size: u32) -> CirclePolynomial {
        let mut sum = CirclePolynomial {
            even: vec![0],
            odd: vec![0],
        };
        for (k, &c) in coefficients.iter().enumerate() {
            let b = basis(k, log_size);
            let scaled = CirclePolynomial {
                even: b.even.iter().map(|&v| mod_mul(v, c, P)).collect(),
                odd: b.odd.iter().map(|&v| mod_mul(v, c, P)).collect(),
            };
            sum = CirclePolynomial {
                even: poly_add(&sum.even, &scaled.even),
                odd: poly_add(&sum.odd, &scaled.odd),
            };
        }
        sum
    }

    /// The vanishing polynomial `v_n(x) = π^{n−1}(x)` of the domain.
    fn vanishing(log_size: u32) -> Vec<u64> {
        let pi = [mod_sub(0, 1, P), 0, 2];
        let mut v = vec![0u64, 1];
        for _ in 1..log_size {
            v = poly_compose(&v, &pi);
        }
        v
    }

    /// The coefficients without trailing zeros, so the leading term is real.
    fn trimmed(p: &[u64]) -> Vec<u64> {
        let mut v = p.to_vec();
        while v.len() > 1 && v.last() == Some(&0) {
            v.pop();
        }
        v
    }

    /// Polynomial remainder of `a` modulo `m`.
    fn poly_rem(a: &[u64], m: &[u64]) -> Vec<u64> {
        let m = trimmed(m);
        let mut rem = trimmed(a);
        let lead = *m.last().expect("modulus has a leading term");
        let lead_inv = crate::domain::contracts::math::mod_inv(lead, P);
        while rem.len() >= m.len() {
            let factor = mod_mul(*rem.last().expect("non-empty"), lead_inv, P);
            let shift = rem.len() - m.len();
            for (i, &c) in m.iter().enumerate() {
                rem[shift + i] = mod_sub(rem[shift + i], mod_mul(factor, c, P), P);
            }
            rem.pop();
        }
        rem
    }

    fn values(n: usize, seed: u64) -> Vec<u64> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state % P
            })
            .collect()
    }

    #[test]
    fn the_forward_transform_interpolates_in_the_fft_basis() {
        for log_size in 1..=5u32 {
            let n = 1usize << log_size;
            let plan = CircleNttPlan::new(n).expect("plan");
            let samples = values(n, 7 + u64::from(log_size));
            let coefficients = plan
                .forward(&Array1::from(samples.clone()))
                .expect("forward");
            let polynomial = interpolant(coefficients.as_slice().expect("contiguous"), log_size);
            for (point, &sample) in plan.domain().iter().zip(&samples) {
                assert_eq!(polynomial.eval(*point), sample, "n={n}");
            }
        }
    }

    /// The pointwise product of two functions over the domain interpolates to
    /// their polynomial product modulo the vanishing polynomial of the domain.
    #[test]
    fn the_pointwise_product_is_the_polynomial_product_modulo_the_vanishing_polynomial() {
        for log_size in 2..=5u32 {
            let n = 1usize << log_size;
            let plan = CircleNttPlan::new(n).expect("plan");
            let left = values(n, 11 + u64::from(log_size));
            let right = values(n, 23 + u64::from(log_size));
            let product: Vec<u64> = left
                .iter()
                .zip(&right)
                .map(|(&a, &b)| mod_mul(a, b, P))
                .collect();
            let coefficients = |v: &[u64]| {
                plan.forward(&Array1::from(v.to_vec()))
                    .expect("forward")
                    .as_slice()
                    .expect("contiguous")
                    .to_vec()
            };
            let (left_poly, right_poly, product_poly) = (
                interpolant(&coefficients(&left), log_size),
                interpolant(&coefficients(&right), log_size),
                interpolant(&coefficients(&product), log_size),
            );
            let difference = left_poly.mul(&right_poly).sub(&product_poly);
            let vanishing_polynomial = vanishing(log_size);
            let zero = |remainder: &[u64]| remainder.iter().all(|&c| c == 0);
            assert!(
                zero(&poly_rem(&difference.even, &vanishing_polynomial))
                    && zero(&poly_rem(&difference.odd, &vanishing_polynomial)),
                "n={n}: f·g − h is not a multiple of v_n"
            );
        }
    }

    #[test]
    fn round_trips_are_exact_up_to_two_sixteen() {
        for log_size in [1u32, 4, 8, 12, 16] {
            let n = 1usize << log_size;
            let plan = CircleNttPlan::new(n).expect("plan");
            let samples = Array1::from(values(n, 99 + u64::from(log_size)));
            let back = plan
                .inverse(&plan.forward(&samples).expect("forward"))
                .expect("inverse");
            assert_eq!(back, samples, "n={n}");
        }
    }

    #[test]
    fn contract_violations_are_typed() {
        assert_eq!(CircleNttPlan::new(0).unwrap_err(), NttError::EmptyLength);
        assert_eq!(CircleNttPlan::new(1).unwrap_err(), NttError::EmptyLength);
        assert_eq!(CircleNttPlan::new(12).unwrap_err(), NttError::NonPowerOfTwo);
        assert_eq!(
            CircleNttPlan::new(1 << 31).unwrap_err(),
            NttError::UnsupportedLength
        );
        assert_eq!(
            CircleNttPlan::with_modulus(8, 998_244_353).unwrap_err(),
            NttError::InvalidModulus
        );
        let plan = CircleNttPlan::new(8).expect("plan");
        assert_eq!(
            plan.forward(&Array1::from(vec![0u64; 4])).unwrap_err(),
            NttError::LengthMismatch
        );
    }
}
