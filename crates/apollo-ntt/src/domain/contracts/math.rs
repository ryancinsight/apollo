//! Modular primitives and structural manipulations.

use super::error::NttError;

/// Verify the supported NTT length contract.
#[must_use]
pub fn is_valid_length(n: usize) -> bool {
    n > 0 && n.is_power_of_two()
}

/// Deterministic primality of a `u64`.
///
/// Miller–Rabin with the first twelve primes as witnesses decides every
/// `n < 3.3 · 10^24`, which covers `u64` (Sorenson and Webster, "Strong
/// pseudoprimes to twelve prime bases", Math. Comp. 86 (2017), 985–1003).
#[must_use]
pub fn is_prime(n: u64) -> bool {
    const WITNESSES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    if n < 2 {
        return false;
    }
    if let Some(&small) = WITNESSES.iter().find(|&&p| n % p == 0) {
        return n == small;
    }
    let odd = (n - 1) >> (n - 1).trailing_zeros();
    let twos = (n - 1).trailing_zeros();
    WITNESSES.iter().all(|&a| {
        let mut x = mod_pow(a, odd, n);
        if x == 1 || x == n - 1 {
            return true;
        }
        (1..twos).any(|_| {
            x = mod_mul(x, x, n);
            x == n - 1
        })
    })
}

/// The transform root `ω = g^((q - 1) / n)` for a power-of-two `n`, after
/// checking the field contract every NTT plan relies on: `q` prime, `n`
/// dividing `q - 1`, and `ω` of order exactly `n`, so the forward transform
/// is invertible with `ω^{-1}` and `n^{-1}`. For a power of two the order is
/// `n` exactly when `ω^(n/2) ≠ 1`.
///
/// # Errors
///
/// [`NttError::InvalidModulus`] for `q < 2`, [`NttError::CompositeModulus`]
/// for a composite `q`, [`NttError::UnsupportedLength`] when `n ∤ q - 1`,
/// [`NttError::NotPrimitiveRoot`] when `ω` has a smaller order.
pub fn transform_root(n: usize, modulus: u64, primitive_root: u64) -> Result<u64, NttError> {
    debug_assert!(
        is_valid_length(n),
        "invariant: callers check the length first"
    );
    if modulus < 2 {
        return Err(NttError::InvalidModulus);
    }
    if !is_prime(modulus) {
        return Err(NttError::CompositeModulus { modulus });
    }
    if (modulus - 1) % n as u64 != 0 {
        return Err(NttError::UnsupportedLength);
    }
    let root = mod_pow(primitive_root, (modulus - 1) / n as u64, modulus);
    let full_order = root != 0 && (n == 1 || mod_pow(root, n as u64 / 2, modulus) != 1);
    if !full_order {
        return Err(NttError::NotPrimitiveRoot {
            n,
            modulus,
            primitive_root,
        });
    }
    Ok(root)
}

/// Modular exponentiation.
#[must_use]
pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base, modulus);
        }
        base = mod_mul(base, base, modulus);
        exp >>= 1;
    }
    result
}

/// Modular multiplication with 128-bit widening.
#[must_use]
pub fn mod_mul(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    ((lhs as u128 * rhs as u128) % modulus as u128) as u64
}

/// Modular inverse via Fermat's little theorem for prime modulus.
#[must_use]
pub fn mod_inv(value: u64, modulus: u64) -> u64 {
    mod_pow(value, modulus - 2, modulus)
}

/// Add modulo `modulus`.
#[must_use]
pub fn mod_add(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    ((lhs as u128 + rhs as u128) % modulus as u128) as u64
}

/// Subtract modulo `modulus`.
#[must_use]
pub fn mod_sub(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    if lhs >= rhs {
        lhs - rhs
    } else {
        modulus - (rhs - lhs)
    }
}

/// Bit-reversal permutation in place.
pub fn bit_reverse_permute(data: &mut [u64]) {
    let n = data.len();
    assert!(
        is_valid_length(n),
        "NTT length must be a non-zero power of two"
    );
    let mut j = 0usize;
    for i in 1..n - 1 {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{is_prime, mod_pow, transform_root};
    use crate::domain::contracts::config::{DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT};
    use crate::domain::contracts::error::NttError;

    fn trial_division(n: u64) -> bool {
        n >= 2 && (2..=n.isqrt()).all(|d| n % d != 0)
    }

    #[test]
    fn primality_agrees_with_trial_division_below_ten_thousand() {
        for n in 0..10_000_u64 {
            assert_eq!(is_prime(n), trial_division(n), "n = {n}");
        }
    }

    #[test]
    fn primality_decides_strong_pseudoprimes_and_large_primes() {
        // Strong pseudoprimes to bases 2, 3, 5 and 7, and a Carmichael
        // number: composite, and each fools a shorter witness list.
        for composite in [2_047_u64, 3_215_031_751, 561, 3_825_123_056_546_413_051] {
            assert!(!is_prime(composite), "{composite} is composite");
        }
        for prime in [DEFAULT_MODULUS, (1 << 31) - 1, 18_446_744_073_709_551_557] {
            assert!(is_prime(prime), "{prime} is prime");
        }
    }

    #[test]
    fn transform_root_rejects_what_breaks_the_transform() {
        // 2 has order 8 modulo 17, so 2^(16/4) = 16 = -1 has order 2, not 4.
        assert_eq!(
            transform_root(4, 17, 2),
            Err(NttError::NotPrimitiveRoot {
                n: 4,
                modulus: 17,
                primitive_root: 2
            })
        );
        assert_eq!(
            transform_root(4, 9, 2),
            Err(NttError::CompositeModulus { modulus: 9 })
        );
        assert_eq!(
            transform_root(4, 17, 34),
            Err(NttError::NotPrimitiveRoot {
                n: 4,
                modulus: 17,
                primitive_root: 34
            })
        );
        assert_eq!(transform_root(32, 17, 3), Err(NttError::UnsupportedLength));
        assert_eq!(transform_root(4, 1, 3), Err(NttError::InvalidModulus));
    }

    #[test]
    fn transform_root_has_order_exactly_n() {
        // 3 generates the units modulo 17; its derived roots have every
        // power-of-two order dividing 16.
        for n in [1_usize, 2, 4, 8, 16] {
            let root = transform_root(n, 17, 3).expect("3 generates modulo 17");
            assert_eq!(mod_pow(root, n as u64, 17), 1, "n = {n}");
            assert!((1..n as u64).all(|k| mod_pow(root, k, 17) != 1), "n = {n}");
        }
        let root = transform_root(1 << 20, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT)
            .expect("the default field serves 2^20");
        assert_eq!(mod_pow(root, 1 << 19, DEFAULT_MODULUS), DEFAULT_MODULUS - 1);
    }
}
