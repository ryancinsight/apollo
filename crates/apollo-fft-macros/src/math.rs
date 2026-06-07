// Canonical prime→smallest-primitive-root mapping. Sourced from
// `shared_primitives.rs` via `include!` — SSOT shared with `apollo-fft`.
include!("shared_primitives.rs");

pub fn find_primitive_root(p: usize) -> usize {
    // Fast lookup against the canonical table (shared with runtime `generator.rs`).
    for &(prime, generator) in PRIMITIVE_ROOTS {
        if prime == p {
            return generator;
        }
    }
    // Dynamic fallback for primes not in the table.
    if p == 2 {
        return 1;
    }
    let p_minus_1 = p - 1;
    let factors = prime_factors(p_minus_1);

    for g in 2..p {
        let mut is_primitive = true;
        for &factor in &factors {
            if mod_pow(g, (p_minus_1 / factor) as u64, p as u64) == 1 {
                is_primitive = false;
                break;
            }
        }
        if is_primitive {
            return g;
        }
    }
    panic!("No primitive root found for {}", p);
}

pub fn prime_factors(mut n: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    if n.is_multiple_of(2) {
        factors.push(2);
        while n.is_multiple_of(2) {
            n /= 2;
        }
    }
    let mut i = 3;
    while i * i <= n {
        if n.is_multiple_of(i) {
            factors.push(i);
            while n.is_multiple_of(i) {
                n /= i;
            }
        }
        i += 2;
    }
    if n > 2 {
        factors.push(n);
    }
    factors
}

pub fn mod_pow(base: usize, mut exp: u64, modulus: u64) -> usize {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1u64;
    let mut base_u64 = (base as u64) % modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base_u64) % modulus;
        }
        exp >>= 1;
        base_u64 = (base_u64 * base_u64) % modulus;
    }
    result as usize
}

#[allow(clippy::many_single_char_names)] // standard Bezout (r, s, t) notation
pub fn extended_gcd(a: isize, b: isize) -> (isize, isize, isize) {
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (1, 0);
    let (mut old_t, mut t) = (0, 1);

    while r != 0 {
        let quotient = old_r / r;

        let temp_r = r;
        r = old_r - quotient * r;
        old_r = temp_r;

        let temp_s = s;
        s = old_s - quotient * s;
        old_s = temp_s;

        let temp_t = t;
        t = old_t - quotient * t;
        old_t = temp_t;
    }
    (old_r, old_s, old_t)
}

/// Compute the modular inverse of `a` modulo `m` via extended GCD.
///
/// This is the single authoritative O(log m) modular-inverse implementation
/// for the macros crate. Returns a value in `[0, m)`. Panics if `gcd(a, m) ≠ 1`.
pub fn mod_inverse_isize(a: isize, m: isize) -> isize {
    let (gcd, x, _) = extended_gcd(a, m);
    assert!(gcd == 1, "Values are not coprime: {} and {}", a, m);
    let res = x % m;
    if res < 0 {
        res + m
    } else {
        res
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ComplexF64 {
    pub re: f64,
    pub im: f64,
}

impl ComplexF64 {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }
}

impl std::ops::Add for ComplexF64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl std::ops::Sub for ComplexF64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl std::ops::Mul for ComplexF64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl std::ops::Div for ComplexF64 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        assert!(denom.abs() > 1e-9, "Division by zero in ComplexF64");
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )
    }
}

pub fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcd_matches_textbook_cases() {
        assert_eq!(gcd(0, 0), 0);
        assert_eq!(gcd(12, 0), 12);
        assert_eq!(gcd(0, 12), 12);
        assert_eq!(gcd(12, 18), 6);
        assert_eq!(gcd(18, 12), 6);
        assert_eq!(gcd(17, 13), 1); // distinct primes
        assert_eq!(gcd(100, 75), 25);
    }

    #[test]
    fn mod_pow_handles_identities_and_modulus_one() {
        assert_eq!(mod_pow(7, 0, 13), 1); // a^0 = 1
        assert_eq!(mod_pow(0, 5, 13), 0); // 0^k = 0 for k > 0
        assert_eq!(mod_pow(5, 1, 13), 5); // a^1 = a mod m
        assert_eq!(mod_pow(123, 456, 1), 0); // modulus 1 → always 0
    }

    #[test]
    fn mod_pow_matches_fermat_little_theorem() {
        // a^(p-1) ≡ 1 (mod p) for a coprime to prime p.
        for &p in &[3usize, 5, 7, 11, 13, 17, 19, 23] {
            for a in 1..p {
                assert_eq!(
                    mod_pow(a, (p - 1) as u64, p as u64),
                    1,
                    "Fermat fails for a={a}, p={p}"
                );
            }
        }
    }

    #[test]
    fn prime_factors_returns_distinct_primes_only() {
        assert_eq!(prime_factors(1), Vec::<usize>::new());
        assert_eq!(prime_factors(2), vec![2]);
        assert_eq!(prime_factors(12), vec![2, 3]); // 2² · 3 → distinct {2, 3}
        assert_eq!(prime_factors(60), vec![2, 3, 5]);
        assert_eq!(prime_factors(97), vec![97]); // prime
        assert_eq!(prime_factors(1000), vec![2, 5]); // 2³ · 5³
    }

    #[test]
    fn extended_gcd_satisfies_bezout_identity() {
        // gcd(a, b) = a*s + b*t
        for &(a, b) in &[(12isize, 18), (17, 13), (100, 75), (1, 1), (7, 0)] {
            let (g, s, t) = extended_gcd(a, b);
            assert_eq!(
                a * s + b * t,
                g,
                "Bezout fails for ({a}, {b}): got g={g}, s={s}, t={t}"
            );
        }
    }

    #[test]
    fn mod_inverse_round_trips_against_mod_pow() {
        // For prime p and a in [1, p), a * a^(-1) ≡ 1 (mod p).
        for &p in &[7isize, 11, 13, 17, 23] {
            for a in 1..p {
                let inv = mod_inverse_isize(a, p);
                assert!((0..p).contains(&inv), "inverse out of [0, {p}): got {inv}");
                assert_eq!(
                    (a * inv).rem_euclid(p),
                    1,
                    "a * a^(-1) ≢ 1 (mod {p}) for a={a}, inv={inv}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "Values are not coprime")]
    fn mod_inverse_panics_when_non_coprime() {
        let _ = mod_inverse_isize(6, 9); // gcd(6, 9) = 3
    }

    #[test]
    fn find_primitive_root_yields_order_p_minus_1() {
        // g is a primitive root iff g^((p-1)/q) ≢ 1 (mod p) for every prime
        // divisor q of p-1.
        for &p in &[3usize, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] {
            let g = find_primitive_root(p);
            assert!(g >= 1 && g < p, "root {g} out of [1, {p})");
            for &q in &prime_factors(p - 1) {
                assert_ne!(
                    mod_pow(g, ((p - 1) / q) as u64, p as u64),
                    1,
                    "root {g} has order < p-1 for p={p}, witness q={q}"
                );
            }
            assert_eq!(
                mod_pow(g, (p - 1) as u64, p as u64),
                1,
                "Fermat sanity: g^(p-1) ≡ 1 (mod {p}) fails for g={g}"
            );
        }
    }

    #[test]
    fn complex_arithmetic_satisfies_field_laws() {
        let a = ComplexF64::new(1.0, 2.0);
        let b = ComplexF64::new(3.0, -1.0);
        // Commutative addition.
        let ab = a + b;
        let ba = b + a;
        assert!((ab.re - ba.re).abs() < 1e-15);
        assert!((ab.im - ba.im).abs() < 1e-15);
        // (a * b) / b = a for nonzero b.
        let q = a * b / b;
        assert!((q.re - a.re).abs() < 1e-12, "real mismatch: {q:?} vs {a:?}");
        assert!((q.im - a.im).abs() < 1e-12, "imag mismatch: {q:?} vs {a:?}");
        // Additive identity.
        let z = ComplexF64::zero();
        let a_plus_z = a + z;
        assert_eq!(a_plus_z.re, a.re);
        assert_eq!(a_plus_z.im, a.im);
    }
}
