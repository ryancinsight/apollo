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
    if n % 2 == 0 {
        factors.push(2);
        while n % 2 == 0 {
            n /= 2;
        }
    }
    let mut i = 3;
    while i * i <= n {
        if n % i == 0 {
            factors.push(i);
            while n % i == 0 {
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
