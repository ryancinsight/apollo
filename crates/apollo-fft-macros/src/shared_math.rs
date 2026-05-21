// ── SSOT shared math utilities ──────────────────────────────────────────────
// This file is the single source of truth for small integer-arithmetic
// helpers used by `apollo-fft` (runtime) via `include!`.
//
// NOTE: `apollo-fft-macros` keeps its own `math.rs` with proc-macro-specific
// versions (e.g. `mod_pow` with `u64` params) and does NOT include this file.
//
// All functions are private (`fn`, not `pub(crate) fn`) so the compiler can
// inline them aggressively within the including module.  Each module that
// `include!`s this file gets its own monomorphized copy.

/// Greatest common divisor (Euclidean algorithm).
#[inline]
#[allow(dead_code)]
const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Modular exponentiation: `base^exp mod modulus`.
#[inline]
#[allow(dead_code)]
fn mod_pow(mut base: usize, mut exp: usize, modulus: usize) -> usize {
    let mut res = 1;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            res = (res * base) % modulus;
        }
        base = (base * base) % modulus;
        exp /= 2;
    }
    res
}

/// Modular multiplicative inverse: `a * x ≡ 1 (mod m)`.
/// Uses extended Euclidean algorithm for O(log m) performance.
/// Panics if no inverse exists (i.e. `gcd(a, m) != 1`).
#[inline]
#[allow(dead_code)]
fn modular_inverse(a: usize, m: usize) -> usize {
    let (mut old_r, mut r) = (a as i64, m as i64);
    let (mut old_s, mut s) = (1i64, 0i64);

    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient * s);
    }

    // old_r is gcd(a, m); old_s is the Bézout coefficient
    debug_assert!(old_r == 1, "gcd({a}, {m}) = {old_r}, inverse does not exist");

    let mut result = old_s % m as i64;
    if result < 0 {
        result += m as i64;
    }
    result as usize
}

/// Prime factorization (with multiplicity).
/// Returns all prime factors in ascending order, e.g. `prime_factors_all(12) = [2, 2, 3]`.
#[inline]
#[allow(dead_code)]
fn prime_factors_all(mut n: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    while n % 2 == 0 {
        factors.push(2);
        n /= 2;
    }
    let mut d = 3;
    while d * d <= n {
        while n % d == 0 {
            factors.push(d);
            n /= d;
        }
        d += 2;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}
