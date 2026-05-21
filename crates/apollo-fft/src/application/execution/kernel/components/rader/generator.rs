//! Prime root generators for Rader's algorithm.
//!
//! [`PRIMITIVE_ROOTS`] mirrors `apollo-fft-macros/src/math.rs`;
//! both definitions are sourced from the same shared file via `include!`.

// ── Shared constant and math sources ────────────────────────────────────────
// These include! macros import canonical constants and helpers from
// single files shared between `apollo-fft-macros` and `apollo-fft`.
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../apollo-fft-macros/src/shared_primitives.rs"
));
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../apollo-fft-macros/src/shared_math.rs"
));

pub(crate) fn primitive_root(p: usize) -> usize {
    // Static lookup — compiler generates a jump table for O(1) dispatch.
    // This mirrors the PRIMITIVE_ROOTS SSOT; the const-consistency test
    // in mod.rs ensures the two stay in sync.
    match p {
        2 => 1,
        3 => 2,
        5 => 2,
        7 => 3,
        11 => 2,
        13 => 2,
        17 => 3,
        19 => 2,
        23 => 5,
        29 => 2,
        31 => 3,
        37 => 2,
        41 => 6,
        43 => 3,
        47 => 5,
        53 => 2,
        59 => 2,
        61 => 2,
        67 => 2,
        71 => 7,
        73 => 5,
        79 => 3,
        83 => 2,
        89 => 3,
        97 => 5,
        101 => 2,
        109 => 6,
        113 => 3,
        127 => 3,
        131 => 2,
        151 => 6,
        167 => 5,
        173 => 2,
        179 => 2,
        181 => 2,
        193 => 5,
        197 => 2,
        199 => 3,
        10007 => 5,
        _ => primitive_root_dynamic(p),
    }
}

fn primitive_root_dynamic(p: usize) -> usize {
    if p == 2 {
        return 1;
    }
    let phi = p - 1;
    let mut factors = prime_factors_all(phi);
    // `prime_factors_all` is documented to return factors in ascending order
    // (trial division from 2 upward), so `dedup()` correctly removes all
    // repeated consecutive entries, leaving unique prime factors of phi.
    debug_assert!(
        factors.windows(2).all(|w| w[0] <= w[1]),
        "prime_factors_all contract violated: output not sorted ascending"
    );
    factors.dedup(); // Only unique prime factors of P-1

    for g in 2..p {
        let mut is_primitive = true;
        for &q in &factors {
            if mod_pow(g, phi / q, p) == 1 {
                is_primitive = false;
                break;
            }
        }
        if is_primitive {
            return g;
        }
    }
    unreachable!("Prime must have a primitive root")
}

pub(crate) const fn inverse_mod(a: usize, m: usize) -> usize {
    let mut m0 = m as i64;
    let mut y = 0i64;
    let mut x = 1i64;
    let mut a_i64 = a as i64;

    if m == 1 {
        return 0;
    }

    while a_i64 > 1 {
        let q = a_i64 / m0;
        let mut t = m0;
        m0 = a_i64 % m0;
        a_i64 = t;
        t = y;
        y = x - q * y;
        x = t;
    }

    if x < 0 {
        x += m as i64;
    }
    x as usize
}
