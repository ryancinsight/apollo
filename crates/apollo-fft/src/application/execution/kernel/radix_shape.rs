//! Mixed-radix factorization and dispatch heuristics for the Cooley-Tukey pipeline.
//!
//! ## Mathematical foundation
//!
//! The Cooley-Tukey DIT algorithm requires N = ∏ rₖ where each rₖ is in the
//! supported prime radix set {2, 3, 5, 7, 11, 13, 17, 23}. Pure powers of two
//! are routed to the more-optimised Stockham power-of-two kernels before composite
//! factorization is attempted.
//!
//! ### Radix ordering (innermost first)
//!
//! Large odd prime radices are placed at the innermost stages, with
//! powers of two at the outermost. This minimises the working-set footprint
//! during early stride-N/r passes (Van Loan 1992, §3.4).
//!
//! ## References
//!
//! - Cooley, J.W. & Tukey, J.W. (1965). *Math. Comp.* 19, 297-301.
//! - Van Loan, C. (1992). *Computational Frameworks for the FFT*. SIAM, §3.4.

#![allow(clippy::same_item_push)]

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/application/numeric/integer_math.rs"
));

/// Factorize `n` into a prime radix sequence for the mixed-radix Cooley-Tukey
/// DIT algorithm, or return `None` if `n` has a prime factor outside
/// {2, 3, 5, 7, 11, 13, 17, 23}.
///
/// ## Mathematical contract
///
/// **Theorem (factorization)**: Every positive integer has a unique prime
/// factorization. For n with all primes in {2, 3, 5, 7, 11, 13, 17, 23} this
/// function returns a sequence whose product equals n and each element is in
/// that prime set.
///
/// **Proof sketch**: Divide out each supported prime exhaustively. If `remaining
/// > 1` after all divisions, n has an unsupported prime → return `None`. □
///
/// ## Complexity
///
/// O(log n) time; O(log n) space for the returned `Vec`.
///
/// ## Failure modes
///
/// Returns `None` when n has an unsupported prime factor or is a pure power
/// of two (handled by the Stockham power-of-two path).
///
/// ## Lowering
///
/// The returned sequence is execution-ready: consecutive `[2, 2]` pairs are
/// lowered to `[4]` before the value leaves this module, enabling the
/// zero-multiplication radix-4 butterfly without an allocation or normalization
/// pass in the composite execution core.
#[inline]
pub(crate) fn factorize_composite(n: usize) -> Option<Vec<usize>> {
    if n <= 1 {
        return Some(Vec::new());
    }
    let mut remaining = n;
    let mut count2 = 0u32;
    let mut count3 = 0u32;
    let mut count5 = 0u32;
    let mut count7 = 0u32;
    let mut count11 = 0u32;
    let mut count13 = 0u32;
    let mut count17 = 0u32;
    let mut count23 = 0u32;
    while remaining % 2 == 0 {
        count2 += 1;
        remaining /= 2;
    }
    while remaining % 3 == 0 {
        count3 += 1;
        remaining /= 3;
    }
    while remaining % 5 == 0 {
        count5 += 1;
        remaining /= 5;
    }
    while remaining % 7 == 0 {
        count7 += 1;
        remaining /= 7;
    }
    while remaining % 11 == 0 {
        count11 += 1;
        remaining /= 11;
    }
    while remaining % 13 == 0 {
        count13 += 1;
        remaining /= 13;
    }
    while remaining % 17 == 0 {
        count17 += 1;
        remaining /= 17;
    }
    while remaining % 23 == 0 {
        count23 += 1;
        remaining /= 23;
    }
    if remaining > 1 {
        return None;
    }
    // Pure power-of-two sizes are handled before composite dispatch.
    if count3 == 0
        && count5 == 0
        && count7 == 0
        && count11 == 0
        && count13 == 0
        && count17 == 0
        && count23 == 0
    {
        return None;
    }
    let mut radices = Vec::new();
    // The radix-2 chain leads, then the odd primes ascending.
    //
    // This order used to be the reverse — largest odd prime first, "for
    // cache-optimal stage shape", a claim with no measurement behind it — and
    // the reverse is what it costs. Both orders run through
    // `composite_forward_with_radices` over identical data in one binary, two
    // runs, `f32`, performance core: leading with the powers of two wins at
    // every length from 60 up, by 13 to 45% — n = 100 +28.5/+34.3%, n = 180
    // +41.6/+45.5%, n = 720 +33.3/+44.3% for the largest-first order. Only
    // n = 12 favours the old shape, by about 10%, where the whole transform is
    // register-resident anyway. Below 60 the two are inside the noise band.
    //
    // Adjacent pairs lower to radix 4 below, so the emitted chain is 4s with a
    // trailing 2 for an odd exponent, which is what the hand-written static
    // table in `dispatch` already spelled out for the lengths it carries;
    // `derived_order_matches_the_static_table` holds the two to each other.
    for _ in 0..count2 {
        radices.push(2usize);
    }
    for _ in 0..count3 {
        radices.push(3);
    }
    for _ in 0..count5 {
        radices.push(5);
    }
    for _ in 0..count7 {
        radices.push(7);
    }
    for _ in 0..count11 {
        radices.push(11);
    }
    for _ in 0..count13 {
        radices.push(13);
    }
    for _ in 0..count17 {
        radices.push(17);
    }
    for _ in 0..count23 {
        radices.push(23);
    }
    Some(lower_radix2_pairs_to_radix4(&radices))
}

/// Test if a number is prime23-smooth (only prime factors 2, 3, 5, 7, 11, 13, 17, 23).
#[inline]
pub(crate) fn is_prime23_smooth(mut n: usize) -> bool {
    if n == 0 {
        return false;
    }
    for &p in &[2, 3, 5, 7, 11, 13, 17, 23] {
        while n % p == 0 {
            n /= p;
        }
    }
    n == 1
}

/// Test if a number is prime.
#[inline]
pub(crate) fn is_prime(n: usize) -> bool {
    if n <= 1 {
        return false;
    }
    if n <= 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    let mut i = 5;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

/// Find a pair of coprime factors N1, N2 such that N1 * N2 = n.
/// Returns None if n is a prime power (no coprime factorization exists).
#[inline]
pub(crate) fn coprime_factors(n: usize) -> Option<(usize, usize)> {
    let factors = prime_factors_all(n);
    if factors.is_empty() {
        return None;
    }

    let mut prime_powers = Vec::new();
    let mut current_prime = factors[0];
    let mut current_power = current_prime;

    for &f in &factors[1..] {
        if f == current_prime {
            current_power *= f;
        } else {
            prime_powers.push(current_power);
            current_prime = f;
            current_power = f;
        }
    }
    prime_powers.push(current_power);

    if prime_powers.len() < 2 {
        return None;
    }

    let n1 = prime_powers.pop().unwrap();
    let n2 = prime_powers.iter().product();
    Some((n1, n2))
}

/// Lower consecutive `[2, 2]` pairs in a prime radix sequence to `[4]`.
///
/// ## Invariant
///
/// `result.iter().product::<usize>() == radices.iter().product::<usize>()`
///
/// ## Rationale
///
/// `dft4` requires 0 multiplications (all ±1, ±i rotations), strictly fewer
/// than two separate `dft2` calls with inter-stage twiddle factors.  A single
/// radix-4 stage also halves the ping-pong pass count for large transforms
/// where the FUSE_THRESHOLD prevents Compose-based fusion.
///
/// Scanning is left-to-right and greedy: `[2, 2, 2]` → `[4, 2]`.
#[inline]
pub(crate) fn lower_radix2_pairs_to_radix4(radices: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(radices.len());
    let mut i = 0;
    while i < radices.len() {
        if i + 1 < radices.len() && radices[i] == 2 && radices[i + 1] == 2 {
            out.push(4);
            i += 2;
        } else {
            out.push(radices[i]);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factorize_product_invariant_holds_for_smooth_sizes() {
        for &n in &[
            3usize, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 21, 22, 23, 24, 25, 28, 33, 34, 35,
            42, 46, 48, 49, 50, 56, 63, 66, 69, 70, 75, 98, 100, 120, 121, 125, 143, 147, 150, 154,
            176, 192, 200, 210, 231, 240, 242, 245, 250, 253, 264, 275, 286, 294, 300, 343, 375,
            384, 392, 450, 500, 506, 528, 575, 588, 600, 616, 625, 650, 675, 686, 700, 726, 750,
            784, 864, 900, 980, 1000, 1100, 1200, 1400, 1470, 1500, 1960, 2000, 2200, 2400, 2500,
            2600, 2700, 2800, 2940, 3000, 4000, 4500, 5000, 6000, 7000, 7500, 10000,
        ] {
            let radices = factorize_composite(n)
                .unwrap_or_else(|| panic!("factorize_composite({n}) returned None"));
            assert_eq!(
                radices.iter().product::<usize>(),
                n,
                "product invariant failed for n={n}"
            );
            for &r in &radices {
                assert!(
                    [2, 3, 4, 5, 7, 11, 13, 17, 23].contains(&r),
                    "unsupported radix {r} for n={n}"
                );
            }
        }
    }

    #[test]
    fn factorize_pure_pot_returns_none() {
        for exp in 1..=20u32 {
            let n = 1usize << exp;
            assert!(
                factorize_composite(n).is_none(),
                "factorize_composite({n}) must be None for pure-PoT"
            );
        }
    }

    #[test]
    fn factorize_unsupported_prime_returns_none() {
        for &n in &[19usize, 29, 31, 37, 38, 41, 43, 58, 361, 551] {
            assert!(
                factorize_composite(n).is_none(),
                "factorize_composite({n}) must be None (has prime > 23 or prime=19)"
            );
        }
    }

    #[test]
    fn factorize_ordering_leads_with_the_power_of_two_chain() {
        // This test previously asserted the opposite — largest odd prime at
        // index 0 — pinning the "innermost first" convention the emission
        // comment claimed. Measurement reversed it: the same kernel over the
        // same data with only the order changed runs 13 to 45% slower in the
        // odd-prime-first shape at every length from 60 up
        // (`dispatch::composite_decomposition`). The assertions are as strict
        // as before, against the order that measured faster.
        //
        // n = 24 = 2³×3: the radix-2 chain leads, odd primes follow.
        let r = factorize_composite(24).unwrap();
        assert_eq!(r, &[4, 2, 3], "n=24 must lead with the power-of-two chain");
        // n = 28 = 2²×7.
        let r = factorize_composite(28).unwrap();
        assert_eq!(r, &[4, 7], "n=28 must lead with the power-of-two chain");
        // n = 46 = 2×23: a lone 2 does not pair, so it stays a 2.
        let r = factorize_composite(46).unwrap();
        assert_eq!(r, &[2, 23], "n=46 must lead with the power-of-two chain");
    }

    #[test]
    fn factorize_emits_only_execution_radices() {
        for &n in &[12usize, 24, 48, 96, 192, 240, 480, 960, 1920] {
            let radices = factorize_composite(n).unwrap();
            for &r in &radices {
                assert!(
                    [2usize, 3, 4, 5, 7, 11, 13, 17, 23].contains(&r),
                    "n={n}: unsupported execution radix {r} emitted"
                );
                assert_ne!(r, 8, "n={n}: composite radix 8 must not be emitted");
                assert_ne!(r, 8, "composite radix 8 must not be emitted");
            }
        }
    }

    #[test]
    fn radix2_pair_lowering_preserves_product_and_uses_radix4_tail() {
        let radices = factorize_composite(192).unwrap();
        assert_eq!(radices, &[4, 4, 4, 3]);
        assert_eq!(radices.iter().product::<usize>(), 192);
    }
}
