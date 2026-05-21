//! Mixed-radix factorization and dispatch heuristics for the Cooley-Tukey pipeline.
//!
//! ## Mathematical foundation
//!
//! The Cooley-Tukey DIT algorithm requires N = ∏ rₖ where each rₖ is in the
//! supported radix set {2, 3, 4, 5, 7, 8}. Since 4 = 2² and 8 = 2³, the
//! underlying prime support is {2, 3, 5, 7}.
//!
//! A size N is **composite-smooth** if N = 2^a·3^b·5^c·7^d with not all
//! of b, c, d equal to zero. Pure powers of two are routed to the more
//! optimised Stockham kernels before composite factorization.
//!
//! ### Radix ordering (innermost first)
//!
//! Odd prime radices {7, 5, 3} are placed at the innermost stages, with
//! powers of two {8, 4, 2} at the outermost, to minimise the working-set
//! footprint during early stride-N/r passes (Van Loan 1992, §3.4).
//!
//! ## References
//!
//! - Cooley, J.W. & Tukey, J.W. (1965). *Math. Comp.* 19, 297-301.
//! - Van Loan, C. (1992). *Computational Frameworks for the FFT*. SIAM, §3.4.

#![allow(clippy::same_item_push)]

/// Factorize `n` into a radix sequence for the mixed-radix Cooley-Tukey DIT
/// algorithm, or return `None` if `n` has a prime factor outside {2, 3, 5, 7}.
///
/// ## Mathematical contract
///
/// **Theorem (factorization)**: Every positive integer has a unique prime
/// factorization. For n with all primes in {2, 3, 5, 7} this function returns
/// a factorization into Apollo's radix set {2, 3, 4, 5, 7, 8}.
///
/// **Proof sketch**: Divide out all factors of 2, 3, 5, 7. If `remaining > 1`
/// after exhaustion, n has an unsupported prime → return `None`. Pack count2
/// greedily into groups of 3 → radix-8, remainder pair → radix-4, singleton
/// → radix-2. Product of returned radices equals n by construction. □
///
/// ## Complexity
///
/// O(log n) time; O(log n) space for the returned `Vec`.
///
/// ## Failure modes
///
/// Returns `None` when n has an unsupported prime factor or is a pure power
/// of two (handled by the Stockham power-of-two path).
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
    while remaining.is_multiple_of(2) {
        count2 += 1;
        remaining /= 2;
    }
    while remaining.is_multiple_of(3) {
        count3 += 1;
        remaining /= 3;
    }
    while remaining.is_multiple_of(5) {
        count5 += 1;
        remaining /= 5;
    }
    while remaining.is_multiple_of(7) {
        count7 += 1;
        remaining /= 7;
    }
    if remaining > 1 {
        return None; // has unsupported prime factor
    }
    // Pure power-of-two sizes are handled before composite dispatch.
    if count3 == 0 && count5 == 0 && count7 == 0 {
        return None;
    }
    let mut radices = Vec::new();
    // Innermost stages: odd prime factors first (7 > 5 > 3 ordering for cache
    // locality and mixed-radix stage shape).
    for _ in 0..count7 {
        radices.push(7usize);
    }
    for _ in 0..count5 {
        radices.push(5usize);
    }
    for _ in 0..count3 {
        radices.push(3usize);
    }
    // Outermost stages: powers of 2 packed greedily into 8 > 4 > 2.
    let mut p2 = count2;
    while p2 >= 3 {
        radices.push(8);
        p2 -= 3;
    }
    if p2 >= 2 {
        radices.push(4);
        p2 -= 2;
    }
    if p2 == 1 {
        radices.push(2);
    }
    Some(radices)
}

/// Empirical Bluestein fallback heuristic for composite sizes with poor cache behaviour.
///
/// ## Heuristic criterion
///
/// For n ∈ [500, 2000], count leading (innermost) DFT-5 stages produced by
/// `factorize_composite`. If ≥ 3 consecutive DFT-5 stages are present, the
/// working set of the first stage is only `prev_len = 1` element — each
/// butterfly processes 5 scalars in isolation, producing poor cache utilization.
/// Bluestein pads to the nearest power of two and uses the vectorized
/// radix-8/Stockham kernel, which is substantially faster for these sizes.
///
/// ## Correctness boundary
///
/// - N=500 = 2²·5³: three DFT-5 innermost stages; triggers heuristic.
/// - N=1000 = 2³·5³: three DFT-5 innermost stages; triggers heuristic.
/// - N=2000 = 2⁴·5³: three DFT-5 innermost stages; triggers heuristic.
/// - N=4000 = 2⁵·5³: outside range [500,2000]; Rayon parallelism recovers perf.
///
/// ## Failure modes
///
/// A false positive routes a composite size through Bluestein (correct, slower
/// worst-case). A false negative uses composite (correct, potentially slower
/// than Bluestein for the affected sizes). Both failure modes produce correct output.
#[inline]
pub(crate) fn should_use_bluestein_instead_of_composite(n: usize) -> bool {
    if (500..=2000).contains(&n) {
        let Some(radices) = factorize_composite(n) else {
            return false; // Not composite; Bluestein will handle it
        };
        let count_leading_fives = radices.iter().take_while(|&&r| r == 5).count();
        if count_leading_fives >= 3 {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factorize_product_invariant_holds_for_smooth_sizes() {
        for &n in &[
            3usize, 5, 6, 7, 9, 10, 12, 14, 15, 18, 21, 24, 25, 28, 35, 42, 48, 49, 50, 56, 63, 70,
            75, 98, 100, 120, 125, 147, 150, 192, 200, 210, 240, 245, 250, 294, 300, 343, 375, 384,
            392, 450, 588, 600, 686, 700, 750, 784, 864, 900, 980, 1000, 1200, 1400, 1470, 1500,
            1960, 2000, 2400, 2500, 2940, 3000, 4000, 4500, 5000, 6000, 7000, 7500, 10000,
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
                    [2, 3, 4, 5, 7, 8].contains(&r),
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
        for &n in &[
            11usize, 13, 17, 19, 22, 23, 26, 29, 31, 33, 34, 38, 46, 58, 121, 143,
        ] {
            assert!(
                factorize_composite(n).is_none(),
                "factorize_composite({n}) must be None (has prime > 7)"
            );
        }
    }

    #[test]
    fn factorize_ordering_is_innermost_first() {
        // n=24 = 3×8: 3 should be innermost (index 0).
        let r = factorize_composite(24).unwrap();
        assert_eq!(r[0], 3, "3-factor must be innermost for n=24");
        // n=28 = 7×4: 7 should be innermost.
        let r = factorize_composite(28).unwrap();
        assert_eq!(r[0], 7, "7-factor must be innermost for n=28");
    }

    #[test]
    fn bluestein_heuristic_triggers_for_5cubed_sizes() {
        assert!(should_use_bluestein_instead_of_composite(500));
        assert!(should_use_bluestein_instead_of_composite(1000));
        assert!(should_use_bluestein_instead_of_composite(2000));
    }

    #[test]
    fn bluestein_heuristic_passes_outside_range_or_non_five() {
        assert!(!should_use_bluestein_instead_of_composite(100));
        assert!(!should_use_bluestein_instead_of_composite(3000));
        assert!(!should_use_bluestein_instead_of_composite(512));
    }
}
