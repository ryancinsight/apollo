/// Factorize `n` into a sequence of small radices suitable for the
/// mixed-radix Cooley-Tukey DIT algorithm, or return `None` if `n` has
/// a prime factor outside the supported set {2, 3, 5}.
///
/// ## Radix selection
///
/// - Powers of 2 are packed greedily into radix-8 (3 factors), then radix-4
///   (2 factors), then radix-2 (1 factor).
/// - Primes 3 and 5 appear individually in the sequence.
///
/// The sequence is ordered **innermost first** (smallest sub-transform first):
/// prime radices precede power-of-two radices so that the smallest butterfly
/// groups are processed in the initial stages, improving cache locality.
///
/// ## Contract
///
/// For the returned `radices`, `radices.iter().product::<usize>() == n` and
/// every element ∈ {2, 3, 4, 5, 8}.
///
/// Returns `None` (caller should fall back to Bluestein) when `n` has any
/// prime factor other than 2, 3, or 5.
#[inline]
pub(crate) fn factorize_composite(n: usize) -> Option<Vec<usize>> {
    if n <= 1 {
        return Some(Vec::new());
    }
    let mut remaining = n;
    let mut count2 = 0u32;
    let mut count3 = 0u32;
    let mut count5 = 0u32;
    while remaining % 2 == 0 { count2 += 1; remaining /= 2; }
    while remaining % 3 == 0 { count3 += 1; remaining /= 3; }
    while remaining % 5 == 0 { count5 += 1; remaining /= 5; }
    if remaining > 1 {
        return None; // has prime factor > 5
    }
    // Pure power-of-two sizes are handled by the existing pow2_dispatch! routing;
    // return None so those sizes continue using the optimised PoT kernels.
    if count3 == 0 && count5 == 0 {
        return None;
    }
    let mut radices = Vec::new();
    // Innermost stages: odd prime factors first (5 > 3 ordering for outer loop
    // peeling matches cache footprint: largest prime groups process last).
    for _ in 0..count5 { radices.push(5usize); }
    for _ in 0..count3 { radices.push(3usize); }
    // Outermost stages: powers of 2 packed greedily into 8 > 4 > 2.
    let mut p2 = count2;
    while p2 >= 3 { radices.push(8); p2 -= 3; }
    if p2 >= 2 { radices.push(4); p2 -= 2; }
    if p2 == 1 { radices.push(2); }
    Some(radices)
}

/// Heuristic: detect composite sizes that would perform worse with mixed-radix
/// compared to Bluestein due to poor cache behavior with many small radices.
///
/// Returns `true` if the composite path should be skipped (fall back to Bluestein).
///
/// Known bad patterns:
/// - Multiple (3+) small radices (3 or 5) before large power-of-2: causes
///   poor cache behavior due to small prev_len in early stages.
/// - Sizes 500, 1000, 2000 range have 3+ DFT-5 stages with low parallelism.
#[inline]
pub(crate) fn should_use_bluestein_instead_of_composite(n: usize) -> bool {
    // Empirically observed bad cases:
    // N=500 (2²×5³): 3 DFT-5 stages too deep for cache
    // N=1000 (2³×5³): 3 DFT-5 stages, composite is 0.6× Bluestein
    // N=2000 (2⁴×5³): 3 DFT-5 stages, break-even, worth avoiding
    if n >= 500 && n <= 2000 {
        let radices = match factorize_composite(n) {
            Some(r) => r,
            None => return false, // Not composite; Bluestein will handle it
        };
        // Count consecutive 5-radix factors at the start (innermost stages).
        let count_leading_fives = radices.iter().take_while(|&&r| r == 5).count();
        if count_leading_fives >= 3 {
            return true; // Too many DFT-5 stages, fall back to Bluestein
        }
    }
    false
}

#[inline]
pub(crate) fn is_power_of_pow2_radix(n: usize, radix_log2: u32) -> bool {
    n.is_power_of_two() && (n.trailing_zeros() % radix_log2 == 0)
}

#[inline]
pub(crate) fn is_power_of_four(n: usize) -> bool {
    is_power_of_pow2_radix(n, 2)
}

#[inline]
pub(crate) fn is_power_of_eight(n: usize) -> bool {
    is_power_of_pow2_radix(n, 3)
}

#[inline]
pub(crate) fn stage_twiddle_slice<T>(twiddles: Option<&[T]>, half: usize) -> Option<&[T]> {
    twiddles.map(|t| &t[(half - 1)..(half - 1 + half)])
}

/// Look up the radix-4 stage twiddle at position `exponent` within a stage slice
/// of nominal length `half`.
///
/// ## Mathematical contract
///
/// In a radix-4 stage with group length `len = 4·half`, the DIT butterfly uses
/// twiddles `W^{j}`, `W^{2j}`, `W^{3j}` for j = 0..half. The exponent may reach
/// `3·half - 1`. For exponents beyond `half - 1` the symmetry
/// `W^{half} = W^{half mod half}·(-1) = -W^{exponent mod half}` holds because
/// `W^{half} = exp(2πi·half/len) = exp(πi) = -1`. Hence:
///
/// ```text
/// stage_twiddle(stage, half, exponent) =
///     if exponent < half:  stage[exponent]
///     else:               -stage[exponent - half]
/// ```
///
/// This avoids storing more than `half` entries per stage while supporting all
/// three twiddle exponents.
///
/// ## Generic bound
///
/// `T` requires `Copy` (to read without consuming) and `Neg<Output = T>`
/// (to negate without allocation). Both `Complex64` and `Complex32` satisfy this.
#[inline]
pub(crate) fn stage_twiddle<T>(stage: &[T], half: usize, exponent: usize) -> T
where
    T: Copy + std::ops::Neg<Output = T>,
{
    if exponent < half {
        stage[exponent]
    } else {
        -stage[exponent - half]
    }
}
