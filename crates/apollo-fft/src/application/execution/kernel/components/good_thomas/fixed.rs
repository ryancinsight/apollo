/// Union of [`super::two_by_prime::DIRECT_PAIR_PRIMES`] and
/// [`super::three_by_prime::THREE_BY_PRIME_PRIMES`].  Canonical 2×prime
/// and 3×prime pairs containing these primes are handled first in
/// `pfa_fft` dispatch by `two_by_prime` or `three_by_prime` before
/// reaching the fixed-size generated codelets.
#[cfg(test)]
pub(super) const FIXED_EXCLUDE_PRIMES: [usize;
    super::two_by_prime::DIRECT_PAIR_PRIMES.len()
        + super::three_by_prime::THREE_BY_PRIME_PRIMES.len()] = {
    let a: &[usize] = super::two_by_prime::DIRECT_PAIR_PRIMES;
    let b: &[usize] = super::three_by_prime::THREE_BY_PRIME_PRIMES;
    let mut merged = [0usize;
        super::two_by_prime::DIRECT_PAIR_PRIMES.len()
            + super::three_by_prime::THREE_BY_PRIME_PRIMES.len()];
    let mut i = 0;
    let mut j = 0;
    while j < a.len() {
        merged[i] = a[j];
        i += 1;
        j += 1;
    }
    j = 0;
    while j < b.len() {
        merged[i] = b[j];
        i += 1;
        j += 1;
    }
    merged
};

apollo_fft_macros::generate_good_thomas_dispatch! {
    short_sizes: [
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 27, 29,
        31, 32, 36, 37, 41, 43, 47, 53,
    ],
    max_n: 200,
    // Must match FIXED_EXCLUDE_PRIMES above (the macro requires literal
    // values; the const is the canonical source of truth).
    direct_pair_primes: [
        // 2×prime — two_by_prime::DIRECT_PAIR_PRIMES
        11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
        // 3×prime — three_by_prime::THREE_BY_PRIME_PRIMES (all 14 primes)
        5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    ],
}
