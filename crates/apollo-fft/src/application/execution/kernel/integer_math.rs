//! Integer arithmetic shared by runtime FFT planning kernels.

/// Compute `base^exp mod modulus` by repeated squaring.
#[inline]
pub(in crate::application::execution::kernel) fn mod_pow(
    mut base: usize,
    mut exp: usize,
    modulus: usize,
) -> usize {
    let mut result = 1;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        base = (base * base) % modulus;
        exp /= 2;
    }
    result
}

/// Return the prime factors of `n`, with multiplicity, in ascending order.
#[inline]
pub(in crate::application::execution::kernel) fn prime_factors_all(mut n: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    while n % 2 == 0 {
        factors.push(2);
        n /= 2;
    }
    let mut divisor = 3;
    while divisor * divisor <= n {
        while n % divisor == 0 {
            factors.push(divisor);
            n /= divisor;
        }
        divisor += 2;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}
