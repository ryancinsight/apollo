use super::super::avx::{fixed_len32_precise_avx_fma, fixed_len64_precise_avx_fma};
use super::super::precision::PreciseStockhamAvxFma;
use super::super::transform::{transform, transform_len4096_four_triples};
use num_complex::Complex64;

pub(crate) unsafe fn forward64_avx_with_scratch(
    data: &mut [Complex64],
    scratch: &mut [Complex64],
    twiddles: &[Complex64],
) {
    // Direct match — no redundant `contains()` pre-check needed.
    // The `_` arm falls through without overhead; the compiler emits a jump
    // table for the two fixed-length cases.
    match data.len() {
        32 => {
            fixed_len32_precise_avx_fma(data, scratch, twiddles);
            return;
        }
        64 => {
            fixed_len64_precise_avx_fma(data, scratch, twiddles);
            return;
        }
        _ => {}
    }
    if data.len() == 4096 && twiddles.get(1).is_some_and(|w| w.im < 0.0) {
        transform_len4096_four_triples::<PreciseStockhamAvxFma>(data, scratch, twiddles);
        return;
    }
    transform::<PreciseStockhamAvxFma>(data, scratch, twiddles, None);
}

#[cfg(test)]
mod tests {
    #[test]
    fn fixed_len_f64_avx_sizes_are_powers_of_two() {
        // The two fixed-length f64 AVX sizes (32, 64) must be powers of two.
        for &n in &[32usize, 64] {
            assert!(
                n.is_power_of_two(),
                "fixed-length f64 AVX size {n} must be a power of two"
            );
        }
    }

    #[test]
    fn fixed_len_f64_avx_sizes_below_4096() {
        for &n in &[32usize, 64] {
            assert!(
                n < 4096,
                "fixed-length f64 AVX size {n} must be below 4096 (the four-triple threshold)"
            );
        }
    }
}
