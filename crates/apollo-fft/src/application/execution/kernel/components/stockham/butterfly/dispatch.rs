use super::super::avx::{fixed_len32_precise_avx_fma, fixed_len64_precise_avx_fma};
use super::super::precision::PreciseStockhamAvxFma;
use super::super::transform::{transform_len4096_four_triples, transform_sized};
use num_complex::Complex64;

pub(crate) unsafe fn forward64_avx_with_scratch(
    data: &mut [Complex64],
    scratch: &mut [Complex64],
    twiddles: &[Complex64],
) {
    // Direct match — no redundant `contains()` pre-check needed.
    // The `_` arm falls through without overhead; the compiler emits a jump
    // table for the fixed-length cases. 128/256 use the generic transform under
    // the same AVX FMA backend (future: add fixed_len128/256 column kernels).
    match data.len() {
        32 => {
            fixed_len32_precise_avx_fma(data, scratch, twiddles);
            return;
        }
        64 => {
            fixed_len64_precise_avx_fma(data, scratch, twiddles);
            return;
        }
        128 | 256 | 512 => {
            // Route explicitly for documentation and to enable future
            // size-specific straight-line schedules without loop overhead.
            // Currently falls through to transform with AvxFma backend.
        }
        _ => {}
    }
    if data.len() == 4096 && twiddles.get(1).is_some_and(|w| w.im < 0.0) {
        transform_len4096_four_triples::<PreciseStockhamAvxFma>(data, scratch, twiddles);
        return;
    }
    let log2 = data.len().trailing_zeros();
    transform_sized::<PreciseStockhamAvxFma>(data, scratch, twiddles, None, log2);
}

/// Sized variant for const LOG2 flow from forward_with_scratch_sized (plan PoT ZST path).
/// Avoids runtime trailing_zeros() and len() in hot AVX PoT sized path; passes const LOG2
/// to transform_sized for better monomorph/DCE to len* bodies.
#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline)]
pub(crate) unsafe fn forward64_avx_with_scratch_sized<const LOG2: u32>(
    data: &mut [Complex64],
    scratch: &mut [Complex64],
    twiddles: &[Complex64],
) {
    let n = 1usize << LOG2;
    debug_assert_eq!(data.len(), n);
    match n {
        32 => {
            fixed_len32_precise_avx_fma(data, scratch, twiddles);
            return;
        }
        64 => {
            fixed_len64_precise_avx_fma(data, scratch, twiddles);
            return;
        }
        128 | 256 | 512 => {
            // Route explicitly; falls through to sized transform.
        }
        _ => {}
    }
    if n == 4096 && twiddles.get(1).is_some_and(|w| w.im < 0.0) {
        transform_len4096_four_triples::<PreciseStockhamAvxFma>(data, scratch, twiddles);
        return;
    }
    transform_sized::<PreciseStockhamAvxFma>(data, scratch, twiddles, None, LOG2);
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
