use super::rader::build_rader_spectrum_vec;
use super::simd::pointwise_mul_c64;
use super::trait_def::MixedRadixScalar;
use super::transpose::transpose_matrix_c64;
use crate::application::execution::kernel::mixed_radix::caches::{
    cached_four_step_twiddles_64, cached_rader_spectrum_64, cached_twiddle_fwd_64,
    cached_twiddle_inv_64, with_stockham_scratch_64,
};
use crate::application::execution::kernel::mixed_radix::traits::{
    forward_short_winograd, inverse_short_winograd,
};
use crate::application::execution::kernel::radix_stage::NormalizeSlice;
use crate::application::execution::kernel::{radix_composite, stockham};
use num_complex::Complex64;
use std::sync::Arc;

impl MixedRadixScalar for f64 {
    type Complex = Complex64;

    #[inline]
    fn complex(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[inline]
    fn cached_twiddle_fwd(n: usize) -> Arc<[Complex64]> {
        cached_twiddle_fwd_64(n)
    }
    #[inline]
    fn cached_twiddle_inv(n: usize) -> Arc<[Complex64]> {
        cached_twiddle_inv_64(n)
    }
    #[inline]
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        with_stockham_scratch_64(n, f)
    }
    #[inline]
    fn with_pfa_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        crate::application::execution::kernel::mixed_radix::caches::with_pfa_scratch_64(n, f)
    }
    #[inline]
    fn with_rader_padded_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        crate::application::execution::kernel::mixed_radix::caches::with_rader_padded_scratch_64(
            n, f,
        )
    }

    #[inline]
    fn cached_rader_spectrum(
        n: usize,
        inverse: bool,
        generator_inverse: usize,
    ) -> Arc<[Complex64]> {
        let key = (n, inverse as usize, generator_inverse);
        cached_rader_spectrum_64(key, |_| {
            build_rader_spectrum_vec::<f64>(n, inverse, generator_inverse)
        })
    }

    #[inline]
    fn cached_four_step_twiddles(
        n: usize,
        n1: usize,
        n2: usize,
        inverse: bool,
    ) -> Arc<[Complex64]> {
        cached_four_step_twiddles_64(n, n1, n2, inverse)
    }
    #[inline]
    fn pointwise_mul(a: &mut [Complex64], b: &[Complex64]) {
        pointwise_mul_c64(a, b);
    }
    #[inline]
    fn stockham_forward(
        data: &mut [Complex64],
        scratch: &mut [Complex64],
        twiddles: &[Complex64],
    ) {
        <f64 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, twiddles);
    }
    #[inline]
    fn stockham_forward_normalized(
        data: &mut [Complex64],
        scratch: &mut [Complex64],
        twiddles: &[Complex64],
        n: usize,
    ) {
        <f64 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, twiddles);
        <Complex64 as NormalizeSlice>::normalize_slice(data, 1.0 / n as f64);
    }
    #[inline(always)]
    fn short_winograd(data: &mut [Complex64], inverse: bool, normalize: bool) -> bool {
        if inverse {
            inverse_short_winograd(data, normalize)
        } else {
            forward_short_winograd(data)
        }
    }
    #[inline]
    fn composite_forward(data: &mut [Complex64], radices: &[usize]) {
        radix_composite::forward_inplace_with_radices(data, radices);
    }
    #[inline]
    fn composite_inverse_unnorm(data: &mut [Complex64], radices: &[usize]) {
        radix_composite::inverse_inplace_unnorm_with_radices(data, radices);
    }
    #[inline]
    fn composite_inverse(data: &mut [Complex64], radices: &[usize]) {
        radix_composite::inverse_inplace_with_radices(data, radices);
    }
    #[inline]
    fn normalize(data: &mut [Complex64], n: usize) {
        <Complex64 as NormalizeSlice>::normalize_slice(data, 1.0 / n as f64);
    }
    #[inline]
    fn transpose_matrix(src: &[Complex64], dst: &mut [Complex64], n1: usize, n2: usize) {
        transpose_matrix_c64(src, dst, n1, n2);
    }
}
