use super::rader::build_rader_spectrum_vec;
use super::simd::pointwise_mul_c32;
use super::trait_def::MixedRadixScalar;
use super::transpose::transpose_matrix_c32;
use crate::application::execution::kernel::mixed_radix::caches::{
    cached_four_step_twiddles_32, cached_rader_spectrum_32, cached_twiddle_fwd_32,
    cached_twiddle_inv_32, with_stockham_scratch_32,
};
use crate::application::execution::kernel::mixed_radix::traits::{
    forward_short_winograd, inverse_short_winograd,
};
use crate::application::execution::kernel::radix_stage::NormalizeSlice;
use crate::application::execution::kernel::{radix_composite, stockham};
use num_complex::Complex32;
use std::sync::Arc;

impl MixedRadixScalar for f32 {
    type Complex = Complex32;

    #[inline]
    fn complex(re: f64, im: f64) -> Complex32 {
        Complex32::new(re as f32, im as f32)
    }

    #[inline]
    fn cached_twiddle_fwd(n: usize) -> Arc<[Complex32]> {
        cached_twiddle_fwd_32(n)
    }
    #[inline]
    fn cached_twiddle_inv(n: usize) -> Arc<[Complex32]> {
        cached_twiddle_inv_32(n)
    }
    #[inline]
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        with_stockham_scratch_32(n, f)
    }
    #[inline]
    fn with_pfa_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        crate::application::execution::kernel::mixed_radix::caches::with_pfa_scratch_32(n, f)
    }
    #[inline]
    fn with_rader_padded_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        crate::application::execution::kernel::mixed_radix::caches::with_rader_padded_scratch_32(
            n, f,
        )
    }

    #[inline]
    fn cached_rader_spectrum(
        n: usize,
        inverse: bool,
        generator_inverse: usize,
    ) -> Arc<[Complex32]> {
        let key = (n, inverse as usize, generator_inverse);
        cached_rader_spectrum_32(key, |_| {
            build_rader_spectrum_vec::<f32>(n, inverse, generator_inverse)
        })
    }

    #[inline]
    fn cached_four_step_twiddles(
        n: usize,
        n1: usize,
        n2: usize,
        inverse: bool,
    ) -> Arc<[Complex32]> {
        cached_four_step_twiddles_32(n, n1, n2, inverse)
    }
    #[inline]
    fn pointwise_mul(a: &mut [Complex32], b: &[Complex32]) {
        pointwise_mul_c32(a, b);
    }
    #[inline]
    fn stockham_forward(
        data: &mut [Complex32],
        scratch: &mut [Complex32],
        twiddles: &[Complex32],
    ) {
        <f32 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, twiddles);
    }
    #[inline]
    fn stockham_forward_normalized(
        data: &mut [Complex32],
        scratch: &mut [Complex32],
        twiddles: &[Complex32],
        n: usize,
    ) {
        <f32 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, twiddles);
        <Complex32 as NormalizeSlice>::normalize_slice(data, 1.0 / n as f32);
    }
    #[inline(always)]
    fn short_winograd(data: &mut [Complex32], inverse: bool, normalize: bool) -> bool {
        if inverse {
            inverse_short_winograd(data, normalize)
        } else {
            forward_short_winograd(data)
        }
    }
    #[inline]
    fn composite_forward(data: &mut [Complex32], radices: &[usize]) {
        radix_composite::forward_inplace_with_radices(data, radices);
    }
    #[inline]
    fn composite_inverse_unnorm(data: &mut [Complex32], radices: &[usize]) {
        radix_composite::inverse_inplace_unnorm_with_radices(data, radices);
    }
    #[inline]
    fn composite_inverse(data: &mut [Complex32], radices: &[usize]) {
        radix_composite::inverse_inplace_with_radices(data, radices);
    }
    #[inline]
    fn normalize(data: &mut [Complex32], n: usize) {
        <Complex32 as NormalizeSlice>::normalize_slice(data, 1.0 / n as f32);
    }
    #[inline]
    fn transpose_matrix(src: &[Complex32], dst: &mut [Complex32], n1: usize, n2: usize) {
        transpose_matrix_c32(src, dst, n1, n2);
    }
}
