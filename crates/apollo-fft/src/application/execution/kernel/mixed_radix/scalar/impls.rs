use super::rader::{
    build_rader_negacyclic_spectra, build_rader_negacyclic_twiddles, build_rader_spectrum_vec,
};
use super::simd::pointwise_mul_reduced;
use super::trait_def::MixedRadixScalar;
use super::transpose::transpose_matrix_reduced;
use crate::application::execution::kernel::mixed_radix::caches::{
    cached_four_step_twiddles, cached_rader_neg_twiddles, cached_rader_negacyclic_spectra,
    cached_rader_spectrum, cached_twiddle_fwd, cached_twiddle_inv, with_pfa_scratch,
    with_rader_padded_scratch, with_stockham_scratch,
};
// Obsolete imports removed
use crate::application::execution::kernel::components::{radix_composite, stockham};
use crate::application::execution::kernel::radix_stage::normalize_inplace;
use num_complex::Complex32;
use std::sync::Arc;

impl MixedRadixScalar for f32 {
    const HALF_CYCLIC_RADER_THRESHOLD: usize =
        crate::application::execution::kernel::components::rader::HALF_CYCLIC_THRESHOLD;

    type Complex = Complex32;

    #[inline]
    fn complex(re: f64, im: f64) -> Complex32 {
        Complex32::new(re as f32, im as f32)
    }

    #[inline]
    fn cached_twiddle_fwd(n: usize) -> Arc<[Complex32]> {
        cached_twiddle_fwd(n)
    }
    #[inline]
    fn cached_twiddle_inv(n: usize) -> Arc<[Complex32]> {
        cached_twiddle_inv(n)
    }
    #[inline]
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        with_stockham_scratch(n, f)
    }
    #[inline]
    fn with_pfa_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        with_pfa_scratch(n, f)
    }
    #[inline]
    fn with_rader_padded_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        with_rader_padded_scratch(n, f)
    }

    #[inline]
    fn cached_rader_spectrum(
        n: usize,
        inverse: bool,
        generator_inverse: usize,
    ) -> Arc<[Complex32]> {
        let key = (n, inverse as usize, generator_inverse);
        cached_rader_spectrum(key, |_| {
            build_rader_spectrum_vec::<f32>(n, inverse, generator_inverse)
        })
    }

    #[inline]
    fn cached_rader_negacyclic_spectra(
        n: usize,
        inverse: bool,
        generator_inverse: usize,
    ) -> (Arc<[Complex32]>, Arc<[Complex32]>) {
        let key = (n, inverse as usize, generator_inverse);
        cached_rader_negacyclic_spectra(key, |_| {
            build_rader_negacyclic_spectra::<f32>(n, inverse, generator_inverse)
        })
    }

    #[inline]
    fn cached_rader_neg_twiddles(m: usize) -> Arc<[Complex32]> {
        cached_rader_neg_twiddles(m, build_rader_negacyclic_twiddles::<f32>)
    }

    #[inline]
    fn cached_four_step_twiddles(
        n: usize,
        n1: usize,
        n2: usize,
        inverse: bool,
    ) -> Arc<[Complex32]> {
        cached_four_step_twiddles(n, n1, n2, inverse)
    }
    #[inline]
    fn pointwise_mul(a: &mut [Complex32], b: &[Complex32]) {
        pointwise_mul_reduced(a, b, false);
    }
    #[inline]
    fn pointwise_mul_conj(a: &mut [Complex32], b: &[Complex32]) {
        pointwise_mul_reduced(a, b, true);
    }
    #[inline]
    fn stockham_forward(data: &mut [Complex32], scratch: &mut [Complex32], twiddles: &[Complex32]) {
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
        normalize_inplace(data, 1.0_f32 / n as f32);
    }
    #[inline(always)]
    fn short_winograd<const INVERSE: bool, const NORMALIZE: bool>(data: &mut [Complex32]) -> bool {
        crate::application::execution::kernel::mixed_radix::traits::short_winograd::<
            Self,
            INVERSE,
            NORMALIZE,
        >(data)
    }
    #[inline]
    fn composite_forward(data: &mut [Complex32], radices: &[usize]) {
        radix_composite::forward_inplace_with_radices(data, radices);
    }
    #[inline]
    fn composite_forward_with_pointwise(
        data: &mut [Complex32],
        radices: &[usize],
        pointwise_spectrum: &[Complex32],
    ) {
        radix_composite::forward_inplace_with_pointwise(data, radices, pointwise_spectrum);
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
        normalize_inplace(data, 1.0_f32 / n as f32);
    }
    #[inline]
    fn transpose_matrix(src: &[Complex32], dst: &mut [Complex32], n1: usize, n2: usize) {
        transpose_matrix_reduced(src, dst, n1, n2);
    }
}
