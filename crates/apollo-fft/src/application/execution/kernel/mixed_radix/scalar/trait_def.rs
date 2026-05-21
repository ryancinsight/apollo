use super::super::traits::ShortWinogradScalar;
use super::transpose::transpose_tiled_scalar;
use crate::application::execution::kernel::components::radix_composite::CompositeCache;
use std::sync::Arc;

pub(crate) mod private {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for f32 {}
}

pub trait MixedRadixScalar:
    private::Sealed + Sized + Copy + 'static + ShortWinogradScalar + CompositeCache
{
    /// Minimum Rader convolution length `N - 1` for the half-cyclic CRT split.
    ///
    /// `usize::MAX` disables automatic production routing while retaining the
    /// strategy for forced benchmark and equivalence tests.
    const HALF_CYCLIC_RADER_THRESHOLD: usize;

    type Complex: Copy
        + Send
        + Sync
        + 'static
        + num_traits::Zero
        + std::ops::Add<Output = Self::Complex>
        + std::ops::Mul<Output = Self::Complex>;

    fn complex(re: f64, im: f64) -> Self::Complex;

    fn cached_twiddle_fwd(n: usize) -> Arc<[Self::Complex]>;
    fn cached_twiddle_inv(n: usize) -> Arc<[Self::Complex]>;

    fn cached_rader_spectrum(
        n: usize,
        inverse: bool,
        generator_inverse: usize,
    ) -> Arc<[Self::Complex]>;

    /// Return precomputed negacyclic Rader convolution spectra for prime length `n`.
    ///
    /// Returns `(cyclic_spectrum, negacyclic_spectrum)` each of length `(n-1)/2`.
    fn cached_rader_negacyclic_spectra(
        n: usize,
        inverse: bool,
        generator_inverse: usize,
    ) -> (Arc<[Self::Complex]>, Arc<[Self::Complex]>);

    /// Return precomputed twist twiddles `e^{i*pi*j/m}` for negacyclic convolution.
    fn cached_rader_neg_twiddles(m: usize) -> Arc<[Self::Complex]>;

    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Self::Complex]) -> R) -> R;
    fn with_pfa_scratch<R>(n: usize, f: impl FnOnce(&mut [Self::Complex]) -> R) -> R;
    fn with_rader_padded_scratch<R>(n: usize, f: impl FnOnce(&mut [Self::Complex]) -> R) -> R;

    fn cached_four_step_twiddles(
        n: usize,
        n1: usize,
        n2: usize,
        inverse: bool,
    ) -> Arc<[Self::Complex]>;

    fn pointwise_mul(a: &mut [Self::Complex], b: &[Self::Complex]);

    fn pointwise_mul_conj(a: &mut [Self::Complex], b: &[Self::Complex]);

    fn stockham_forward(
        data: &mut [Self::Complex],
        scratch: &mut [Self::Complex],
        twiddles: &[Self::Complex],
    );

    #[inline]
    fn stockham_forward_normalized(
        data: &mut [Self::Complex],
        scratch: &mut [Self::Complex],
        twiddles: &[Self::Complex],
        n: usize,
    ) {
        Self::stockham_forward(data, scratch, twiddles);
        Self::normalize(data, n);
    }

    fn short_winograd<const INVERSE: bool, const NORMALIZE: bool>(
        data: &mut [Self::Complex],
    ) -> bool;

    fn composite_forward(data: &mut [Self::Complex], radices: &[usize]);
    fn composite_forward_with_pointwise(
        data: &mut [Self::Complex],
        radices: &[usize],
        pointwise_spectrum: &[Self::Complex],
    );
    fn composite_inverse_unnorm(data: &mut [Self::Complex], radices: &[usize]);
    fn composite_inverse(data: &mut [Self::Complex], radices: &[usize]);

    fn normalize(data: &mut [Self::Complex], n: usize);

    fn transpose_matrix(src: &[Self::Complex], dst: &mut [Self::Complex], n1: usize, n2: usize) {
        transpose_tiled_scalar(src, dst, n1, n2);
    }
}
