use super::transpose::transpose_tiled_scalar;
use std::sync::Arc;

pub(crate) mod private {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for f32 {}
}

pub(crate) trait MixedRadixScalar: private::Sealed + Sized + Copy + 'static {
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

    fn short_winograd(data: &mut [Self::Complex], inverse: bool, normalize: bool) -> bool;

    fn composite_forward(data: &mut [Self::Complex], radices: &[usize]);
    fn composite_inverse_unnorm(data: &mut [Self::Complex], radices: &[usize]);
    fn composite_inverse(data: &mut [Self::Complex], radices: &[usize]);

    fn normalize(data: &mut [Self::Complex], n: usize);

    fn transpose_matrix(src: &[Self::Complex], dst: &mut [Self::Complex], n1: usize, n2: usize) {
        transpose_tiled_scalar(src, dst, n1, n2);
    }
}
