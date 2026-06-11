use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::plan::fft::dimension_1d::executors::static_fft_dispatch;
use core::marker::PhantomData;
use ndarray::Array1;
use num_complex::Complex;
use leto::ArrayViewMut1;

/// Zero-sized 1D FFT plan for compile-time-known lengths.
///
/// The length is encoded as `N`, so execution routes through const-generic
/// branches that monomorphize per size instead of storing runtime executor
/// function pointers.
#[derive(Clone, Copy, Debug, Default)]
pub struct StaticFftPlan1D<F: MixedRadixScalar, const N: usize> {
    precision: PhantomData<F>,
}

impl<F: MixedRadixScalar, const N: usize> StaticFftPlan1D<F, N> {
    /// Construct a zero-sized static plan.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            precision: PhantomData,
        }
    }

    /// Return the compile-time plan length.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        N
    }

    /// Return whether the compile-time plan length is zero.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        N == 0
    }
}

impl<F: MixedRadixScalar<Complex = Complex<F>>, const N: usize> StaticFftPlan1D<F, N> {
    /// Forward transform of a complex signal in-place.
    #[inline]
    pub fn forward_complex_inplace(&self, data: &mut Array1<F::Complex>) {
        self.forward_complex_slice_inplace(data.as_slice_mut().expect("Array must be contiguous"));
    }

    /// Inverse transform of a complex signal in-place with normalization.
    #[inline]
    pub fn inverse_complex_inplace(&self, data: &mut Array1<F::Complex>) {
        self.inverse_complex_slice_inplace(data.as_slice_mut().expect("Array must be contiguous"));
    }

    /// Forward transform of a complex Leto view in-place.
    #[inline]
    pub fn forward_complex_leto_inplace(&self, mut data: ArrayViewMut1<'_, F::Complex>) {
        self.forward_complex_slice_inplace(data.as_mut_slice_memory_order().expect("Array must be contiguous"));
    }

    /// Inverse transform of a complex Leto view in-place with normalization.
    #[inline]
    pub fn inverse_complex_leto_inplace(&self, mut data: ArrayViewMut1<'_, F::Complex>) {
        self.inverse_complex_slice_inplace(data.as_mut_slice_memory_order().expect("Array must be contiguous"));
    }

    /// Inverse transform of a complex Leto view in-place without normalization.
    #[inline]
    pub fn inverse_complex_leto_unnorm_inplace(&self, mut data: ArrayViewMut1<'_, F::Complex>) {
        self.inverse_complex_slice_unnorm_inplace(data.as_mut_slice_memory_order().expect("Array must be contiguous"));
    }

    /// Forward transform of a complex slice in-place.
    #[inline]
    pub fn forward_complex_slice_inplace(&self, slice: &mut [F::Complex]) {
        static_fft_dispatch::<F, N, false, false>(slice);
    }

    /// Inverse transform of a complex slice in-place with normalization.
    #[inline]
    pub fn inverse_complex_slice_inplace(&self, slice: &mut [F::Complex]) {
        static_fft_dispatch::<F, N, true, true>(slice);
    }

    /// Inverse transform of a complex slice in-place without normalization.
    #[inline]
    pub fn inverse_complex_slice_unnorm_inplace(&self, slice: &mut [F::Complex]) {
        static_fft_dispatch::<F, N, true, false>(slice);
    }
}
