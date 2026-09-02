use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::plan::fft::dimension_1d::executors::static_fft_dispatch;
use crate::application::execution::plan::fft::layout::with_c_order_view;
use core::marker::PhantomData;
use eunomia::Complex;
use leto::Array1;
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
    ///
    /// Any valid layout is accepted: a C-dense view transforms in place and a
    /// strided view is staged through thread-local scratch and written back
    /// in its logical order, as the 2-D and 3-D plans do.
    ///
    /// # Panics
    ///
    /// Panics if the view length differs from `N`.
    #[inline]
    pub fn forward_complex_leto_inplace(&self, data: ArrayViewMut1<'_, F::Complex>)
    where
        F::Complex: PlanScratch,
    {
        with_c_order_view(data, |mut view| {
            self.forward_complex_slice_inplace(
                view.as_mut_slice()
                    .expect("invariant: a C-order view over its dense block is a slice"),
            );
        });
    }

    /// Inverse transform of a complex Leto view in-place with normalization.
    ///
    /// Any valid layout is accepted: a C-dense view transforms in place and a
    /// strided view is staged through thread-local scratch and written back
    /// in its logical order, as the 2-D and 3-D plans do.
    ///
    /// # Panics
    ///
    /// Panics if the view length differs from `N`.
    #[inline]
    pub fn inverse_complex_leto_inplace(&self, data: ArrayViewMut1<'_, F::Complex>)
    where
        F::Complex: PlanScratch,
    {
        with_c_order_view(data, |mut view| {
            self.inverse_complex_slice_inplace(
                view.as_mut_slice()
                    .expect("invariant: a C-order view over its dense block is a slice"),
            );
        });
    }

    /// Inverse transform of a complex Leto view in-place without normalization.
    ///
    /// Any valid layout is accepted: a C-dense view transforms in place and a
    /// strided view is staged through thread-local scratch and written back
    /// in its logical order, as the 2-D and 3-D plans do.
    ///
    /// # Panics
    ///
    /// Panics if the view length differs from `N`.
    #[inline]
    pub fn inverse_complex_leto_unnorm_inplace(&self, data: ArrayViewMut1<'_, F::Complex>)
    where
        F::Complex: PlanScratch,
    {
        with_c_order_view(data, |mut view| {
            self.inverse_complex_slice_unnorm_inplace(
                view.as_mut_slice()
                    .expect("invariant: a C-order view over its dense block is a slice"),
            );
        });
    }

    /// Forward transform of a complex slice in-place.
    ///
    /// # Panics
    ///
    /// Panics if `slice.len()` differs from `N`.
    #[inline]
    pub fn forward_complex_slice_inplace(&self, slice: &mut [F::Complex]) {
        static_fft_dispatch::<F, N, false, false>(slice);
    }

    /// Inverse transform of a complex slice in-place with normalization.
    ///
    /// # Panics
    ///
    /// Panics if `slice.len()` differs from `N`.
    #[inline]
    pub fn inverse_complex_slice_inplace(&self, slice: &mut [F::Complex]) {
        static_fft_dispatch::<F, N, true, true>(slice);
    }

    /// Inverse transform of a complex slice in-place without normalization.
    ///
    /// # Panics
    ///
    /// Panics if `slice.len()` differs from `N`.
    #[inline]
    pub fn inverse_complex_slice_unnorm_inplace(&self, slice: &mut [F::Complex]) {
        static_fft_dispatch::<F, N, true, false>(slice);
    }
}
