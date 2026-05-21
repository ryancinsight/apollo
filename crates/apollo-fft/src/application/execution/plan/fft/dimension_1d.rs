//! 1D FFT plan.
//!
//! Apollo-owned 1D FFT implementation based on `MixedRadixScalar`.

use crate::application::execution::kernel::mixed_radix::{dispatch_inplace, MixedRadixScalar};
use crate::domain::metadata::shape::Shape1D;
use ndarray::Array1;
use num_complex::Complex;
use std::sync::Arc;

/// Reusable 1D FFT plan generic over `MixedRadixScalar`.
pub struct FftPlan1D<F: MixedRadixScalar> {
    n: usize,
    twiddle_fwd: Option<Arc<[F::Complex]>>,
    twiddle_inv: Option<Arc<[F::Complex]>>,
}

impl<F: MixedRadixScalar> std::fmt::Debug for FftPlan1D<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPlan1D").field("n", &self.n).finish()
    }
}

impl<F: MixedRadixScalar<Complex = Complex<F>>> FftPlan1D<F> {
    /// Create a new 1D plan.
    #[must_use]
    pub fn new(shape: Shape1D) -> Self {
        let is_pow2 = shape.n.is_power_of_two() && shape.n > 1;
        Self {
            n: shape.n,
            twiddle_fwd: if is_pow2 {
                Some(F::cached_twiddle_fwd(shape.n))
            } else {
                None
            },
            twiddle_inv: if is_pow2 {
                Some(F::cached_twiddle_inv(shape.n))
            } else {
                None
            },
        }
    }

    /// Return the plan length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Return the validated shape owned by this plan.
    #[must_use]
    pub fn shape(&self) -> Shape1D {
        Shape1D { n: self.n }
    }

    /// Forward transform of a complex signal in-place.
    pub fn forward_complex_inplace(&self, data: &mut Array1<F::Complex>) {
        self.forward_complex_slice_inplace(data.as_slice_mut().expect("Array must be contiguous"));
    }

    /// Forward transform of a complex slice in-place.
    pub fn forward_complex_slice_inplace(&self, slice: &mut [F::Complex]) {
        if let Some(tw) = &self.twiddle_fwd {
            dispatch_inplace::<F, false, false>(slice, Some(tw.as_ref()));
        } else {
            crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(slice);
        }
    }

    /// Inverse transform of a complex signal in-place with normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array1<F::Complex>) {
        self.inverse_complex_slice_inplace(data.as_slice_mut().expect("Array must be contiguous"));
    }

    /// Inverse transform of a complex slice in-place with normalization.
    pub fn inverse_complex_slice_inplace(&self, slice: &mut [F::Complex]) {
        if let Some(tw) = &self.twiddle_inv {
            dispatch_inplace::<F, true, true>(slice, Some(tw.as_ref()));
        } else {
            crate::application::execution::kernel::mixed_radix::inverse_inplace::<F>(slice);
        }
    }

    /// Forward transform of a complex signal (allocating).
    #[must_use]
    pub fn forward_complex(&self, input: &Array1<F::Complex>) -> Array1<F::Complex> {
        let mut output = input.clone();
        self.forward_complex_inplace(&mut output);
        output
    }

    /// Inverse transform of a complex signal (allocating).
    #[must_use]
    pub fn inverse_complex(&self, input: &Array1<F::Complex>) -> Array1<F::Complex> {
        let mut output = input.clone();
        self.inverse_complex_inplace(&mut output);
        output
    }
}
