//! 1D FFT plan.
//!
//! Apollo-owned 1D FFT implementation based on `MixedRadixScalar`.

use crate::application::execution::kernel::mixed_radix::{dispatch_inplace, MixedRadixScalar};
use crate::domain::metadata::shape::Shape1D;
use ndarray::Array1;
use num_complex::Complex;
use std::sync::Arc;

/// Reusable 1D FFT plan generic over `MixedRadixScalar`.
enum PlanStrategy<F: MixedRadixScalar> {
    Identity,
    ShortWinograd,
    PowerOfTwo {
        twiddle_fwd: Arc<[F::Complex]>,
        twiddle_inv: Arc<[F::Complex]>,
    },
    GoodThomas {
        n1: usize,
        n2: usize,
    },
    Composite {
        radices: Arc<[usize]>,
    },
    Rader,
}

impl<F: MixedRadixScalar> Clone for PlanStrategy<F> {
    fn clone(&self) -> Self {
        match self {
            Self::Identity => Self::Identity,
            Self::ShortWinograd => Self::ShortWinograd,
            Self::PowerOfTwo {
                twiddle_fwd,
                twiddle_inv,
            } => Self::PowerOfTwo {
                twiddle_fwd: twiddle_fwd.clone(),
                twiddle_inv: twiddle_inv.clone(),
            },
            Self::GoodThomas { n1, n2 } => Self::GoodThomas { n1: *n1, n2: *n2 },
            Self::Composite { radices } => Self::Composite {
                radices: radices.clone(),
            },
            Self::Rader => Self::Rader,
        }
    }
}

/// Reusable 1D FFT plan generic over `MixedRadixScalar`.
pub struct FftPlan1D<F: MixedRadixScalar> {
    n: usize,
    twiddle_fwd: Option<Arc<[F::Complex]>>,
    twiddle_inv: Option<Arc<[F::Complex]>>,
    strategy: PlanStrategy<F>,
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
        let n = shape.n;
        let strategy = if n <= 1 {
            PlanStrategy::Identity
        } else if n.is_power_of_two() {
            PlanStrategy::PowerOfTwo {
                twiddle_fwd: F::cached_twiddle_fwd(n),
                twiddle_inv: F::cached_twiddle_inv(n),
            }
        } else if crate::application::execution::kernel::mixed_radix::traits::is_short_winograd_size(n) {
            PlanStrategy::ShortWinograd
        } else if let Some((n1, n2)) = crate::application::execution::kernel::mixed_radix::caches::cached_coprime_factors(n) {
            PlanStrategy::GoodThomas { n1, n2 }
        } else if let Some(radices) = crate::application::execution::kernel::mixed_radix::caches::cached_prime23_radices(n) {
            PlanStrategy::Composite { radices }
        } else {
            PlanStrategy::Rader
        };

        Self {
            n,
            twiddle_fwd: match &strategy {
                PlanStrategy::PowerOfTwo { twiddle_fwd, .. } => Some(twiddle_fwd.clone()),
                _ => None,
            },
            twiddle_inv: match &strategy {
                PlanStrategy::PowerOfTwo { twiddle_inv, .. } => Some(twiddle_inv.clone()),
                _ => None,
            },
            strategy,
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

    /// Inverse transform of a complex signal in-place with normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array1<F::Complex>) {
        self.inverse_complex_slice_inplace(data.as_slice_mut().expect("Array must be contiguous"));
    }

    /// Inverse transform of a complex signal in-place without normalization.
    pub fn inverse_complex_unnorm_inplace(&self, data: &mut Array1<F::Complex>) {
        self.inverse_complex_slice_unnorm_inplace(data.as_slice_mut().expect("Array must be contiguous"));
    }

    /// Forward transform of a complex slice in-place.
    pub fn forward_complex_slice_inplace(&self, slice: &mut [F::Complex]) {
        match &self.strategy {
            PlanStrategy::Identity => {}
            PlanStrategy::ShortWinograd => {
                F::short_winograd::<false, false>(slice);
            }
            PlanStrategy::PowerOfTwo { twiddle_fwd, .. } => {
                F::pot_inplace::<false, false>(slice, twiddle_fwd);
            }
            PlanStrategy::GoodThomas { n1, n2 } => {
                crate::application::execution::kernel::components::good_thomas::pfa_fft::<F, false>(slice, *n1, *n2);
            }
            PlanStrategy::Composite { radices } => {
                F::composite_forward(slice, radices);
            }
            PlanStrategy::Rader => {
                crate::application::execution::kernel::components::rader::rader_fft::<F, false>(slice);
            }
        }
    }

    /// Inverse transform of a complex slice in-place with normalization.
    pub fn inverse_complex_slice_inplace(&self, slice: &mut [F::Complex]) {
        let n = self.n;
        match &self.strategy {
            PlanStrategy::Identity => {}
            PlanStrategy::ShortWinograd => {
                F::short_winograd::<true, true>(slice);
            }
            PlanStrategy::PowerOfTwo { twiddle_inv, .. } => {
                F::pot_inplace::<true, true>(slice, twiddle_inv);
            }
            PlanStrategy::GoodThomas { n1, n2 } => {
                crate::application::execution::kernel::components::good_thomas::pfa_fft::<F, true>(slice, *n1, *n2);
                F::normalize(slice, n);
            }
            PlanStrategy::Composite { radices } => {
                F::composite_inverse(slice, radices);
            }
            PlanStrategy::Rader => {
                crate::application::execution::kernel::components::rader::rader_fft::<F, true>(slice);
                F::normalize(slice, n);
            }
        }
    }

    /// Inverse transform of a complex slice in-place without normalization.
    pub fn inverse_complex_slice_unnorm_inplace(&self, slice: &mut [F::Complex]) {
        match &self.strategy {
            PlanStrategy::Identity => {}
            PlanStrategy::ShortWinograd => {
                F::short_winograd::<true, false>(slice);
            }
            PlanStrategy::PowerOfTwo { twiddle_inv, .. } => {
                F::pot_inplace::<true, false>(slice, twiddle_inv);
            }
            PlanStrategy::GoodThomas { n1, n2 } => {
                crate::application::execution::kernel::components::good_thomas::pfa_fft::<F, true>(slice, *n1, *n2);
            }
            PlanStrategy::Composite { radices } => {
                F::composite_inverse_unnorm(slice, radices);
            }
            PlanStrategy::Rader => {
                crate::application::execution::kernel::components::rader::rader_fft::<F, true>(slice);
            }
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
