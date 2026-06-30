use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::domain::metadata::shape::Shape1D;
use core::marker::PhantomData;
use leto::ArrayViewMut1;
use leto::Array1;
use eunomia::Complex;
use std::borrow::Cow;
use std::sync::Arc;

use super::executors::{
    exec_composite_forward, exec_composite_inverse, exec_composite_inverse_unnorm,
    exec_good_thomas_forward, exec_good_thomas_inverse, exec_good_thomas_inverse_unnorm,
    exec_identity, exec_pot_forward_16, exec_pot_forward_2, exec_pot_forward_32,
    exec_pot_forward_4, exec_pot_forward_512, exec_pot_forward_64, exec_pot_forward_8,
    exec_pot_forward_generic, exec_pot_forward_sized, exec_pot_inverse_16, exec_pot_inverse_2,
    exec_pot_inverse_32, exec_pot_inverse_4, exec_pot_inverse_512, exec_pot_inverse_64,
    exec_pot_inverse_8, exec_pot_inverse_generic, exec_pot_inverse_sized,
    exec_pot_inverse_unnorm_16, exec_pot_inverse_unnorm_2, exec_pot_inverse_unnorm_32,
    exec_pot_inverse_unnorm_4, exec_pot_inverse_unnorm_512, exec_pot_inverse_unnorm_64,
    exec_pot_inverse_unnorm_8, exec_pot_inverse_unnorm_generic, exec_pot_inverse_unnorm_sized,
    exec_rader_forward, exec_rader_inverse, exec_rader_inverse_unnorm, exec_winograd_forward,
    exec_winograd_inverse, exec_winograd_inverse_unnorm, runtime_tiny_direct_dispatch,
};
use super::helpers::{arc_to_cow, PlanStrategy};

/// Reusable 1D FFT plan generic over `MixedRadixScalar`.
pub struct FftPlan1D<F: MixedRadixScalar> {
    pub(crate) n: usize,
    #[cfg(test)]
    pub(crate) strategy: PlanStrategy<F>,

    // Cached variables to completely bypass enum matching in the hot path:
    pub(crate) n1: usize,
    pub(crate) n2: usize,
    pub(crate) radices: Option<Cow<'static, [usize]>>,
    pub(crate) twiddle_fwd: Option<Arc<[F::Complex]>>,
    pub(crate) twiddle_inv: Option<Arc<[F::Complex]>>,

    // Function pointers for execution routing:
    pub(crate) forward_impl: fn(&Self, &mut [F::Complex]),
    pub(crate) inverse_impl: fn(&Self, &mut [F::Complex]),
    pub(crate) inverse_unnorm_impl: fn(&Self, &mut [F::Complex]),
}

impl<F: MixedRadixScalar> Clone for FftPlan1D<F> {
    fn clone(&self) -> Self {
        Self {
            n: self.n,
            #[cfg(test)]
            strategy: self.strategy.clone(),
            n1: self.n1,
            n2: self.n2,
            radices: self.radices.clone(),
            twiddle_fwd: self.twiddle_fwd.clone(),
            twiddle_inv: self.twiddle_inv.clone(),
            forward_impl: self.forward_impl,
            inverse_impl: self.inverse_impl,
            inverse_unnorm_impl: self.inverse_unnorm_impl,
        }
    }
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
        let strategy: PlanStrategy<F> = if n <= 1 {
            PlanStrategy::Identity
        } else if n.is_power_of_two() {
            let log2 = n.trailing_zeros();
            PlanStrategy::PowerOfTwo {
                twiddle_fwd: F::cached_twiddle_fwd(n),
                twiddle_inv: F::cached_twiddle_inv(n),
                log2,
                pot: PhantomData,
            }
        } else if n == 385 {
            PlanStrategy::Composite {
                radices: Cow::Borrowed(&[11, 5, 7]),
            }
        } else if n == 180 {
            PlanStrategy::Composite {
                radices: Cow::Borrowed(&[5, 3, 3, 4]),
            }
        } else if n == 144 {
            PlanStrategy::Composite {
                radices: Cow::Borrowed(&[4, 4, 3, 3]),
            }
        } else if n == 176 {
            PlanStrategy::Composite {
                radices: Cow::Borrowed(&[11, 4, 4]),
            }
        } else if n == 200 {
            PlanStrategy::Composite {
                radices: Cow::Borrowed(F::COMPOSITE_RADICES_200),
            }
        } else if F::use_generated_codelet_plan(n) {
            PlanStrategy::ShortWinograd
        } else if n == 72 && !F::FORCE_COMPOSITE_72 {
            PlanStrategy::GoodThomas { n1: 9, n2: 8 }
        } else if n == 511 {
            PlanStrategy::GoodThomas { n1: 73, n2: 7 }
        } else if n == 36 {
            PlanStrategy::Composite {
                radices: Cow::Borrowed(&[4, 3, 3]),
            }
        } else if n == 48 {
            PlanStrategy::Composite {
                radices: Cow::Borrowed(&[4, 4, 3]),
            }
        } else if n == 63 && F::FORCE_COMPOSITE_63 {
            PlanStrategy::Composite {
                radices: Cow::Borrowed(&[3, 3, 7]),
            }
        } else if n == 72 && F::FORCE_COMPOSITE_72 {
            PlanStrategy::Composite {
                radices: Cow::Borrowed(&[4, 2, 3, 3]),
            }
        } else if n == 72 {
            PlanStrategy::GoodThomas { n1: 9, n2: 8 }
        } else if n == 90 {
            PlanStrategy::Composite {
                radices: Cow::Borrowed(&[2, 3, 3, 5]),
            }
        } else if n == 198 {
            PlanStrategy::Composite {
                radices: Cow::Borrowed(&[2, 3, 3, 11]),
            }
        } else if crate::application::execution::kernel::mixed_radix::traits::is_short_winograd_size(n)
            && (n <= 64 || F::use_generated_codelet_plan(n))
        {
            PlanStrategy::ShortWinograd
        } else if let Some(radices) = crate::application::execution::kernel::mixed_radix::caches::cached_prime23_radices(n) {
            PlanStrategy::Composite {
                radices: arc_to_cow(radices),
            }
        } else if let Some((n1, n2)) = crate::application::execution::kernel::mixed_radix::caches::cached_coprime_factors(n)
            .filter(|&(n1, n2)| crate::application::execution::kernel::components::good_thomas::has_static_coprime_codelet(n1, n2))
        {
            PlanStrategy::GoodThomas { n1, n2 }
        } else if let Some((n1, n2)) = crate::application::execution::kernel::mixed_radix::caches::cached_coprime_factors(n) {
            PlanStrategy::GoodThomas { n1, n2 }
        } else {
            PlanStrategy::Rader
        };

        // Cache parameters and assign specialized function pointers
        let mut n1 = 0;
        let mut n2 = 0;
        let mut radices_field = None;
        let mut twiddle_fwd = None;
        let mut twiddle_inv = None;

        let mut forward_impl: fn(&Self, &mut [F::Complex]) = exec_identity::<F>;
        let mut inverse_impl: fn(&Self, &mut [F::Complex]) = exec_identity::<F>;
        let mut inverse_unnorm_impl: fn(&Self, &mut [F::Complex]) = exec_identity::<F>;

        match &strategy {
            PlanStrategy::Identity => {}
            PlanStrategy::ShortWinograd => {
                macro_rules! assign_winograd {
                    ($size:expr) => {
                        if n == $size {
                            forward_impl = exec_winograd_forward::<F, $size>;
                            inverse_impl = exec_winograd_inverse::<F, $size>;
                            inverse_unnorm_impl = exec_winograd_inverse_unnorm::<F, $size>;
                        }
                    };
                }
                assign_winograd!(2);
                assign_winograd!(3);
                assign_winograd!(4);
                assign_winograd!(5);
                assign_winograd!(6);
                assign_winograd!(7);
                assign_winograd!(8);
                assign_winograd!(9);
                assign_winograd!(10);
                assign_winograd!(11);
                assign_winograd!(12);
                assign_winograd!(13);
                assign_winograd!(14);
                assign_winograd!(15);
                assign_winograd!(16);
                assign_winograd!(17);
                assign_winograd!(18);
                assign_winograd!(19);
                assign_winograd!(20);
                assign_winograd!(21);
                assign_winograd!(22);
                assign_winograd!(23);
                assign_winograd!(24);
                assign_winograd!(25);
                assign_winograd!(26);
                assign_winograd!(27);
                assign_winograd!(28);
                assign_winograd!(29);
                assign_winograd!(30);
                assign_winograd!(31);
                assign_winograd!(32);
                assign_winograd!(33);
                assign_winograd!(34);
                assign_winograd!(35);
                assign_winograd!(36);
                assign_winograd!(37);
                assign_winograd!(38);
                assign_winograd!(39);
                assign_winograd!(40);
                assign_winograd!(41);
                assign_winograd!(42);
                assign_winograd!(43);
                assign_winograd!(44);
                assign_winograd!(45);
                assign_winograd!(46);
                assign_winograd!(47);
                assign_winograd!(48);
                assign_winograd!(49);
                assign_winograd!(50);
                assign_winograd!(51);
                assign_winograd!(52);
                assign_winograd!(53);
                assign_winograd!(54);
                assign_winograd!(55);
                assign_winograd!(56);
                assign_winograd!(58);
                assign_winograd!(60);
                assign_winograd!(62);
                assign_winograd!(63);
                assign_winograd!(64);
                assign_winograd!(72);
                assign_winograd!(81);
                assign_winograd!(96);
                assign_winograd!(99);
                assign_winograd!(108);
                assign_winograd!(112);
                assign_winograd!(120);
                assign_winograd!(121);
                assign_winograd!(126);
                assign_winograd!(128);
                assign_winograd!(144);
                assign_winograd!(154);
                assign_winograd!(168);
                assign_winograd!(180);
                assign_winograd!(189);
                assign_winograd!(222);
                assign_winograd!(242);
                assign_winograd!(246);
                assign_winograd!(259);
                assign_winograd!(275);
                assign_winograd!(280);
                assign_winograd!(296);
                assign_winograd!(363);
                assign_winograd!(400);
                assign_winograd!(484);
            }
            PlanStrategy::PowerOfTwo {
                twiddle_fwd: fwd,
                twiddle_inv: inv,
                log2,
                pot: _,
            } => {
                twiddle_fwd = Some(fwd.clone());
                twiddle_inv = Some(inv.clone());
                match *log2 {
                    1 => {
                        forward_impl = exec_pot_forward_2::<F>;
                        inverse_impl = exec_pot_inverse_2::<F>;
                        inverse_unnorm_impl = exec_pot_inverse_unnorm_2::<F>;
                    }
                    2 => {
                        forward_impl = exec_pot_forward_4::<F>;
                        inverse_impl = exec_pot_inverse_4::<F>;
                        inverse_unnorm_impl = exec_pot_inverse_unnorm_4::<F>;
                    }
                    3 => {
                        forward_impl = exec_pot_forward_8::<F>;
                        inverse_impl = exec_pot_inverse_8::<F>;
                        inverse_unnorm_impl = exec_pot_inverse_unnorm_8::<F>;
                    }
                    4 => {
                        forward_impl = exec_pot_forward_16::<F>;
                        inverse_impl = exec_pot_inverse_16::<F>;
                        inverse_unnorm_impl = exec_pot_inverse_unnorm_16::<F>;
                    }
                    5 => {
                        forward_impl = exec_pot_forward_32::<F>;
                        inverse_impl = exec_pot_inverse_32::<F>;
                        inverse_unnorm_impl = exec_pot_inverse_unnorm_32::<F>;
                    }
                    6 => {
                        forward_impl = exec_pot_forward_64::<F>;
                        inverse_impl = exec_pot_inverse_64::<F>;
                        inverse_unnorm_impl = exec_pot_inverse_unnorm_64::<F>;
                    }
                    7 => {
                        forward_impl = exec_pot_forward_sized::<F, 7>;
                        inverse_impl = exec_pot_inverse_sized::<F, 7>;
                        inverse_unnorm_impl = exec_pot_inverse_unnorm_sized::<F, 7>;
                    }
                    8 => {
                        forward_impl = exec_pot_forward_sized::<F, 8>;
                        inverse_impl = exec_pot_inverse_sized::<F, 8>;
                        inverse_unnorm_impl = exec_pot_inverse_unnorm_sized::<F, 8>;
                    }
                    9 => {
                        forward_impl = exec_pot_forward_512::<F>;
                        inverse_impl = exec_pot_inverse_512::<F>;
                        inverse_unnorm_impl = exec_pot_inverse_unnorm_512::<F>;
                    }
                    10 => {
                        forward_impl = exec_pot_forward_sized::<F, 10>;
                        inverse_impl = exec_pot_inverse_sized::<F, 10>;
                        inverse_unnorm_impl = exec_pot_inverse_unnorm_sized::<F, 10>;
                    }
                    _ => {
                        forward_impl = exec_pot_forward_generic::<F>;
                        inverse_impl = exec_pot_inverse_generic::<F>;
                        inverse_unnorm_impl = exec_pot_inverse_unnorm_generic::<F>;
                    }
                }
            }
            &PlanStrategy::GoodThomas {
                n1: factor1,
                n2: factor2,
            } => {
                n1 = factor1;
                n2 = factor2;
                forward_impl = exec_good_thomas_forward::<F>;
                inverse_impl = exec_good_thomas_inverse::<F>;
                inverse_unnorm_impl = exec_good_thomas_inverse_unnorm::<F>;
            }
            PlanStrategy::Composite { radices } => {
                radices_field = Some(Cow::clone(radices));
                forward_impl = exec_composite_forward::<F>;
                inverse_impl = exec_composite_inverse::<F>;
                inverse_unnorm_impl = exec_composite_inverse_unnorm::<F>;
            }
            PlanStrategy::Rader => {
                forward_impl = exec_rader_forward::<F>;
                inverse_impl = exec_rader_inverse::<F>;
                inverse_unnorm_impl = exec_rader_inverse_unnorm::<F>;
            }
        }

        Self {
            n,
            #[cfg(test)]
            strategy,
            n1,
            n2,
            radices: radices_field,
            twiddle_fwd,
            twiddle_inv,
            forward_impl,
            inverse_impl,
            inverse_unnorm_impl,
        }
    }

    /// Return the plan length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Return whether the plan length is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
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
        self.inverse_complex_slice_unnorm_inplace(
            data.as_slice_mut().expect("Array must be contiguous"),
        );
    }

    /// Forward transform of a complex Leto view in-place.
    pub fn forward_complex_leto_inplace(&self, mut data: ArrayViewMut1<'_, F::Complex>) {
        self.forward_complex_slice_inplace(
            data.as_mut_slice_memory_order()
                .expect("Array must be contiguous"),
        );
    }

    /// Inverse transform of a complex Leto view in-place with normalization.
    pub fn inverse_complex_leto_inplace(&self, mut data: ArrayViewMut1<'_, F::Complex>) {
        self.inverse_complex_slice_inplace(
            data.as_mut_slice_memory_order()
                .expect("Array must be contiguous"),
        );
    }

    /// Inverse transform of a complex Leto view in-place without normalization.
    pub fn inverse_complex_leto_unnorm_inplace(&self, mut data: ArrayViewMut1<'_, F::Complex>) {
        self.inverse_complex_slice_unnorm_inplace(
            data.as_mut_slice_memory_order()
                .expect("Array must be contiguous"),
        );
    }

    /// Forward transform of a complex slice in-place.
    #[inline]
    pub fn forward_complex_slice_inplace(&self, slice: &mut [F::Complex]) {
        if runtime_tiny_direct_dispatch::<F, false, false>(self.n, slice) {
            return;
        }
        (self.forward_impl)(self, slice);
    }

    /// Inverse transform of a complex slice in-place with normalization.
    #[inline]
    pub fn inverse_complex_slice_inplace(&self, slice: &mut [F::Complex]) {
        if runtime_tiny_direct_dispatch::<F, true, true>(self.n, slice) {
            return;
        }
        (self.inverse_impl)(self, slice);
    }

    /// Inverse transform of a complex slice in-place without normalization.
    #[inline]
    pub fn inverse_complex_slice_unnorm_inplace(&self, slice: &mut [F::Complex]) {
        if runtime_tiny_direct_dispatch::<F, true, false>(self.n, slice) {
            return;
        }
        (self.inverse_unnorm_impl)(self, slice);
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
