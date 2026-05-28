//! 1D FFT plan.
//!
//! Apollo-owned 1D FFT implementation based on `MixedRadixScalar`.

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use crate::application::execution::kernel::mixed_radix::traits::ShortDft;
use crate::domain::metadata::shape::Shape1D;
use ndarray::Array1;
use num_complex::Complex;
use std::sync::Arc;

/// Reusable 1D FFT plan strategy generic over `MixedRadixScalar`.
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
    strategy: PlanStrategy<F>,
    
    // Cached variables to completely bypass enum matching in the hot path:
    n1: usize,
    n2: usize,
    input_perm: Option<Arc<[usize]>>,
    output_perm: Option<Arc<[usize]>>,
    sub_plan1: Option<Arc<FftPlan1D<F>>>,
    sub_plan2: Option<Arc<FftPlan1D<F>>>,
    radices: Option<Arc<[usize]>>,
    twiddle_fwd: Option<Arc<[F::Complex]>>,
    twiddle_inv: Option<Arc<[F::Complex]>>,
    
    // Rader-specific cached variables:
    rader_gather: Option<Arc<[usize]>>,
    rader_kernel_fwd: Option<Arc<[F::Complex]>>,
    rader_kernel_inv: Option<Arc<[F::Complex]>>,
    
    // Function pointers for execution routing:
    forward_impl: fn(&Self, &mut [F::Complex]),
    inverse_impl: fn(&Self, &mut [F::Complex]),
    inverse_unnorm_impl: fn(&Self, &mut [F::Complex]),
}

impl<F: MixedRadixScalar> Clone for FftPlan1D<F> {
    fn clone(&self) -> Self {
        Self {
            n: self.n,
            strategy: self.strategy.clone(),
            n1: self.n1,
            n2: self.n2,
            input_perm: self.input_perm.clone(),
            output_perm: self.output_perm.clone(),
            sub_plan1: self.sub_plan1.clone(),
            sub_plan2: self.sub_plan2.clone(),
            radices: self.radices.clone(),
            twiddle_fwd: self.twiddle_fwd.clone(),
            twiddle_inv: self.twiddle_inv.clone(),
            rader_gather: self.rader_gather.clone(),
            rader_kernel_fwd: self.rader_kernel_fwd.clone(),
            rader_kernel_inv: self.rader_kernel_inv.clone(),
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
        let strategy = if n <= 1 {
            PlanStrategy::Identity
        } else if n.is_power_of_two() {
            PlanStrategy::PowerOfTwo {
                twiddle_fwd: F::cached_twiddle_fwd(n),
                twiddle_inv: F::cached_twiddle_inv(n),
            }
        } else if crate::application::execution::kernel::mixed_radix::traits::is_short_winograd_size(n) {
            PlanStrategy::ShortWinograd
        } else if let Some((n1, n2)) = crate::application::execution::kernel::mixed_radix::caches::cached_coprime_factors(n)
            .filter(|&(n1, n2)| crate::application::execution::kernel::components::good_thomas::has_static_coprime_codelet(n1, n2))
        {
            PlanStrategy::GoodThomas { n1, n2 }
        } else if let Some(radices) = crate::application::execution::kernel::mixed_radix::caches::cached_prime23_radices(n) {
            PlanStrategy::Composite { radices }
        } else if let Some((n1, n2)) = crate::application::execution::kernel::mixed_radix::caches::cached_coprime_factors(n) {
            PlanStrategy::GoodThomas { n1, n2 }
        } else {
            PlanStrategy::Rader
        };

        // Cache parameters and assign specialized function pointers
        let mut n1 = 0;
        let mut n2 = 0;
        let mut input_perm = None;
        let mut output_perm = None;
        let mut sub_plan1 = None;
        let mut sub_plan2 = None;
        let mut radices_field = None;
        let mut twiddle_fwd = None;
        let mut twiddle_inv = None;
        let mut rader_gather = None;
        let mut rader_kernel_fwd = None;
        let mut rader_kernel_inv = None;

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
                // List all supported winograd sizes:
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
                assign_winograd!(81);
                assign_winograd!(128);
            }
            PlanStrategy::PowerOfTwo { twiddle_fwd: fwd, twiddle_inv: inv } => {
                twiddle_fwd = Some(fwd.clone());
                twiddle_inv = Some(inv.clone());
                match n {
                    2 => {
                        forward_impl = exec_pot_forward_2::<F>;
                        inverse_impl = exec_pot_inverse_2::<F>;
                        inverse_unnorm_impl = exec_pot_inverse_unnorm_2::<F>;
                    }
                    4 => {
                        forward_impl = exec_pot_forward_4::<F>;
                        inverse_impl = exec_pot_inverse_4::<F>;
                        inverse_unnorm_impl = exec_pot_inverse_unnorm_4::<F>;
                    }
                    8 => {
                        forward_impl = exec_pot_forward_8::<F>;
                        inverse_impl = exec_pot_inverse_8::<F>;
                        inverse_unnorm_impl = exec_pot_inverse_unnorm_8::<F>;
                    }
                    16 => {
                        forward_impl = exec_pot_forward_16::<F>;
                        inverse_impl = exec_pot_inverse_16::<F>;
                        inverse_unnorm_impl = exec_pot_inverse_unnorm_16::<F>;
                    }
                    32 => {
                        forward_impl = exec_pot_forward_32::<F>;
                        inverse_impl = exec_pot_inverse_32::<F>;
                        inverse_unnorm_impl = exec_pot_inverse_unnorm_32::<F>;
                    }
                    64 => {
                        forward_impl = exec_pot_forward_64::<F>;
                        inverse_impl = exec_pot_inverse_64::<F>;
                        inverse_unnorm_impl = exec_pot_inverse_unnorm_64::<F>;
                    }
                    _ => {
                        forward_impl = exec_pot_forward_generic::<F>;
                        inverse_impl = exec_pot_inverse_generic::<F>;
                        inverse_unnorm_impl = exec_pot_inverse_unnorm_generic::<F>;
                    }
                }
            }
            &PlanStrategy::GoodThomas { n1: factor1, n2: factor2 } => {
                n1 = factor1;
                n2 = factor2;
                let (ip, op) = crate::application::execution::kernel::mixed_radix::caches::cached_pfa_perm(n1, n2);
                input_perm = Some(ip);
                output_perm = Some(op);
                sub_plan1 = Some(Arc::new(FftPlan1D::new(Shape1D::new(n1).unwrap())));
                sub_plan2 = Some(Arc::new(FftPlan1D::new(Shape1D::new(n2).unwrap())));

                forward_impl = exec_good_thomas_forward::<F>;
                inverse_impl = exec_good_thomas_inverse::<F>;
                inverse_unnorm_impl = exec_good_thomas_inverse_unnorm::<F>;
            }
            PlanStrategy::Composite { radices } => {
                radices_field = Some(radices.clone());
                forward_impl = exec_composite_forward::<F>;
                inverse_impl = exec_composite_inverse::<F>;
                inverse_unnorm_impl = exec_composite_inverse_unnorm::<F>;
            }
            PlanStrategy::Rader => {
                let (g, g_inv) = crate::application::execution::kernel::components::rader::generator::primitive_root_and_inverse(n);
                let gather = crate::application::execution::kernel::components::rader::cached_generator_order(n, g);
                rader_gather = Some(gather);
                
                let k_fwd = F::cached_rader_spectrum(n, false, g_inv);
                let k_inv = F::cached_rader_spectrum(n, true, g_inv);
                rader_kernel_fwd = Some(k_fwd);
                rader_kernel_inv = Some(k_inv);
                
                sub_plan1 = Some(Arc::new(FftPlan1D::new(Shape1D::new(n - 1).unwrap())));

                forward_impl = exec_rader_forward::<F>;
                inverse_impl = exec_rader_inverse::<F>;
                inverse_unnorm_impl = exec_rader_inverse_unnorm::<F>;
            }
        }

        Self {
            n,
            strategy,
            n1,
            n2,
            input_perm,
            output_perm,
            sub_plan1,
            sub_plan2,
            radices: radices_field,
            twiddle_fwd,
            twiddle_inv,
            rader_gather,
            rader_kernel_fwd,
            rader_kernel_inv,
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
    #[inline(always)]
    pub fn forward_complex_slice_inplace(&self, slice: &mut [F::Complex]) {
        (self.forward_impl)(self, slice);
    }

    /// Inverse transform of a complex slice in-place with normalization.
    #[inline(always)]
    pub fn inverse_complex_slice_inplace(&self, slice: &mut [F::Complex]) {
        (self.inverse_impl)(self, slice);
    }

    /// Inverse transform of a complex slice in-place without normalization.
    #[inline(always)]
    pub fn inverse_complex_slice_unnorm_inplace(&self, slice: &mut [F::Complex]) {
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

// ── Specialized Executors ───────────────────────────────────────────────────

// 1. Identity
fn exec_identity<F: MixedRadixScalar>(_: &FftPlan1D<F>, _: &mut [F::Complex]) {}

// 2. ShortWinograd
fn exec_winograd_forward<F: MixedRadixScalar<Complex = Complex<F>>, const N: usize>(_: &FftPlan1D<F>, slice: &mut [F::Complex])
where
    F: ShortDft<N>,
{
    if let Ok(data) = slice.try_into() {
        F::dft::<false>(data);
    }
}

fn exec_winograd_inverse<F: MixedRadixScalar<Complex = Complex<F>>, const N: usize>(_: &FftPlan1D<F>, slice: &mut [F::Complex])
where
    F: ShortDft<N>,
{
    if let Ok(data) = slice.try_into() {
        F::dft::<true>(data);
        F::normalize(slice, N);
    }
}

fn exec_winograd_inverse_unnorm<F: MixedRadixScalar<Complex = Complex<F>>, const N: usize>(_: &FftPlan1D<F>, slice: &mut [F::Complex])
where
    F: ShortDft<N>,
{
    if let Ok(data) = slice.try_into() {
        F::dft::<true>(data);
    }
}

// 3. PowerOfTwo small sizes
macro_rules! define_pot_executors {
    ($size:expr, $fwd:ident, $inv:ident, $inv_un:ident) => {
        fn $fwd<F: MixedRadixScalar<Complex = Complex<F>>>(_: &FftPlan1D<F>, slice: &mut [F::Complex]) {
            unsafe { F::small_pot_inplace_sized::<$size, false, false>(slice); }
        }
        fn $inv<F: MixedRadixScalar<Complex = Complex<F>>>(_: &FftPlan1D<F>, slice: &mut [F::Complex]) {
            unsafe { F::small_pot_inplace_sized::<$size, true, true>(slice); }
        }
        fn $inv_un<F: MixedRadixScalar<Complex = Complex<F>>>(_: &FftPlan1D<F>, slice: &mut [F::Complex]) {
            unsafe { F::small_pot_inplace_sized::<$size, true, false>(slice); }
        }
    };
}
define_pot_executors!(2, exec_pot_forward_2, exec_pot_inverse_2, exec_pot_inverse_unnorm_2);
define_pot_executors!(4, exec_pot_forward_4, exec_pot_inverse_4, exec_pot_inverse_unnorm_4);
define_pot_executors!(8, exec_pot_forward_8, exec_pot_inverse_8, exec_pot_inverse_unnorm_8);
define_pot_executors!(16, exec_pot_forward_16, exec_pot_inverse_16, exec_pot_inverse_unnorm_16);
define_pot_executors!(32, exec_pot_forward_32, exec_pot_inverse_32, exec_pot_inverse_unnorm_32);
define_pot_executors!(64, exec_pot_forward_64, exec_pot_inverse_64, exec_pot_inverse_unnorm_64);

// 4. PowerOfTwo generic sizes (using cached twiddles)
fn exec_pot_forward_generic<F: MixedRadixScalar<Complex = Complex<F>>>(plan: &FftPlan1D<F>, slice: &mut [F::Complex]) {
    if let Some(tw) = &plan.twiddle_fwd {
        F::pot_inplace::<false, false>(slice, tw);
    }
}
fn exec_pot_inverse_generic<F: MixedRadixScalar<Complex = Complex<F>>>(plan: &FftPlan1D<F>, slice: &mut [F::Complex]) {
    if let Some(tw) = &plan.twiddle_inv {
        F::pot_inplace::<true, true>(slice, tw);
    }
}
fn exec_pot_inverse_unnorm_generic<F: MixedRadixScalar<Complex = Complex<F>>>(plan: &FftPlan1D<F>, slice: &mut [F::Complex]) {
    if let Some(tw) = &plan.twiddle_inv {
        F::pot_inplace::<true, false>(slice, tw);
    }
}

// 5. Good-Thomas (Static or Generic)
fn exec_good_thomas_forward<F: MixedRadixScalar<Complex = Complex<F>>>(plan: &FftPlan1D<F>, slice: &mut [F::Complex]) {
    let n = plan.n;
    let n1 = plan.n1;
    let n2 = plan.n2;
    let input_perm = plan.input_perm.as_ref().unwrap();
    let output_perm = plan.output_perm.as_ref().unwrap();
    let sub1 = plan.sub_plan1.as_ref().unwrap();
    let sub2 = plan.sub_plan2.as_ref().unwrap();

    F::with_pfa_scratch(n + n1, |scratch| {
        let (matrix, col_buf) = scratch.split_at_mut(n);

        // Gather out-of-place into matrix
        let n4 = (n / 4) * 4;
        let mut j = 0usize;
        while j < n4 {
            unsafe {
                *matrix.get_unchecked_mut(j) = *slice.get_unchecked(*input_perm.get_unchecked(j));
                *matrix.get_unchecked_mut(j + 1) = *slice.get_unchecked(*input_perm.get_unchecked(j + 1));
                *matrix.get_unchecked_mut(j + 2) = *slice.get_unchecked(*input_perm.get_unchecked(j + 2));
                *matrix.get_unchecked_mut(j + 3) = *slice.get_unchecked(*input_perm.get_unchecked(j + 3));
            }
            j += 4;
        }
        while j < n {
            unsafe {
                *matrix.get_unchecked_mut(j) = *slice.get_unchecked(*input_perm.get_unchecked(j));
            }
            j += 1;
        }

        // Transform rows (size n2) using sub2
        for i1 in 0..n1 {
            let row_start = i1 * n2;
            let row_slice = &mut matrix[row_start..row_start + n2];
            sub2.forward_complex_slice_inplace(row_slice);
        }

        // Transform columns (size n1) using sub1
        for i2 in 0..n2 {
            unsafe {
                *col_buf.get_unchecked_mut(0) = *matrix.get_unchecked(i2);
                *col_buf.get_unchecked_mut(1) = *matrix.get_unchecked(n2 + i2);
                if n1 > 2 {
                    *col_buf.get_unchecked_mut(2) = *matrix.get_unchecked(2 * n2 + i2);
                }
                if n1 > 3 {
                    *col_buf.get_unchecked_mut(3) = *matrix.get_unchecked(3 * n2 + i2);
                }
                for i1 in 4..n1 {
                    *col_buf.get_unchecked_mut(i1) = *matrix.get_unchecked(i1 * n2 + i2);
                }
            }

            sub1.forward_complex_slice_inplace(col_buf);

            // Scatter directly to slice
            for i1 in 0..n1 {
                unsafe {
                    let out_idx = *output_perm.get_unchecked(i2 * n1 + i1);
                    *slice.get_unchecked_mut(out_idx) = *col_buf.get_unchecked(i1);
                }
            }
        }
    });
}

fn exec_good_thomas_inverse<F: MixedRadixScalar<Complex = Complex<F>>>(plan: &FftPlan1D<F>, slice: &mut [F::Complex]) {
    let n = plan.n;
    let n1 = plan.n1;
    let n2 = plan.n2;
    let input_perm = plan.input_perm.as_ref().unwrap();
    let output_perm = plan.output_perm.as_ref().unwrap();
    let sub1 = plan.sub_plan1.as_ref().unwrap();
    let sub2 = plan.sub_plan2.as_ref().unwrap();

    F::with_pfa_scratch(n + n1, |scratch| {
        let (matrix, col_buf) = scratch.split_at_mut(n);

        // Gather out-of-place into matrix
        let n4 = (n / 4) * 4;
        let mut j = 0usize;
        while j < n4 {
            unsafe {
                *matrix.get_unchecked_mut(j) = *slice.get_unchecked(*input_perm.get_unchecked(j));
                *matrix.get_unchecked_mut(j + 1) = *slice.get_unchecked(*input_perm.get_unchecked(j + 1));
                *matrix.get_unchecked_mut(j + 2) = *slice.get_unchecked(*input_perm.get_unchecked(j + 2));
                *matrix.get_unchecked_mut(j + 3) = *slice.get_unchecked(*input_perm.get_unchecked(j + 3));
            }
            j += 4;
        }
        while j < n {
            unsafe {
                *matrix.get_unchecked_mut(j) = *slice.get_unchecked(*input_perm.get_unchecked(j));
            }
            j += 1;
        }

        // Transform rows (size n2) using sub2
        for i1 in 0..n1 {
            let row_start = i1 * n2;
            let row_slice = &mut matrix[row_start..row_start + n2];
            sub2.inverse_complex_slice_unnorm_inplace(row_slice);
        }

        // Transform columns (size n1) using sub1
        for i2 in 0..n2 {
            unsafe {
                *col_buf.get_unchecked_mut(0) = *matrix.get_unchecked(i2);
                *col_buf.get_unchecked_mut(1) = *matrix.get_unchecked(n2 + i2);
                if n1 > 2 {
                    *col_buf.get_unchecked_mut(2) = *matrix.get_unchecked(2 * n2 + i2);
                }
                if n1 > 3 {
                    *col_buf.get_unchecked_mut(3) = *matrix.get_unchecked(3 * n2 + i2);
                }
                for i1 in 4..n1 {
                    *col_buf.get_unchecked_mut(i1) = *matrix.get_unchecked(i1 * n2 + i2);
                }
            }

            sub1.inverse_complex_slice_unnorm_inplace(col_buf);

            // Scatter directly to slice
            for i1 in 0..n1 {
                unsafe {
                    let out_idx = *output_perm.get_unchecked(i2 * n1 + i1);
                    *slice.get_unchecked_mut(out_idx) = *col_buf.get_unchecked(i1);
                }
            }
        }
    });
    F::normalize(slice, n);
}

fn exec_good_thomas_inverse_unnorm<F: MixedRadixScalar<Complex = Complex<F>>>(plan: &FftPlan1D<F>, slice: &mut [F::Complex]) {
    let n = plan.n;
    let n1 = plan.n1;
    let n2 = plan.n2;
    let input_perm = plan.input_perm.as_ref().unwrap();
    let output_perm = plan.output_perm.as_ref().unwrap();
    let sub1 = plan.sub_plan1.as_ref().unwrap();
    let sub2 = plan.sub_plan2.as_ref().unwrap();

    F::with_pfa_scratch(n + n1, |scratch| {
        let (matrix, col_buf) = scratch.split_at_mut(n);

        // Gather out-of-place into matrix
        let n4 = (n / 4) * 4;
        let mut j = 0usize;
        while j < n4 {
            unsafe {
                *matrix.get_unchecked_mut(j) = *slice.get_unchecked(*input_perm.get_unchecked(j));
                *matrix.get_unchecked_mut(j + 1) = *slice.get_unchecked(*input_perm.get_unchecked(j + 1));
                *matrix.get_unchecked_mut(j + 2) = *slice.get_unchecked(*input_perm.get_unchecked(j + 2));
                *matrix.get_unchecked_mut(j + 3) = *slice.get_unchecked(*input_perm.get_unchecked(j + 3));
            }
            j += 4;
        }
        while j < n {
            unsafe {
                *matrix.get_unchecked_mut(j) = *slice.get_unchecked(*input_perm.get_unchecked(j));
            }
            j += 1;
        }

        // Transform rows (size n2) using sub2
        for i1 in 0..n1 {
            let row_start = i1 * n2;
            let row_slice = &mut matrix[row_start..row_start + n2];
            sub2.inverse_complex_slice_unnorm_inplace(row_slice);
        }

        // Transform columns (size n1) using sub1
        for i2 in 0..n2 {
            unsafe {
                *col_buf.get_unchecked_mut(0) = *matrix.get_unchecked(i2);
                *col_buf.get_unchecked_mut(1) = *matrix.get_unchecked(n2 + i2);
                if n1 > 2 {
                    *col_buf.get_unchecked_mut(2) = *matrix.get_unchecked(2 * n2 + i2);
                }
                if n1 > 3 {
                    *col_buf.get_unchecked_mut(3) = *matrix.get_unchecked(3 * n2 + i2);
                }
                for i1 in 4..n1 {
                    *col_buf.get_unchecked_mut(i1) = *matrix.get_unchecked(i1 * n2 + i2);
                }
            }

            sub1.inverse_complex_slice_unnorm_inplace(col_buf);

            // Scatter directly to slice
            for i1 in 0..n1 {
                unsafe {
                    let out_idx = *output_perm.get_unchecked(i2 * n1 + i1);
                    *slice.get_unchecked_mut(out_idx) = *col_buf.get_unchecked(i1);
                }
            }
        }
    });
}

// 6. Composite
fn exec_composite_forward<F: MixedRadixScalar<Complex = Complex<F>>>(plan: &FftPlan1D<F>, slice: &mut [F::Complex]) {
    if let Some(radices) = &plan.radices {
        F::composite_forward(slice, radices);
    }
}
fn exec_composite_inverse<F: MixedRadixScalar<Complex = Complex<F>>>(plan: &FftPlan1D<F>, slice: &mut [F::Complex]) {
    if let Some(radices) = &plan.radices {
        F::composite_inverse(slice, radices);
    }
}
fn exec_composite_inverse_unnorm<F: MixedRadixScalar<Complex = Complex<F>>>(plan: &FftPlan1D<F>, slice: &mut [F::Complex]) {
    if let Some(radices) = &plan.radices {
        F::composite_inverse_unnorm(slice, radices);
    }
}

// 7. Rader
fn exec_rader_forward<F: MixedRadixScalar<Complex = Complex<F>>>(plan: &FftPlan1D<F>, slice: &mut [F::Complex]) {
    let n = plan.n;
    if crate::application::execution::kernel::components::rader::static_rader::try_static_rader::<F, false>(slice, n) {
        return;
    }
    let gather = plan.rader_gather.as_ref().unwrap();
    let kernel_fwd = plan.rader_kernel_fwd.as_ref().unwrap();
    let sub = plan.sub_plan1.as_ref().unwrap();
    let l = n - 1;
    let x0 = slice[0];

    F::with_rader_padded_scratch(l, |padded| {
        let sum_x = crate::application::execution::kernel::components::rader::gather_sum_slice::<F>(slice, padded, gather);
        sub.forward_complex_slice_inplace(padded);
        F::pointwise_mul(padded, kernel_fwd);
        sub.inverse_complex_slice_inplace(padded);

        slice[0] = x0 + sum_x;
        crate::application::execution::kernel::components::rader::scatter_slice::<F>(slice, padded, x0, gather);
    });
}

fn exec_rader_inverse<F: MixedRadixScalar<Complex = Complex<F>>>(plan: &FftPlan1D<F>, slice: &mut [F::Complex]) {
    let n = plan.n;
    if crate::application::execution::kernel::components::rader::static_rader::try_static_rader::<F, true>(slice, n) {
        F::normalize(slice, n);
        return;
    }
    let gather = plan.rader_gather.as_ref().unwrap();
    let kernel_inv = plan.rader_kernel_inv.as_ref().unwrap();
    let sub = plan.sub_plan1.as_ref().unwrap();
    let l = n - 1;
    let x0 = slice[0];

    F::with_rader_padded_scratch(l, |padded| {
        let sum_x = crate::application::execution::kernel::components::rader::gather_sum_slice::<F>(slice, padded, gather);
        sub.forward_complex_slice_inplace(padded);
        F::pointwise_mul(padded, kernel_inv);
        sub.inverse_complex_slice_inplace(padded);

        slice[0] = x0 + sum_x;
        crate::application::execution::kernel::components::rader::scatter_slice::<F>(slice, padded, x0, gather);
    });
    F::normalize(slice, n);
}

fn exec_rader_inverse_unnorm<F: MixedRadixScalar<Complex = Complex<F>>>(plan: &FftPlan1D<F>, slice: &mut [F::Complex]) {
    let n = plan.n;
    if crate::application::execution::kernel::components::rader::static_rader::try_static_rader::<F, true>(slice, n) {
        return;
    }
    let gather = plan.rader_gather.as_ref().unwrap();
    let kernel_inv = plan.rader_kernel_inv.as_ref().unwrap();
    let sub = plan.sub_plan1.as_ref().unwrap();
    let l = n - 1;
    let x0 = slice[0];

    F::with_rader_padded_scratch(l, |padded| {
        let sum_x = crate::application::execution::kernel::components::rader::gather_sum_slice::<F>(slice, padded, gather);
        sub.forward_complex_slice_inplace(padded);
        F::pointwise_mul(padded, kernel_inv);
        sub.inverse_complex_slice_inplace(padded);

        slice[0] = x0 + sum_x;
        crate::application::execution::kernel::components::rader::scatter_slice::<F>(slice, padded, x0, gather);
    });
}
