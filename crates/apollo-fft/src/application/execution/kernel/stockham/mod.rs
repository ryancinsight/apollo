//! Natural-order radix-2 FFT execution through Stockham autosort.
//!
//! Computes power-of-two FFTs through a Stockham autosort pass over a
//! separate scratch buffer, writing the final spectrum in natural order.
//! No standalone reordering pass is executed before or after the butterflies.
//!
//! # Module layout
//!
//! ```text
//! stockham/
//!   mod.rs          — StockhamKernel public trait + impls, module re-exports
//!   stage.rs        — scalar stage_impl<C>, L1 residency helpers
//!   avx/            — x86_64 AVX/FMA kernels for f64 and f32
//!   precision.rs    — StockhamPrecision, StockhamFusion traits + impls
//!   transform.rs    — transform<P>, transform_len4096_four_triples<P>
//!   butterfly.rs    — packed types, build_butterfly512, fixed-len kernels
//! ```

#![allow(clippy::many_single_char_names)]
#![allow(clippy::empty_line_after_doc_comments)]

pub(crate) mod avx;
pub(crate) mod butterfly;
pub(crate) mod precision;
pub(crate) mod stage;
pub(crate) mod transform;

use butterfly::{forward64_avx_with_scratch, forward32_avx_with_scratch};
use num_complex::{Complex32, Complex64};

pub(crate) trait StockhamKernel: Sized {
    type Complex;

    /// Forward radix-2 Stockham FFT into natural order using caller-provided scratch.
    ///
    /// `data` and `scratch` must have the same length (a power of two).
    /// `twiddles` must be the output of the matching `build_forward_twiddle_table_*` call.
    fn forward_with_scratch(
        data: &mut [Self::Complex],
        scratch: &mut [Self::Complex],
        twiddles: &[Self::Complex],
    );
}

impl StockhamKernel for f64 {
    type Complex = Complex64;

    #[inline]
    fn forward_with_scratch(
        data: &mut [Complex64],
        scratch: &mut [Complex64],
        twiddles: &[Complex64],
    ) {
        let n = data.len();
        debug_assert_eq!(scratch.len(), n, "stockham scratch length mismatch");
        debug_assert!(n.is_power_of_two());
        if n <= 1 {
            return;
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
        {
            unsafe { forward64_avx_with_scratch(data, scratch, twiddles) };
            return;
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")))]
        {
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx")
                && std::arch::is_x86_feature_detected!("fma")
            {
                unsafe { forward64_avx_with_scratch(data, scratch, twiddles) };
                return;
            }
            transform::<F64Stockham>(data, scratch, twiddles, None);
        }
    }
}

impl StockhamKernel for f32 {
    type Complex = Complex32;

    #[inline]
    fn forward_with_scratch(
        data: &mut [Complex32],
        scratch: &mut [Complex32],
        twiddles: &[Complex32],
    ) {
        let n = data.len();
        debug_assert_eq!(scratch.len(), n, "stockham scratch length mismatch");
        debug_assert!(n.is_power_of_two());
        if n <= 1 {
            return;
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
        {
            unsafe { forward32_avx_with_scratch(data, scratch, twiddles) };
            return;
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")))]
        {
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx")
                && std::arch::is_x86_feature_detected!("fma")
            {
                unsafe { forward32_avx_with_scratch(data, scratch, twiddles) };
                return;
            }
            transform::<F32Stockham>(data, scratch, twiddles, None);
        }
    }
}


#[cfg(test)]
pub(crate) mod tests;
