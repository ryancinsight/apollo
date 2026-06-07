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

#[cfg(not(target_arch = "x86_64"))]
use crate::application::execution::kernel::pot::StockhamAutosort;
#[cfg(not(target_arch = "x86_64"))]
use crate::with_pot_zst;
use butterfly::{
    forward32_avx_with_scratch, forward32_avx_with_scratch_sized, forward64_avx_with_scratch,
    forward64_avx_with_scratch_sized,
};
use num_complex::{Complex32, Complex64};
use transform::transform_sized;
#[cfg(not(target_arch = "x86_64"))]
use transform::transform_with_strategy;

/// Dispatch to ZST-driven `transform_with_strategy` for hot LOG2 (5..=10),
/// with fallback to `transform_sized` for other values. Eliminates duplication
/// across `f64`/`f32` `StockhamKernel` impls.
///
/// Uses the shared `with_pot_zst!` macro from `pot` for ZST construction.
#[cfg(not(target_arch = "x86_64"))]
macro_rules! zst_stockham_dispatch {
    ($log2_val:expr, $precision:ty, $data:expr, $scratch:expr, $twiddles:expr) => {
        match $log2_val {
            5 => with_pot_zst!(5, _s, {
                transform_with_strategy::<StockhamAutosort, 5, $precision>(
                    _s, $data, $scratch, $twiddles, None,
                );
            }),
            6 => with_pot_zst!(6, _s, {
                transform_with_strategy::<StockhamAutosort, 6, $precision>(
                    _s, $data, $scratch, $twiddles, None,
                );
            }),
            7 => with_pot_zst!(7, _s, {
                transform_with_strategy::<StockhamAutosort, 7, $precision>(
                    _s, $data, $scratch, $twiddles, None,
                );
            }),
            8 => with_pot_zst!(8, _s, {
                transform_with_strategy::<StockhamAutosort, 8, $precision>(
                    _s, $data, $scratch, $twiddles, None,
                );
            }),
            9 => with_pot_zst!(9, _s, {
                transform_with_strategy::<StockhamAutosort, 9, $precision>(
                    _s, $data, $scratch, $twiddles, None,
                );
            }),
            10 => with_pot_zst!(10, _s, {
                transform_with_strategy::<StockhamAutosort, 10, $precision>(
                    _s, $data, $scratch, $twiddles, None,
                );
            }),
            _ => {
                transform_sized::<$precision>($data, $scratch, $twiddles, None, $log2_val);
            }
        }
    };
}

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

    /// Sized forward (const LOG2) for full monomorphization from plan PoT sized arms.
    /// Enables const LOG2 to flow to transform_sized / with_strategy / len* bodies,
    /// eliminating runtime trailing_zeros + match in hot PoT paths (elevates zero-cost
    /// monomorph + ZST threading). Body mirrors forward_with_scratch but uses const LOG2.
    fn forward_with_scratch_sized<const LOG2: u32>(
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
        #[cfg(target_arch = "x86_64")]
        {
            // If the size is one of our hand-optimized FMA/AVX fixed-length sizes,
            // route to the AVX/FMA path immediately if the CPU supports it,
            // bypassing the generic AVX-512 Stockham loop.
            if matches!(
                n,
                2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 4096 | 32768
            ) {
                #[cfg(all(target_feature = "avx", target_feature = "fma"))]
                {
                    unsafe { forward64_avx_with_scratch(data, scratch, twiddles) };
                    return;
                }
                #[cfg(not(all(target_feature = "avx", target_feature = "fma")))]
                {
                    if std::arch::is_x86_feature_detected!("avx")
                        && std::arch::is_x86_feature_detected!("fma")
                    {
                        unsafe { forward64_avx_with_scratch(data, scratch, twiddles) };
                        return;
                    }
                }
            }
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            let log2 = data.len().trailing_zeros() as u32;
            transform_sized::<precision::PreciseStockhamAvx512>(
                data, scratch, twiddles, None, log2,
            );
            return;
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                let log2 = data.len().trailing_zeros();
                transform_sized::<precision::PreciseStockhamAvx512>(
                    data, scratch, twiddles, None, log2,
                );
                return;
            }
            #[cfg(all(target_feature = "avx", target_feature = "fma"))]
            {
                unsafe { forward64_avx_with_scratch(data, scratch, twiddles) };
                return;
            }
            #[cfg(not(all(target_feature = "avx", target_feature = "fma")))]
            {
                if std::arch::is_x86_feature_detected!("avx")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    unsafe { forward64_avx_with_scratch(data, scratch, twiddles) };
                    return;
                }
                let log2 = data.len().trailing_zeros() as u32;
                transform_sized::<precision::PreciseStockham>(data, scratch, twiddles, None, log2);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let log2 = data.len().trailing_zeros() as u32;
            zst_stockham_dispatch!(log2, precision::PreciseStockham, data, scratch, twiddles);
        }
    }

    /// Sized variant (const LOG2). Mirrors the above but uses const LOG2 for transform
    /// calls and with_strategy arms (no runtime trailing_zeros). The const param + monomorph
    /// context from caller (pot_inplace_sized<LOG2> etc) allows DCE / direct len* selection.
    #[inline]
    fn forward_with_scratch_sized<const LOG2: u32>(
        data: &mut [Complex64],
        scratch: &mut [Complex64],
        twiddles: &[Complex64],
    ) {
        let n = 1usize << LOG2;
        debug_assert_eq!(scratch.len(), n, "stockham scratch length mismatch");
        debug_assert!(n.is_power_of_two());
        if n <= 1 {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if matches!(
                n,
                2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 4096 | 32768
            ) {
                #[cfg(all(target_feature = "avx", target_feature = "fma"))]
                {
                    unsafe { forward64_avx_with_scratch_sized::<LOG2>(data, scratch, twiddles) };
                    return;
                }
                #[cfg(not(all(target_feature = "avx", target_feature = "fma")))]
                {
                    if std::arch::is_x86_feature_detected!("avx")
                        && std::arch::is_x86_feature_detected!("fma")
                    {
                        unsafe {
                            forward64_avx_with_scratch_sized::<LOG2>(data, scratch, twiddles)
                        };
                        return;
                    }
                }
            }
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            transform_sized::<precision::PreciseStockhamAvx512>(
                data, scratch, twiddles, None, LOG2,
            );
            return;
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                transform_sized::<precision::PreciseStockhamAvx512>(
                    data, scratch, twiddles, None, LOG2,
                );
                return;
            }
            #[cfg(all(target_feature = "avx", target_feature = "fma"))]
            {
                unsafe { forward64_avx_with_scratch_sized::<LOG2>(data, scratch, twiddles) };
                return;
            }
            #[cfg(not(all(target_feature = "avx", target_feature = "fma")))]
            {
                if std::arch::is_x86_feature_detected!("avx")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    unsafe { forward64_avx_with_scratch_sized::<LOG2>(data, scratch, twiddles) };
                    return;
                }
                transform_sized::<precision::PreciseStockham>(data, scratch, twiddles, None, LOG2);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            zst_stockham_dispatch!(LOG2, precision::PreciseStockham, data, scratch, twiddles);
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
        #[cfg(target_arch = "x86_64")]
        {
            // If the size is one of our hand-optimized FMA/AVX fixed-length sizes,
            // route to the AVX/FMA path immediately if the CPU supports it,
            // bypassing the generic AVX-512 Stockham loop.
            if matches!(
                n,
                2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 4096 | 32768
            ) {
                #[cfg(all(target_feature = "avx", target_feature = "fma"))]
                {
                    unsafe { forward32_avx_with_scratch(data, scratch, twiddles) };
                    return;
                }
                #[cfg(not(all(target_feature = "avx", target_feature = "fma")))]
                {
                    if std::arch::is_x86_feature_detected!("avx")
                        && std::arch::is_x86_feature_detected!("fma")
                    {
                        unsafe { forward32_avx_with_scratch(data, scratch, twiddles) };
                        return;
                    }
                }
            }
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            let log2 = data.len().trailing_zeros() as u32;
            transform_sized::<precision::ReducedStockhamAvx512>(
                data, scratch, twiddles, None, log2,
            );
            return;
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                let log2 = data.len().trailing_zeros();
                transform_sized::<precision::ReducedStockhamAvx512>(
                    data, scratch, twiddles, None, log2,
                );
                return;
            }
            #[cfg(all(target_feature = "avx", target_feature = "fma"))]
            {
                unsafe { forward32_avx_with_scratch(data, scratch, twiddles) };
                return;
            }
            #[cfg(not(all(target_feature = "avx", target_feature = "fma")))]
            {
                if std::arch::is_x86_feature_detected!("avx")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    unsafe { forward32_avx_with_scratch(data, scratch, twiddles) };
                    return;
                }
                let log2 = data.len().trailing_zeros() as u32;
                transform_sized::<precision::ReducedStockham>(data, scratch, twiddles, None, log2);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let log2 = data.len().trailing_zeros() as u32;
            zst_stockham_dispatch!(log2, precision::ReducedStockham, data, scratch, twiddles);
        }
    }

    /// Sized variant (const LOG2) for f32. Mirrors forward_with_scratch but uses
    /// const LOG2 for transform_sized calls, enabling const propagation from plan
    /// pot_inplace_sized<LOG2> callers into monomorphized Stockham bodies.
    #[inline]
    fn forward_with_scratch_sized<const LOG2: u32>(
        data: &mut [Complex32],
        scratch: &mut [Complex32],
        twiddles: &[Complex32],
    ) {
        let n = 1usize << LOG2;
        debug_assert_eq!(scratch.len(), n, "stockham scratch length mismatch");
        debug_assert!(n.is_power_of_two());
        if n <= 1 {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if matches!(
                n,
                2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 4096 | 32768
            ) {
                #[cfg(all(target_feature = "avx", target_feature = "fma"))]
                {
                    unsafe { forward32_avx_with_scratch_sized::<LOG2>(data, scratch, twiddles) };
                    return;
                }
                #[cfg(not(all(target_feature = "avx", target_feature = "fma")))]
                {
                    if std::arch::is_x86_feature_detected!("avx")
                        && std::arch::is_x86_feature_detected!("fma")
                    {
                        unsafe {
                            forward32_avx_with_scratch_sized::<LOG2>(data, scratch, twiddles)
                        };
                        return;
                    }
                }
            }
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            transform_sized::<precision::ReducedStockhamAvx512>(
                data, scratch, twiddles, None, LOG2,
            );
            return;
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                transform_sized::<precision::ReducedStockhamAvx512>(
                    data, scratch, twiddles, None, LOG2,
                );
                return;
            }
            #[cfg(all(target_feature = "avx", target_feature = "fma"))]
            {
                unsafe { forward32_avx_with_scratch_sized::<LOG2>(data, scratch, twiddles) };
                return;
            }
            #[cfg(not(all(target_feature = "avx", target_feature = "fma")))]
            {
                if std::arch::is_x86_feature_detected!("avx")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    unsafe { forward32_avx_with_scratch_sized::<LOG2>(data, scratch, twiddles) };
                    return;
                }
                transform_sized::<precision::ReducedStockham>(data, scratch, twiddles, None, LOG2);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            transform_sized::<precision::ReducedStockham>(data, scratch, twiddles, None, LOG2);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
