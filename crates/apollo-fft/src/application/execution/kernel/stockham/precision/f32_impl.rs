#![allow(clippy::wildcard_imports)]
use super::super::avx::*;
use super::super::butterfly::{stage_pair_impl, stage_quad_impl, stage_triple_impl};
use super::super::stage::stage_impl;
use super::super::stage::stockham_f32_stage_is_l1_resident;
use super::traits::*;
use crate::application::execution::kernel::radix_stage::normalize_inplace_c32;
use num_complex::Complex32;

#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
impl StockhamPrecision for F32Stockham {
    type Real = f32;
    type Complex = Complex32;

    const MAX_FUSED_STAGES: u32 = 4;

    #[inline]
    fn stage_triple_enabled(stride: usize, n: usize, input_is_data: bool) -> bool {
        let _ = (stride, n, input_is_data);
        false
    }

    #[inline]
    fn stage(src: &[Complex32], dst: &mut [Complex32], radix: usize, twiddles: &[Complex32]) {
        stage_impl(src, dst, radix, twiddles);
    }

    #[inline]
    fn stage_pair(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
    ) {
        stage_pair_impl(src, dst, radix, first_twiddles, second_twiddles);
    }

    #[inline]
    fn stage_triple(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
        third_twiddles: &[Complex32],
    ) {
        stage_triple_impl(
            src,
            dst,
            radix,
            first_twiddles,
            second_twiddles,
            third_twiddles,
        );
    }

    #[inline]
    fn stage_quad(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
        third_twiddles: &[Complex32],
        fourth_twiddles: &[Complex32],
    ) {
        stage_quad_impl(
            src,
            dst,
            radix,
            first_twiddles,
            second_twiddles,
            third_twiddles,
            fourth_twiddles,
        );
    }

    #[inline]
    fn scale(data: &mut [Complex32], scale: f32) {
        normalize_inplace_c32(data, scale);
    }
}
#[cfg(target_arch = "x86_64")]
pub(crate) struct F32StockhamAvxFma;

#[cfg(target_arch = "x86_64")]
impl private::Sealed for F32StockhamAvxFma {}

#[cfg(target_arch = "x86_64")]
impl StockhamRadix16AvxLeaf for F32StockhamAvxFma {
    #[inline]
    unsafe fn stage_quad_groups_eight_avx_fma(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
        third_twiddles: &[Complex32],
        fourth_twiddles: &[Complex32],
    ) {
        unsafe {
            stockham_quad_groups_eight32(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
                third_twiddles,
                fourth_twiddles,
            )
        };
    }
}

#[cfg(target_arch = "x86_64")]
impl StockhamPrecision for F32StockhamAvxFma {
    type Real = f32;
    type Complex = Complex32;

    const MAX_FUSED_STAGES: u32 = 4;

    #[inline]
    fn stage_triple_enabled(stride: usize, n: usize, input_is_data: bool) -> bool {
        let groups = n / (stride << 1);
        groups > 4 || (groups == 4 && !input_is_data)
    }

    #[inline]
    fn stage_quad_enabled(stride: usize, n: usize, input_is_data: bool) -> bool {
        let groups = n / (stride << 1);
        let _ = (stride, groups, n, input_is_data);
        false
    }

    #[inline]
    fn stage(src: &[Complex32], dst: &mut [Complex32], radix: usize, twiddles: &[Complex32]) {
        let groups = src.len() / (radix << 1);
        if groups == 1 && radix >= 2 {
            unsafe { stage32_groups_one_avx_fma(src, dst, radix, twiddles) };
        } else if groups >= 4 {
            unsafe { stage32_avx_fma(src, dst, radix, twiddles) };
        } else {
            stage_impl(src, dst, radix, twiddles);
        }
    }

    #[inline]
    fn stage_pair(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
    ) {
        let groups = src.len() / (radix << 1);
        if radix == 1 {
            if src.len() >= 16 {
                unsafe { stage_pair32_radix1_avx_fma(src, dst, second_twiddles) };
            } else {
                stage_pair_impl(src, dst, radix, first_twiddles, second_twiddles);
            }
        } else if groups >= 8 {
            unsafe { stage_pair32_avx_fma(src, dst, radix, first_twiddles, second_twiddles) };
        } else if groups == 4 {
            unsafe {
                stage_pair32_quarter_groups_two_avx_fma(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                )
            };
        } else if groups == 2 {
            unsafe {
                stage_pair32_groups_two_avx_fma(src, dst, radix, first_twiddles, second_twiddles)
            };
        } else {
            stage_pair_impl(src, dst, radix, first_twiddles, second_twiddles);
        }
    }

    #[inline]
    fn stage_triple(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
        third_twiddles: &[Complex32],
    ) {
        let groups = src.len() / (radix << 1);
        if radix == 1 && groups >= 8 && src.len() >= 32 {
            unsafe { stage_triple32_radix1_avx_fma(src, dst, second_twiddles, third_twiddles) };
        } else if groups >= 16 {
            if stockham_f32_stage_is_l1_resident(src.len()) {
                unsafe {
                    stage_triple32_low_live_avx_fma(
                        src,
                        dst,
                        radix,
                        groups,
                        first_twiddles,
                        second_twiddles,
                        third_twiddles,
                    )
                };
            } else {
                unsafe {
                    stage_triple32_avx_fma(
                        src,
                        dst,
                        radix,
                        groups,
                        first_twiddles,
                        second_twiddles,
                        third_twiddles,
                    )
                };
            }
        } else if groups == 8 {
            unsafe {
                stage_triple32_quarter_groups_two_avx_fma(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                )
            };
        } else if groups == 4 {
            unsafe {
                stage_triple32_quarter_groups_one_avx_fma(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                )
            };
        } else {
            stage_triple_impl(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
                third_twiddles,
            );
        }
    }

    #[inline]
    fn stage_quad(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
        third_twiddles: &[Complex32],
        fourth_twiddles: &[Complex32],
    ) {
        let groups = src.len() / (radix << 1);
        if groups == 8 {
            unsafe {
                <Self as StockhamRadix16AvxLeaf>::stage_quad_groups_eight_avx_fma(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                    fourth_twiddles,
                )
            };
        } else {
            stage_quad_impl(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
                third_twiddles,
                fourth_twiddles,
            );
        }
    }

    fn scale(data: &mut [Complex32], scale: f32) {
        normalize_inplace_c32(data, scale);
    }
}
