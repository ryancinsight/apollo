use super::super::avx::{
    backend::StockhamAvxBackend,
    generic::{
        base::stage_avx_fma,
        pair::{stage_pair_avx_fma, stage_pair_radix1_avx_fma},
        triple::{
            stage_triple_avx_fma, stage_triple_low_live_avx_fma, stage_triple_radix1_avx_fma,
        },
    },
};
use super::super::butterfly::{stage_pair_impl, stage_quad_impl, stage_triple_impl};
use super::super::stage::stage_impl;
use super::super::stage::stockham_reduced_stage_is_l1_resident;
#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
use super::traits::ReducedStockham;
use super::traits::{private, StockhamPrecision};
use crate::application::execution::kernel::radix_stage::normalize_inplace;
use num_complex::Complex32;

#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
impl StockhamPrecision for ReducedStockham {
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
        stage_impl::<_, 1024>(src, dst, radix, twiddles);
    }

    #[inline]
    fn stage_pair(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
    ) {
        stage_pair_impl::<_, 1024>(src, dst, radix, first_twiddles, second_twiddles);
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
        stage_triple_impl::<_, 1024>(
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
        stage_quad_impl::<_, 1024>(
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
        normalize_inplace(data, scale);
    }
}
#[cfg(target_arch = "x86_64")]
pub(crate) struct ReducedStockhamAvxFma;

#[cfg(target_arch = "x86_64")]
impl private::Sealed for ReducedStockhamAvxFma {}

#[cfg(target_arch = "x86_64")]
impl StockhamPrecision for ReducedStockhamAvxFma {
    type Real = f32;
    type Complex = Complex32;

    const MAX_FUSED_STAGES: u32 = 4;

    #[inline]
    fn stage_triple_enabled(stride: usize, n: usize, input_is_data: bool) -> bool {
        let groups = n / (stride << 1);
        groups > 4 || (groups == 4 && !input_is_data)
    }

    #[inline]
    fn stage_quad_enabled(stride: usize, n: usize, _input_is_data: bool) -> bool {
        n / (stride << 1) == 8
    }

    #[inline]
    fn stage(src: &[Complex32], dst: &mut [Complex32], radix: usize, twiddles: &[Complex32]) {
        let groups = src.len() / (radix << 1);
        if groups == 1 && radix >= 2 {
            unsafe { <f32 as StockhamAvxBackend>::stage_groups_one(src, dst, radix, twiddles) };
        } else if groups >= 4 {
            unsafe { stage_avx_fma::<f32>(src, dst, radix, twiddles) };
        } else {
            stage_impl::<_, 1024>(src, dst, radix, twiddles);
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
                unsafe { stage_pair_radix1_avx_fma::<f32>(src, dst, second_twiddles) };
            } else {
                stage_pair_impl::<_, 1024>(src, dst, radix, first_twiddles, second_twiddles);
            }
        } else if groups >= 8 {
            unsafe { stage_pair_avx_fma::<f32>(src, dst, radix, first_twiddles, second_twiddles) };
        } else if groups == 4 {
            unsafe {
                <f32 as StockhamAvxBackend>::stage_pair_quarter_groups_two(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                )
            };
        } else if groups == 2 {
            unsafe {
                <f32 as StockhamAvxBackend>::stage_pair_groups_two(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                )
            };
        } else {
            stage_pair_impl::<_, 1024>(src, dst, radix, first_twiddles, second_twiddles);
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
            unsafe {
                stage_triple_radix1_avx_fma::<f32>(src, dst, second_twiddles, third_twiddles)
            };
        } else if groups >= 16 {
            if stockham_reduced_stage_is_l1_resident(src.len()) {
                unsafe {
                    stage_triple_low_live_avx_fma::<f32>(
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
                    stage_triple_avx_fma::<f32>(
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
                <f32 as StockhamAvxBackend>::stage_triple_quarter_groups_two(
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
                <f32 as StockhamAvxBackend>::stage_triple_quarter_groups_one(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                )
            };
        } else {
            stage_triple_impl::<_, 1024>(
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
                <f32 as StockhamAvxBackend>::stockham_quad_groups_eight(
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
            stage_quad_impl::<_, 1024>(
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
        normalize_inplace(data, scale);
    }
}
#[cfg(target_arch = "x86_64")]
pub(crate) struct ReducedStockhamAvx512;

#[cfg(target_arch = "x86_64")]
impl private::Sealed for ReducedStockhamAvx512 {}

#[cfg(target_arch = "x86_64")]
impl StockhamPrecision for ReducedStockhamAvx512 {
    type Real = f32;
    type Complex = Complex32;

    const MAX_FUSED_STAGES: u32 = 4;

    #[inline]
    fn stage_triple_enabled(stride: usize, n: usize, input_is_data: bool) -> bool {
        let groups = n / (stride << 1);
        groups > 4 || (groups == 4 && !input_is_data)
    }

    #[inline]
    fn stage_quad_enabled(stride: usize, n: usize, _input_is_data: bool) -> bool {
        n / (stride << 1) == 8
    }

    #[inline]
    fn stage(src: &[Complex32], dst: &mut [Complex32], radix: usize, twiddles: &[Complex32]) {
        let groups = src.len() / (radix << 1);
        if groups == 1 && radix >= 2 {
            unsafe {
                <crate::application::execution::kernel::components::stockham::avx::reduced::avx512_backend::Avx512BackendReduced as StockhamAvxBackend>::stage_groups_one(src, dst, radix, twiddles)
            };
        } else if groups >= 8 {
            // avx512 f32 COMPLEX_PER_VECTOR is 8
            unsafe {
                stage_avx_fma::<crate::application::execution::kernel::components::stockham::avx::reduced::avx512_backend::Avx512BackendReduced>(src, dst, radix, twiddles)
            };
        } else {
            <ReducedStockhamAvxFma as StockhamPrecision>::stage(src, dst, radix, twiddles);
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
                unsafe {
                    stage_pair_radix1_avx_fma::<crate::application::execution::kernel::components::stockham::avx::reduced::avx512_backend::Avx512BackendReduced>(src, dst, second_twiddles)
                };
            } else {
                <ReducedStockhamAvxFma as StockhamPrecision>::stage_pair(src, dst, radix, first_twiddles, second_twiddles);
            }
        } else if groups == 8 && radix >= 2 {
            // avx512 pairs require multiples of 8
            unsafe {
                <crate::application::execution::kernel::components::stockham::avx::reduced::avx512_backend::Avx512BackendReduced as StockhamAvxBackend>::stage_pair_groups_two(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                )
            };
        } else if groups >= 16 {
            unsafe {
                stage_pair_avx_fma::<crate::application::execution::kernel::components::stockham::avx::reduced::avx512_backend::Avx512BackendReduced>(src, dst, radix, first_twiddles, second_twiddles)
            };
        } else {
            <ReducedStockhamAvxFma as StockhamPrecision>::stage_pair(src, dst, radix, first_twiddles, second_twiddles);
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
        if radix == 1 && groups >= 32 {
            unsafe {
                stage_triple_radix1_avx_fma::<crate::application::execution::kernel::components::stockham::avx::reduced::avx512_backend::Avx512BackendReduced>(src, dst, second_twiddles, third_twiddles)
            };
        } else if groups >= 32 {
            if stockham_reduced_stage_is_l1_resident(src.len()) {
                unsafe {
                    stage_triple_low_live_avx_fma::<crate::application::execution::kernel::components::stockham::avx::reduced::avx512_backend::Avx512BackendReduced>(
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
                    stage_triple_avx_fma::<crate::application::execution::kernel::components::stockham::avx::reduced::avx512_backend::Avx512BackendReduced>(
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
        } else {
            <ReducedStockhamAvxFma as StockhamPrecision>::stage_triple(
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
                <crate::application::execution::kernel::components::stockham::avx::reduced::avx512_backend::Avx512BackendReduced
                    as StockhamAvxBackend>::stockham_quad_groups_eight_low_live(
                    src, dst, radix, first_twiddles, second_twiddles, third_twiddles, fourth_twiddles,
                )
            }
        } else {
            <ReducedStockhamAvxFma as StockhamPrecision>::stage_quad(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
                third_twiddles,
                fourth_twiddles,
            )
        }
    }

    fn scale(data: &mut [Complex32], scale: f32) {
        normalize_inplace(data, scale);
    }
}
