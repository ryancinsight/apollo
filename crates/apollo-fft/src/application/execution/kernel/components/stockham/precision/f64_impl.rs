use super::super::avx::f64::triple_2::stage_triple64_groups_eight_avx_fma;
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
use super::super::stage::stockham_precise_stage_is_l1_resident;

#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
use super::traits::F64Stockham;
use super::traits::{private, StockhamPrecision};
use crate::application::execution::kernel::radix_stage::normalize_inplace;
use num_complex::Complex64;

#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
impl StockhamPrecision for F64Stockham {
    type Real = f64;
    type Complex = Complex64;

    const MAX_FUSED_STAGES: u32 = 4;

    #[inline]
    fn stage(src: &[Complex64], dst: &mut [Complex64], radix: usize, twiddles: &[Complex64]) {
        stage_impl::<_, 512>(src, dst, radix, twiddles);
    }

    #[inline]
    fn stage_pair(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
    ) {
        stage_pair_impl::<_, 512>(src, dst, radix, first_twiddles, second_twiddles);
    }

    #[inline]
    fn stage_triple(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
        third_twiddles: &[Complex64],
    ) {
        stage_triple_impl::<_, 512>(
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
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
        third_twiddles: &[Complex64],
        fourth_twiddles: &[Complex64],
    ) {
        stage_quad_impl::<_, 512>(
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
    fn scale(data: &mut [Complex64], scale: f64) {
        normalize_inplace(data, scale);
    }
}
#[cfg(target_arch = "x86_64")]
pub(crate) struct F64StockhamAvxFma;

#[cfg(target_arch = "x86_64")]
impl private::Sealed for F64StockhamAvxFma {}

#[cfg(target_arch = "x86_64")]
impl StockhamPrecision for F64StockhamAvxFma {
    type Real = f64;
    type Complex = Complex64;

    const MAX_FUSED_STAGES: u32 = 4;

    #[inline]
    fn stage_triple_enabled(stride: usize, n: usize, input_is_data: bool) -> bool {
        // `groups == 4` means exactly three stages remain and is a zero-copy
        // win only when the source is scratch, because the radix-8 autosort
        // suffix then writes the final ping-pong pass into `data`.
        // `groups > 4` leaves at least one additional pass after the radix-8
        // stage; the fused stage still reduces arithmetic scheduling overhead
        // without changing the final ping-pong parity.
        let groups = n / (stride << 1);
        groups > 4 || (groups == 4 && !input_is_data)
    }

    #[inline]
    fn stage_quad_enabled(stride: usize, n: usize, _input_is_data: bool) -> bool {
        n / (stride << 1) == 8
    }

    #[inline]
    fn stage(src: &[Complex64], dst: &mut [Complex64], radix: usize, twiddles: &[Complex64]) {
        let groups = src.len() / (radix << 1);
        if groups == 1 && radix >= 2 {
            unsafe { <f64 as StockhamAvxBackend>::stage_groups_one(src, dst, radix, twiddles) };
        } else if groups >= 2 {
            unsafe { stage_avx_fma::<f64>(src, dst, radix, twiddles) };
        } else {
            stage_impl::<_, 512>(src, dst, radix, twiddles);
        }
    }

    #[inline]
    fn stage_pair(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
    ) {
        let groups = src.len() / (radix << 1);
        if radix == 1 {
            if src.len() >= 8 {
                unsafe { stage_pair_radix1_avx_fma::<f64>(src, dst, second_twiddles) };
            } else {
                stage_pair_impl::<_, 512>(src, dst, radix, first_twiddles, second_twiddles);
            }
        } else if groups == 2 && radix >= 2 {
            unsafe {
                <f64 as StockhamAvxBackend>::stage_pair_groups_two(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                )
            };
        } else if groups >= 4 {
            unsafe { stage_pair_avx_fma::<f64>(src, dst, radix, first_twiddles, second_twiddles) };
        } else {
            stage_pair_impl::<_, 512>(src, dst, radix, first_twiddles, second_twiddles);
        }
    }

    #[inline]
    fn stage_triple(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
        third_twiddles: &[Complex64],
    ) {
        let groups = src.len() / (radix << 1);
        if radix == 1 && groups >= 8 && stockham_precise_stage_is_l1_resident(src.len()) {
            unsafe {
                stage_triple_radix1_avx_fma::<f64>(src, dst, second_twiddles, third_twiddles)
            };
        } else if groups == 8 {
            unsafe {
                stage_triple64_groups_eight_avx_fma(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                )
            };
        } else if groups >= 8 {
            if stockham_precise_stage_is_l1_resident(src.len()) {
                unsafe {
                    stage_triple_low_live_avx_fma::<f64>(
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
                    stage_triple_avx_fma::<f64>(
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
        } else if groups == 4 {
            unsafe {
                <f64 as StockhamAvxBackend>::stage_triple_quarter_groups_one(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                )
            };
        } else {
            stage_triple_impl::<_, 512>(
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
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
        third_twiddles: &[Complex64],
        fourth_twiddles: &[Complex64],
    ) {
        let groups = src.len() / (radix << 1);
        if groups == 8 {
            unsafe {
                <f64 as StockhamAvxBackend>::stockham_quad_groups_eight_low_live(
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
            stage_quad_impl::<_, 512>(
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

    fn scale(data: &mut [Complex64], scale: f64) {
        normalize_inplace(data, scale);
    }
}
#[cfg(target_arch = "x86_64")]
pub(crate) struct F64StockhamAvx512;

#[cfg(target_arch = "x86_64")]
impl private::Sealed for F64StockhamAvx512 {}

#[cfg(target_arch = "x86_64")]
impl StockhamPrecision for F64StockhamAvx512 {
    type Real = f64;
    type Complex = Complex64;

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
    fn stage(src: &[Complex64], dst: &mut [Complex64], radix: usize, twiddles: &[Complex64]) {
        let groups = src.len() / (radix << 1);
        if groups == 1 && radix >= 2 {
            unsafe {
                <crate::application::execution::kernel::components::stockham::avx::f64::avx512_backend::Avx512BackendF64 as StockhamAvxBackend>::stage_groups_one(src, dst, radix, twiddles)
            };
        } else if groups >= 4 {
            // avx512 COMPLEX_PER_VECTOR is 4
            unsafe {
                stage_avx_fma::<crate::application::execution::kernel::components::stockham::avx::f64::avx512_backend::Avx512BackendF64>(src, dst, radix, twiddles)
            };
        } else {
            stage_impl::<_, 512>(src, dst, radix, twiddles);
        }
    }

    #[inline]
    fn stage_pair(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
    ) {
        let groups = src.len() / (radix << 1);
        if radix == 1 {
            if src.len() >= 8 {
                unsafe {
                    stage_pair_radix1_avx_fma::<crate::application::execution::kernel::components::stockham::avx::f64::avx512_backend::Avx512BackendF64>(src, dst, second_twiddles)
                };
            } else {
                stage_pair_impl::<_, 512>(src, dst, radix, first_twiddles, second_twiddles);
            }
        } else if groups == 4 && radix >= 2 {
            // avx512 pairs require multiples of 4
            unsafe {
                <crate::application::execution::kernel::components::stockham::avx::f64::avx512_backend::Avx512BackendF64 as StockhamAvxBackend>::stage_pair_groups_two(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                )
            };
        } else if groups >= 8 {
            unsafe {
                stage_pair_avx_fma::<crate::application::execution::kernel::components::stockham::avx::f64::avx512_backend::Avx512BackendF64>(src, dst, radix, first_twiddles, second_twiddles)
            };
        } else {
            stage_pair_impl::<_, 512>(src, dst, radix, first_twiddles, second_twiddles);
        }
    }

    #[inline]
    fn stage_triple(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
        third_twiddles: &[Complex64],
    ) {
        let groups = src.len() / (radix << 1);
        if radix == 1 && groups >= 16 && stockham_precise_stage_is_l1_resident(src.len()) {
            unsafe {
                stage_triple_radix1_avx_fma::<crate::application::execution::kernel::components::stockham::avx::f64::avx512_backend::Avx512BackendF64>(src, dst, second_twiddles, third_twiddles)
            };
        } else if groups >= 16 {
            if stockham_precise_stage_is_l1_resident(src.len()) {
                unsafe {
                    stage_triple_low_live_avx_fma::<crate::application::execution::kernel::components::stockham::avx::f64::avx512_backend::Avx512BackendF64>(
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
                    stage_triple_avx_fma::<crate::application::execution::kernel::components::stockham::avx::f64::avx512_backend::Avx512BackendF64>(
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
            stage_triple_impl::<_, 512>(
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
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
        third_twiddles: &[Complex64],
        fourth_twiddles: &[Complex64],
    ) {
        let groups = src.len() / (radix << 1);
        if groups == 8 {
            unsafe {
                <crate::application::execution::kernel::components::stockham::avx::f64::avx512_backend::Avx512BackendF64
                    as StockhamAvxBackend>::stockham_quad_groups_eight_low_live(
                    src, dst, radix, first_twiddles, second_twiddles, third_twiddles, fourth_twiddles,
                )
            }
        } else {
            stage_quad_impl::<_, 512>(
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

    fn scale(data: &mut [Complex64], scale: f64) {
        normalize_inplace(data, scale);
    }
}
