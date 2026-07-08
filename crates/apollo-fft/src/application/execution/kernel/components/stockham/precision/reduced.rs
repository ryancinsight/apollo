use super::super::avx::{
    backend::StockhamAvxBackend,
    generic::{
        base::stage_avx_fma,
        pair::{stage_pair_avx_fma, stage_pair_radix1_avx_fma},
        triple::{
            stage_triple_avx_fma, stage_triple_low_live_avx_fma, stage_triple_radix1_avx_fma,
            stage_triple_radix1_n1024_avx_fma, stage_triple_radix1_n128_avx_fma,
            stage_triple_radix1_n256_avx_fma, stage_triple_radix1_n32768_avx_fma,
            stage_triple_radix1_n32_avx_fma, stage_triple_radix1_n512_avx_fma,
            stage_triple_radix1_n64_avx_fma,
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
use eunomia::Complex32;

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
        let n = src.len();
        if radix == 1 && n == 32 {
            // Scalar f32 unroll for len32 (worst PoT): explicit 4x j0 one (same as f64 scalar).
            let quarter_groups = 4usize;
            let eighth_n = 4usize;
            let quarter_n = 8usize;
            let half_n = 16usize;
            let w2b = second_twiddles[1];
            let w3b = third_twiddles[1];
            let w3c = third_twiddles[2];
            let w3d = third_twiddles[3];
            for k in 0..4 {
                super::super::butterfly::stage_triple_scalar_one_j0_impl(
                    src,
                    dst,
                    0,
                    0,
                    quarter_groups,
                    eighth_n,
                    quarter_n,
                    half_n,
                    k,
                    w2b,
                    w3b,
                    w3c,
                    w3d,
                );
            }
            return;
        }
        if radix == 1 && n == 64 {
            // Scalar f32 unroll for len64 (PoT worst 64): 8 explicit j0 one (additive to n32).
            let quarter_groups = 8usize;
            let eighth_n = 8usize;
            let quarter_n = 16usize;
            let half_n = 32usize;
            let w2b = second_twiddles[1];
            let w3b = third_twiddles[1];
            let w3c = third_twiddles[2];
            let w3d = third_twiddles[3];
            for k in 0..8 {
                super::super::butterfly::stage_triple_scalar_one_j0_impl(
                    src,
                    dst,
                    0,
                    0,
                    quarter_groups,
                    eighth_n,
                    quarter_n,
                    half_n,
                    k,
                    w2b,
                    w3b,
                    w3c,
                    w3d,
                );
            }
            return;
        }
        if radix == 1 && n == 128 {
            // Scalar f32 unroll for len128 (PoT 128 md ~1.27x f32): 16 explicit j0 one (additive).
            let quarter_groups = 16usize;
            let eighth_n = 16usize;
            let quarter_n = 32usize;
            let half_n = 64usize;
            let w2b = second_twiddles[1];
            let w3b = third_twiddles[1];
            let w3c = third_twiddles[2];
            let w3d = third_twiddles[3];
            for k in 0..16 {
                super::super::butterfly::stage_triple_scalar_one_j0_impl(
                    src,
                    dst,
                    0,
                    0,
                    quarter_groups,
                    eighth_n,
                    quarter_n,
                    half_n,
                    k,
                    w2b,
                    w3b,
                    w3c,
                    w3d,
                );
            }
            return;
        }
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
        // Enable triples when AVX can process ≥2 full vectors per group.
        // For f32 AVX (8 complex/vector), groups ≥ 2 gives enough work;
        // for groups == 2, require the triple writes to scratch (second call
        // so !input_is_data) to avoid live-range conflicts.
        groups > 2 || (groups == 2 && !input_is_data)
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
        let n = src.len();
        let groups = n / (radix << 1);
        if radix == 1 && n == 32 {
            // Per-LOG2 unroll for md-worst PoT 32 f32 (len32): n32 special no-loop.
            unsafe {
                stage_triple_radix1_n32_avx_fma::<f32>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 64 {
            // Per-LOG2 unroll for md-worst PoT 64 f32 (len64 first pass): n64 special no-loop.
            // Additive to prior n32 + ZST for 64. Targets controlling 1.97x from benchmark_results.
            unsafe {
                stage_triple_radix1_n64_avx_fma::<f32>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 128 {
            // Per-LOG2 unroll for PoT 128 f32 (len128 first pass, current md 1.27x): n128 special no-loop.
            // Additive to n64 + ZST LOG2=7.
            unsafe {
                stage_triple_radix1_n128_avx_fma::<f32>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 256 {
            // Per-LOG2 unroll for n=256 radix1 (len256 first pass + p=256 f32 pads for n113 etc): n256 special.
            // Targets 256/512/32768 PoT in md (f32 256 1.137x, 32768 2.64x; 32/64 benefit). Additive to 128 + ZST.
            unsafe {
                stage_triple_radix1_n256_avx_fma::<f32>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 512 {
            // Per-LOG2 unroll for n=512 radix1 (len512 first pass + p=512 f32 pads for n113/257 etc): n512 special.
            // Targets 512 f32 (now 0.733x) / f64 1.241x / 32768 PoT in md. Additive to n256 + ZST LOG2=9.
            unsafe {
                stage_triple_radix1_n512_avx_fma::<f32>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 1024 {
            // Per-LOG2 unroll for n=1024 radix1 (len1024 first pass + p=1024 f32 pads): n1024 special.
            // Targets 1024/32768 PoT (md 32768 f64 2.75x); f32 avx sub for rader bluestein pads. Additive to n512 + ZST.
            unsafe {
                stage_triple_radix1_n1024_avx_fma::<f32>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 32768 {
            // 4x unrolled k loop for n=32768 radix1 first pass (len32768, md f64 2.75x).
            // Additive to n1024. 4 do_one/iter for higher ILP. Uniform step (avx f32 step=4 + avx512).
            unsafe {
                stage_triple_radix1_n32768_avx_fma::<f32>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && groups >= 8 && n >= 32 {
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
                <ReducedStockhamAvxFma as StockhamPrecision>::stage_pair(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                );
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
            <ReducedStockhamAvxFma as StockhamPrecision>::stage_pair(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
            );
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
        let n = src.len();
        let groups = n / (radix << 1);
        if radix == 1 && n == 32 {
            unsafe {
                stage_triple_radix1_n32_avx_fma::<crate::application::execution::kernel::components::stockham::avx::reduced::avx512_backend::Avx512BackendReduced>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 64 {
            unsafe {
                stage_triple_radix1_n64_avx_fma::<crate::application::execution::kernel::components::stockham::avx::reduced::avx512_backend::Avx512BackendReduced>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 128 {
            unsafe {
                stage_triple_radix1_n128_avx_fma::<crate::application::execution::kernel::components::stockham::avx::reduced::avx512_backend::Avx512BackendReduced>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 512 {
            unsafe {
                stage_triple_radix1_n512_avx_fma::<crate::application::execution::kernel::components::stockham::avx::reduced::avx512_backend::Avx512BackendReduced>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 1024 {
            unsafe {
                stage_triple_radix1_n1024_avx_fma::<crate::application::execution::kernel::components::stockham::avx::reduced::avx512_backend::Avx512BackendReduced>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 32768 {
            // 4x (upgraded); wired for reduced avx512 backend (f32 step may be 16).
            unsafe {
                stage_triple_radix1_n32768_avx_fma::<crate::application::execution::kernel::components::stockham::avx::reduced::avx512_backend::Avx512BackendReduced>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && groups >= 32 {
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
