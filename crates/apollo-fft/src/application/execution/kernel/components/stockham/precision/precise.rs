#[cfg(target_arch = "x86_64")]
use super::super::avx::precise::triple_2::stage_triple_groups_eight_precise_avx_fma;
#[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
use super::super::stage::stockham_precise_stage_is_l1_resident;

#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
use super::traits::PreciseStockham;
use super::traits::{private, StockhamPrecision};
use crate::application::execution::kernel::radix_stage::normalize_inplace;
use eunomia::Complex64;

#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
impl StockhamPrecision for PreciseStockham {
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
        let n = src.len();
        if radix == 1 && n == 32 {
            // Scalar unroll for len32 (PoT worst): 4 explicit j0 one_impl (quarter_groups=4).
            // Removes inner k/j loop for this monomorph. Same value as impl.
            // (Inner-Fn from stage.rs used directly; fits per-LOG2 specialization.)
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
            // Scalar unroll for len64 (md-worst PoT 64 from benchmark_results): 8 explicit j0 one_impl.
            // Removes inner loop for this monomorph (additive to n32 + ZST LOG2=6). Same ops.
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
            // Scalar unroll for len128 (PoT 128 in md ~1.27x f32): 16 explicit j0 one_impl (quarter_groups=16).
            // Removes inner loop (additive to n64 + ZST LOG2=7). Same ops as impl.
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
        if radix == 1 && n == 256 {
            // Scalar unroll for len256 (PoT 256/512/32768 in md): 32 explicit j0 one (quarter=32).
            // Additive to 128. Targets controlling PoT f32 (256 1.137x, 32768 2.64x).
            let quarter_groups = 32usize;
            let eighth_n = 32usize;
            let quarter_n = 64usize;
            let half_n = 128usize;
            let w2b = second_twiddles[1];
            let w3b = third_twiddles[1];
            let w3c = third_twiddles[2];
            let w3d = third_twiddles[3];
            for k in 0..32 {
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
pub(crate) struct PreciseStockhamAvxFma;

#[cfg(target_arch = "x86_64")]
impl private::Sealed for PreciseStockhamAvxFma {}

#[cfg(target_arch = "x86_64")]
impl StockhamPrecision for PreciseStockhamAvxFma {
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
        let n = src.len();
        let groups = n / (radix << 1);
        if radix == 1 && n == 32 {
            // Per-LOG2 unroll for md-worst PoT 32 (len32 first pass): explicit no-loop
            // vector iters via specialized (Inner-Fn shared). Additive to ZST/const-LOG2.
            unsafe {
                stage_triple_radix1_n32_avx_fma::<f64>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 64 {
            // Per-LOG2 unroll for md-worst PoT 64 (len64 first pass): explicit no-loop
            // vector iters via n64 special (DCE + ILP). Additive to n32 + ZST for LOG2=6.
            unsafe {
                stage_triple_radix1_n64_avx_fma::<f64>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 128 {
            // Per-LOG2 unroll for PoT 128 (len128 first pass, md ~1.27x f32): n128 special no-loop.
            // Additive to prior + ZST LOG2=7. Targets remaining PoT.
            unsafe {
                stage_triple_radix1_n128_avx_fma::<f64>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 256 {
            // Per-LOG2 unroll for n=256 (len256 first pass + pads): n256 special.
            // Targets 256/512/32768 PoT (md f32 256 1.137x, 32768 2.64x). Additive to 128 + ZST LOG2=8.
            unsafe {
                stage_triple_radix1_n256_avx_fma::<f64>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 512 {
            // Per-LOG2 unroll for n=512 (len512 first pass + p=512 f32 pads for n113/257 etc): n512 special.
            // Targets 512 f64 1.241x / 32768 2.75x PoT in md. Additive to n256 + ZST LOG2=9.
            unsafe {
                stage_triple_radix1_n512_avx_fma::<f64>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 1024 {
            // Per-LOG2 unroll for n=1024 (len1024 first pass + p=1024 f32 pads): n1024 special.
            // Targets 1024/32768 PoT structure (md 32768 f64 2.75x). Additive to n512 + ZST LOG2=10. f32 sub for rader pads.
            unsafe {
                stage_triple_radix1_n1024_avx_fma::<f64>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 32768 {
            // 4x unrolled k loop for n=32768 radix1 first pass (len32768, md f64 2.75x worst remaining PoT).
            // Additive to n1024. 4 do_one/iter for higher ILP / lower overhead vs prior 2x. Uniform for avx512 step.
            unsafe {
                stage_triple_radix1_n32768_avx_fma::<f64>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && groups >= 8 {
            unsafe {
                stage_triple_radix1_avx_fma::<f64>(src, dst, second_twiddles, third_twiddles)
            };
        } else if groups == 8 {
            unsafe {
                stage_triple_groups_eight_precise_avx_fma(
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
pub(crate) struct PreciseStockhamAvx512;

#[cfg(target_arch = "x86_64")]
impl private::Sealed for PreciseStockhamAvx512 {}

#[cfg(target_arch = "x86_64")]
impl StockhamPrecision for PreciseStockhamAvx512 {
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
                <crate::application::execution::kernel::components::stockham::avx::precise::avx512_backend::Avx512BackendPrecise as StockhamAvxBackend>::stage_groups_one(src, dst, radix, twiddles)
            };
        } else if groups >= 4 {
            // avx512 COMPLEX_PER_VECTOR is 4
            unsafe {
                stage_avx_fma::<crate::application::execution::kernel::components::stockham::avx::precise::avx512_backend::Avx512BackendPrecise>(src, dst, radix, twiddles)
            };
        } else {
            <PreciseStockhamAvxFma as StockhamPrecision>::stage(src, dst, radix, twiddles);
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
                    stage_pair_radix1_avx_fma::<crate::application::execution::kernel::components::stockham::avx::precise::avx512_backend::Avx512BackendPrecise>(src, dst, second_twiddles)
                };
            } else {
                <PreciseStockhamAvxFma as StockhamPrecision>::stage_pair(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                );
            }
        } else if groups == 4 && radix >= 2 {
            // avx512 pairs require multiples of 4
            unsafe {
                <crate::application::execution::kernel::components::stockham::avx::precise::avx512_backend::Avx512BackendPrecise as StockhamAvxBackend>::stage_pair_groups_two(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                )
            };
        } else if groups >= 8 {
            unsafe {
                stage_pair_avx_fma::<crate::application::execution::kernel::components::stockham::avx::precise::avx512_backend::Avx512BackendPrecise>(src, dst, radix, first_twiddles, second_twiddles)
            };
        } else {
            <PreciseStockhamAvxFma as StockhamPrecision>::stage_pair(
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
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
        third_twiddles: &[Complex64],
    ) {
        let n = src.len();
        let groups = n / (radix << 1);
        if radix == 1 && n == 32 {
            unsafe {
                stage_triple_radix1_n32_avx_fma::<crate::application::execution::kernel::components::stockham::avx::precise::avx512_backend::Avx512BackendPrecise>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 64 {
            unsafe {
                stage_triple_radix1_n64_avx_fma::<crate::application::execution::kernel::components::stockham::avx::precise::avx512_backend::Avx512BackendPrecise>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 128 {
            unsafe {
                stage_triple_radix1_n128_avx_fma::<crate::application::execution::kernel::components::stockham::avx::precise::avx512_backend::Avx512BackendPrecise>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 256 {
            unsafe {
                stage_triple_radix1_n256_avx_fma::<crate::application::execution::kernel::components::stockham::avx::precise::avx512_backend::Avx512BackendPrecise>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 512 {
            unsafe {
                stage_triple_radix1_n512_avx_fma::<crate::application::execution::kernel::components::stockham::avx::precise::avx512_backend::Avx512BackendPrecise>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 1024 {
            unsafe {
                stage_triple_radix1_n1024_avx_fma::<crate::application::execution::kernel::components::stockham::avx::precise::avx512_backend::Avx512BackendPrecise>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && n == 32768 {
            // 4x (upgraded); wired for avx512 backend too (step may be 8/16).
            unsafe {
                stage_triple_radix1_n32768_avx_fma::<crate::application::execution::kernel::components::stockham::avx::precise::avx512_backend::Avx512BackendPrecise>(src, dst, second_twiddles, third_twiddles)
            };
        } else if radix == 1 && groups >= 16 {
            unsafe {
                stage_triple_radix1_avx_fma::<crate::application::execution::kernel::components::stockham::avx::precise::avx512_backend::Avx512BackendPrecise>(src, dst, second_twiddles, third_twiddles)
            };
        } else if groups >= 16 {
            if stockham_precise_stage_is_l1_resident(src.len()) {
                unsafe {
                    stage_triple_low_live_avx_fma::<crate::application::execution::kernel::components::stockham::avx::precise::avx512_backend::Avx512BackendPrecise>(
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
                    stage_triple_avx_fma::<crate::application::execution::kernel::components::stockham::avx::precise::avx512_backend::Avx512BackendPrecise>(
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
            <PreciseStockhamAvxFma as StockhamPrecision>::stage_triple(
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
                <crate::application::execution::kernel::components::stockham::avx::precise::avx512_backend::Avx512BackendPrecise
                    as StockhamAvxBackend>::stockham_quad_groups_eight_low_live(
                    src, dst, radix, first_twiddles, second_twiddles, third_twiddles, fourth_twiddles,
                )
            }
        } else {
            <PreciseStockhamAvxFma as StockhamPrecision>::stage_quad(
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
