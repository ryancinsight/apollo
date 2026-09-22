//! One generic implementation for the two AVX-512 Stockham precisions.
//!
//! The precise and reduced wide-lane markers share one dispatch shape — try
//! the double-width lanes, delegate to the AVX/FMA sibling otherwise — and
//! differ only in tuning carried by [`StockhamWideBackend`]. The dispatch
//! below is written once and monomorphizes per backend to the code the
//! hand-written impls produced.
//!
//! The AVX/FMA and scalar markers are intentionally not part of this
//! generic: their dispatch trees differ in branch structure between
//! precisions (the reduced tree carries wide-pair and quarter-group arms
//! the precise tree lacks), so merging them would parameterize branch
//! existence rather than thresholds.

#[cfg(target_arch = "x86_64")]
use super::super::avx::backend::{StockhamAvxBackend, StockhamPairGroups, StockhamWideBackend};
#[cfg(target_arch = "x86_64")]
use super::super::avx::precise::avx512_backend::Avx512BackendPrecise;
#[cfg(target_arch = "x86_64")]
use super::super::avx::reduced::avx512_backend::Avx512BackendReduced;
#[cfg(target_arch = "x86_64")]
use super::super::butterfly::{
    stage_groups_one_lanes, stage_lanes, stage_pair_lanes, stage_pair_radix_one_lanes,
    stage_triple_lanes, stage_triple_radix_one_lanes,
};
#[cfg(target_arch = "x86_64")]
use super::precise::PreciseStockhamAvxFma;
#[cfg(target_arch = "x86_64")]
use super::reduced::ReducedStockhamAvxFma;
#[cfg(target_arch = "x86_64")]
use super::traits::{private, StockhamPrecision};
#[cfg(target_arch = "x86_64")]
use crate::application::execution::kernel::radix_stage::{normalize_inplace, NormalizeSlice};
#[cfg(target_arch = "x86_64")]
use eunomia::{Complex32, Complex64};

/// Zero-sized marker selecting the wide-lane dispatch of backend `B`.
#[cfg(target_arch = "x86_64")]
pub(crate) struct StockhamAvx512<B>(std::marker::PhantomData<B>);

#[cfg(target_arch = "x86_64")]
impl<B> private::Sealed for StockhamAvx512<B> {}

#[cfg(target_arch = "x86_64")]
impl<B: StockhamWideBackend> StockhamPrecision for StockhamAvx512<B>
where
    B::Complex: NormalizeSlice<Scale = B::Real>,
{
    type Real = B::Real;
    type Complex = B::Complex;

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
    fn stage(src: &[B::Complex], dst: &mut [B::Complex], radix: usize, twiddles: &[B::Complex]) {
        let groups = src.len() / (radix << 1);
        if groups == 1 && radix >= 2 {
            if !B::stage_one(src, dst, radix, twiddles) {
                <B::Sibling as StockhamPrecision>::stage(src, dst, radix, twiddles);
            }
        } else if groups < B::STAGE_NARROW || !B::stage(src, dst, radix, twiddles) {
            <B::Sibling as StockhamPrecision>::stage(src, dst, radix, twiddles);
        }
    }

    #[inline]
    fn stage_pair(
        src: &[B::Complex],
        dst: &mut [B::Complex],
        radix: usize,
        first_twiddles: &[B::Complex],
        second_twiddles: &[B::Complex],
    ) {
        let groups = src.len() / (radix << 1);
        if radix == 1 {
            if src.len() < B::PAIR_RADIX_ONE_MIN || !B::stage_pair_one(src, dst, second_twiddles) {
                <B::Sibling as StockhamPrecision>::stage_pair(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                );
            }
        } else if groups == B::PAIR_WIDE_GROUPS && radix >= 2 {
            // Wide pairs require multiples of the family width.
            // SAFETY: this precision is selected only after AVX-512F was established on the host
            // (`stockham/mod.rs`), and the group count with the stage loop's stride-aligned slices
            // is the kernel's shape.
            unsafe {
                <B as StockhamPairGroups>::stage_pair_groups_two(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                )
            };
        } else if groups < B::PAIR_NARROW
            || !B::stage_pair(src, dst, radix, first_twiddles, second_twiddles)
        {
            <B::Sibling as StockhamPrecision>::stage_pair(
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
        src: &[B::Complex],
        dst: &mut [B::Complex],
        radix: usize,
        first_twiddles: &[B::Complex],
        second_twiddles: &[B::Complex],
        third_twiddles: &[B::Complex],
    ) {
        let n = src.len();
        let groups = n / (radix << 1);
        if radix == 1 && groups >= B::TRIPLE_WIDE {
            if !B::stage_triple_one(src, dst, second_twiddles, third_twiddles) {
                <B::Sibling as StockhamPrecision>::stage_triple(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                );
            }
        } else if groups >= B::TRIPLE_WIDE {
            if !B::stage_triple(
                src,
                dst,
                radix,
                first_twiddles,
                second_twiddles,
                third_twiddles,
            ) {
                <B::Sibling as StockhamPrecision>::stage_triple(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                );
            }
        } else {
            <B::Sibling as StockhamPrecision>::stage_triple(
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
        src: &[B::Complex],
        dst: &mut [B::Complex],
        radix: usize,
        first_twiddles: &[B::Complex],
        second_twiddles: &[B::Complex],
        third_twiddles: &[B::Complex],
        fourth_twiddles: &[B::Complex],
    ) {
        let groups = src.len() / (radix << 1);
        if groups == 8 {
            // SAFETY: this precision is selected only after AVX-512F was established on the host
            // (`stockham/mod.rs`), and the group count with the stage loop's stride-aligned slices
            // is the kernel's shape.
            unsafe {
                <B as StockhamAvxBackend>::stockham_quad_groups_eight_low_live(
                    src,
                    dst,
                    radix,
                    first_twiddles,
                    second_twiddles,
                    third_twiddles,
                    fourth_twiddles,
                )
            }
        } else {
            <B::Sibling as StockhamPrecision>::stage_quad(
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

    #[inline]
    fn scale(data: &mut [B::Complex], scale: B::Real) {
        normalize_inplace(data, scale);
    }
}

/// Precise (f64) wide-lane tuning: 8-wide lanes with the precise thresholds,
/// delegating to the AVX/FMA sibling.
#[cfg(target_arch = "x86_64")]
impl StockhamWideBackend for Avx512BackendPrecise {
    type Sibling = PreciseStockhamAvxFma;
    const STAGE_NARROW: usize = 4;
    const PAIR_RADIX_ONE_MIN: usize = 8;
    const PAIR_WIDE_GROUPS: usize = 4;
    const PAIR_NARROW: usize = 8;
    const TRIPLE_WIDE: usize = 16;

    #[inline]
    fn stage_one(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        twiddles: &[Complex64],
    ) -> bool {
        stage_groups_one_lanes::<f64, 8>(src, dst, radix, twiddles)
    }

    #[inline]
    fn stage(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        twiddles: &[Complex64],
    ) -> bool {
        stage_lanes::<f64, 8>(src, dst, radix, twiddles)
    }

    #[inline]
    fn stage_pair_one(
        src: &[Complex64],
        dst: &mut [Complex64],
        second_twiddles: &[Complex64],
    ) -> bool {
        stage_pair_radix_one_lanes::<f64, 8>(src, dst, second_twiddles)
    }

    #[inline]
    fn stage_pair(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
    ) -> bool {
        stage_pair_lanes::<f64, 8>(src, dst, radix, first_twiddles, second_twiddles)
    }

    #[inline]
    fn stage_triple_one(
        src: &[Complex64],
        dst: &mut [Complex64],
        second_twiddles: &[Complex64],
        third_twiddles: &[Complex64],
    ) -> bool {
        stage_triple_radix_one_lanes::<f64, 8>(src, dst, second_twiddles, third_twiddles)
    }

    #[inline]
    fn stage_triple(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
        third_twiddles: &[Complex64],
    ) -> bool {
        stage_triple_lanes::<f64, 8>(
            src,
            dst,
            radix,
            first_twiddles,
            second_twiddles,
            third_twiddles,
        )
    }
}

/// Reduced (f32) wide-lane tuning: 16-wide lanes with the reduced thresholds,
/// delegating to the AVX/FMA sibling.
#[cfg(target_arch = "x86_64")]
impl StockhamWideBackend for Avx512BackendReduced {
    type Sibling = ReducedStockhamAvxFma;
    const STAGE_NARROW: usize = 8;
    const PAIR_RADIX_ONE_MIN: usize = 16;
    const PAIR_WIDE_GROUPS: usize = 8;
    const PAIR_NARROW: usize = 16;
    const TRIPLE_WIDE: usize = 32;

    #[inline]
    fn stage_one(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        twiddles: &[Complex32],
    ) -> bool {
        stage_groups_one_lanes::<f32, 16>(src, dst, radix, twiddles)
    }

    #[inline]
    fn stage(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        twiddles: &[Complex32],
    ) -> bool {
        stage_lanes::<f32, 16>(src, dst, radix, twiddles)
    }

    #[inline]
    fn stage_pair_one(
        src: &[Complex32],
        dst: &mut [Complex32],
        second_twiddles: &[Complex32],
    ) -> bool {
        stage_pair_radix_one_lanes::<f32, 16>(src, dst, second_twiddles)
    }

    #[inline]
    fn stage_pair(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
    ) -> bool {
        stage_pair_lanes::<f32, 16>(src, dst, radix, first_twiddles, second_twiddles)
    }

    #[inline]
    fn stage_triple_one(
        src: &[Complex32],
        dst: &mut [Complex32],
        second_twiddles: &[Complex32],
        third_twiddles: &[Complex32],
    ) -> bool {
        stage_triple_radix_one_lanes::<f32, 16>(src, dst, second_twiddles, third_twiddles)
    }

    #[inline]
    fn stage_triple(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
        third_twiddles: &[Complex32],
    ) -> bool {
        stage_triple_lanes::<f32, 16>(
            src,
            dst,
            radix,
            first_twiddles,
            second_twiddles,
            third_twiddles,
        )
    }
}
