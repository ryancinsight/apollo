//! Concrete `MixedRadixScalar` implementations for the two production
//! floating-point precisions.
//!
//! The two impls are kept together rather than split per type because they
//! perform identical trait wiring; the only differences are the concrete
//! complex element type and the precision-specific SIMD/transpose routines
//! (`pointwise_mul_reduced`/`_precise`, `transpose_matrix_reduced`/`_precise`).
//! The precision-tagged names refer to SIMD lane density, not to the type
//! suffix, so they remain compliant with the naming policy in CLAUDE.md.

use super::rader::{
    build_rader_negacyclic_spectra, build_rader_negacyclic_twiddles, build_rader_spectrum_vec,
};
use super::simd::{pointwise_mul_precise, pointwise_mul_reduced};
use super::small_pot::{
    small_pot_inplace_precise, small_pot_inplace_reduced, small_pot_inplace_sized_precise,
    small_pot_inplace_sized_reduced,
};
use super::trait_def::MixedRadixScalar;
use super::transpose::{transpose_matrix_precise, transpose_matrix_reduced};
use super::twiddle_constants::{
    TWIDDLES_FWD_PRECISE, TWIDDLES_FWD_REDUCED, TWIDDLES_INV_PRECISE, TWIDDLES_INV_REDUCED,
};
use crate::application::execution::kernel::components::{radix_composite, stockham};
use crate::application::execution::kernel::mixed_radix::caches::{
    cached_four_step_twiddles, cached_rader_neg_twiddles, cached_rader_negacyclic_spectra,
    cached_rader_spectrum, cached_twiddle_fwd, cached_twiddle_inv, with_bluestein_scratch,
    with_pfa_scratch, with_rader_padded_scratch, with_stockham_scratch,
};
use crate::application::execution::kernel::pot::{PoTStrategy, SizedPoT};
use crate::application::execution::kernel::radix_stage::normalize_inplace;
use eunomia::{Complex32, Complex64};
use std::sync::Arc;

// ── AVX/SIMD helpers (shared by f32 and f64 impls) ─────────────────────────

impl MixedRadixScalar for f32 {
    const HALF_CYCLIC_RADER_THRESHOLD: usize = 32;
    const HALF_CYCLIC_RADER_PRIMES: &'static [usize] = &[];
    const COMPOSITE_RADICES_200: &'static [usize] = &[4, 2, 5, 5];
    const FORCE_COMPOSITE_63: bool = true;
    const FORCE_COMPOSITE_72: bool = true;
    const PREFER_BLUESTEIN_MID_RADER: bool = true;
    const BLUESTEIN_PAD_POWER_OF_TWO: bool = true;
    const BLUESTEIN_NATIVE_PHASE_TRIG: bool = true;

    type Complex = Complex32;

    #[inline]
    fn complex(re: f64, im: f64) -> Complex32 {
        Complex32::new(re as f32, im as f32)
    }

    #[inline]
    fn cached_twiddle_fwd(n: usize) -> Arc<[Complex32]> {
        cached_twiddle_fwd(n)
    }
    #[inline]
    fn cached_twiddle_inv(n: usize) -> Arc<[Complex32]> {
        cached_twiddle_inv(n)
    }
    #[inline]
    fn with_twiddle_fwd<R>(n: usize, f: impl FnOnce(&[Complex32]) -> R) -> R {
        crate::application::execution::kernel::mixed_radix::caches::twiddle::with_twiddle_fwd(n, f)
    }
    #[inline]
    fn with_twiddle_inv<R>(n: usize, f: impl FnOnce(&[Complex32]) -> R) -> R {
        crate::application::execution::kernel::mixed_radix::caches::twiddle::with_twiddle_inv(n, f)
    }
    #[inline]
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        with_stockham_scratch(n, f)
    }
    #[inline]
    fn with_pfa_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        with_pfa_scratch(n, f)
    }
    #[inline]
    fn with_rader_padded_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        with_rader_padded_scratch(n, f)
    }
    #[inline]
    fn with_bluestein_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
        with_bluestein_scratch(n, f)
    }

    #[inline]
    fn cached_rader_spectrum<const INVERSE: bool>(
        n: usize,
        generator_inverse: usize,
    ) -> Arc<[Complex32]> {
        let key = (n, INVERSE as usize, generator_inverse);
        cached_rader_spectrum(key, |_| {
            build_rader_spectrum_vec::<f32, INVERSE>(n, generator_inverse)
        })
    }

    #[inline]
    fn cached_rader_negacyclic_spectra<const INVERSE: bool>(
        n: usize,
        generator_inverse: usize,
    ) -> (Arc<[Complex32]>, Arc<[Complex32]>) {
        let key = (n, INVERSE as usize, generator_inverse);
        cached_rader_negacyclic_spectra(key, |_| {
            build_rader_negacyclic_spectra::<f32, INVERSE>(n, generator_inverse)
        })
    }

    #[inline]
    fn cached_rader_neg_twiddles(m: usize) -> Arc<[Complex32]> {
        cached_rader_neg_twiddles(m, build_rader_negacyclic_twiddles::<f32>)
    }

    #[inline]
    fn cached_four_step_twiddles<const INVERSE: bool>(
        n: usize,
        n1: usize,
        n2: usize,
    ) -> Arc<[Complex32]> {
        cached_four_step_twiddles::<Complex32, INVERSE>(n, n1, n2)
    }
    #[inline]
    fn pointwise_mul(a: &mut [Complex32], b: &[Complex32]) {
        pointwise_mul_reduced::<false>(a, b);
    }
    #[inline]
    fn pointwise_mul_conj(a: &mut [Complex32], b: &[Complex32]) {
        pointwise_mul_reduced::<true>(a, b);
    }
    #[inline]
    fn stockham_forward(data: &mut [Complex32], scratch: &mut [Complex32], twiddles: &[Complex32]) {
        <f32 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, twiddles);
    }
    #[inline]
    fn stockham_forward_normalized(
        data: &mut [Complex32],
        scratch: &mut [Complex32],
        twiddles: &[Complex32],
        n: usize,
    ) {
        <f32 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, twiddles);
        normalize_inplace(data, 1.0_f32 / n as f32);
    }

    #[cfg_attr(debug_assertions, inline(never))]
    #[cfg_attr(not(debug_assertions), inline)]
    fn stockham_forward_sized<const LOG2: u32>(
        data: &mut [Complex32],
        scratch: &mut [Complex32],
        twiddles: &[Complex32],
    ) {
        <f32 as stockham::StockhamKernel>::forward_with_scratch_sized::<LOG2>(
            data, scratch, twiddles,
        );
    }

    #[cfg_attr(debug_assertions, inline(never))]
    #[cfg_attr(not(debug_assertions), inline)]
    fn stockham_forward_normalized_sized<const LOG2: u32>(
        data: &mut [Complex32],
        scratch: &mut [Complex32],
        twiddles: &[Complex32],
    ) {
        Self::stockham_forward_sized::<LOG2>(data, scratch, twiddles);
        normalize_inplace(data, 1.0_f32 / (1usize << LOG2) as f32);
    }

    #[inline]
    fn short_winograd<const INVERSE: bool, const NORMALIZE: bool>(data: &mut [Complex32]) -> bool {
        crate::application::execution::kernel::mixed_radix::traits::short_winograd::<
            Self,
            INVERSE,
            NORMALIZE,
        >(data)
    }
    #[inline]
    unsafe fn small_pot_inplace<const INVERSE: bool, const NORMALIZE: bool>(
        data: &mut [Complex32],
    ) -> bool {
        small_pot_inplace_reduced::<INVERSE, NORMALIZE>(data)
    }
    #[inline]
    fn composite_forward(data: &mut [Complex32], radices: &[usize]) {
        radix_composite::forward_inplace_with_radices(data, radices);
    }
    #[inline]
    fn composite_forward_with_pointwise(
        data: &mut [Complex32],
        radices: &[usize],
        pointwise_spectrum: &[Complex32],
    ) {
        radix_composite::forward_inplace_with_pointwise(data, radices, pointwise_spectrum);
    }
    #[inline]
    fn composite_inverse_unnorm(data: &mut [Complex32], radices: &[usize]) {
        radix_composite::inverse_inplace_unnorm_with_radices(data, radices);
    }
    #[inline]
    fn composite_inverse(data: &mut [Complex32], radices: &[usize]) {
        radix_composite::inverse_inplace_with_radices(data, radices);
    }
    #[inline]
    fn normalize(data: &mut [Complex32], n: usize) {
        normalize_inplace(data, 1.0_f32 / n as f32);
    }
    #[inline]
    fn transpose_matrix(src: &[Complex32], dst: &mut [Complex32], n1: usize, n2: usize) {
        transpose_matrix_reduced(src, dst, n1, n2);
    }

    #[inline]
    unsafe fn small_pot_inplace_sized<
        const N: usize,
        const INVERSE: bool,
        const NORMALIZE: bool,
    >(
        data: &mut [Complex32],
    ) {
        small_pot_inplace_sized_reduced::<N, INVERSE, NORMALIZE>(data);
    }

    #[inline]
    fn pot_inplace<const INVERSE: bool, const NORMALIZE: bool>(
        data: &mut [Self::Complex],
        twiddles: &[Self::Complex],
    ) {
        let n = data.len();
        match n {
            2 => unsafe {
                Self::small_pot_inplace_sized::<2, INVERSE, NORMALIZE>(data);
            },
            4 => unsafe {
                Self::small_pot_inplace_sized::<4, INVERSE, NORMALIZE>(data);
            },
            8 => unsafe {
                Self::small_pot_inplace_sized::<8, INVERSE, NORMALIZE>(data);
            },
            16 => unsafe {
                Self::small_pot_inplace_sized::<16, INVERSE, NORMALIZE>(data);
            },
            32 => unsafe {
                Self::small_pot_inplace_sized::<32, INVERSE, NORMALIZE>(data);
            },
            64 => unsafe {
                Self::small_pot_inplace_sized::<64, INVERSE, NORMALIZE>(data);
            },
            _ => {
                Self::with_scratch(n, |scratch| {
                    if INVERSE && NORMALIZE {
                        Self::stockham_forward_normalized(data, scratch, twiddles, n);
                    } else {
                        Self::stockham_forward(data, scratch, twiddles);
                    }
                });
            }
        }
    }

    fn pot_inplace_sized<
        const INVERSE: bool,
        const NORMALIZE: bool,
        S: PoTStrategy,
        const LOG2: u32,
    >(
        data: &mut [Self::Complex],
        twiddles: &[Self::Complex],
        _s: SizedPoT<S, LOG2>,
    ) {
        // const LOG2 drives selection (zero-cost monomorph); for <=64 preserve direct small
        // no-scratch path (memory efficiency + best for 32/64 which have dedicated AVX fixed column);
        // for 128+ (md-worst PoT) use stockham sized path so const LOG2 flows end-to-end to
        // kernel forward_with_scratch_sized -> transform_sized / with_strategy / len* bodies.
        match LOG2 {
            1 => unsafe {
                Self::small_pot_inplace_sized::<2, INVERSE, NORMALIZE>(data);
            },
            2 => unsafe {
                Self::small_pot_inplace_sized::<4, INVERSE, NORMALIZE>(data);
            },
            3 => unsafe {
                Self::small_pot_inplace_sized::<8, INVERSE, NORMALIZE>(data);
            },
            4 => unsafe {
                Self::small_pot_inplace_sized::<16, INVERSE, NORMALIZE>(data);
            },
            5 => unsafe {
                Self::small_pot_inplace_sized::<32, INVERSE, NORMALIZE>(data);
            },
            6 => unsafe {
                Self::small_pot_inplace_sized::<64, INVERSE, NORMALIZE>(data);
            },
            _ => {
                let n = 1usize << LOG2;
                Self::with_scratch(n, |scratch| {
                    if INVERSE && NORMALIZE {
                        Self::stockham_forward_normalized_sized::<LOG2>(data, scratch, twiddles);
                    } else {
                        Self::stockham_forward_sized::<LOG2>(data, scratch, twiddles);
                    }
                });
            }
        }
    }

    #[inline]
    fn small_pot_twiddles<const INVERSE: bool>(n: usize) -> &'static [Self::Complex] {
        let idx = n.trailing_zeros() as usize;
        if INVERSE {
            &TWIDDLES_INV_REDUCED[idx]
        } else {
            &TWIDDLES_FWD_REDUCED[idx]
        }
    }

    fn use_generated_codelet_plan(n: usize) -> bool {
        matches!(
            n,
            72 | 81
                | 96
                | 99
                | 108
                | 112
                | 120
                | 121
                | 126
                | 128
                | 144
                | 154
                | 168
                | 180
                | 189
                | 222
                | 242
                | 246
                | 259
                | 275
                | 280
                | 296
                | 363
                | 400
                | 484
        )
    }
}

impl MixedRadixScalar for f64 {
    const HALF_CYCLIC_RADER_THRESHOLD: usize = 32;
    const HALF_CYCLIC_RADER_PRIMES: &'static [usize] = &[67];
    const COMPOSITE_RADICES_200: &'static [usize] = &[4, 5, 5, 2];
    const FORCE_COMPOSITE_63: bool = false;
    const FORCE_COMPOSITE_72: bool = false;
    const PREFER_BLUESTEIN_MID_RADER: bool = false;
    const BLUESTEIN_PAD_POWER_OF_TWO: bool = true;
    const BLUESTEIN_NATIVE_PHASE_TRIG: bool = false;

    type Complex = Complex64;

    #[inline]
    fn complex(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[inline]
    fn cached_twiddle_fwd(n: usize) -> Arc<[Complex64]> {
        cached_twiddle_fwd(n)
    }
    #[inline]
    fn cached_twiddle_inv(n: usize) -> Arc<[Complex64]> {
        cached_twiddle_inv(n)
    }
    #[inline]
    fn with_twiddle_fwd<R>(n: usize, f: impl FnOnce(&[Complex64]) -> R) -> R {
        crate::application::execution::kernel::mixed_radix::caches::twiddle::with_twiddle_fwd(n, f)
    }
    #[inline]
    fn with_twiddle_inv<R>(n: usize, f: impl FnOnce(&[Complex64]) -> R) -> R {
        crate::application::execution::kernel::mixed_radix::caches::twiddle::with_twiddle_inv(n, f)
    }
    #[inline]
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        with_stockham_scratch(n, f)
    }
    #[inline]
    fn with_pfa_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        with_pfa_scratch(n, f)
    }
    #[inline]
    fn with_rader_padded_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        with_rader_padded_scratch(n, f)
    }
    #[inline]
    fn with_bluestein_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
        with_bluestein_scratch(n, f)
    }

    #[inline]
    fn cached_rader_spectrum<const INVERSE: bool>(
        n: usize,
        generator_inverse: usize,
    ) -> Arc<[Complex64]> {
        let key = (n, INVERSE as usize, generator_inverse);
        cached_rader_spectrum(key, |_| {
            build_rader_spectrum_vec::<f64, INVERSE>(n, generator_inverse)
        })
    }

    #[inline]
    fn cached_rader_negacyclic_spectra<const INVERSE: bool>(
        n: usize,
        generator_inverse: usize,
    ) -> (Arc<[Complex64]>, Arc<[Complex64]>) {
        let key = (n, INVERSE as usize, generator_inverse);
        cached_rader_negacyclic_spectra(key, |_| {
            build_rader_negacyclic_spectra::<f64, INVERSE>(n, generator_inverse)
        })
    }

    #[inline]
    fn cached_rader_neg_twiddles(m: usize) -> Arc<[Complex64]> {
        cached_rader_neg_twiddles(m, build_rader_negacyclic_twiddles::<f64>)
    }

    #[inline]
    fn cached_four_step_twiddles<const INVERSE: bool>(
        n: usize,
        n1: usize,
        n2: usize,
    ) -> Arc<[Complex64]> {
        cached_four_step_twiddles::<Complex64, INVERSE>(n, n1, n2)
    }
    #[inline]
    fn pointwise_mul(a: &mut [Complex64], b: &[Complex64]) {
        pointwise_mul_precise::<false>(a, b);
    }
    #[inline]
    fn pointwise_mul_conj(a: &mut [Complex64], b: &[Complex64]) {
        pointwise_mul_precise::<true>(a, b);
    }
    #[inline]
    fn stockham_forward(data: &mut [Complex64], scratch: &mut [Complex64], twiddles: &[Complex64]) {
        <f64 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, twiddles);
    }
    #[inline]
    fn stockham_forward_normalized(
        data: &mut [Complex64],
        scratch: &mut [Complex64],
        twiddles: &[Complex64],
        n: usize,
    ) {
        <f64 as stockham::StockhamKernel>::forward_with_scratch(data, scratch, twiddles);
        normalize_inplace(data, 1.0_f64 / n as f64);
    }

    #[cfg_attr(debug_assertions, inline(never))]
    #[cfg_attr(not(debug_assertions), inline)]
    fn stockham_forward_sized<const LOG2: u32>(
        data: &mut [Complex64],
        scratch: &mut [Complex64],
        twiddles: &[Complex64],
    ) {
        <f64 as stockham::StockhamKernel>::forward_with_scratch_sized::<LOG2>(
            data, scratch, twiddles,
        );
    }

    #[cfg_attr(debug_assertions, inline(never))]
    #[cfg_attr(not(debug_assertions), inline)]
    fn stockham_forward_normalized_sized<const LOG2: u32>(
        data: &mut [Complex64],
        scratch: &mut [Complex64],
        twiddles: &[Complex64],
    ) {
        Self::stockham_forward_sized::<LOG2>(data, scratch, twiddles);
        normalize_inplace(data, 1.0_f64 / (1usize << LOG2) as f64);
    }

    #[inline]
    fn short_winograd<const INVERSE: bool, const NORMALIZE: bool>(data: &mut [Complex64]) -> bool {
        crate::application::execution::kernel::mixed_radix::traits::short_winograd::<
            Self,
            INVERSE,
            NORMALIZE,
        >(data)
    }
    #[inline]
    unsafe fn small_pot_inplace<const INVERSE: bool, const NORMALIZE: bool>(
        data: &mut [Complex64],
    ) -> bool {
        small_pot_inplace_precise::<INVERSE, NORMALIZE>(data)
    }
    #[inline]
    fn composite_forward(data: &mut [Complex64], radices: &[usize]) {
        radix_composite::forward_inplace_with_radices(data, radices);
    }
    #[inline]
    fn composite_forward_with_pointwise(
        data: &mut [Complex64],
        radices: &[usize],
        pointwise_spectrum: &[Complex64],
    ) {
        radix_composite::forward_inplace_with_pointwise(data, radices, pointwise_spectrum);
    }
    #[inline]
    fn composite_inverse_unnorm(data: &mut [Complex64], radices: &[usize]) {
        radix_composite::inverse_inplace_unnorm_with_radices(data, radices);
    }
    #[inline]
    fn composite_inverse(data: &mut [Complex64], radices: &[usize]) {
        radix_composite::inverse_inplace_with_radices(data, radices);
    }
    #[inline]
    fn normalize(data: &mut [Complex64], n: usize) {
        normalize_inplace(data, 1.0_f64 / n as f64);
    }
    #[inline]
    fn transpose_matrix(src: &[Complex64], dst: &mut [Complex64], n1: usize, n2: usize) {
        transpose_matrix_precise(src, dst, n1, n2);
    }

    #[inline]
    unsafe fn small_pot_inplace_sized<
        const N: usize,
        const INVERSE: bool,
        const NORMALIZE: bool,
    >(
        data: &mut [Complex64],
    ) {
        small_pot_inplace_sized_precise::<N, INVERSE, NORMALIZE>(data);
    }

    #[inline]
    fn pot_inplace<const INVERSE: bool, const NORMALIZE: bool>(
        data: &mut [Self::Complex],
        twiddles: &[Self::Complex],
    ) {
        let n = data.len();
        match n {
            2 => unsafe {
                Self::small_pot_inplace_sized::<2, INVERSE, NORMALIZE>(data);
            },
            4 => unsafe {
                Self::small_pot_inplace_sized::<4, INVERSE, NORMALIZE>(data);
            },
            8 => unsafe {
                Self::small_pot_inplace_sized::<8, INVERSE, NORMALIZE>(data);
            },
            16 => unsafe {
                Self::small_pot_inplace_sized::<16, INVERSE, NORMALIZE>(data);
            },
            32 => unsafe {
                Self::small_pot_inplace_sized::<32, INVERSE, NORMALIZE>(data);
            },
            64 => unsafe {
                Self::small_pot_inplace_sized::<64, INVERSE, NORMALIZE>(data);
            },
            _ => {
                Self::with_scratch(n, |scratch| {
                    if INVERSE && NORMALIZE {
                        Self::stockham_forward_normalized(data, scratch, twiddles, n);
                    } else {
                        Self::stockham_forward(data, scratch, twiddles);
                    }
                });
            }
        }
    }

    fn pot_inplace_sized<
        const INVERSE: bool,
        const NORMALIZE: bool,
        S: PoTStrategy,
        const LOG2: u32,
    >(
        data: &mut [Self::Complex],
        twiddles: &[Self::Complex],
        _s: SizedPoT<S, LOG2>,
    ) {
        // const LOG2 drives selection (zero-cost monomorph); for <=64 preserve direct small
        // no-scratch path (memory efficiency + best for 32/64 which have dedicated AVX fixed column);
        // for 128+ (md-worst PoT) use stockham sized path so const LOG2 flows end-to-end to
        // kernel forward_with_scratch_sized -> transform_sized / with_strategy / len* bodies.
        // twiddles passed directly as &[Complex] (zero-copy reference from plan).
        match LOG2 {
            1 => unsafe {
                Self::small_pot_inplace_sized::<2, INVERSE, NORMALIZE>(data);
            },
            2 => unsafe {
                Self::small_pot_inplace_sized::<4, INVERSE, NORMALIZE>(data);
            },
            3 => unsafe {
                Self::small_pot_inplace_sized::<8, INVERSE, NORMALIZE>(data);
            },
            4 => unsafe {
                Self::small_pot_inplace_sized::<16, INVERSE, NORMALIZE>(data);
            },
            5 => unsafe {
                Self::small_pot_inplace_sized::<32, INVERSE, NORMALIZE>(data);
            },
            6 => unsafe {
                Self::small_pot_inplace_sized::<64, INVERSE, NORMALIZE>(data);
            },
            _ => {
                let n = 1usize << LOG2;
                Self::with_scratch(n, |scratch| {
                    if INVERSE && NORMALIZE {
                        Self::stockham_forward_normalized_sized::<LOG2>(data, scratch, twiddles);
                    } else {
                        Self::stockham_forward_sized::<LOG2>(data, scratch, twiddles);
                    }
                });
            }
        }
    }

    #[inline]
    fn small_pot_twiddles<const INVERSE: bool>(n: usize) -> &'static [Self::Complex] {
        let idx = n.trailing_zeros() as usize;
        if INVERSE {
            &TWIDDLES_INV_PRECISE[idx]
        } else {
            &TWIDDLES_FWD_PRECISE[idx]
        }
    }

    fn use_generated_codelet_plan(n: usize) -> bool {
        matches!(
            n,
            72 | 81
                | 96
                | 99
                | 108
                | 112
                | 120
                | 121
                | 126
                | 128
                | 144
                | 154
                | 168
                | 180
                | 189
                | 222
                | 242
                | 246
                | 259
                | 275
                | 280
                | 296
                | 363
                | 400
                | 484
        )
    }
}
