use super::super::precision_bridge::{run_via_complex32, Complex32Bridge};
use super::super::radix_shape::should_use_bluestein_instead_of_composite;
use super::super::radix_stage::normalize_inplace_c32;
use super::super::{bluestein, radix_composite, stockham};
use super::caches::{
    cached_composite_radices, cached_twiddle_fwd_32, cached_twiddle_inv_32,
    with_stockham_scratch_32,
};
use super::traits::{forward_short_winograd, inverse_short_winograd};

/// In-place forward FFT (unnormalized) for compact storage routed through `Complex32`.
///
/// ## Dispatch
///
/// All power-of-two sizes promote to f32, run the Stockham f32 autosort kernel
/// (no bit-reversal), and demote back to f16. Non-PoT 2/3/5/7-smooth sizes use
/// the composite mixed-radix path via `run_via_complex32`. Other lengths use
/// Bluestein-f32.
///
#[inline]
pub(crate) fn forward_compact_storage<S: Complex32Bridge>(data: &mut [S]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let n = data.len();
        run_via_complex32(data, |buf| {
            if forward_short_winograd(buf) {
                return;
            }
            let tw = cached_twiddle_fwd_32(n);
            with_stockham_scratch_32(n, |scratch| {
                <f32 as stockham::StockhamKernel>::forward_with_scratch(buf, scratch, tw.as_ref());
            });
        });
    } else {
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                run_via_complex32(data, |buf| {
                    radix_composite::forward_inplace_with_radices(buf, &radices)
                });
                return;
            }
        }
        run_via_complex32(data, bluestein::forward_inplace_32);
    }
}

/// In-place inverse FFT (unnormalized) for compact storage routed through `Complex32`.
///
/// PoT sizes: promote compact storage to f32, run Stockham f32 with inverse
/// twiddles and no 1/N scale, then demote back to storage precision.
#[inline]
pub(crate) fn inverse_unnorm_compact_storage<S: Complex32Bridge>(data: &mut [S]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let n = data.len();
        run_via_complex32(data, |buf| {
            if inverse_short_winograd(buf, false) {
                return;
            }
            let tw = cached_twiddle_inv_32(n);
            with_stockham_scratch_32(n, |scratch| {
                <f32 as stockham::StockhamKernel>::forward_with_scratch(buf, scratch, tw.as_ref());
            });
        });
    } else {
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                run_via_complex32(data, |buf| {
                    radix_composite::inverse_inplace_unnorm_with_radices(buf, &radices)
                });
                return;
            }
        }
        run_via_complex32(data, bluestein::inverse_inplace_unnorm_32);
    }
}

/// In-place inverse FFT normalized by 1/N for compact storage routed through `Complex32`.
///
/// PoT sizes: promote compact storage to f32, run Stockham f32 with inverse
/// twiddles, apply 1/N scale, then demote back to storage precision.
#[inline]
pub(crate) fn inverse_compact_storage<S: Complex32Bridge>(data: &mut [S]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let n = data.len();
        run_via_complex32(data, |buf| {
            if inverse_short_winograd(buf, true) {
                return;
            }
            let tw = cached_twiddle_inv_32(n);
            with_stockham_scratch_32(n, |scratch| {
                <f32 as stockham::StockhamKernel>::forward_with_scratch(buf, scratch, tw.as_ref());
            });
            normalize_inplace_c32(buf, 1.0f32 / n as f32);
        });
    } else {
        if !should_use_bluestein_instead_of_composite(data.len()) {
            if let Some(radices) = cached_composite_radices(data.len()) {
                run_via_complex32(data, |buf| {
                    radix_composite::inverse_inplace_with_radices(buf, &radices)
                });
                return;
            }
        }
        run_via_complex32(data, bluestein::inverse_inplace_32);
    }
}
