//! Vectorized interleaved↔planar boundaries for the batched driver.
//!
//! The per-pass attribution (`RESIDENT_SECTIONS=1`) put 45% of the batched
//! route's budget in data movement. With the native AVX2 interleave network
//! (hermes `HS-AVX2-INTERLEAVE-OVERRIDES`) and capability-hoisted view-chunk
//! access, the reinterleave pass measured 1252 -> 900 TSC at N = 1024 pinned
//! and is kept. The mirrored deinterleave kernel measured *slower* than its
//! scalar loop (1431 -> 1520) — LLVM already auto-vectorizes that loop well,
//! and the earlier "boundary vectorization loses" verdict still holds on the
//! load side — so only the store-side kernel exists here.
//!
//! The kernel gates on the four-lane width and reports `false` otherwise;
//! the driver falls back to the scalar loop, which stays the reference
//! implementation for every other width.

use super::BatchedPlanCache;
use hermes_simd::{LaneKernel, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

/// Padded planar planes back into interleaved rows, natural order.
pub(crate) struct InterleaveRows<'a, T> {
    /// Padded real plane.
    pub(crate) re: &'a [T],
    /// Padded imaginary plane.
    pub(crate) im: &'a [T],
    /// Interleaved output, `2 * n` lanes.
    pub(crate) data: &'a mut [T],
    /// Row length in complexes.
    pub(crate) m: usize,
    /// Padded plane row stride.
    pub(crate) stride: usize,
}

impl<T: BatchedPlanCache> LaneKernel<T> for InterleaveRows<'_, T> {
    /// Whether the dispatched width handled the pass.
    type Output = bool;

    #[expect(
        clippy::inline_always,
        reason = "the body must inline into the dispatcher's target-feature \
                  frame (hermes LaneKernel contract)"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) -> bool {
        if <A as SimdStorage<T>>::LANE_COUNT != 4 || self.m % 4 != 0 || self.stride % 4 != 0 {
            return false;
        }
        let re_view = simd.view(self.re);
        let im_view = simd.view(self.im);
        let mut data_view = simd.view_mut(self.data);
        for row in 0..self.m {
            let src_c = row * self.stride / 4;
            let dst_c = row * self.m / 2;
            for r in 0..self.m / 4 {
                let e = Vector::from_view_chunk(&re_view, src_c + r);
                let o = Vector::from_view_chunk(&im_view, src_c + r);
                let (lo, hi) = e.interleave(o);
                lo.store_to_view_chunk(&mut data_view, dst_c + 2 * r);
                hi.store_to_view_chunk(&mut data_view, dst_c + 2 * r + 1);
            }
        }
        true
    }
}
