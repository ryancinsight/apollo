//! Planar-register row transform — ADR 0041, increment one.
//!
//! **Measured verdict: the planar premise is falsified.** Pinned per-pass
//! attribution at N = 1024 puts the planar row pass at ~6.3k TSC against the
//! interleaved kernel's ~5.6k — planar trades the interleaved multiply's
//! stage shuffles for boundary deinterleave/interleave networks and
//! in-register permutes of its own (~96 shuffle-class ops per row either
//! way), and even this host's efficiency core — the pinned measurement
//! core — had the shuffle throughput to keep the interleaved form from
//! being shuffle-port-bound; the performance-core case is implied, not
//! measured. Both register-row
//! shapes cost ~3x a batched streaming pass in arithmetic alone, so neither
//! reaches the reference construction. RustFFT's f64/1024 plan (read from
//! source, `avx_planner.rs`, f64 impl) is
//! `MixedRadix8xnAvx(Butterfly128Avx64)`: a hand-written 128-point
//! interleaved AVX base butterfly composed with an 8xn mixed-radix layer
//! carrying full-length scratch — a large L1-resident base transform, not
//! per-row register residency. The module stays test-gated as the differential oracle and
//! measurement instrument for that direction (ADR 0041 revision).
//!
//! Original design, kept for the record:
//!
//! Same five-pass driver shape as the interleaved resident kernel, with each
//! 32-sample row held as 8 real + 8 imaginary registers instead of 16
//! interleaved ones. Every whole-register stage (distances 16, 8, 4) is pure
//! add/sub/FMA traffic with zero shuffles; the two in-register stages use one
//! cross-half or adjacent permute per plane; and the interleaved↔planar
//! conversion at the row boundary is the native four-shuffle
//! deinterleave/interleave network (hermes `HS-AVX2-INTERLEAVE-OVERRIDES`).
//! The four-step matrix fold at load is four FMA-class ops per register pair
//! where the interleaved form pays three shuffles per multiply.
//!
//! Stage semantics, lane order, and output ordering are identical to
//! `ResidentRows` — DIF, natural in, bit-reversed out — so the driver reuses
//! the same transposes, rev-baked matrix, and closing involution, and the two
//! kernels are differential oracles for each other.

use super::{exact_lanes_supported, ResidentPlan, ResidentPlanCache, ROW};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::Complex;
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an outlined \
              load re-enters baseline codegen per call"
)]
#[inline(always)]
fn load<T, A>(data: &[T], at: usize) -> Vector<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    Vector::load_unaligned_from_slice(&data[at..at + <A as SimdStorage<T>>::LANE_COUNT])
        .expect("invariant: the caller bounds `at` by the slice length")
}

/// A butterfly result: `(u_re, u_im, v_re, v_im)`.
type Quad<T, A> = (Vector<T, A>, Vector<T, A>, Vector<T, A>, Vector<T, A>);

/// One planar whole-register DIF butterfly: returns `(u_re, u_im, v_re,
/// v_im)` with the difference twiddled by `(w_re, w_im)`. Eight FMA-class
/// operations, no shuffles.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope"
)]
#[inline(always)]
fn butterfly<T, A>(
    rx: Vector<T, A>,
    ix: Vector<T, A>,
    ry: Vector<T, A>,
    iy: Vector<T, A>,
    wr: Vector<T, A>,
    wi: Vector<T, A>,
) -> Quad<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let ur = rx + ry;
    let ui = ix + iy;
    let vr = rx - ry;
    let vi = ix - iy;
    (ur, ui, vr.mul_sub(wr, vi * wi), vr.mul_add(wi, vi * wr))
}

/// Every row of the `m x m` matrix through the planar register-resident DIF
/// transform; `APPLY_MATRIX` folds the four-step matrix in at load.
pub(crate) struct PlanarRows<'a, T, const APPLY_MATRIX: bool> {
    /// Interleaved samples, `2 * m * m` lanes.
    pub(crate) data: &'a mut [T],
    pub(crate) plan: &'a ResidentPlan<T>,
}

impl<T, const APPLY_MATRIX: bool> LaneKernel<T> for PlanarRows<'_, T, APPLY_MATRIX>
where
    T: LaneScalar + MixedRadixScalar,
{
    /// Whether the dispatched width handled the rows.
    type Output = bool;

    #[expect(
        clippy::inline_always,
        reason = "the row body must inline into the dispatcher's \
                  target-feature frame (hermes LaneKernel contract for large \
                  bodies)"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "one straight-line register-resident transform; splitting it \
                  moves live registers across call boundaries"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, _simd: Simd<T, A>) -> bool {
        if <A as SimdStorage<T>>::LANE_COUNT != 4 {
            return false;
        }
        let row_lanes = 2 * ROW;

        // Stage twiddles are row-invariant: planar table offsets are
        // W_32 at 0..16, W_16 at 16..24, W_8 at 24..28.
        let st_re = &self.plan.stages_re;
        let st_im = &self.plan.stages_im;

        // Distance-2 helpers: the butterfly is `x + sign * swap_pairs(x)`
        // with sign [1, 1, -1, -1], and the lane-3 twiddle is `W_4^1 =
        // (0, w4_im)`, applied by blending the cross-plane products in.
        let zero = T::from_precise(0.0);
        let one = T::from_precise(1.0);
        let neg = T::from_precise(-1.0);
        let sgn2 = Vector::<T, A>::load_unaligned_from_slice(&[one, one, neg, neg])
            .expect("invariant: four-lane constant");
        let sgn1 = Vector::<T, A>::load_unaligned_from_slice(&[one, neg, one, neg])
            .expect("invariant: four-lane constant");
        let mask3 = Vector::<T, A>::load_unaligned_from_slice(&[zero, zero, zero, neg])
            .expect("invariant: four-lane constant");
        let w4_pos = Vector::<T, A>::splat(self.plan.w4_im);
        let w4_neg = Vector::<T, A>::splat(self.plan.w4_im_neg);

        for row in 0..ROW {
            let base = row * row_lanes;
            let base_m = row * ROW;
            // Named locals: the eight-element register arrays fell to the
            // stack under the loop form (planar rows1 measured 6629 TSC,
            // above the interleaved kernel), exactly as the interleaved
            // body's history records. Named bindings are the one form the
            // register allocator reliably keeps in registers.
            let lo0 = load::<T, A>(self.data, base);
            let hi0 = load::<T, A>(self.data, base + 4);
            let (e0, o0) = lo0.deinterleave(hi0);
            let lo1 = load::<T, A>(self.data, base + 8);
            let hi1 = load::<T, A>(self.data, base + 12);
            let (e1, o1) = lo1.deinterleave(hi1);
            let lo2 = load::<T, A>(self.data, base + 16);
            let hi2 = load::<T, A>(self.data, base + 20);
            let (e2, o2) = lo2.deinterleave(hi2);
            let lo3 = load::<T, A>(self.data, base + 24);
            let hi3 = load::<T, A>(self.data, base + 28);
            let (e3, o3) = lo3.deinterleave(hi3);
            let lo4 = load::<T, A>(self.data, base + 32);
            let hi4 = load::<T, A>(self.data, base + 36);
            let (e4, o4) = lo4.deinterleave(hi4);
            let lo5 = load::<T, A>(self.data, base + 40);
            let hi5 = load::<T, A>(self.data, base + 44);
            let (e5, o5) = lo5.deinterleave(hi5);
            let lo6 = load::<T, A>(self.data, base + 48);
            let hi6 = load::<T, A>(self.data, base + 52);
            let (e6, o6) = lo6.deinterleave(hi6);
            let lo7 = load::<T, A>(self.data, base + 56);
            let hi7 = load::<T, A>(self.data, base + 60);
            let (e7, o7) = lo7.deinterleave(hi7);
            let (mut re0, mut im0) = (e0, o0);
            let (mut re1, mut im1) = (e1, o1);
            let (mut re2, mut im2) = (e2, o2);
            let (mut re3, mut im3) = (e3, o3);
            let (mut re4, mut im4) = (e4, o4);
            let (mut re5, mut im5) = (e5, o5);
            let (mut re6, mut im6) = (e6, o6);
            let (mut re7, mut im7) = (e7, o7);
            if APPLY_MATRIX {
                let mr = load::<T, A>(&self.plan.matrix_re, base_m);
                let mi = load::<T, A>(&self.plan.matrix_im, base_m);
                re0 = e0.mul_sub(mr, o0 * mi);
                im0 = e0.mul_add(mi, o0 * mr);
                let mr = load::<T, A>(&self.plan.matrix_re, base_m + 4);
                let mi = load::<T, A>(&self.plan.matrix_im, base_m + 4);
                re1 = e1.mul_sub(mr, o1 * mi);
                im1 = e1.mul_add(mi, o1 * mr);
                let mr = load::<T, A>(&self.plan.matrix_re, base_m + 8);
                let mi = load::<T, A>(&self.plan.matrix_im, base_m + 8);
                re2 = e2.mul_sub(mr, o2 * mi);
                im2 = e2.mul_add(mi, o2 * mr);
                let mr = load::<T, A>(&self.plan.matrix_re, base_m + 12);
                let mi = load::<T, A>(&self.plan.matrix_im, base_m + 12);
                re3 = e3.mul_sub(mr, o3 * mi);
                im3 = e3.mul_add(mi, o3 * mr);
                let mr = load::<T, A>(&self.plan.matrix_re, base_m + 16);
                let mi = load::<T, A>(&self.plan.matrix_im, base_m + 16);
                re4 = e4.mul_sub(mr, o4 * mi);
                im4 = e4.mul_add(mi, o4 * mr);
                let mr = load::<T, A>(&self.plan.matrix_re, base_m + 20);
                let mi = load::<T, A>(&self.plan.matrix_im, base_m + 20);
                re5 = e5.mul_sub(mr, o5 * mi);
                im5 = e5.mul_add(mi, o5 * mr);
                let mr = load::<T, A>(&self.plan.matrix_re, base_m + 24);
                let mi = load::<T, A>(&self.plan.matrix_im, base_m + 24);
                re6 = e6.mul_sub(mr, o6 * mi);
                im6 = e6.mul_add(mi, o6 * mr);
                let mr = load::<T, A>(&self.plan.matrix_re, base_m + 28);
                let mi = load::<T, A>(&self.plan.matrix_im, base_m + 28);
                re7 = e7.mul_sub(mr, o7 * mi);
                im7 = e7.mul_add(mi, o7 * mr);
            }

            // Distance 16 (L = 32, table at 0): pairs (r, r + 4).
            let wr = load::<T, A>(st_re, 0);
            let wi = load::<T, A>(st_im, 0);
            let (ur, ui, vr, vi) = butterfly(re0, im0, re4, im4, wr, wi);
            re0 = ur;
            im0 = ui;
            re4 = vr;
            im4 = vi;
            let wr = load::<T, A>(st_re, 4);
            let wi = load::<T, A>(st_im, 4);
            let (ur, ui, vr, vi) = butterfly(re1, im1, re5, im5, wr, wi);
            re1 = ur;
            im1 = ui;
            re5 = vr;
            im5 = vi;
            let wr = load::<T, A>(st_re, 8);
            let wi = load::<T, A>(st_im, 8);
            let (ur, ui, vr, vi) = butterfly(re2, im2, re6, im6, wr, wi);
            re2 = ur;
            im2 = ui;
            re6 = vr;
            im6 = vi;
            let wr = load::<T, A>(st_re, 12);
            let wi = load::<T, A>(st_im, 12);
            let (ur, ui, vr, vi) = butterfly(re3, im3, re7, im7, wr, wi);
            re3 = ur;
            im3 = ui;
            re7 = vr;
            im7 = vi;

            // Distance 8 (L = 16, table at 16): pairs (r, r + 2) per half.
            let wr = load::<T, A>(st_re, 16);
            let wi = load::<T, A>(st_im, 16);
            let (ur, ui, vr, vi) = butterfly(re0, im0, re2, im2, wr, wi);
            re0 = ur;
            im0 = ui;
            re2 = vr;
            im2 = vi;
            let wr = load::<T, A>(st_re, 20);
            let wi = load::<T, A>(st_im, 20);
            let (ur, ui, vr, vi) = butterfly(re1, im1, re3, im3, wr, wi);
            re1 = ur;
            im1 = ui;
            re3 = vr;
            im3 = vi;
            let wr = load::<T, A>(st_re, 16);
            let wi = load::<T, A>(st_im, 16);
            let (ur, ui, vr, vi) = butterfly(re4, im4, re6, im6, wr, wi);
            re4 = ur;
            im4 = ui;
            re6 = vr;
            im6 = vi;
            let wr = load::<T, A>(st_re, 20);
            let wi = load::<T, A>(st_im, 20);
            let (ur, ui, vr, vi) = butterfly(re5, im5, re7, im7, wr, wi);
            re5 = ur;
            im5 = ui;
            re7 = vr;
            im7 = vi;

            // Distance 4 (L = 8, table at 24): shared twiddle register pair.
            let w8r = load::<T, A>(st_re, 24);
            let w8i = load::<T, A>(st_im, 24);
            let (ur, ui, vr, vi) = butterfly(re0, im0, re1, im1, w8r, w8i);
            re0 = ur;
            im0 = ui;
            re1 = vr;
            im1 = vi;
            let (ur, ui, vr, vi) = butterfly(re2, im2, re3, im3, w8r, w8i);
            re2 = ur;
            im2 = ui;
            re3 = vr;
            im3 = vi;
            let (ur, ui, vr, vi) = butterfly(re4, im4, re5, im5, w8r, w8i);
            re4 = ur;
            im4 = ui;
            re5 = vr;
            im5 = vi;
            let (ur, ui, vr, vi) = butterfly(re6, im6, re7, im7, w8r, w8i);
            re6 = ur;
            im6 = ui;
            re7 = vr;
            im7 = vi;

            // Distance 2 (L = 4): in-register cross-half butterfly, then
            // the lane-3 `W_4^1` twiddle as a cross-plane blend.
            let rb = re0.mul_add(sgn2, re0.swap_pairs());
            let ib = im0.mul_add(sgn2, im0.swap_pairs());
            re0 = mask3.blend(ib * w4_neg, rb);
            im0 = mask3.blend(rb * w4_pos, ib);
            let rb = re1.mul_add(sgn2, re1.swap_pairs());
            let ib = im1.mul_add(sgn2, im1.swap_pairs());
            re1 = mask3.blend(ib * w4_neg, rb);
            im1 = mask3.blend(rb * w4_pos, ib);
            let rb = re2.mul_add(sgn2, re2.swap_pairs());
            let ib = im2.mul_add(sgn2, im2.swap_pairs());
            re2 = mask3.blend(ib * w4_neg, rb);
            im2 = mask3.blend(rb * w4_pos, ib);
            let rb = re3.mul_add(sgn2, re3.swap_pairs());
            let ib = im3.mul_add(sgn2, im3.swap_pairs());
            re3 = mask3.blend(ib * w4_neg, rb);
            im3 = mask3.blend(rb * w4_pos, ib);
            let rb = re4.mul_add(sgn2, re4.swap_pairs());
            let ib = im4.mul_add(sgn2, im4.swap_pairs());
            re4 = mask3.blend(ib * w4_neg, rb);
            im4 = mask3.blend(rb * w4_pos, ib);
            let rb = re5.mul_add(sgn2, re5.swap_pairs());
            let ib = im5.mul_add(sgn2, im5.swap_pairs());
            re5 = mask3.blend(ib * w4_neg, rb);
            im5 = mask3.blend(rb * w4_pos, ib);
            let rb = re6.mul_add(sgn2, re6.swap_pairs());
            let ib = im6.mul_add(sgn2, im6.swap_pairs());
            re6 = mask3.blend(ib * w4_neg, rb);
            im6 = mask3.blend(rb * w4_pos, ib);
            let rb = re7.mul_add(sgn2, re7.swap_pairs());
            let ib = im7.mul_add(sgn2, im7.swap_pairs());
            re7 = mask3.blend(ib * w4_neg, rb);
            im7 = mask3.blend(rb * w4_pos, ib);

            // Distance 1 (L = 2): adjacent-lane butterfly, twiddle one.
            re0 = re0.mul_add(sgn1, re0.swap_adjacent());
            im0 = im0.mul_add(sgn1, im0.swap_adjacent());
            re1 = re1.mul_add(sgn1, re1.swap_adjacent());
            im1 = im1.mul_add(sgn1, im1.swap_adjacent());
            re2 = re2.mul_add(sgn1, re2.swap_adjacent());
            im2 = im2.mul_add(sgn1, im2.swap_adjacent());
            re3 = re3.mul_add(sgn1, re3.swap_adjacent());
            im3 = im3.mul_add(sgn1, im3.swap_adjacent());
            re4 = re4.mul_add(sgn1, re4.swap_adjacent());
            im4 = im4.mul_add(sgn1, im4.swap_adjacent());
            re5 = re5.mul_add(sgn1, re5.swap_adjacent());
            im5 = im5.mul_add(sgn1, im5.swap_adjacent());
            re6 = re6.mul_add(sgn1, re6.swap_adjacent());
            im6 = im6.mul_add(sgn1, im6.swap_adjacent());
            re7 = re7.mul_add(sgn1, re7.swap_adjacent());
            im7 = im7.mul_add(sgn1, im7.swap_adjacent());

            // Reinterleave and store.
            let (lo, hi) = re0.interleave(im0);
            lo.store_unaligned_to_slice(&mut self.data[base..base + 4])
                .expect("invariant: one register per slot");
            hi.store_unaligned_to_slice(&mut self.data[base + 4..base + 8])
                .expect("invariant: one register per slot");
            let (lo, hi) = re1.interleave(im1);
            lo.store_unaligned_to_slice(&mut self.data[base + 8..base + 12])
                .expect("invariant: one register per slot");
            hi.store_unaligned_to_slice(&mut self.data[base + 12..base + 16])
                .expect("invariant: one register per slot");
            let (lo, hi) = re2.interleave(im2);
            lo.store_unaligned_to_slice(&mut self.data[base + 16..base + 20])
                .expect("invariant: one register per slot");
            hi.store_unaligned_to_slice(&mut self.data[base + 20..base + 24])
                .expect("invariant: one register per slot");
            let (lo, hi) = re3.interleave(im3);
            lo.store_unaligned_to_slice(&mut self.data[base + 24..base + 28])
                .expect("invariant: one register per slot");
            hi.store_unaligned_to_slice(&mut self.data[base + 28..base + 32])
                .expect("invariant: one register per slot");
            let (lo, hi) = re4.interleave(im4);
            lo.store_unaligned_to_slice(&mut self.data[base + 32..base + 36])
                .expect("invariant: one register per slot");
            hi.store_unaligned_to_slice(&mut self.data[base + 36..base + 40])
                .expect("invariant: one register per slot");
            let (lo, hi) = re5.interleave(im5);
            lo.store_unaligned_to_slice(&mut self.data[base + 40..base + 44])
                .expect("invariant: one register per slot");
            hi.store_unaligned_to_slice(&mut self.data[base + 44..base + 48])
                .expect("invariant: one register per slot");
            let (lo, hi) = re6.interleave(im6);
            lo.store_unaligned_to_slice(&mut self.data[base + 48..base + 52])
                .expect("invariant: one register per slot");
            hi.store_unaligned_to_slice(&mut self.data[base + 52..base + 56])
                .expect("invariant: one register per slot");
            let (lo, hi) = re7.interleave(im7);
            lo.store_unaligned_to_slice(&mut self.data[base + 56..base + 60])
                .expect("invariant: one register per slot");
            hi.store_unaligned_to_slice(&mut self.data[base + 60..base + 64])
                .expect("invariant: one register per slot");
        }
        true
    }
}

/// The planar four-step driver: identical pass structure to
/// [`super::four_step_resident`], dispatching the planar row kernel.
pub(crate) fn four_step_planar<T, const INVERSE: bool>(data: &mut [Complex<T>]) -> bool
where
    T: ResidentPlanCache,
    Complex<T>: bytemuck::Pod,
{
    if data.len() != ROW * ROW {
        return false;
    }
    // Match the interleaved driver: capability resolution precedes plan
    // construction and every in-place permutation.
    if !exact_lanes_supported::<4, T>() {
        return false;
    }
    let plan = T::cached_resident_plan::<INVERSE>(ROW * ROW);

    #[cfg(all(test, windows, target_arch = "x86_64"))]
    macro_rules! sect {
        ($label:literal, $body:block) => {{
            let t0 = unsafe { core::arch::x86_64::_rdtsc() };
            let out = $body;
            let t1 = unsafe { core::arch::x86_64::_rdtsc() };
            static SECTIONS: std::sync::LazyLock<bool> =
                std::sync::LazyLock::new(|| std::env::var_os("RESIDENT_SECTIONS").is_some());
            if *SECTIONS {
                eprintln!("PSECT {} {}", $label, t1 - t0);
            }
            out
        }};
    }
    #[cfg(not(all(test, windows, target_arch = "x86_64")))]
    macro_rules! sect {
        ($label:literal, $body:block) => {
            $body
        };
    }

    sect!("t1", { super::transpose_samples(data, ROW) });
    {
        let flat: &mut [T] = bytemuck::cast_slice_mut(data);
        if !sect!("rows1", {
            hermes_simd::vectorize_lanes::<4, T, _>(PlanarRows::<T, false> {
                data: flat,
                plan: plan.as_ref(),
            })
            .unwrap_or(false)
        }) {
            super::transpose_samples(data, ROW);
            return false;
        }
    }
    sect!("t2", { super::transpose_samples(data, ROW) });
    {
        let flat: &mut [T] = bytemuck::cast_slice_mut(data);
        let handled = sect!("rows2", {
            hermes_simd::vectorize_lanes::<4, T, _>(PlanarRows::<T, true> {
                data: flat,
                plan: plan.as_ref(),
            })
        });
        debug_assert_eq!(handled, Some(true), "width accepted the first pass");
    }
    sect!("untangle", { super::untangle_output(data, ROW) });
    true
}

#[cfg(test)]
#[path = "planar_tests.rs"]
mod tests;
