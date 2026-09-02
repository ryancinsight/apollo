//! Decimation-in-frequency stage set for the planar four-step's second axis.
//!
//! ## Why a second stage set exists
//!
//! The deinterleave earns bit-reversed rows for free by writing each row to
//! `rev(row)`, which is exactly the order a decimation-in-time stage set wants.
//! The transpose between the two axes then destroys that order, so the route
//! used to run a whole pass restoring it — 4.9 to 6.2% of the route, on the
//! order of the transpose itself, and its entire content was undoing the pass
//! before it (`gap_audit.md#planar-pass-attribution`).
//!
//! Decimation in frequency inverts both ends of that. It consumes natural
//! order, which is what the transpose leaves, and produces bit-reversed order,
//! which the sink absorbs for free by reading `rev(row)` — the same trick the
//! deinterleave already uses on the source side. The repair pass has nowhere
//! left to be.
//!
//! ## Why it is a sibling rather than a parameter
//!
//! The two are different algorithms, not two configurations of one. DIT
//! multiplies the odd operand *before* the butterfly and DIF multiplies the
//! difference *after* it, so nearly every arithmetic line differs; a const
//! parameter would monomorphize to these same two bodies while making both
//! harder to read. What they genuinely share — the twiddle table, the load and
//! store helpers, the fold contract — is shared.
//!
//! The table is shared exactly: stage sub-length `l` occupies `l / 2` entries
//! at offset `l / 2 - 1`, holding `W_l^j`. DIT walks those stages upward and
//! DIF downward, over the same values.

use super::{load, store};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector};

/// All stages of `batch` independent length-`len` transforms over planar data,
/// decimated in frequency.
///
/// Input rows in natural order; output rows bit-reversed. Dispatch happens once
/// for the whole stage set, as [`super::BatchedStages`] documents.
pub(super) struct BatchedStagesDif<'a, T> {
    pub(super) re: &'a mut [T],
    pub(super) im: &'a mut [T],
    pub(super) tw: &'a [(T, T)],
    /// Planar four-step twiddles multiplied into the first stage's loads, or
    /// `None`. Row-major with row stride `batch`, rows in the same *natural*
    /// order the data rows now carry — the mirror of the bit-reversed planes
    /// the decimation-in-time set required.
    pub(super) fold: Option<(&'a [T], &'a [T])>,
    pub(super) batch: usize,
    pub(super) stride: usize,
    pub(super) len: usize,
}

impl<T> LaneKernel<T> for BatchedStagesDif<'_, T>
where
    T: LaneScalar + MixedRadixScalar,
{
    type Output = ();

    #[expect(
        clippy::inline_always,
        reason = "large LaneKernel::call body must fold into the dispatcher's target-feature scope"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
        let lanes = <A as SimdStorage<T>>::LANE_COUNT;
        let b = self.batch;
        let s = self.stride;
        let mut l = self.len;

        // An odd `log2(len)` leaves one stage unpaired. It runs first, at the
        // widest distance, so the four-step fold still rides the pass that
        // reads every element exactly once — the placement the DIT set gives
        // it at `l == 2`.
        if self.len.trailing_zeros() % 2 == 1 {
            let half = l >> 1;
            let base = half - 1;
            for j in 0..half {
                let (twr, twi) = self.tw[base + j];
                let (wr, wi) = (simd.splat(twr), simd.splat(twi));
                let ia = j * s;
                let ib = ia + half * s;
                let ta = j * b;
                let tb = ta + half * b;
                let mut k = 0;
                while k + lanes <= b {
                    let mut ar = load::<T, A>(self.re, ia + k);
                    let mut ai = load::<T, A>(self.im, ia + k);
                    let mut br = load::<T, A>(self.re, ib + k);
                    let mut bi = load::<T, A>(self.im, ib + k);
                    if let Some((pr, pi)) = self.fold {
                        let tw = |r: Vector<T, A>, i: Vector<T, A>, at: usize| {
                            let vr = load::<T, A>(pr, at + k);
                            let vi = load::<T, A>(pi, at + k);
                            (vr.mul_add(r, -(vi * i)), vr.mul_add(i, vi * r))
                        };
                        (ar, ai) = tw(ar, ai, ta);
                        (br, bi) = tw(br, bi, tb);
                    }
                    // The sum keeps its place and the difference carries the
                    // twiddle. That ordering is what decimation in frequency
                    // is; the time-decimated set multiplies first instead.
                    let (dr, di) = (ar - br, ai - bi);
                    store::<T, A>(ar + br, self.re, ia + k);
                    store::<T, A>(ai + bi, self.im, ia + k);
                    store::<T, A>(wr.mul_add(dr, -(wi * di)), self.re, ib + k);
                    store::<T, A>(wr.mul_add(di, wi * dr), self.im, ib + k);
                    k += lanes;
                }
                for k in k..b {
                    let (mut ar, mut ai) = (self.re[ia + k], self.im[ia + k]);
                    let (mut br, mut bi) = (self.re[ib + k], self.im[ib + k]);
                    if let Some((pr, pi)) = self.fold {
                        let tw = |r: T, i: T, at: usize| {
                            let (vr, vi) = (pr[at + k], pi[at + k]);
                            (vr * r - vi * i, vr * i + vi * r)
                        };
                        (ar, ai) = tw(ar, ai, ta);
                        (br, bi) = tw(br, bi, tb);
                    }
                    let (dr, di) = (ar - br, ai - bi);
                    self.re[ia + k] = ar + br;
                    self.im[ia + k] = ai + bi;
                    self.re[ib + k] = twr * dr - twi * di;
                    self.im[ib + k] = twr * di + twi * dr;
                }
            }
            l >>= 1;
        }

        // Two stages per pass over the data, for the reason the DIT set gives:
        // at these sizes the traffic binds, so stage `l` fuses with stage
        // `l / 2` and each of the four operands is loaded and stored once
        // rather than twice.
        while l >= 4 {
            let quarter = l >> 2;
            let half = l >> 1;
            let groups = self.len / l;
            // Stage `l` holds `W_l^j` at `l / 2 - 1`; stage `l / 2` holds
            // `W_(l/2)^j` at `l / 4 - 1`.
            let wide = half - 1;
            let narrow = quarter - 1;
            let folding = l == self.len;
            for j in 0..quarter {
                let (w1r, w1i) = self.tw[wide + j];
                let (w2r, w2i) = self.tw[wide + j + quarter];
                let (w3r, w3i) = self.tw[narrow + j];
                let (v1r, v1i) = (simd.splat(w1r), simd.splat(w1i));
                let (v2r, v2i) = (simd.splat(w2r), simd.splat(w2i));
                let (v3r, v3i) = (simd.splat(w3r), simd.splat(w3i));
                let fold = if folding { self.fold } else { None };
                for g in 0..groups {
                    let row_a = g * l + j;
                    let ia = row_a * s;
                    let ib = ia + quarter * s;
                    let ic = ia + half * s;
                    let id = ic + quarter * s;
                    let ta = row_a * b;
                    let tb = ta + quarter * b;
                    let tc = ta + half * b;
                    let td = tc + quarter * b;
                    let mut k = 0;
                    while k + lanes <= b {
                        let mut ar = load::<T, A>(self.re, ia + k);
                        let mut ai = load::<T, A>(self.im, ia + k);
                        let mut br = load::<T, A>(self.re, ib + k);
                        let mut bi = load::<T, A>(self.im, ib + k);
                        let mut cr = load::<T, A>(self.re, ic + k);
                        let mut ci = load::<T, A>(self.im, ic + k);
                        let mut dr = load::<T, A>(self.re, id + k);
                        let mut di = load::<T, A>(self.im, id + k);
                        if let Some((pr, pi)) = fold {
                            let tw = |r: Vector<T, A>, i: Vector<T, A>, at: usize| {
                                let vr = load::<T, A>(pr, at + k);
                                let vi = load::<T, A>(pi, at + k);
                                (vr.mul_add(r, -(vi * i)), vr.mul_add(i, vi * r))
                            };
                            (ar, ai) = tw(ar, ai, ta);
                            (br, bi) = tw(br, bi, tb);
                            (cr, ci) = tw(cr, ci, tc);
                            (dr, di) = tw(dr, di, td);
                        }

                        // Stage `l`, distance `l / 2`: (a,c) against `W_l^j`
                        // and (b,d) against `W_l^(j + l/4)`.
                        let (uar, uai) = (ar + cr, ai + ci);
                        let (ubr, ubi) = (br + dr, bi + di);
                        let (scr, sci) = (ar - cr, ai - ci);
                        let (sdr, sdi) = (br - dr, bi - di);
                        let ucr = v1r.mul_add(scr, -(v1i * sci));
                        let uci = v1r.mul_add(sci, v1i * scr);
                        let udr = v2r.mul_add(sdr, -(v2i * sdi));
                        let udi = v2r.mul_add(sdi, v2i * sdr);

                        // Stage `l / 2`, distance `l / 4`: (a,b) and (c,d),
                        // both against `W_(l/2)^j`. Neither operand has left a
                        // register.
                        let (sbr, sbi) = (uar - ubr, uai - ubi);
                        let (sfr, sfi) = (ucr - udr, uci - udi);

                        store::<T, A>(uar + ubr, self.re, ia + k);
                        store::<T, A>(uai + ubi, self.im, ia + k);
                        store::<T, A>(v3r.mul_add(sbr, -(v3i * sbi)), self.re, ib + k);
                        store::<T, A>(v3r.mul_add(sbi, v3i * sbr), self.im, ib + k);
                        store::<T, A>(ucr + udr, self.re, ic + k);
                        store::<T, A>(uci + udi, self.im, ic + k);
                        store::<T, A>(v3r.mul_add(sfr, -(v3i * sfi)), self.re, id + k);
                        store::<T, A>(v3r.mul_add(sfi, v3i * sfr), self.im, id + k);
                        k += lanes;
                    }
                    // Scalar remainder when the batch is not a lane multiple.
                    for k in k..b {
                        let (mut ar, mut ai) = (self.re[ia + k], self.im[ia + k]);
                        let (mut br, mut bi) = (self.re[ib + k], self.im[ib + k]);
                        let (mut cr, mut ci) = (self.re[ic + k], self.im[ic + k]);
                        let (mut dr, mut di) = (self.re[id + k], self.im[id + k]);
                        if let Some((pr, pi)) = fold {
                            let tw = |r: T, i: T, at: usize| {
                                let (vr, vi) = (pr[at + k], pi[at + k]);
                                (vr * r - vi * i, vr * i + vi * r)
                            };
                            (ar, ai) = tw(ar, ai, ta);
                            (br, bi) = tw(br, bi, tb);
                            (cr, ci) = tw(cr, ci, tc);
                            (dr, di) = tw(dr, di, td);
                        }

                        let (uar, uai) = (ar + cr, ai + ci);
                        let (ubr, ubi) = (br + dr, bi + di);
                        let (scr, sci) = (ar - cr, ai - ci);
                        let (sdr, sdi) = (br - dr, bi - di);
                        let ucr = w1r * scr - w1i * sci;
                        let uci = w1r * sci + w1i * scr;
                        let udr = w2r * sdr - w2i * sdi;
                        let udi = w2r * sdi + w2i * sdr;

                        let (sbr, sbi) = (uar - ubr, uai - ubi);
                        let (sfr, sfi) = (ucr - udr, uci - udi);

                        self.re[ia + k] = uar + ubr;
                        self.im[ia + k] = uai + ubi;
                        self.re[ib + k] = w3r * sbr - w3i * sbi;
                        self.im[ib + k] = w3r * sbi + w3i * sbr;
                        self.re[ic + k] = ucr + udr;
                        self.im[ic + k] = uci + udi;
                        self.re[id + k] = w3r * sfr - w3i * sfi;
                        self.im[id + k] = w3r * sfi + w3i * sfr;
                    }
                }
            }
            l >>= 2;
        }
    }
}
