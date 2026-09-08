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

use super::{load, reverse_row, store, store_interleaved};
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
    /// Interleaved output written by the last pass in place of the planes,
    /// or `None` to leave the result in the planes. Rows of `batch`
    /// complexes as `2 * batch` reals; plane row `p` lands in output row
    /// `rev(p)`, the permutation the reinterleave pass used to absorb. Writing
    /// here deletes that pass: the last stage pair stores every element
    /// exactly once, and one register interleave plus two interleaved stores
    /// replace the two plane stores.
    pub(super) sink: Option<&'a mut [T]>,
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
        let row_bits = self.len.trailing_zeros();
        let mut sink = self.sink;
        let mut l = self.len;

        // An odd `log2(len)` leaves one stage unpaired. It runs first, at the
        // widest distance, so the four-step fold still rides the pass that
        // reads every element exactly once — the placement the DIT set gives
        // it at `l == 2`.
        if self.len.trailing_zeros() % 2 == 1 {
            let half = l >> 1;
            let base = half - 1;
            // When this is the whole transform it is also the last pass.
            let last = self.len == 2;
            for j in 0..half {
                let (twr, twi) = self.tw[base + j];
                let (wr, wi) = (simd.splat(twr), simd.splat(twi));
                let ia = j * s;
                let ib = ia + half * s;
                let ta = j * b;
                let tb = ta + half * b;
                let oa = reverse_row(j, row_bits) * b * 2;
                let ob = reverse_row(j + half, row_bits) * b * 2;
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
                    let (oar, oai) = (ar + br, ai + bi);
                    let (obr, obi) = (wr.mul_add(dr, -(wi * di)), wr.mul_add(di, wi * dr));
                    match (last, sink.as_deref_mut()) {
                        (true, Some(out)) => {
                            store_interleaved::<T, A>(oar, oai, out, oa + 2 * k);
                            store_interleaved::<T, A>(obr, obi, out, ob + 2 * k);
                        }
                        _ => {
                            store::<T, A>(oar, self.re, ia + k);
                            store::<T, A>(oai, self.im, ia + k);
                            store::<T, A>(obr, self.re, ib + k);
                            store::<T, A>(obi, self.im, ib + k);
                        }
                    }
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
                    let (oar, oai) = (ar + br, ai + bi);
                    let (obr, obi) = (twr * dr - twi * di, twr * di + twi * dr);
                    match (last, sink.as_deref_mut()) {
                        (true, Some(out)) => {
                            out[oa + 2 * k] = oar;
                            out[oa + 2 * k + 1] = oai;
                            out[ob + 2 * k] = obr;
                            out[ob + 2 * k + 1] = obi;
                        }
                        _ => {
                            self.re[ia + k] = oar;
                            self.im[ia + k] = oai;
                            self.re[ib + k] = obr;
                            self.im[ib + k] = obi;
                        }
                    }
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
            // The pair (4, 2) is the last pass of every stage set that has a
            // paired loop at all.
            let last = l == 4;
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
                    let oa = reverse_row(row_a, row_bits) * b * 2;
                    let ob = reverse_row(row_a + quarter, row_bits) * b * 2;
                    let oc = reverse_row(row_a + half, row_bits) * b * 2;
                    let od = reverse_row(row_a + half + quarter, row_bits) * b * 2;
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

                        let (oar, oai) = (uar + ubr, uai + ubi);
                        let (obr, obi) =
                            (v3r.mul_add(sbr, -(v3i * sbi)), v3r.mul_add(sbi, v3i * sbr));
                        let (ocr, oci) = (ucr + udr, uci + udi);
                        let (odr, odi) =
                            (v3r.mul_add(sfr, -(v3i * sfi)), v3r.mul_add(sfi, v3i * sfr));
                        match (last, sink.as_deref_mut()) {
                            (true, Some(out)) => {
                                store_interleaved::<T, A>(oar, oai, out, oa + 2 * k);
                                store_interleaved::<T, A>(obr, obi, out, ob + 2 * k);
                                store_interleaved::<T, A>(ocr, oci, out, oc + 2 * k);
                                store_interleaved::<T, A>(odr, odi, out, od + 2 * k);
                            }
                            _ => {
                                store::<T, A>(oar, self.re, ia + k);
                                store::<T, A>(oai, self.im, ia + k);
                                store::<T, A>(obr, self.re, ib + k);
                                store::<T, A>(obi, self.im, ib + k);
                                store::<T, A>(ocr, self.re, ic + k);
                                store::<T, A>(oci, self.im, ic + k);
                                store::<T, A>(odr, self.re, id + k);
                                store::<T, A>(odi, self.im, id + k);
                            }
                        }
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

                        let (oar, oai) = (uar + ubr, uai + ubi);
                        let (obr, obi) = (w3r * sbr - w3i * sbi, w3r * sbi + w3i * sbr);
                        let (ocr, oci) = (ucr + udr, uci + udi);
                        let (odr, odi) = (w3r * sfr - w3i * sfi, w3r * sfi + w3i * sfr);
                        match (last, sink.as_deref_mut()) {
                            (true, Some(out)) => {
                                out[oa + 2 * k] = oar;
                                out[oa + 2 * k + 1] = oai;
                                out[ob + 2 * k] = obr;
                                out[ob + 2 * k + 1] = obi;
                                out[oc + 2 * k] = ocr;
                                out[oc + 2 * k + 1] = oci;
                                out[od + 2 * k] = odr;
                                out[od + 2 * k + 1] = odi;
                            }
                            _ => {
                                self.re[ia + k] = oar;
                                self.im[ia + k] = oai;
                                self.re[ib + k] = obr;
                                self.im[ib + k] = obi;
                                self.re[ic + k] = ocr;
                                self.im[ic + k] = oci;
                                self.re[id + k] = odr;
                                self.im[id + k] = odi;
                            }
                        }
                    }
                }
            }
            l >>= 2;
        }
    }
}
