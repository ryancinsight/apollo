//! The batched four-step over interleaved data, in place.
//!
//! **Status: correct, measured, and declined** — compiled for tests only,
//! where it serves as the independent-implementation differential oracle for
//! the planar kernel. Pinned on a P-core it is 16 to 37% slower than the
//! planar sibling at every covered size (256 through 65536), and roughly even
//! on an E-core only at the largest: the three-shuffles-per-multiply cost of
//! interleaved butterflies outweighs the conversion and transpose seams it
//! removes. That verdict is the module's chief product — it localizes the
//! remaining gap to the reference engines in radix depth per pass, not in the
//! planar layout choice.
//!
//! The planar sibling in the parent module buys shuffle-free butterflies by
//! splitting samples into real and imaginary planes — and pays for the split
//! at every boundary: a deinterleave pass in, an interleave pass out, two
//! scratch planes, and a transpose whose fused twiddle multiply is scalar
//! because the planes and the interleaved twiddle table disagree on layout.
//! The section profile priced those seams at about a third of the driver at
//! mid sizes (`gap_audit`, `ATLAS-APOLLO-BATCHED-SEAMS-2026-08-27`).
//!
//! This kernel keeps the data in its natural interleaved layout end to end,
//! operating **in place** — no scratch at all. A butterfly's twiddle multiply
//! becomes three shuffles and one alternating FMA via
//! [`hermes_simd::ComplexReg`] — the trade the reference engines make — and in
//! exchange every seam disappears: no conversions, no planes, and the
//! four-step twiddle pass is a plain elementwise vector multiply because the
//! cached matrix is already interleaved.
//!
//! Layout: sample `j` of sub-transform `b` at `data[j * m + b]` — natural
//! row-major order, rows are sub-transform elements, columns are the batch.
//! Every butterfly pairs whole rows, so vector lanes never mix batches and no
//! intra-register shuffle vocabulary is needed beyond the complex multiply.

use super::{BatchedPlan, BatchedPlanCache};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::Complex;
use hermes_simd::{ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage};

/// Tile edge for the in-place transpose, matching the planar sibling's
/// associativity-derived choice: rows are a power of two apart, so wide tiles
/// alias in L1 once `m * 16` bytes reaches the set period.
const TRANSPOSE_TILE: usize = 8;

/// Swaps whole rows of `m` samples per the plan's bit-reversal list.
fn permute_rows<T>(data: &mut [Complex<T>], swaps: &[(u32, u32)], m: usize)
where
    T: Copy,
{
    for &(i, j) in swaps {
        let (i, j) = (i as usize * m, j as usize * m);
        let (low, high) = data.split_at_mut(j);
        low[i..i + m].swap_with_slice(&mut high[..m]);
    }
}

/// In-place tiled transpose of the `m x m` sample matrix.
fn transpose_samples<T: Copy>(data: &mut [Complex<T>], m: usize) {
    for ib in (0..m).step_by(TRANSPOSE_TILE) {
        let ie = (ib + TRANSPOSE_TILE).min(m);
        for jb in (ib..m).step_by(TRANSPOSE_TILE) {
            let je = (jb + TRANSPOSE_TILE).min(m);
            for i in ib..ie {
                let start = if jb == ib { i + 1 } else { jb };
                for j in start.max(jb)..je {
                    data.swap(i * m + j, j * m + i);
                }
            }
        }
    }
}

/// Elementwise product of `data` and the four-step twiddle matrix, both
/// interleaved, as one vector pass.
struct TwiddlePass<'a, T> {
    data: &'a mut [T],
    twiddles: &'a [T],
}

impl<T> LaneKernel<T> for TwiddlePass<'_, T>
where
    T: LaneScalar + MixedRadixScalar,
{
    type Output = ();

    fn call<A: SimdArch + SimdKernel<T>>(self, _simd: Simd<T, A>) {
        let lanes = <A as SimdStorage<T>>::LANE_COUNT;
        let total = self.data.len();
        let mut at = 0usize;
        while at + lanes <= total {
            let v = ComplexReg::<T, A>::from_interleaved(load::<T, A>(self.data, at));
            let w = ComplexReg::<T, A>::from_interleaved(load::<T, A>(self.twiddles, at));
            store::<T, A>((v * w).into_interleaved(), self.data, at);
            at += lanes;
        }
        // Scalar remainder for arrays narrower than one register (small `n`
        // on wide backends); a skipped tail would be silent wrong answers.
        let mut sample = at / 2;
        while 2 * sample < total {
            let (ar, ai) = (self.data[2 * sample], self.data[2 * sample + 1]);
            let (wr, wi) = (self.twiddles[2 * sample], self.twiddles[2 * sample + 1]);
            self.data[2 * sample] = ar * wr - ai * wi;
            self.data[2 * sample + 1] = ar * wi + ai * wr;
            sample += 1;
        }
    }
}

/// All fused stages of `m` interleaved sub-transforms of length `m`.
struct InterleavedStages<'a, T> {
    /// Flat interleaved samples, `2 * m * m` lanes.
    data: &'a mut [T],
    tw: &'a [(T, T)],
    /// Sub-transform length and batch count (square split).
    len: usize,
}

#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an outlined \
              call would re-enter baseline codegen per load"
)]
#[inline(always)]
fn load<T, A>(data: &[T], at: usize) -> hermes_simd::Vector<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    hermes_simd::Vector::load_unaligned_from_slice(
        &data[at..at + <A as SimdStorage<T>>::LANE_COUNT],
    )
    .expect("invariant: the caller bounds `at` by the slice length")
}

#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an outlined \
              call would re-enter baseline codegen per store"
)]
#[inline(always)]
fn store<T, A>(v: hermes_simd::Vector<T, A>, data: &mut [T], at: usize)
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    v.store_unaligned_to_slice(&mut data[at..at + <A as SimdStorage<T>>::LANE_COUNT])
        .expect("invariant: the caller bounds `at` by the slice length")
}

impl<T> LaneKernel<T> for InterleavedStages<'_, T>
where
    T: LaneScalar + MixedRadixScalar,
{
    type Output = ();

    fn call<A: SimdArch + SimdKernel<T>>(self, _simd: Simd<T, A>) {
        let lanes = <A as SimdStorage<T>>::LANE_COUNT;
        // Lanes per row: `m` interleaved samples.
        let row = 2 * self.len;
        let splat = |w: (T, T)| ComplexReg::<T, A>::splat(Complex::new(w.0, w.1));
        let mut twx = 0usize;
        let mut l = 2usize;

        // Radix-4 fusion, mirroring the planar kernel stage for stage: fusing
        // stage `l` with `2l` loads each operand row once per pass instead of
        // twice, and the three twiddles are loop-invariant broadcasts.
        while l * 2 <= self.len {
            let half = l >> 1;
            let groups = self.len / (2 * l);
            for j in 0..half {
                let w1 = splat(self.tw[twx + j]);
                let w2 = splat(self.tw[twx + half + j]);
                let w3 = splat(self.tw[twx + half + j + half]);
                for g in 0..groups {
                    let ia = (g * 2 * l + j) * row;
                    let ib = ia + half * row;
                    let ic = ia + l * row;
                    let id = ic + half * row;
                    let mut k = 0;
                    while k + lanes <= row {
                        let a = ComplexReg::<T, A>::from_interleaved(load(self.data, ia + k));
                        let b = ComplexReg::<T, A>::from_interleaved(load(self.data, ib + k));
                        let c = ComplexReg::<T, A>::from_interleaved(load(self.data, ic + k));
                        let d = ComplexReg::<T, A>::from_interleaved(load(self.data, id + k));

                        // Stage `l`: (a, b) and (c, d), both against W_l^j.
                        let (ua, ub) = a.butterfly(b * w1);
                        let (uc, ud) = c.butterfly(d * w1);

                        // Stage `2l`: (ua, uc) against W_2l^j, (ub, ud)
                        // against W_2l^(j + l/2), operands still in registers.
                        let vc = uc * w2;
                        let vd = ud * w3;
                        store((ua + vc).into_interleaved(), self.data, ia + k);
                        store((ub + vd).into_interleaved(), self.data, ib + k);
                        store((ua - vc).into_interleaved(), self.data, ic + k);
                        store((ub - vd).into_interleaved(), self.data, id + k);
                        k += lanes;
                    }
                    // Scalar remainder: rows narrower than one register.
                    let (w1s, w2s, w3s) = (
                        self.tw[twx + j],
                        self.tw[twx + half + j],
                        self.tw[twx + half + j + half],
                    );
                    while k < row {
                        let mul =
                            |x: (T, T), w: (T, T)| (x.0 * w.0 - x.1 * w.1, x.0 * w.1 + x.1 * w.0);
                        let at = |base: usize| (self.data[base + k], self.data[base + k + 1]);
                        let (a, b, c, d) = (at(ia), at(ib), at(ic), at(id));
                        let tb = mul(b, w1s);
                        let td = mul(d, w1s);
                        let (ua, ub) = ((a.0 + tb.0, a.1 + tb.1), (a.0 - tb.0, a.1 - tb.1));
                        let (uc, ud) = ((c.0 + td.0, c.1 + td.1), (c.0 - td.0, c.1 - td.1));
                        let vc = mul(uc, w2s);
                        let vd = mul(ud, w3s);
                        let put = |data: &mut [T], base: usize, v: (T, T)| {
                            data[base + k] = v.0;
                            data[base + k + 1] = v.1;
                        };
                        put(self.data, ia, (ua.0 + vc.0, ua.1 + vc.1));
                        put(self.data, ib, (ub.0 + vd.0, ub.1 + vd.1));
                        put(self.data, ic, (ua.0 - vc.0, ua.1 - vc.1));
                        put(self.data, id, (ub.0 - vd.0, ub.1 - vd.1));
                        k += 2;
                    }
                }
            }
            twx += half + l;
            l <<= 2;
        }

        // One radix-2 stage remains when `log2(len)` is odd.
        if l <= self.len {
            let half = l >> 1;
            let groups = self.len / l;
            for j in 0..half {
                let w = splat(self.tw[twx + j]);
                for g in 0..groups {
                    let lo = (g * l + j) * row;
                    let hi = lo + half * row;
                    let mut k = 0;
                    while k + lanes <= row {
                        let a = ComplexReg::<T, A>::from_interleaved(load(self.data, lo + k));
                        let b = ComplexReg::<T, A>::from_interleaved(load(self.data, hi + k));
                        let (u, v) = a.butterfly(b * w);
                        store(u.into_interleaved(), self.data, lo + k);
                        store(v.into_interleaved(), self.data, hi + k);
                        k += lanes;
                    }
                    // Scalar remainder: rows narrower than one register.
                    let ws = self.tw[twx + j];
                    while k < row {
                        let (br, bi) = (self.data[hi + k], self.data[hi + k + 1]);
                        let (tr, ti) = (br * ws.0 - bi * ws.1, br * ws.1 + bi * ws.0);
                        let (ar, ai) = (self.data[lo + k], self.data[lo + k + 1]);
                        self.data[lo + k] = ar + tr;
                        self.data[lo + k + 1] = ai + ti;
                        self.data[hi + k] = ar - tr;
                        self.data[hi + k + 1] = ai - ti;
                        k += 2;
                    }
                }
            }
        }
    }
}

/// Runs the stage set over rows the caller has already bit-reversed.
fn run_stages<T>(data: &mut [Complex<T>], plan: &BatchedPlan<T>)
where
    T: BatchedPlanCache<Complex = Complex<T>>,
{
    let flat: &mut [T] = bytemuck::cast_slice_mut(data);
    hermes_simd::vectorize(InterleavedStages {
        data: flat,
        tw: &plan.tw,
        len: plan.len,
    });
}

/// In-place four-step transform over interleaved data: no scratch, no planar
/// conversion, one vector twiddle pass, one pure transpose.
///
/// # Panics
///
/// If `data.len()` is not an even power of two of at least 4, or the row width
/// `2 * sqrt(n)` does not divide the dispatched lane count's requirements —
/// callers admit lengths through the same gate as the planar sibling.
pub(crate) fn four_step_interleaved<T, const INVERSE: bool>(data: &mut [Complex<T>])
where
    T: BatchedPlanCache<Complex = Complex<T>>,
{
    let n = data.len();
    let k = n.trailing_zeros();
    assert!(
        n.is_power_of_two() && k % 2 == 0 && n >= 4,
        "requires an even power of two of at least 4"
    );
    let m = 1usize << (k / 2);

    let plan = T::cached_plan::<INVERSE>(m);

    // 1. First axis: bit-reverse rows, then the stage set. The input is
    //    already batch-major for this direction.
    permute_rows(data, &plan.swaps, m);
    run_stages(data, plan.as_ref());

    // 2. Four-step twiddle W_n^{b*k1}: the cached matrix is interleaved like
    //    the data, so this is one elementwise vector pass — the layout
    //    agreement the planar kernel never had.
    let twiddles = T::cached_four_step_twiddles::<INVERSE>(n, m, m);
    let flat: &mut [T] = bytemuck::cast_slice_mut(data);
    hermes_simd::vectorize(TwiddlePass {
        data: flat,
        twiddles: bytemuck::cast_slice(&twiddles),
    });

    // 3. Transpose so the second axis becomes batch-major, then its stage set;
    //    the result lands at the natural output index.
    transpose_samples(data, m);
    permute_rows(data, &plan.swaps, m);
    run_stages(data, plan.as_ref());
}

#[cfg(test)]
mod tests;
