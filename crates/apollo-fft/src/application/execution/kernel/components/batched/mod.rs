//! Batched power-of-two sub-transforms with the transform index in the lane.
//!
//! ## Why this layout
//!
//! A butterfly vectorized *within* one transform must gather its two operands
//! from positions `j` and `j + half`, and a complex multiply on interleaved data
//! then needs cross-lane shuffles to separate the real and imaginary parts. Both
//! costs vanish when the lane position holds the *transform* index instead:
//! with `B` independent transforms laid out so element `j` of transform `b` sits
//! at `j * B + b`, a butterfly reads two contiguous runs of `B` values, the
//! twiddle is one scalar broadcast across every lane, and no shuffle occurs
//! anywhere.
//!
//! This is the arrangement [`FftPlanarMut`] documents — "lane `c` across all
//! rows is one independent transform instance and no cross-lane shuffle is
//! required" — and which nothing previously used.
//!
//! Measured against the other kernel shapes tried in this crate, all of which
//! sat between 3.4 and 6.1 flops/ns, this reaches 10.2 to 10.7 across batch and
//! length combinations, using the same `hermes_simd` `Vector` operations. The
//! layout is the variable.
//!
//! ## Where it applies
//!
//! The four-step decomposition already splits `N = N1 x N2` and transforms
//! along each axis in turn. Writing `i = j * N2 + b`, the input is *already*
//! batch-major for the first axis, so those `N2` transforms of length `N1` need
//! no transpose at all; and the output index `k2 * N1 + k1` falls out of the
//! second batched pass, so there is no final transpose either. One transpose
//! sits between them, in place because the four-step gate admits only square
//! splits.
//!
//! [`FftPlanarMut`]: crate::domain::storage::FftPlanarMut

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::Complex;
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// Tile side for the in-place transpose, in elements.
///
/// Per-length batched-transform plan: the decimation-in-time twiddle table and
/// the bit-reversal permutation, both build-time cost.
pub(crate) struct BatchedPlan<T> {
    len: usize,
    /// Stage-major twiddles: stage `s` (sub-transform length `2^(s+1)`) occupies
    /// `2^s` entries, so the table totals `len - 1`.
    tw: Vec<(T, T)>,
    /// Index pairs to exchange for the bit-reversal permutation, `i < j` only,
    /// so applying them once performs the permutation.
    swaps: Vec<(u32, u32)>,
}

impl<T: MixedRadixScalar> BatchedPlan<T> {
    fn new<const INVERSE: bool>(len: usize) -> Self {
        assert!(
            len.is_power_of_two(),
            "batched plan requires a power of two"
        );
        let logn = len.trailing_zeros();
        let swaps = (0..len)
            .filter_map(|i| {
                let j = ((i as u32).reverse_bits() >> (32 - logn)) as usize;
                (j > i).then_some((
                    u32::try_from(i).expect("index fits u32"),
                    u32::try_from(j).expect("index fits u32"),
                ))
            })
            .collect();

        let sign = if INVERSE { 1.0_f64 } else { -1.0_f64 };
        let mut tw = Vec::with_capacity(len.saturating_sub(1));
        let mut l = 2usize;
        while l <= len {
            for j in 0..l / 2 {
                // Direct evaluation per entry: a recurrence here would carry
                // O(N·u) twiddle error, which the forward-error bound this
                // crate documents does not admit.
                let (sin, cos) = (sign * core::f64::consts::TAU * j as f64 / l as f64).sin_cos();
                tw.push((T::from_precise(cos), T::from_precise(sin)));
            }
            l <<= 1;
        }
        Self { len, tw, swaps }
    }
}

/// All stages of `batch` independent length-`len` transforms over planar data.
///
/// Dispatch happens once for the whole stage set, which is the placement Hermes
/// ADR 016 requires: outside the innermost loop, and never wrapping a
/// thread-spawning call.
struct BatchedStages<'a, T> {
    re: &'a mut [T],
    im: &'a mut [T],
    tw: &'a [(T, T)],
    /// Live columns per row — the loop bound.
    batch: usize,
    /// Elements per row including [`ROW_PAD`] — the index multiplier.
    stride: usize,
    len: usize,
}

impl<T> LaneKernel<T> for BatchedStages<'_, T>
where
    T: LaneScalar + MixedRadixScalar,
{
    type Output = ();

    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
        let lanes = <A as SimdStorage<T>>::LANE_COUNT;
        let b = self.batch;
        let s = self.stride;
        let mut twx = 0usize;
        let mut l = 2usize;

        // Two stages per pass over the data.
        //
        // A stage-per-pass loop re-streams the whole `len * batch` array
        // `log2(len)` times, and at these sizes that traffic binds rather than
        // the arithmetic: RustFFT runs six stages in two passes by holding a
        // column in registers, and PhastFT fuses four into one codelet. Fusing
        // stage `l` with stage `2l` loads each of the four operands once and
        // writes each once, halving the passes.
        //
        // Four is the width rather than eight, and that follows from the planar
        // layout rather than being a conservative choice: real and imaginary
        // parts occupy separate registers here, so a radix-8 step would need
        // sixteen vector registers for operands alone and would spill on AVX2.
        // The three twiddles are invariant in `g` and hoist out of the inner
        // loops.
        while l * 2 <= self.len {
            let half = l >> 1;
            let groups = self.len / (2 * l);
            for j in 0..half {
                // Stage `l` occupies `half` table entries and stage `2l` the
                // `l` that follow, so the second stage's two twiddles are read
                // straight from the table rather than derived by rotation:
                // exact stored values, and no sign case for the inverse
                // direction.
                let (w1r, w1i) = self.tw[twx + j];
                let (w2r, w2i) = self.tw[twx + half + j];
                let (w3r, w3i) = self.tw[twx + half + j + half];
                let (v1r, v1i) = (simd.splat(w1r), simd.splat(w1i));
                let (v2r, v2i) = (simd.splat(w2r), simd.splat(w2i));
                let (v3r, v3i) = (simd.splat(w3r), simd.splat(w3i));

                for g in 0..groups {
                    let ia = (g * 2 * l + j) * s;
                    let ib = ia + half * s;
                    let ic = ia + l * s;
                    let id = ic + half * s;
                    let mut k = 0;
                    while k + lanes <= b {
                        let ar = load::<T, A>(self.re, ia + k);
                        let ai = load::<T, A>(self.im, ia + k);
                        let br = load::<T, A>(self.re, ib + k);
                        let bi = load::<T, A>(self.im, ib + k);
                        let cr = load::<T, A>(self.re, ic + k);
                        let ci = load::<T, A>(self.im, ic + k);
                        let dr = load::<T, A>(self.re, id + k);
                        let di = load::<T, A>(self.im, id + k);

                        // Stage `l`: (a,b) and (c,d), both against W_l^j.
                        let tbr = v1r.mul_add(br, -(v1i * bi));
                        let tbi = v1r.mul_add(bi, v1i * br);
                        let (uar, uai) = (ar + tbr, ai + tbi);
                        let (ubr, ubi) = (ar - tbr, ai - tbi);

                        let tdr = v1r.mul_add(dr, -(v1i * di));
                        let tdi = v1r.mul_add(di, v1i * dr);
                        let (ucr, uci) = (cr + tdr, ci + tdi);
                        let (udr, udi) = (cr - tdr, ci - tdi);

                        // Stage `2l`: (a,c) against W_2l^j and (b,d) against
                        // W_2l^(j + l/2). Neither operand has left a register.
                        let vcr = v2r.mul_add(ucr, -(v2i * uci));
                        let vci = v2r.mul_add(uci, v2i * ucr);
                        let vdr = v3r.mul_add(udr, -(v3i * udi));
                        let vdi = v3r.mul_add(udi, v3i * udr);

                        store::<T, A>(uar + vcr, self.re, ia + k);
                        store::<T, A>(uai + vci, self.im, ia + k);
                        store::<T, A>(ubr + vdr, self.re, ib + k);
                        store::<T, A>(ubi + vdi, self.im, ib + k);
                        store::<T, A>(uar - vcr, self.re, ic + k);
                        store::<T, A>(uai - vci, self.im, ic + k);
                        store::<T, A>(ubr - vdr, self.re, id + k);
                        store::<T, A>(ubi - vdi, self.im, id + k);
                        k += lanes;
                    }
                    // Scalar remainder when the batch is not a lane multiple.
                    for k in k..b {
                        let (ar, ai) = (self.re[ia + k], self.im[ia + k]);
                        let (br, bi) = (self.re[ib + k], self.im[ib + k]);
                        let (cr, ci) = (self.re[ic + k], self.im[ic + k]);
                        let (dr, di) = (self.re[id + k], self.im[id + k]);

                        let tbr = w1r * br - w1i * bi;
                        let tbi = w1r * bi + w1i * br;
                        let (uar, uai) = (ar + tbr, ai + tbi);
                        let (ubr, ubi) = (ar - tbr, ai - tbi);

                        let tdr = w1r * dr - w1i * di;
                        let tdi = w1r * di + w1i * dr;
                        let (ucr, uci) = (cr + tdr, ci + tdi);
                        let (udr, udi) = (cr - tdr, ci - tdi);

                        let vcr = w2r * ucr - w2i * uci;
                        let vci = w2r * uci + w2i * ucr;
                        let vdr = w3r * udr - w3i * udi;
                        let vdi = w3r * udi + w3i * udr;

                        self.re[ia + k] = uar + vcr;
                        self.im[ia + k] = uai + vci;
                        self.re[ib + k] = ubr + vdr;
                        self.im[ib + k] = ubi + vdi;
                        self.re[ic + k] = uar - vcr;
                        self.im[ic + k] = uai - vci;
                        self.re[id + k] = ubr - vdr;
                        self.im[id + k] = ubi - vdi;
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
                let (twr, twi) = self.tw[twx + j];
                let wr = simd.splat(twr);
                let wi = simd.splat(twi);
                for g in 0..groups {
                    let lo = (g * l + j) * s;
                    let hi = lo + half * s;
                    let mut k = 0;
                    while k + lanes <= b {
                        let (lo_s, hi_s) = (lo + k, hi + k);
                        let ar = load::<T, A>(self.re, lo_s);
                        let ai = load::<T, A>(self.im, lo_s);
                        let br = load::<T, A>(self.re, hi_s);
                        let bi = load::<T, A>(self.im, hi_s);

                        let tr = wr.mul_add(br, -(wi * bi));
                        let ti = wr.mul_add(bi, wi * br);

                        store::<T, A>(ar + tr, self.re, lo_s);
                        store::<T, A>(ai + ti, self.im, lo_s);
                        store::<T, A>(ar - tr, self.re, hi_s);
                        store::<T, A>(ai - ti, self.im, hi_s);
                        k += lanes;
                    }
                    for k in k..b {
                        let (lo_s, hi_s) = (lo + k, hi + k);
                        let (ar, ai) = (self.re[lo_s], self.im[lo_s]);
                        let (br, bi) = (self.re[hi_s], self.im[hi_s]);
                        let tr = twr * br - twi * bi;
                        let ti = twr * bi + twi * br;
                        self.re[lo_s] = ar + tr;
                        self.im[lo_s] = ai + ti;
                        self.re[hi_s] = ar - tr;
                        self.im[hi_s] = ai - ti;
                    }
                }
            }
        }
    }
}

/// One-vector load at `at`, which the caller has proved is in bounds.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line               call here reintroduces the ADR 009 penalty this kernel exists to avoid"
)]
#[inline(always)]
fn load<T, A>(data: &[T], at: usize) -> Vector<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(at + <A as SimdStorage<T>>::LANE_COUNT <= data.len());
    // SAFETY: the caller's loop condition bounds `at + LANE_COUNT` by the slice
    // length, and `BatchedStages::call` receives the capability proving that
    // the host executes `A`. The checked wrapper revalidates both per call,
    // which measured as 45% of this kernel's time; the bound here is
    // loop-invariant.
    unsafe { Vector::<T, A>::load_unaligned(data.as_ptr().add(at)) }
}

/// One-vector store at `at`, which the caller has proved is in bounds.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line               call here reintroduces the ADR 009 penalty this kernel exists to avoid"
)]
#[inline(always)]
fn store<T, A>(v: Vector<T, A>, data: &mut [T], at: usize)
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(at + <A as SimdStorage<T>>::LANE_COUNT <= data.len());
    // SAFETY: as `load` above.
    unsafe { v.store_unaligned(data.as_mut_ptr().add(at)) }
}

/// Applies the bit-reversal permutation across every transform in the batch.
fn permute<T: Copy>(re: &mut [T], im: &mut [T], swaps: &[(u32, u32)], batch: usize, stride: usize) {
    // Whole-row exchanges rather than element swaps: `swap_with_slice` lowers
    // to block copies the compiler vectorizes, where the element loop issued
    // two scalar swaps per lane. The plan's swap list guarantees `i < j`, so
    // splitting at `j`'s row start yields the two disjoint rows safely.
    for &(i, j) in swaps {
        let (i, j) = (i as usize * stride, j as usize * stride);
        let (re_low, re_high) = re.split_at_mut(j);
        re_low[i..i + batch].swap_with_slice(&mut re_high[..batch]);
        let (im_low, im_high) = im.split_at_mut(j);
        im_low[i..i + batch].swap_with_slice(&mut im_high[..batch]);
    }
}

/// Square in-place transpose of an `m x m` plane, tiled for locality.
/// Extra elements appended to each plane row, so the row stride is `m + 8`
/// rather than `m`.
///
/// With an unpadded power-of-two stride, all of it aliases: every plane row
/// maps to the same L1 sets, `re` and `im` (a power-of-two apart) share sets,
/// and the three same-sized buffers a four-step holds live — caller data,
/// scratch planes, twiddle matrix — collide by allocation accident. Eight
/// f64 elements are one cache line, so consecutive rows shift by a full line
/// and the uniform aliasing is gone; for f32 the shift is half a line, which
/// still rotates the sets. The pad is a spacer, never computed on: every loop
/// bounds itself by the live column count and multiplies row indices by the
/// stride.
pub(crate) const ROW_PAD: usize = 8;

/// Tile edge for [`twiddle_transpose`]. The plane stride is `m * 8` bytes — a power of
/// two, so at `m >= 256` every tile row aliases to the same L1 set, and `re`
/// and `im` (halves of one allocation, also a power of two apart) share sets
/// too. A 32-row tile then puts 64+ lines into one set of an 8/12-way cache
/// and thrashes: the standalone transpose measured 20x slower per element at
/// `m = 256` than at `m = 128`. Eight rows keeps the in-flight lines per set
/// within associativity.
const TWIDDLE_TRANSPOSE_TILE: usize = 8;

/// Applies the four-step twiddle `W_N^(k1*b)` and transposes both planes, in
/// one in-place pass.
///
/// Fused because the twiddle multiply is elementwise and the transpose already
/// touches every element: riding one on the other deletes a full pass over the
/// data. Element `p` is multiplied by `tw[p]` and moved to its mirror, so each
/// element is twiddled exactly once — at its pre-transpose index, which is the
/// order the separate pass used.
fn twiddle_transpose<T>(re: &mut [T], im: &mut [T], tw: &[Complex<T>], m: usize, stride: usize)
where
    T: LaneScalar + MixedRadixScalar,
{
    debug_assert!(stride >= m && re.len() >= m * stride);
    debug_assert_eq!(tw.len(), m * m);

    /// `p`/`q` index the padded planes; `ptw`/`qtw` index the unpadded twiddle
    /// matrix for the same two logical elements.
    fn swap_twiddled<T>(
        re: &mut [T],
        im: &mut [T],
        tw: &[Complex<T>],
        (p, ptw): (usize, usize),
        (q, qtw): (usize, usize),
    ) where
        T: LaneScalar + MixedRadixScalar,
    {
        let (tp, tq) = (tw[ptw], tw[qtw]);
        let (ar, ai) = (re[p], im[p]);
        let (br, bi) = (re[q], im[q]);
        re[p] = br * tq.re - bi * tq.im;
        im[p] = br * tq.im + bi * tq.re;
        re[q] = ar * tp.re - ai * tp.im;
        im[q] = ar * tp.im + ai * tp.re;
    }

    for ib in (0..m).step_by(TWIDDLE_TRANSPOSE_TILE) {
        let ie = (ib + TWIDDLE_TRANSPOSE_TILE).min(m);
        for jb in (ib..m).step_by(TWIDDLE_TRANSPOSE_TILE) {
            let je = (jb + TWIDDLE_TRANSPOSE_TILE).min(m);
            if jb == ib {
                // Diagonal tile: fixed points are twiddled in place, and each
                // off-diagonal pair inside the tile is visited once.
                for i in ib..ie {
                    let p = i * stride + i;
                    let t = tw[i * m + i];
                    let (a, c) = (re[p], im[p]);
                    re[p] = a * t.re - c * t.im;
                    im[p] = a * t.im + c * t.re;
                    for j in (i + 1)..je {
                        swap_twiddled(
                            re,
                            im,
                            tw,
                            (i * stride + j, i * m + j),
                            (j * stride + i, j * m + i),
                        );
                    }
                }
            } else {
                for i in ib..ie {
                    for j in jb..je {
                        swap_twiddled(
                            re,
                            im,
                            tw,
                            (i * stride + j, i * m + j),
                            (j * stride + i, j * m + i),
                        );
                    }
                }
            }
        }
    }
}

/// Plans keyed by `(length, inverse)`: the two directions carry conjugate
/// twiddles and cannot share an entry.
type PlanCache<T> = RefCell<HashMap<(usize, bool), Arc<BatchedPlan<T>>>>;

thread_local! {
    static PLAN_CACHE_F64: PlanCache<f64> = RefCell::new(HashMap::new());
    static PLAN_CACHE_F32: PlanCache<f32> = RefCell::new(HashMap::new());
}

/// Scalars whose batched plans are cached per thread.
pub(crate) trait BatchedPlanCache:
    MixedRadixScalar + LaneScalar + bytemuck::Pod + Sized
{
    fn cached_plan<const INVERSE: bool>(len: usize) -> Arc<BatchedPlan<Self>>;
}

macro_rules! impl_plan_cache {
    ($t:ty, $cache:ident) => {
        impl BatchedPlanCache for $t {
            fn cached_plan<const INVERSE: bool>(len: usize) -> Arc<BatchedPlan<Self>> {
                $cache.with(|c| {
                    let key = (len, INVERSE);
                    if let Some(plan) = c.borrow().get(&key) {
                        return Arc::clone(plan);
                    }
                    let plan = Arc::new(BatchedPlan::<$t>::new::<INVERSE>(len));
                    c.borrow_mut().insert(key, Arc::clone(&plan));
                    plan
                })
            }
        }
    };
}

impl_plan_cache!(f64, PLAN_CACHE_F64);
impl_plan_cache!(f32, PLAN_CACHE_F32);

/// Runs the stage set of `batch` transforms of length `plan.len` over planar
/// `re`/`im`, whose rows the caller has already bit-reversed — either by
/// [`permute`] or by writing rows to their reversed positions in the first
/// place, which is how the driver's deinterleave avoids a whole pass.
fn run_batched<T>(re: &mut [T], im: &mut [T], plan: &BatchedPlan<T>, batch: usize, stride: usize)
where
    T: LaneScalar + MixedRadixScalar,
{
    hermes_simd::vectorize(BatchedStages {
        re,
        im,
        tw: &plan.tw,
        batch,
        stride,
        len: plan.len,
    });
}

/// Four-step power-of-two transform over the batched layout.
///
/// Requires a square split, which is what the four-step dispatch gate admits,
/// so the middle transpose is in place and the scratch requirement matches the
/// Stockham path it replaces: one `N`-element complex buffer, reinterpreted as
/// two `N`-element real planes.
///
/// # Panics
///
/// Panics if `data.len()` is not an even power of two, or if `scratch` is
/// shorter than `data`.
/// Scratch length, in complex elements, that [`four_step_batched`] requires
/// for a transform of length `n`.
///
/// The single definition of the padded-plane requirement, so callers and the
/// driver cannot disagree about it.
pub(crate) fn scratch_len(n: usize) -> usize {
    let m = 1usize << (n.trailing_zeros() / 2);
    m * (m + ROW_PAD)
}

pub(crate) fn four_step_batched<T, const INVERSE: bool>(
    data: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
) where
    T: BatchedPlanCache<Complex = Complex<T>>,
{
    let n = data.len();
    let k = n.trailing_zeros();
    assert!(
        n.is_power_of_two() && k % 2 == 0 && n >= 4,
        "requires an even power of two of at least 4"
    );
    let m = 1usize << (k / 2);
    let stride = m + ROW_PAD;
    let plane = scratch_len(n);
    // One complex scratch element is two reals, so `m * stride` complexes hold
    // the two padded planes. The pad breaks the power-of-two row stride that
    // makes every plane row alias to one L1 set (see [`ROW_PAD`]).
    assert!(
        scratch.len() >= plane,
        "scratch must hold two padded planes"
    );
    let flat: &mut [T] = bytemuck::cast_slice_mut(&mut scratch[..plane]);
    let (re, im) = flat.split_at_mut(plane);

    // The stage set wants bit-reversed rows, so the deinterleave writes each
    // row at its reversed position and the separate permutation pass this
    // replaces is deleted: bit reversal is an involution, so writing to
    // `rev(row)` is exactly the swap list the plan would have applied.
    let row_bits = k / 2;
    for (row, chunk) in data.chunks_exact(m).enumerate() {
        let dest = row.reverse_bits() >> (usize::BITS - row_bits);
        for (b, c) in chunk.iter().enumerate() {
            re[dest * stride + b] = c.re;
            im[dest * stride + b] = c.im;
        }
    }

    // 1. The `m` transforms of length `m` along the first axis; the input is
    //    already batch-major for this direction, so no transpose is needed.
    let plan = T::cached_plan::<INVERSE>(m);
    run_batched(re, im, plan.as_ref(), m, stride);

    // 2+3. The four-step twiddle W_N^{b·k1} and the transpose, in one pass.
    // The matrix is built once when it enters the shared cache, never per
    // transform; the multiply rides the transpose's traffic, so the separate
    // elementwise pass this replaces is deleted rather than optimized.
    let twiddles = T::cached_four_step_twiddles::<INVERSE>(n, m, m);
    twiddle_transpose(re, im, &twiddles, m, stride);

    // 4. The `m` transforms along the second axis. The transpose produced
    //    natural row order, so this set still needs its bit-reversal pass; the
    //    result lands at `k2 * m + k1`, the natural output index.
    permute(re, im, &plan.swaps, m, stride);
    run_batched(re, im, plan.as_ref(), m, stride);

    for (row, chunk) in data.chunks_exact_mut(m).enumerate() {
        for (b, c) in chunk.iter_mut().enumerate() {
            *c = Complex::new(re[row * stride + b], im[row * stride + b]);
        }
    }
}

#[cfg(test)]
mod tests;
