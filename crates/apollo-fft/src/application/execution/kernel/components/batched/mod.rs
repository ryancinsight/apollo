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
    /// Planar four-step twiddle planes multiplied into the first stage's
    /// loads, or `None` for a plain stage set. Row-major with row stride
    /// `batch`, rows in the same bit-reversed order the data rows carry.
    fold: Option<(&'a [T], &'a [T])>,
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

                // The four-step twiddle rides the first stage's loads:
                // `l == 2` is the pass that reads every element exactly once,
                // so multiplying there deletes the standalone pass the scalar
                // transpose multiply used to be.
                let fold = if l == 2 { self.fold } else { None };
                for g in 0..groups {
                    let row_a = g * 2 * l + j;
                    let ia = row_a * s;
                    let ib = ia + half * s;
                    let ic = ia + l * s;
                    let id = ic + half * s;
                    let ta = row_a * b;
                    let tb = ta + half * b;
                    let tc = ta + l * b;
                    let td = tc + half * b;
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
                            let tw = |r: hermes_simd::Vector<T, A>,
                                      i: hermes_simd::Vector<T, A>,
                                      at: usize| {
                                let wr = load::<T, A>(pr, at + k);
                                let wi = load::<T, A>(pi, at + k);
                                (wr.mul_add(r, -(wi * i)), wr.mul_add(i, wi * r))
                            };
                            (ar, ai) = tw(ar, ai, ta);
                            (br, bi) = tw(br, bi, tb);
                            (cr, ci) = tw(cr, ci, tc);
                            (dr, di) = tw(dr, di, td);
                        }

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
                        let (mut ar, mut ai) = (self.re[ia + k], self.im[ia + k]);
                        let (mut br, mut bi) = (self.re[ib + k], self.im[ib + k]);
                        let (mut cr, mut ci) = (self.re[ic + k], self.im[ic + k]);
                        let (mut dr, mut di) = (self.re[id + k], self.im[id + k]);
                        if let Some((pr, pi)) = fold {
                            let tw = |r: T, i: T, at: usize| {
                                let (wr, wi) = (pr[at + k], pi[at + k]);
                                (wr * r - wi * i, wr * i + wi * r)
                            };
                            (ar, ai) = tw(ar, ai, ta);
                            (br, bi) = tw(br, bi, tb);
                            (cr, ci) = tw(cr, ci, tc);
                            (dr, di) = tw(dr, di, td);
                        }

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

        // One radix-2 stage remains when `log2(len)` is odd; when the whole
        // transform is that single stage, the first-stage fold applies here.
        if l <= self.len {
            let half = l >> 1;
            let groups = self.len / l;
            let fold = if l == 2 { self.fold } else { None };
            for j in 0..half {
                let (twr, twi) = self.tw[twx + j];
                let wr = simd.splat(twr);
                let wi = simd.splat(twi);
                for g in 0..groups {
                    let row_lo = g * l + j;
                    let lo = row_lo * s;
                    let hi = lo + half * s;
                    let tlo = row_lo * b;
                    let thi = tlo + half * b;
                    let mut k = 0;
                    while k + lanes <= b {
                        let (lo_s, hi_s) = (lo + k, hi + k);
                        let mut ar = load::<T, A>(self.re, lo_s);
                        let mut ai = load::<T, A>(self.im, lo_s);
                        let mut br = load::<T, A>(self.re, hi_s);
                        let mut bi = load::<T, A>(self.im, hi_s);
                        if let Some((pr, pi)) = fold {
                            let tw = |r: hermes_simd::Vector<T, A>,
                                      i: hermes_simd::Vector<T, A>,
                                      at: usize| {
                                let vr = load::<T, A>(pr, at + k);
                                let vi = load::<T, A>(pi, at + k);
                                (vr.mul_add(r, -(vi * i)), vr.mul_add(i, vi * r))
                            };
                            (ar, ai) = tw(ar, ai, tlo);
                            (br, bi) = tw(br, bi, thi);
                        }

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
                        let (mut ar, mut ai) = (self.re[lo_s], self.im[lo_s]);
                        let (mut br, mut bi) = (self.re[hi_s], self.im[hi_s]);
                        if let Some((pr, pi)) = fold {
                            let tw = |r: T, i: T, at: usize| {
                                let (vr, vi) = (pr[at + k], pi[at + k]);
                                (vr * r - vi * i, vr * i + vi * r)
                            };
                            (ar, ai) = tw(ar, ai, tlo);
                            (br, bi) = tw(br, bi, thi);
                        }
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

/// Transposes both `m x m` planes in place, tiled.
///
/// Pure exchange: the four-step twiddle that used to ride this pass as a
/// scalar multiply — 26% of the driver at N = 256 — now rides stage-set-2's
/// first-stage vector loads instead, which the twiddle matrix's symmetry
/// (`W^(j*b)` equals its own transpose) makes exactly equivalent.
fn transpose_planes<T: Copy>(re: &mut [T], im: &mut [T], m: usize, stride: usize) {
    debug_assert!(stride >= m && re.len() >= m * stride);
    for ib in (0..m).step_by(TWIDDLE_TRANSPOSE_TILE) {
        let ie = (ib + TWIDDLE_TRANSPOSE_TILE).min(m);
        for jb in (ib..m).step_by(TWIDDLE_TRANSPOSE_TILE) {
            let je = (jb + TWIDDLE_TRANSPOSE_TILE).min(m);
            for i in ib..ie {
                let start = if jb == ib { i + 1 } else { jb };
                for j in start.max(jb)..je {
                    re.swap(i * stride + j, j * stride + i);
                    im.swap(i * stride + j, j * stride + i);
                }
            }
        }
    }
}

/// Plans keyed by `(length, inverse)`: the two directions carry conjugate
/// twiddles and cannot share an entry.
/// Planar, row-permuted four-step twiddle planes.
///
/// The interleaved `W_n^(j*b)` matrix cannot feed a vector multiply against
/// planar data, which is why the transpose's fused multiply was scalar — the
/// measured 26% of the driver at N = 256. These planes fix the layout
/// disagreement once, at build: split into real and imaginary planes, with
/// rows in bit-reversed order so stage-set-2's loads (which run after the row
/// permutation) index them directly. The matrix is symmetric (`W^(j*b)` is
/// its own transpose), which is what lets the multiply move to the other side
/// of the transpose at all.
pub(crate) struct FourStepPlanes<T> {
    pub(crate) re: Box<[T]>,
    pub(crate) im: Box<[T]>,
}

impl<T: MixedRadixScalar<Complex = Complex<T>>> FourStepPlanes<T> {
    fn new<const INVERSE: bool>(n: usize, m: usize) -> Self
    where
        Complex<T>: crate::application::execution::kernel::twiddle_table::TwiddleOutput,
    {
        // The uncached builder: the interleaved matrix is a build transient
        // here, split into planes and dropped, so batched sizes cache exactly
        // one twiddle representation. The threaded four-step's sizes cache
        // exactly one too — the interleaved matrix its fused
        // transpose-multiply reads with better locality — and the two size
        // ranges are disjoint by the routing thresholds. An explicit
        // vector rewrite of the driver's boundary loops was measured against
        // this same baseline and declined: the compiler already vectorizes
        // those canonical interleave patterns, and the added dispatch
        // round-trips made every size 2 to 7% slower.
        let interleaved =
            crate::application::execution::kernel::mixed_radix::caches::build_four_step_twiddles::<
                Complex<T>,
                INVERSE,
            >(n, m, m);
        let bits = m.trailing_zeros();
        let mut re = vec![T::from_precise(0.0); m * m].into_boxed_slice();
        let mut im = vec![T::from_precise(0.0); m * m].into_boxed_slice();
        for row in 0..m {
            let src = row.reverse_bits() >> (usize::BITS - bits);
            for col in 0..m {
                re[row * m + col] = interleaved[src * m + col].re;
                im[row * m + col] = interleaved[src * m + col].im;
            }
        }
        Self { re, im }
    }
}

type PlanCache<T> = RefCell<HashMap<(usize, bool), Arc<BatchedPlan<T>>>>;
type PlanesCache<T> = RefCell<HashMap<(usize, bool), Arc<FourStepPlanes<T>>>>;

thread_local! {
    static PLAN_CACHE_F64: PlanCache<f64> = RefCell::new(HashMap::new());
    static PLAN_CACHE_F32: PlanCache<f32> = RefCell::new(HashMap::new());
    static PLANES_CACHE_F64: PlanesCache<f64> = RefCell::new(HashMap::new());
    static PLANES_CACHE_F32: PlanesCache<f32> = RefCell::new(HashMap::new());
}

/// Scalars whose batched plans are cached per thread.
pub(crate) trait BatchedPlanCache:
    MixedRadixScalar + LaneScalar + bytemuck::Pod + Sized
{
    fn cached_plan<const INVERSE: bool>(len: usize) -> Arc<BatchedPlan<Self>>;
    fn cached_four_step_planes<const INVERSE: bool>(
        n: usize,
        m: usize,
    ) -> Arc<FourStepPlanes<Self>>;
}

macro_rules! impl_plan_cache {
    ($t:ty, $cache:ident, $planes:ident) => {
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

            fn cached_four_step_planes<const INVERSE: bool>(
                n: usize,
                m: usize,
            ) -> Arc<FourStepPlanes<Self>> {
                $planes.with(|c| {
                    let key = (n, INVERSE);
                    if let Some(planes) = c.borrow().get(&key) {
                        return Arc::clone(planes);
                    }
                    let planes = Arc::new(FourStepPlanes::<$t>::new::<INVERSE>(n, m));
                    c.borrow_mut().insert(key, Arc::clone(&planes));
                    planes
                })
            }
        }
    };
}

impl_plan_cache!(f64, PLAN_CACHE_F64, PLANES_CACHE_F64);
impl_plan_cache!(f32, PLAN_CACHE_F32, PLANES_CACHE_F32);

/// Runs the stage set of `batch` transforms of length `plan.len` over planar
/// `re`/`im`, whose rows the caller has already bit-reversed — either by
/// [`permute`] or by writing rows to their reversed positions in the first
/// place, which is how the driver's deinterleave avoids a whole pass.
fn run_batched<T>(
    re: &mut [T],
    im: &mut [T],
    plan: &BatchedPlan<T>,
    fold: Option<(&[T], &[T])>,
    batch: usize,
    stride: usize,
) where
    T: LaneScalar + MixedRadixScalar,
{
    hermes_simd::vectorize(BatchedStages {
        re,
        im,
        tw: &plan.tw,
        fold,
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

/// Whether [`four_step_batched`] covers a transform of length `n`.
///
/// The single definition of the planar route's domain: an even power of two
/// — the square split the driver is written for — below the point where the
/// row transforms are worth threading.
pub(crate) fn planar_applies(n: usize) -> bool {
    n.is_power_of_two()
        && n.trailing_zeros() % 2 == 0
        && n >= 4
        && n < crate::application::execution::kernel::components::four_step::PARALLEL_ROW_THRESHOLD
}

/// Whether [`four_step_split_batched`] covers a transform of length `n`.
///
/// An odd power of two has no square split, so it decimates once; the route
/// applies when both halves are then planar. The lower bound is the one the
/// unfused decimation already carried — below it the plan hands these
/// lengths to the base kernels and codelets, which never reach here, so
/// widening the domain would change only what tests exercise.
pub(crate) fn planar_split_applies(n: usize) -> bool {
    n.is_power_of_two() && n.trailing_zeros() % 2 == 1 && n >= 512 && planar_applies(n / 2)
}

/// Scratch, in complex elements, that [`four_step_split_batched`] requires.
///
/// Both half-planes are live at once, which is what lets the combine read
/// them together; against the unfused route — a full `n`-element decimation
/// buffer plus one half-plane nested inside it — this is the smaller peak.
pub(crate) fn split_scratch_len(n: usize) -> usize {
    2 * scratch_len(n / 2)
}

/// Per-pass attribution instrument, compiled only into test builds; the
/// release pinned probes run as tests, so they see it, while production
/// builds carry nothing.
#[cfg(all(test, windows, target_arch = "x86_64"))]
macro_rules! sect {
    ($label:literal, $body:block) => {{
        let t0 = unsafe { core::arch::x86_64::_rdtsc() };
        let out = $body;
        let t1 = unsafe { core::arch::x86_64::_rdtsc() };
        static SECTIONS: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var_os("RESIDENT_SECTIONS").is_some());
        if *SECTIONS {
            eprintln!("BSECT {} {}", $label, t1 - t0);
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

/// Transforms one length-`n` subsequence of `src` into the padded planes of
/// `scratch`, leaving the result there rather than interleaving it back.
///
/// `STEP` and `OFFSET` select the subsequence: `STEP = 1, OFFSET = 0` reads
/// `src` whole, and `STEP = 2` reads the even or odd half of a radix-2
/// decimation directly out of the caller's buffer. Reading the parent in
/// place is the point — it is what lets [`super::four_step::four_step_fft`]
/// decimate an odd power of two without first materializing the two
/// subsequences (`gap_audit.md#reference-standing`).
///
/// Returns the plane geometry `(m, stride)` the result is addressed by:
/// element `j` of the transform is at plane index `(j / m) * stride + j % m`.
///
/// # Panics
///
/// Panics if the subsequence length is not an even power of two of at least
/// four, if `src` cannot supply it at the given step and offset, or if
/// `scratch` is shorter than [`scratch_len`].
pub(crate) fn four_step_planes<T, const INVERSE: bool, const STEP: usize, const OFFSET: usize>(
    src: &[Complex<T>],
    scratch: &mut [Complex<T>],
) -> (usize, usize)
where
    T: BatchedPlanCache<Complex = Complex<T>>,
{
    let n = src.len() / STEP;
    let k = n.trailing_zeros();
    assert!(
        n.is_power_of_two() && k % 2 == 0 && n >= 4,
        "requires an even power of two of at least 4"
    );
    assert!(OFFSET < STEP && src.len() >= n * STEP, "subsequence bounds");
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
    // The scalar loop stays deliberately: a vectorized sibling built on the
    // native deinterleave network measured slower (1431 -> 1520 TSC pinned) —
    // LLVM already auto-vectorizes this loop well (see `boundary`).
    // Chunking by `m * STEP` keeps the inner index provably in range, so the
    // strided read costs no bounds check the sequential one did not.
    sect!("deint", {
        for (row, chunk) in src.chunks_exact(m * STEP).enumerate().take(m) {
            let dest = row.reverse_bits() >> (usize::BITS - row_bits);
            for b in 0..m {
                let c = chunk[b * STEP + OFFSET];
                re[dest * stride + b] = c.re;
                im[dest * stride + b] = c.im;
            }
        }
    });

    // 1. The `m` transforms of length `m` along the first axis; the input is
    //    already batch-major for this direction, so no transpose is needed.
    let plan = T::cached_plan::<INVERSE>(m);
    sect!("stages1", {
        run_batched(re, im, plan.as_ref(), None, m, stride)
    });

    // 2. Transpose so the second axis becomes batch-major. Pure exchange:
    //    the four-step twiddle now rides stage-set-2's first loads below.
    sect!("transpose", {
        let handled = hermes_simd::vectorize_lanes::<4, T, _>(boundary::TransposePlanes {
            re,
            im,
            m,
            stride,
        })
        .unwrap_or(false);
        if !handled {
            transpose_planes(re, im, m, stride);
        }
    });

    // 3. The `m` transforms along the second axis, with the four-step twiddle
    //    W_N^{b·k1} folded into the first stage's loads — the matrix is
    //    symmetric, so applying it after the transpose is identical, and its
    //    planar row-permuted planes are built once in the shared cache. The
    //    transpose produced natural row order, so the bit-reversal pass still
    //    runs; the result lands at `k2 * m + k1`, the natural output index.
    let planes = T::cached_four_step_planes::<INVERSE>(n, m);
    sect!("permute", { permute(re, im, &plan.swaps, m, stride) });
    sect!("stages2", {
        run_batched(
            re,
            im,
            plan.as_ref(),
            Some((&planes.re, &planes.im)),
            m,
            stride,
        )
    });

    (m, stride)
}

/// In-place four-step FFT over the padded planar layout.
///
/// # Panics
///
/// Panics if `data.len()` is not an even power of two of at least four, or
/// if `scratch` is shorter than [`scratch_len`].
pub(crate) fn four_step_batched<T, const INVERSE: bool>(
    data: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
) where
    T: BatchedPlanCache<Complex = Complex<T>>,
{
    let (m, stride) = four_step_planes::<T, INVERSE, 1, 0>(data, scratch);
    let plane = scratch_len(data.len());
    let flat: &mut [T] = bytemuck::cast_slice_mut(&mut scratch[..plane]);
    let (re, im) = flat.split_at_mut(plane);

    sect!("reint", {
        let handled = hermes_simd::vectorize_lanes::<4, T, _>(boundary::InterleaveRows {
            re,
            im,
            data: bytemuck::cast_slice_mut(data),
            m,
            stride,
        })
        .unwrap_or(false);
        if !handled {
            for (row, chunk) in data.chunks_exact_mut(m).enumerate() {
                for (b, c) in chunk.iter_mut().enumerate() {
                    *c = Complex::new(re[row * stride + b], im[row * stride + b]);
                }
            }
        }
    });
}

/// In-place four-step FFT for an odd power of two, decimated once.
///
/// `X[j] = E[j] + W_N^j O[j]` and `X[j + N/2] = E[j] - W_N^j O[j]`, with `E`
/// and `O` the transforms of the even- and odd-indexed samples. Both halves
/// are even powers, so each takes the planar route — and each takes it
/// *from `data` directly*, at stride two, so the decimation never
/// materializes. The combine then rides the pass that would have
/// interleaved the halves back.
///
/// That fusion is the whole point of the route. Transforming the halves as
/// free-standing inputs costs three extra passes over `n`, which at
/// n = 8192 measured as the entire deficit against the reference
/// (`gap_audit.md#reference-standing`).
///
/// # Panics
///
/// Panics unless [`planar_split_applies`] accepts `data.len()`, or if
/// `scratch` is shorter than [`split_scratch_len`].
pub(crate) fn four_step_split_batched<T, const INVERSE: bool>(
    data: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
) where
    T: BatchedPlanCache<Complex = Complex<T>>,
{
    let n = data.len();
    assert!(planar_split_applies(n), "requires a planar odd power of two");
    let half = n / 2;
    let plane = scratch_len(half);
    assert!(
        scratch.len() >= 2 * plane,
        "scratch must hold both half-planes"
    );
    // The stage-major table ends with the length-`n` stage, whose `n / 2`
    // entries are `W_N^j` in order; earlier stages occupy `n / 2 - 1` slots.
    let twiddles = if INVERSE {
        T::cached_twiddle_inv(n)
    } else {
        T::cached_twiddle_fwd(n)
    };
    let combine = &twiddles[half - 1..n - 1];

    let (even, odd) = scratch.split_at_mut(plane);
    let (m, stride) = four_step_planes::<T, INVERSE, 2, 0>(data, even);
    four_step_planes::<T, INVERSE, 2, 1>(data, odd);
    combine_planar_halves(data, even, odd, m, stride, combine);
}

/// Combines two planar half-transforms into `data` in one pass.
///
/// `even` and `odd` hold the transforms of the even- and odd-indexed
/// subsequences in the padded plane layout [`four_step_planes`] returns.
/// This writes `X[j] = E[j] + W_N^j O[j]` and `X[j + N/2] = E[j] - W_N^j
/// O[j]`, so the butterfly rides the pass that would have interleaved each
/// half back on its own — the pass, and the half-sized buffers it would
/// have written to, are what the fusion removes.
///
/// # Panics
///
/// Panics if `data` is not twice the half-transform length, or if
/// `twiddles` is shorter than that half.
pub(crate) fn combine_planar_halves<T>(
    data: &mut [Complex<T>],
    even: &[Complex<T>],
    odd: &[Complex<T>],
    m: usize,
    stride: usize,
    twiddles: &[Complex<T>],
) where
    T: BatchedPlanCache<Complex = Complex<T>>,
{
    let half = m * m;
    assert_eq!(data.len(), 2 * half, "combine spans both halves");
    assert!(twiddles.len() >= half, "one rotation per output pair");
    let plane = m * stride;
    let (e_re, e_im) = bytemuck::cast_slice::<_, T>(&even[..plane]).split_at(plane);
    let (o_re, o_im) = bytemuck::cast_slice::<_, T>(&odd[..plane]).split_at(plane);
    let (low, high) = data.split_at_mut(half);

    sect!("combine", {
        for row in 0..m {
            let base = row * stride;
            for b in 0..m {
                let j = row * m + b;
                let e = Complex::new(e_re[base + b], e_im[base + b]);
                let o = Complex::new(o_re[base + b], o_im[base + b]);
                let rotated = o * twiddles[j];
                low[j] = e + rotated;
                high[j] = e - rotated;
            }
        }
    });
}

pub(crate) mod boundary;

// Test-gated deliberately: the interleaved in-place kernel is correct and
// measured — and slower than the planar sibling by 16 to 37% pinned on a
// P-core at every covered size, because the shuffle cost of interleaved
// butterflies outweighs the planar seams it removes. It stays as the
// independent-implementation differential oracle for this module; the
// pinned probe that declined it is committed beside it.
#[cfg(test)]
pub(crate) mod interleaved;

// Windows-gated: pins threads through Win32 to control the hybrid scheduler.
#[cfg(all(test, windows))]
mod pinned_ladder;

#[cfg(test)]
mod tests;
