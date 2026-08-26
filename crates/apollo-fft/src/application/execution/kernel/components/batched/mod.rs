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
/// A row-by-row transpose strides its destination by the row length and misses
/// on nearly every store once the array outruns L1. Tiling bounds both
/// footprints: `32 x 32` f64 pairs is 16 KB across the two planes, which stays
/// resident while the tile is processed.
const TRANSPOSE_TILE: usize = 32;

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
    batch: usize,
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
        let mut twx = 0usize;
        let mut l = 2usize;
        while l <= self.len {
            let half = l >> 1;
            let groups = self.len / l;
            for j in 0..half {
                let (twr, twi) = self.tw[twx + j];
                let wr = simd.splat(twr);
                let wi = simd.splat(twi);
                for g in 0..groups {
                    let lo = (g * l + j) * b;
                    let hi = lo + half * b;
                    let mut k = 0;
                    while k + lanes <= b {
                        let (lo_s, hi_s) = (lo + k, hi + k);
                        // Every operand is contiguous and same-component, so
                        // this is elementwise with a broadcast twiddle: no
                        // shuffle, and one fused multiply-add per component.
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
                    // Scalar remainder when the batch is not a lane multiple.
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
            twx += half;
            l <<= 1;
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
fn permute<T: Copy>(re: &mut [T], im: &mut [T], swaps: &[(u32, u32)], batch: usize) {
    for &(i, j) in swaps {
        let (i, j) = (i as usize * batch, j as usize * batch);
        for k in 0..batch {
            re.swap(i + k, j + k);
            im.swap(i + k, j + k);
        }
    }
}

/// Square in-place transpose of an `m x m` plane, tiled for locality.
fn transpose_square<T: Copy>(plane: &mut [T], m: usize) {
    debug_assert_eq!(plane.len(), m * m);
    for ib in (0..m).step_by(TRANSPOSE_TILE) {
        let ie = (ib + TRANSPOSE_TILE).min(m);
        // The diagonal tile transposes within itself; off-diagonal tiles are
        // exchanged with their mirror, so each pair is visited once.
        for jb in (ib..m).step_by(TRANSPOSE_TILE) {
            let je = (jb + TRANSPOSE_TILE).min(m);
            for i in ib..ie {
                let start = if jb == ib { i + 1 } else { jb };
                for j in start.max(jb)..je {
                    plane.swap(i * m + j, j * m + i);
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

/// Runs `batch` transforms of length `plan.len` over planar `re`/`im`.
fn run_batched<T>(re: &mut [T], im: &mut [T], plan: &BatchedPlan<T>, batch: usize)
where
    T: LaneScalar + MixedRadixScalar,
{
    permute(re, im, &plan.swaps, batch);
    hermes_simd::vectorize(BatchedStages {
        re,
        im,
        tw: &plan.tw,
        batch,
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
pub(crate) fn four_step_batched<T, const INVERSE: bool>(
    data: &mut [Complex<T>],
    scratch: &mut [Complex<T>],
) where
    T: BatchedPlanCache<Complex = Complex<T>>,
{
    let n = data.len();
    let k = n.trailing_zeros();
    assert!(
        n.is_power_of_two() && k % 2 == 0,
        "requires an even power of two"
    );
    assert!(scratch.len() >= n, "scratch must hold the whole transform");
    let m = 1usize << (k / 2);

    // One complex scratch element is two reals, so an N-element complex buffer
    // is exactly the two N-element planes this needs.
    let flat: &mut [T] = bytemuck::cast_slice_mut(&mut scratch[..n]);
    let (re, im) = flat.split_at_mut(n);

    for (i, c) in data.iter().enumerate() {
        re[i] = c.re;
        im[i] = c.im;
    }

    // 1. The `m` transforms of length `m` along the first axis; the input is
    //    already batch-major for this direction, so no transpose is needed.
    let plan = T::cached_plan::<INVERSE>(m);
    run_batched(re, im, plan.as_ref(), m);

    // 2. Elementwise four-step twiddle, W_N^{b·k1}. Direct evaluation is
    // paid once when the matrix enters the shared cache, never per transform.
    let twiddles = T::cached_four_step_twiddles::<INVERSE>(n, m, m);
    for k1 in 0..m {
        for b in 0..m {
            let i = k1 * m + b;
            let twiddle = twiddles[i];
            let (a, c) = (re[i], im[i]);
            re[i] = a * twiddle.re - c * twiddle.im;
            im[i] = a * twiddle.im + c * twiddle.re;
        }
    }

    // 3. Transpose so the second axis becomes batch-major.
    transpose_square(re, m);
    transpose_square(im, m);

    // 4. The `m` transforms along the second axis. The result lands at
    //    `k2 * m + k1`, which is the natural output index, so nothing follows.
    run_batched(re, im, plan.as_ref(), m);

    for (i, c) in data.iter_mut().enumerate() {
        *c = Complex::new(re[i], im[i]);
    }
}

#[cfg(test)]
mod tests;
