//! Register-level operations on interleaved complexes: the fused complex
//! multiply, the unaligned load and store at complex offsets, the first
//! stage's arm scatter, and the pointwise spectrum a convolution's last
//! pass applies.

use eunomia::Complex;
use hermes_simd::{ComplexReg, LaneScalar, SimdArch, SimdKernel, SimdStorage, Vector};

/// The widest register a pass admits, in complexes: the arm scatter holds
/// `R.div_ceil(per) * per <= 8` transposed columns for every radix up to 7.
pub(in super::super) const MAX_COMPLEXES_PER_REGISTER: usize = 8;

/// `x * w` on interleaved complexes: even lanes `xr wr - xi wi`, odd lanes
/// `xr wi + xi wr`, one fused multiply-add-subtract per register as the
/// AVX2 `cmul` computed it.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line multiply reintroduces the ADR 009 penalty"
)]
#[inline(always)]
pub(in super::super) fn cmul<T, A>(x: Vector<T, A>, w: Vector<T, A>) -> Vector<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    w.dup_even().fmaddsub(x, w.dup_odd() * x.swap_adjacent())
}

/// One register of interleaved complexes at complex offset `at` of `data`.
///
/// # Safety
/// `2 * at + LANE_COUNT <= 2 * data.len()`: the register stays inside the
/// slice, which the callers assert once per pass.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line load reintroduces the ADR 009 penalty"
)]
#[inline(always)]
pub(in super::super) unsafe fn load<T, A>(data: &[Complex<T>], at: usize) -> Vector<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(2 * at + <A as SimdStorage<T>>::LANE_COUNT <= 2 * data.len());
    // SAFETY: the caller's contract; a `Complex<T>` is two `T` in order.
    unsafe { Vector::<T, A>::load_unaligned(data.as_ptr().cast::<T>().add(2 * at)) }
}

/// One register of interleaved complexes stored at complex offset `at`.
///
/// # Safety
/// As [`load`].
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line store reintroduces the ADR 009 penalty"
)]
#[inline(always)]
pub(in super::super) unsafe fn store<T, A>(v: Vector<T, A>, data: &mut [Complex<T>], at: usize)
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(2 * at + <A as SimdStorage<T>>::LANE_COUNT <= 2 * data.len());
    // SAFETY: the caller's contract; a `Complex<T>` is two `T` in order.
    unsafe { v.store_unaligned(data.as_mut_ptr().cast::<T>().add(2 * at)) }
}

/// The complexes an arm scatter of radix `R` writes past its `per` groups:
/// the zero padding of its last transpose tile, `per - R % per` when `R`
/// does not fill whole tiles, none for two arms (a pair interleave) or a
/// radix that is a multiple of the width.
#[inline]
pub(in super::super) fn scatter_spill<T, A, const R: usize>() -> usize
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let per = <A as SimdStorage<T>>::LANE_COUNT / 2;
    if R == 2 || per == 0 || R % per == 0 {
        0
    } else {
        per - R % per
    }
}

/// Writes a first stage's arms: register `b[k]` holds arm `k` of `per`
/// consecutive groups, and each group's `R` arms are consecutive in `dst`,
/// group `i` from complex offset `at + R * i`.
///
/// Two arms are one pair interleave. Otherwise the arms go through square
/// complex transposes of `per` rows, `R.div_ceil(per)` tiles zero-padded
/// past `R`: column `i` of tile `t` is arms `t * per..` of group `i`, stored
/// at `at + R * i + t * per`. A padded tile's column runs
/// [`scatter_spill`] complexes past its group into the next group's slot,
/// which the next store (groups ascending, tiles ascending within a group)
/// overwrites; the last store's run-over lands past `at + R * per`.
///
/// # Safety
/// `at + R * per + scatter_spill::<T, A, R>() <= dst.len()`, and the width
/// is at most [`MAX_COMPLEXES_PER_REGISTER`] complexes.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line scatter reintroduces the ADR 009 penalty"
)]
#[inline(always)]
pub(in super::super) unsafe fn store_arms<T, A, const R: usize>(
    b: [Vector<T, A>; R],
    dst: &mut [Complex<T>],
    at: usize,
) where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let per = <A as SimdStorage<T>>::LANE_COUNT / 2;
    debug_assert!(
        (1..=MAX_COMPLEXES_PER_REGISTER).contains(&per)
            && at + R * per + scatter_spill::<T, A, R>() <= dst.len()
    );
    if R == 2 {
        let (first, second) = b[0].interleave_pairs(b[1]);
        // SAFETY: the caller's contract; two registers cover `2 * per` complexes.
        unsafe {
            store(first, dst, at);
            store(second, dst, at + per);
        }
        return;
    }
    let tiles = R.div_ceil(per);
    // Every tile's columns, tile-major: `columns[t * per + i]` is column `i`
    // of tile `t`.
    let mut columns = [ComplexReg::<T, A>::zero(); MAX_COMPLEXES_PER_REGISTER];
    for t in 0..tiles {
        let tile = &mut columns[t * per..(t + 1) * per];
        for (row, arm) in tile.iter_mut().zip(&b[t * per..]) {
            *row = ComplexReg::from_interleaved(*arm);
        }
        ComplexReg::transpose_square(tile);
    }
    for i in 0..per {
        for t in 0..tiles {
            // SAFETY: the caller's contract bounds the last store's run-over.
            unsafe {
                store(
                    columns[t * per + i].into_interleaved(),
                    dst,
                    at + R * i + t * per,
                );
            }
        }
    }
}

/// Multiplies `dst` by `factors` element-wise, register-wide then scalar:
/// the pointwise spectrum a convolution's last pass applies.
///
/// # Panics
/// Panics if `factors` is shorter than `dst`.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope with the pass that calls it"
)]
#[inline(always)]
pub(in super::super) fn apply_pointwise<T, A>(dst: &mut [Complex<T>], factors: &[Complex<T>])
where
    T: LaneScalar + eunomia::RealField,
    A: SimdArch + SimdKernel<T>,
{
    assert!(
        factors.len() >= dst.len(),
        "invariant: the pointwise spectrum covers the output"
    );
    let per = <A as SimdStorage<T>>::LANE_COUNT / 2;
    let n = dst.len();
    let mut i = 0;
    while per > 0 && i + per <= n {
        // SAFETY: `i + per <= n <= factors.len()`, so both registers stay
        // inside their slices.
        unsafe {
            let d = load::<T, A>(dst, i);
            let p = load::<T, A>(factors, i);
            store(cmul(d, p), dst, i);
        }
        i += per;
    }
    for (d, p) in dst[i..].iter_mut().zip(&factors[i..n]) {
        *d = Complex::new(d.re * p.re - d.im * p.im, d.re * p.im + d.im * p.re);
    }
}
