//! Register-level operations on interleaved complexes: the fused complex
//! multiply, the unaligned load and store at complex offsets, the arm
//! scatters of the stages narrower than a register, and the pointwise
//! spectrum a convolution's last pass applies.

use core::cmp::Ordering;
use eunomia::Complex;
use hermes_simd::{BitMask, LaneScalar, Mask, SimdArch, SimdKernel, SimdStorage, Vector};

/// The widest register a pass admits, in complexes: the first stage's
/// scatter has a pair decimation for 1, 2, 4 and 8 complexes per register.
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

/// Arm `k` of a first stage's registers, zero past the radix: the padding
/// row of a transpose tile.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line select reintroduces the ADR 009 penalty"
)]
#[inline(always)]
fn arm_or_zero<T, A, const R: usize>(b: [Vector<T, A>; R], k: usize) -> Vector<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    if k < R {
        b[k]
    } else {
        Vector::zero()
    }
}

/// Writes a first stage's arms: register `b[k]` holds arm `k` of `per`
/// consecutive groups, and each group's `R` arms are consecutive in `dst`,
/// group `i` from complex offset `at + R * i`.
///
/// Two arms are one pair interleave. Otherwise the arms go through square
/// complex transposes of `per` rows, `R.div_ceil(per)` tiles zero-padded
/// past `R`, each the backend's pair decimation (a square read as one flat
/// pair sequence is its stride-`per` decimation, so output `i` is column
/// `i`): column `i` of tile `t` is arms `t * per..` of group `i`, stored at
/// `at + R * i + t * per`. The last tile's padding runs [`scatter_spill`]
/// complexes past its group into the next group's leading slots, so the
/// tiles store last-first: the run-over lands only where a later store of
/// a lower tile (or the next group's own last tile, higher than the spill)
/// overwrites it, and the final group's run-over lands past `at + R * per`.
/// Every register is a tuple or array element at a constant index, so none
/// touches the stack.
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
    for t in (0..R.div_ceil(per)).rev() {
        let base = t * per;
        // SAFETY: the caller's contract bounds every store, the last tile's
        // run-over included.
        unsafe {
            match per {
                // One complex per register: each register is one arm of the
                // one group.
                1 => store(b[base], dst, at + base),
                2 => {
                    let (c0, c1) = arm_or_zero(b, base).deinterleave_pairs(arm_or_zero(b, base + 1));
                    store(c0, dst, at + base);
                    store(c1, dst, at + R + base);
                }
                4 => {
                    let (c0, c1, c2, c3) = arm_or_zero(b, base).deinterleave_pairs4(
                        arm_or_zero(b, base + 1),
                        arm_or_zero(b, base + 2),
                        arm_or_zero(b, base + 3),
                    );
                    store(c0, dst, at + base);
                    store(c1, dst, at + R + base);
                    store(c2, dst, at + 2 * R + base);
                    store(c3, dst, at + 3 * R + base);
                }
                8 => {
                    let columns = arm_or_zero(b, base).deinterleave_pairs8(
                        arm_or_zero(b, base + 1),
                        arm_or_zero(b, base + 2),
                        arm_or_zero(b, base + 3),
                        arm_or_zero(b, base + 4),
                        arm_or_zero(b, base + 5),
                        arm_or_zero(b, base + 6),
                        arm_or_zero(b, base + 7),
                    );
                    for (i, column) in columns.into_iter().enumerate() {
                        store(column, dst, at + i * R + base);
                    }
                }
                _ => unreachable!(
                    "invariant: the lane count is a power of two, so a register holds 1, 2, 4 or 8 complexes up to MAX_COMPLEXES_PER_REGISTER"
                ),
            }
        }
    }
}

/// Writes the arms of a stage whose rows are half a register: register
/// `b[k]` holds arm `k` of two consecutive groups, `prev_len = per / 2`
/// complexes each, and the two groups' outputs are contiguous in `dst`
/// from `at`, each group its `R` arms in order. The output stream is the
/// low halves of `b[0..R]` then their high halves, packed two halves per
/// register: `(b[2i].lo, b[2i+1].lo)` while both are low halves, the
/// middle register of an odd radix `(b[R-1].lo, b[0].hi)`, then
/// `(b[2i-R].hi, b[2i+1-R].hi)`. Every register is a half interleave or a
/// half blend of two arms; no padding, no run-over.
///
/// # Safety
/// `at + R * per <= dst.len()`.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line scatter reintroduces the ADR 009 penalty"
)]
#[inline(always)]
pub(in super::super) unsafe fn store_arm_halves<T, A, const R: usize>(
    b: [Vector<T, A>; R],
    dst: &mut [Complex<T>],
    at: usize,
) where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let per = <A as SimdStorage<T>>::LANE_COUNT / 2;
    debug_assert!(at + R * per <= dst.len());
    for i in 0..R {
        let packed = match (2 * i + 1).cmp(&R) {
            Ordering::Less => b[2 * i].interleave_halves(b[2 * i + 1]).0,
            Ordering::Equal => b[R - 1].blend_halves(b[0]),
            Ordering::Greater => b[2 * i - R].interleave_halves(b[2 * i + 1 - R]).1,
        };
        // SAFETY: the caller's contract; `R` registers cover `R * per` complexes.
        unsafe { store(packed, dst, at + i * per) };
    }
}

/// One register holding `row` twice, for a stage whose twiddle rows are
/// half a register: the multiplier of two groups' arms at once.
///
/// # Panics
/// Panics if `row` is not half a register long.
#[inline]
pub(in super::super) fn duplicated_row<T, A>(row: &[Complex<T>]) -> Vector<T, A>
where
    T: LaneScalar + eunomia::RealField,
    A: SimdArch + SimdKernel<T>,
{
    let per = <A as SimdStorage<T>>::LANE_COUNT / 2;
    assert!(
        2 * row.len() == per && per <= MAX_COMPLEXES_PER_REGISTER,
        "invariant: a duplicated row is half a register"
    );
    let zero = Complex::new(T::from_f64(0.0), T::from_f64(0.0));
    let mut pair = [zero; MAX_COMPLEXES_PER_REGISTER];
    pair[..row.len()].copy_from_slice(row);
    pair[row.len()..per].copy_from_slice(row);
    // SAFETY: `pair` holds `MAX_COMPLEXES_PER_REGISTER >= per` complexes.
    unsafe { load::<T, A>(&pair, 0) }
}

/// The mask of the first `count` complexes of a register: the ragged tail
/// of a row, `count < per`.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line mask reintroduces the ADR 009 penalty"
)]
#[inline(always)]
pub(in super::super) fn prefix_mask<T, A>(count: usize) -> Mask<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(2 * count <= <A as SimdStorage<T>>::LANE_COUNT);
    // SAFETY: the kernel runs inside its backend's dispatch frame, which
    // proved host support.
    unsafe { Mask::from_bitmask(BitMask((1_u64 << (2 * count)) - 1)) }
}

/// The first `count` complexes at complex offset `at` of `data`, zero
/// beyond them: a ragged tail loaded without reading past the row.
///
/// # Safety
/// `at + count <= data.len()`, and `mask` is [`prefix_mask`] of `count`.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line load reintroduces the ADR 009 penalty"
)]
#[inline(always)]
pub(in super::super) unsafe fn load_prefix<T, A>(
    data: &[Complex<T>],
    at: usize,
    count: usize,
    mask: Mask<T, A>,
) -> Vector<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(at + count <= data.len());
    // SAFETY: the caller's contract; only the `2 count` masked lanes are
    // read, and they lie inside the slice.
    unsafe {
        Vector::<T, A>::masked_load_partial(
            data.as_ptr().cast::<T>().add(2 * at),
            2 * count,
            mask,
            Vector::zero(),
        )
    }
}

/// Stores the first `count` complexes of `v` at complex offset `at`.
///
/// # Safety
/// As [`load_prefix`].
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line store reintroduces the ADR 009 penalty"
)]
#[inline(always)]
pub(in super::super) unsafe fn store_prefix<T, A>(
    v: Vector<T, A>,
    data: &mut [Complex<T>],
    at: usize,
    count: usize,
    mask: Mask<T, A>,
) where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(at + count <= data.len());
    // SAFETY: the caller's contract; only the `2 count` masked lanes are
    // written, and they lie inside the slice.
    unsafe { v.masked_store_partial(data.as_mut_ptr().cast::<T>().add(2 * at), 2 * count, mask) }
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
