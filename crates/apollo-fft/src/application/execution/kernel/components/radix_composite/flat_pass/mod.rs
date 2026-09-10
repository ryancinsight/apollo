//! Flat Stockham passes over the dispatched register width, one generic
//! kernel per radix.
//!
//! The interleaved-complex arithmetic rides hermes' sub-lane primitives
//! (`dup_even`, `dup_odd`, `swap_adjacent`, `fmaddsub`): the same products
//! and the same fused structure as the AVX2 intrinsics these kernels
//! replace, so every backend `vectorize` dispatches (AVX2, AVX-512, NEON)
//! shares one body and one per-element operation order, and the scalar
//! backend declines to the scalar pass. A `Complex<T>` is two `T` reals in
//! order, so a register of `LANE_COUNT` reals holds `LANE_COUNT / 2`
//! complexes and a complex offset is twice a real one.

mod radix2;

pub(super) use radix2::FlatPassR2;

use eunomia::Complex;
use hermes_simd::{LaneScalar, SimdArch, SimdKernel, SimdStorage, Vector};

/// `x * w` on interleaved complexes: even lanes `xr wr - xi wi`, odd lanes
/// `xr wi + xi wr`, one fused multiply-add-subtract per register as the
/// AVX2 `cmul` computed it.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line multiply reintroduces the ADR 009 penalty"
)]
#[inline(always)]
pub(super) fn cmul<T, A>(x: Vector<T, A>, w: Vector<T, A>) -> Vector<T, A>
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
pub(super) unsafe fn load<T, A>(data: &[Complex<T>], at: usize) -> Vector<T, A>
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
pub(super) unsafe fn store<T, A>(v: Vector<T, A>, data: &mut [Complex<T>], at: usize)
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(2 * at + <A as SimdStorage<T>>::LANE_COUNT <= 2 * data.len());
    // SAFETY: the caller's contract; a `Complex<T>` is two `T` in order.
    unsafe { v.store_unaligned(data.as_mut_ptr().cast::<T>().add(2 * at)) }
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
pub(super) fn apply_pointwise<T, A>(dst: &mut [Complex<T>], factors: &[Complex<T>])
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
