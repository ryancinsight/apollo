//! Unchecked register loads and stores over the planes and the caller's
//! interleaved rows, and the row map between the two.

use hermes_simd::{LaneScalar, SimdArch, SimdKernel, SimdStorage, Vector};

/// One-vector load at `at`, which the caller has proved is in bounds.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an out-of-line               call here reintroduces the ADR 009 penalty this kernel exists to avoid"
)]
#[inline(always)]
pub(super) fn load<T, A>(data: &[T], at: usize) -> Vector<T, A>
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
pub(super) fn store<T, A>(v: Vector<T, A>, data: &mut [T], at: usize)
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    debug_assert!(at + <A as SimdStorage<T>>::LANE_COUNT <= data.len());
    // SAFETY: as `load` above.
    unsafe { v.store_unaligned(data.as_mut_ptr().add(at)) }
}

/// Two adjacent vectors of interleaved complexes at `at` (in reals), split
/// into their real and imaginary lanes in the plane column order
/// ([`super::lane_order::LaneOrder`]): the sub-lane unpack alone, no cross-lane permute.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope, as `load`"
)]
#[inline(always)]
pub(super) fn load_interleaved<T, A>(data: &[T], at: usize) -> (Vector<T, A>, Vector<T, A>)
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let lanes = <A as SimdStorage<T>>::LANE_COUNT;
    load::<T, A>(data, at).deinterleave_sublanes(load::<T, A>(data, at + lanes))
}

/// The counterpart of [`load_interleaved`]: real and imaginary lanes in plane
/// column order stored as two adjacent vectors of interleaved complexes at
/// `at` (in reals), in memory order.
#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope, as `store`"
)]
#[inline(always)]
pub(super) fn store_interleaved<T, A>(re: Vector<T, A>, im: Vector<T, A>, data: &mut [T], at: usize)
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    let lanes = <A as SimdStorage<T>>::LANE_COUNT;
    let (lo, hi) = re.interleave_sublanes(im);
    store::<T, A>(lo, data, at);
    store::<T, A>(hi, data, at + lanes);
}

/// Bit-reverses a row index within `bits` bits; the map between plane rows
/// and interleaved rows on both sides of the stage sets.
#[inline]
pub(super) fn reverse_row(row: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        row.reverse_bits() >> (usize::BITS - bits)
    }
}
