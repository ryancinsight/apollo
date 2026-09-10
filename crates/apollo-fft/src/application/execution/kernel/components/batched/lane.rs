//! One lane of butterfly arithmetic: the SIMD vector or the scalar element.
//!
//! The butterflies are written once over [`Lane`], which both implement, so
//! the vector loop and its scalar remainder run the same arithmetic. The
//! vector form fuses each complex multiply's products (`w.re * x.re - w.im *
//! x.im` as one fused multiply-add against the exactly negated second
//! product); the scalar form keeps the two-operation product the remainder
//! always used. Both are the forms the stage sets carried before the
//! grouping was generalised, so the outputs of every existing test are
//! unchanged bit for bit.

use core::ops::{Add, Mul, Sub};

use hermes_simd::{LaneScalar, SimdArch, SimdKernel, Vector};

/// One lane of butterfly arithmetic: a SIMD vector or a scalar element.
pub(crate) trait Lane:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
{
    /// `self * b + c`.
    fn fma(self, b: Self, c: Self) -> Self;
    /// `self * b - c`.
    fn fms(self, b: Self, c: Self) -> Self;
}

impl<T, A> Lane for Vector<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    #[expect(
        clippy::inline_always,
        reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
    )]
    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        self.mul_add(b, c)
    }

    #[expect(
        clippy::inline_always,
        reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
    )]
    #[inline(always)]
    fn fms(self, b: Self, c: Self) -> Self {
        self.mul_add(b, -c)
    }
}

impl Lane for f64 {
    #[expect(
        clippy::inline_always,
        reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
    )]
    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        self * b + c
    }

    #[expect(
        clippy::inline_always,
        reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
    )]
    #[inline(always)]
    fn fms(self, b: Self, c: Self) -> Self {
        self * b - c
    }
}

impl Lane for f32 {
    #[expect(
        clippy::inline_always,
        reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
    )]
    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        self * b + c
    }

    #[expect(
        clippy::inline_always,
        reason = "must fold into the caller's target-feature scope; an out-of-line butterfly reintroduces the ADR 009 penalty"
    )]
    #[inline(always)]
    fn fms(self, b: Self, c: Self) -> Self {
        self * b - c
    }
}
