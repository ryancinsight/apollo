//! Sealed `BluesteinScalar` trait — monomorphic dispatch surface for Bluestein kernels.
//!
//! Eliminates all `_64`/`_32` function suffixes in the pointwise kernel layer by
//! encoding the ISA-specific AVX/FMA dispatch as associated functions on this trait.
//! All generic helper functions in `pointwise.rs` are parameterized over `C: BluesteinScalar`.

use num_complex::{Complex32, Complex64};

/// Sealing module — prevents external implementations of `BluesteinScalar`.
pub(super) mod private {
    pub trait Sealed {}
}

/// Zero-cost dispatch surface for Bluestein pointwise operations.
///
/// Each `impl BluesteinScalar` delegates the AVX bodies to the type-specific
/// `unsafe fn` definitions in `avx_f64.rs` / `avx_f32.rs`.  All generic code
/// above this boundary sees only the trait methods and is monomorphized by the
/// compiler into machine code identical to the former hand-written specializations.
///
/// # Safety contract
/// Every `avx_*` associated function is `unsafe` and may only be called from
/// within the `if has_avx_fma() && ...` guard in `pointwise.rs`.
pub(crate) trait BluesteinScalar:
    Copy
    + Send
    + Sync
    + Default
    + private::Sealed
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::MulAssign
{
    /// `size_of::<Self>()` — used for `write_bytes` zero-fill.
    const BYTE_SIZE: usize;
    /// Minimum slice length for which AVX processing is profitable.
    const SIMD_MIN: usize;

    // ── AVX/FMA kernels (ISA-dispatched per concrete type) ────────────────────

    /// `dst[i] = input[i] * factors[i]` (complex multiplication).
    ///
    /// # Safety
    /// Caller must hold AVX+FMA CPU feature guarantee and ensure all slices
    /// have equal length ≥ `SIMD_MIN`.
    unsafe fn avx_fill_mul(dst: &mut [Self], input: &[Self], factors: &[Self]);

    /// `dst[i] = conj_mul(input[i], factors[i])` — conjugate multiply.
    ///
    /// # Safety
    /// Same as `avx_fill_mul`.
    unsafe fn avx_fill_mul_conj(dst: &mut [Self], input: &[Self], factors: &[Self]);

    /// `dst[i] *= twiddle[i]` in-place.
    ///
    /// # Safety
    /// Same as `avx_fill_mul`.
    unsafe fn avx_mul_inplace(dst: &mut [Self], twiddle: &[Self]);

    /// In-place inverse twiddle kernel: element 0 is conjugate-multiplied by
    /// `twiddle[0]`; elements `1..` are conjugate-multiplied in reverse order.
    ///
    /// # Safety
    /// Same as `avx_fill_mul`.
    unsafe fn avx_mul_inplace_inverse(dst: &mut [Self], twiddle: &[Self]);

    /// Parallel chunk variant of the inverse twiddle kernel.
    ///
    /// # Safety
    /// `factor_base < twiddle.len()` and `factor_base + 1 >= dst.len()`.
    unsafe fn avx_mul_inplace_inverse_chunk(dst: &mut [Self], twiddle: &[Self], factor_base: usize);

    // ── Scalar fallbacks (always safe) ────────────────────────────────────────

    /// Conjugate multiply: `conj(a) * b` — used for inverse chirp convolution.
    ///
    /// For `Complex<f>`: `re = a.re*b.re + a.im*b.im`, `im = a.im*b.re - a.re*b.im`.
    fn conj_mul(a: Self, b: Self) -> Self;
}

// ── Complex64 impl ────────────────────────────────────────────────────────────

impl private::Sealed for Complex64 {}

impl BluesteinScalar for Complex64 {
    const BYTE_SIZE: usize = size_of::<Self>();
    const SIMD_MIN: usize = 8;

    #[inline]
    unsafe fn avx_fill_mul(dst: &mut [Self], input: &[Self], factors: &[Self]) {
        super::avx_f64::mul_complex_pointwise_64_avx_from_input(dst, input, factors);
    }

    #[inline]
    unsafe fn avx_fill_mul_conj(dst: &mut [Self], input: &[Self], factors: &[Self]) {
        super::avx_f64::mul_complex_pointwise_64_avx_from_input_conj(dst, input, factors);
    }

    #[inline]
    unsafe fn avx_mul_inplace(dst: &mut [Self], twiddle: &[Self]) {
        super::avx_f64::mul_complex_pointwise_64_avx_inplace(dst, twiddle);
    }

    #[inline]
    unsafe fn avx_mul_inplace_inverse(dst: &mut [Self], twiddle: &[Self]) {
        super::avx_f64::mul_complex_pointwise_64_avx_inplace_inverse(dst, twiddle);
    }

    #[inline]
    unsafe fn avx_mul_inplace_inverse_chunk(
        dst: &mut [Self],
        twiddle: &[Self],
        factor_base: usize,
    ) {
        super::avx_f64::mul_complex_pointwise_64_avx_inplace_inverse_chunk(
            dst,
            twiddle,
            factor_base,
        );
    }

    #[inline]
    fn conj_mul(a: Self, b: Self) -> Self {
        let re = a.re * b.re + a.im * b.im;
        let im = a.im * b.re - a.re * b.im;
        Self::new(re, im)
    }
}

// ── Complex32 impl ────────────────────────────────────────────────────────────

impl private::Sealed for Complex32 {}

impl BluesteinScalar for Complex32 {
    const BYTE_SIZE: usize = size_of::<Self>();
    const SIMD_MIN: usize = 8;

    #[inline]
    unsafe fn avx_fill_mul(dst: &mut [Self], input: &[Self], factors: &[Self]) {
        super::avx_f32::mul_complex_pointwise_32_avx_from_input(dst, input, factors);
    }

    #[inline]
    unsafe fn avx_fill_mul_conj(dst: &mut [Self], input: &[Self], factors: &[Self]) {
        super::avx_f32::mul_complex_pointwise_32_avx_from_input_conj(dst, input, factors);
    }

    #[inline]
    unsafe fn avx_mul_inplace(dst: &mut [Self], twiddle: &[Self]) {
        super::avx_f32::mul_complex_pointwise_32_avx_inplace(dst, twiddle);
    }

    #[inline]
    unsafe fn avx_mul_inplace_inverse(dst: &mut [Self], twiddle: &[Self]) {
        super::avx_f32::mul_complex_pointwise_32_avx_inplace_inverse(dst, twiddle);
    }

    #[inline]
    unsafe fn avx_mul_inplace_inverse_chunk(
        dst: &mut [Self],
        twiddle: &[Self],
        factor_base: usize,
    ) {
        super::avx_f32::mul_complex_pointwise_32_avx_inplace_inverse_chunk(
            dst,
            twiddle,
            factor_base,
        );
    }

    #[inline]
    fn conj_mul(a: Self, b: Self) -> Self {
        let re = a.re * b.re + a.im * b.im;
        let im = a.im * b.re - a.re * b.im;
        Self::new(re, im)
    }
}
