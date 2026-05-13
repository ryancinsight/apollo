//! Plan-owned FFT workspace allocation helpers.

#![allow(clippy::uninit_vec)]

use half::f16;
use num_complex::{Complex, Complex32, Complex64};

mod sealed {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for half::f16 {}
    impl Sealed for num_complex::Complex<half::f16> {}
    impl Sealed for num_complex::Complex32 {}
    impl Sealed for num_complex::Complex64 {}
}

/// FFT workspace elements whose bit patterns are valid and that have no drop glue.
///
/// This trait is sealed so the uninitialized workspace allocation contract
/// cannot be extended outside this module to types with stricter validity
/// invariants.
pub(crate) trait UninitWorkspaceElement: Copy + sealed::Sealed {}

impl UninitWorkspaceElement for f32 {}
impl UninitWorkspaceElement for f64 {}
impl UninitWorkspaceElement for f16 {}
impl UninitWorkspaceElement for Complex<f16> {}
impl UninitWorkspaceElement for Complex32 {}
impl UninitWorkspaceElement for Complex64 {}

/// Allocate a workspace without zero-filling it.
///
/// # Contract
///
/// The returned vector has length `len`, but its element values are unspecified.
/// Callers may only use it for work buffers whose full contents are overwritten
/// before the first read. `UninitWorkspaceElement` is sealed to Apollo's current
/// FFT scratch types: `f16`, `f32`, `f64`, `Complex<f16>`, `Complex32`, and `Complex64`.
#[inline]
pub(crate) fn uninit_copy_vec<T: UninitWorkspaceElement>(len: usize) -> Vec<T> {
    let mut values = Vec::with_capacity(len);
    // SAFETY: `T` is restricted by the sealed trait above to FFT scratch
    // scalars with no invalid bit patterns and no destructor. Callers must
    // satisfy the overwrite-before-read contract stated above.
    unsafe { values.set_len(len) };
    values
}
