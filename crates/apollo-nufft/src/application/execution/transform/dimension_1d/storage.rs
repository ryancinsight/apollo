//! Typed complex storage views for 1D NUFFT paths.

use apollo_fft::{CpuStorage, F16};
use eunomia::{Complex32, Complex64};

/// Complex storage admitted by typed NUFFT paths.
///
/// The conversion ladder and precision profile come from the shared
/// [`CpuStorage`] vocabulary; this trait adds only the reinterpret
/// views the device conversion helpers use to skip a copy when the
/// host layout already matches an accelerator element.
pub trait NufftComplexStorage: CpuStorage<Complex64> {
    /// View slice as `Complex32` if layout is identical.
    #[inline]
    fn as_c32_slice(slice: &[Self]) -> Option<&[Complex32]> {
        let _ = slice;
        None
    }

    /// View mutable slice as `Complex32` if layout is identical.
    #[inline]
    fn as_c32_slice_mut(slice: &mut [Self]) -> Option<&mut [Complex32]> {
        let _ = slice;
        None
    }

    /// View slice as `Complex64` if layout is identical.
    #[inline]
    fn as_c64_slice(slice: &[Self]) -> Option<&[Complex64]> {
        let _ = slice;
        None
    }

    /// View mutable slice as `Complex64` if layout is identical.
    #[inline]
    fn as_c64_slice_mut(slice: &mut [Self]) -> Option<&mut [Complex64]> {
        let _ = slice;
        None
    }
}

impl NufftComplexStorage for Complex64 {
    #[inline]
    fn as_c64_slice(slice: &[Self]) -> Option<&[Complex64]> {
        Some(slice)
    }

    #[inline]
    fn as_c64_slice_mut(slice: &mut [Self]) -> Option<&mut [Complex64]> {
        Some(slice)
    }
}

impl NufftComplexStorage for Complex32 {
    #[inline]
    fn as_c32_slice(slice: &[Self]) -> Option<&[Complex32]> {
        Some(slice)
    }

    #[inline]
    fn as_c32_slice_mut(slice: &mut [Self]) -> Option<&mut [Complex32]> {
        Some(slice)
    }
}

impl NufftComplexStorage for [F16; 2] {}
