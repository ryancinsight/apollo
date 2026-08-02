//! Shared typed-storage contract for `f32`-arithmetic GPU transforms
//! (ADR 0037).

use crate::{f16, PrecisionProfile};

mod sealed {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for crate::f16 {}
}

/// Storage whose declared compute profile is the native `f32` GPU contract.
///
/// This sealed capability admits `f32` and the explicit mixed `f16`/`f32`
/// profile. `f64` is intentionally excluded so GPU typed APIs cannot silently
/// narrow high-accuracy storage to `f32` arithmetic.
///
/// ```compile_fail
/// use apollo_fft::GpuStorage;
///
/// fn require_gpu_storage<T: GpuStorage>() {}
/// require_gpu_storage::<f64>();
/// ```
pub trait GpuStorage: Copy + Send + Sync + 'static + sealed::Sealed {
    /// Precision profile this storage declares.
    const PROFILE: PrecisionProfile;

    /// Convert storage into the concrete `f32` accelerator contract.
    fn to_gpu(self) -> f32;

    /// Convert a concrete `f32` accelerator result back to storage.
    fn from_gpu(value: f32) -> Self;

    /// View slice as `f32` if layout is identical.
    #[inline]
    fn as_f32_slice(slice: &[Self]) -> Option<&[f32]> {
        let _ = slice;
        None
    }

    /// View mutable slice as `f32` if layout is identical.
    #[inline]
    fn as_f32_slice_mut(slice: &mut [Self]) -> Option<&mut [f32]> {
        let _ = slice;
        None
    }
}

impl GpuStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_gpu(self) -> f32 {
        self
    }

    fn from_gpu(value: f32) -> Self {
        value
    }

    #[inline]
    fn as_f32_slice(slice: &[Self]) -> Option<&[f32]> {
        Some(slice)
    }

    #[inline]
    fn as_f32_slice_mut(slice: &mut [Self]) -> Option<&mut [f32]> {
        Some(slice)
    }
}

impl GpuStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_gpu(self) -> f32 {
        self.to_f32()
    }

    fn from_gpu(value: f32) -> Self {
        f16::from_f32(value)
    }
}
