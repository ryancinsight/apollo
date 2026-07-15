//! Concrete storage admitted by the SDFT accelerator boundary.

use apollo_fft::{f16, PrecisionProfile};
use eunomia::Complex32;

mod sealed {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for apollo_fft::f16 {}
    impl Sealed for eunomia::Complex32 {}
    impl Sealed for [apollo_fft::f16; 2] {}
}

/// Real input storage admitted by the concrete `f32` SDFT forward kernel.
///
/// Native `f32` storage borrows directly into the Hephaestus upload. Reduced
/// `f16` storage converts through Mnemosyne scratch. The CPU owner's `f64`
/// representation is intentionally excluded so GPU execution never narrows
/// an SDFT window implicitly.
///
/// ```compile_fail
/// use apollo_sdft::SdftGpuRealStorage;
///
/// fn require_gpu_storage<T: SdftGpuRealStorage>() {}
/// require_gpu_storage::<f64>();
/// ```
pub trait SdftGpuRealStorage: Copy + Send + Sync + 'static + sealed::Sealed {
    /// Precision profile required by this representation.
    const PROFILE: PrecisionProfile;

    /// Convert this value into the concrete accelerator input representation.
    fn to_gpu(self) -> f32;

    /// Borrow storage as concrete accelerator values when layouts match.
    fn as_gpu_slice(slice: &[Self]) -> Option<&[f32]>;
}

impl SdftGpuRealStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_gpu(self) -> f32 {
        self
    }

    fn as_gpu_slice(slice: &[Self]) -> Option<&[f32]> {
        Some(slice)
    }
}

impl SdftGpuRealStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_gpu(self) -> f32 {
        self.to_f32()
    }

    fn as_gpu_slice(_: &[Self]) -> Option<&[f32]> {
        None
    }
}

/// Complex output storage admitted by the concrete SDFT forward kernel.
///
/// Native [`Complex32`] storage receives Hephaestus downloads directly.
/// Reduced `[f16; 2]` storage converts once after the `f32` kernel. CPU
/// `Complex64` output remains outside this boundary so high-accuracy callers
/// must use the CPU plan or make an explicit quantization decision.
///
/// ```compile_fail
/// use apollo_sdft::SdftGpuBinStorage;
/// use eunomia::Complex64;
///
/// fn require_gpu_storage<T: SdftGpuBinStorage>() {}
/// require_gpu_storage::<Complex64>();
/// ```
pub trait SdftGpuBinStorage: Copy + Send + Sync + 'static + sealed::Sealed {
    /// Precision profile required by this representation.
    const PROFILE: PrecisionProfile;

    /// Convert concrete accelerator output into this representation.
    fn from_gpu(value: Complex32) -> Self;

    /// View mutable storage as accelerator values when layouts match.
    fn as_gpu_slice_mut(slice: &mut [Self]) -> Option<&mut [Complex32]>;
}

impl SdftGpuBinStorage for Complex32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn from_gpu(value: Complex32) -> Self {
        value
    }

    fn as_gpu_slice_mut(slice: &mut [Self]) -> Option<&mut [Complex32]> {
        Some(slice)
    }
}

impl SdftGpuBinStorage for [f16; 2] {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn from_gpu(value: Complex32) -> Self {
        [f16::from_f32(value.re), f16::from_f32(value.im)]
    }

    fn as_gpu_slice_mut(_: &mut [Self]) -> Option<&mut [Complex32]> {
        None
    }
}
