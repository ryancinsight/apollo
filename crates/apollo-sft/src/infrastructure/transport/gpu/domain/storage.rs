//! Concrete storage admitted by the SFT accelerator boundary.

use apollo_fft::{f16, PrecisionProfile};
use eunomia::Complex32;

mod sealed {
    pub trait Sealed {}

    impl Sealed for eunomia::Complex32 {}
    impl Sealed for [apollo_fft::f16; 2] {}
}

/// Storage admitted by the concrete `Complex32` accelerator kernel.
///
/// Native [`Complex32`] values borrow directly into the Hephaestus upload.
/// Mixed `[f16; 2]` values convert through a Mnemosyne scratch buffer. The CPU
/// owner's `Complex64` storage is deliberately excluded: accelerator callers
/// must choose concrete `f32` storage rather than silently narrowing a
/// high-accuracy sparse transform.
///
/// ```compile_fail
/// use apollo_sft::SftGpuStorage;
/// use eunomia::Complex64;
///
/// fn require_gpu_storage<T: SftGpuStorage>() {}
/// require_gpu_storage::<Complex64>();
/// ```
pub trait SftGpuStorage: Copy + Send + Sync + 'static + sealed::Sealed {
    /// Precision profile required by this representation.
    const PROFILE: PrecisionProfile;

    /// Convert host storage into the concrete `Complex32` accelerator representation.
    fn to_gpu(self) -> Complex32;

    /// Convert concrete accelerator output back into host storage.
    fn from_gpu(value: Complex32) -> Self;

    /// View host storage as concrete accelerator values when layouts match.
    fn as_gpu_slice(slice: &[Self]) -> Option<&[Complex32]>;

    /// View mutable host storage as accelerator values when layouts match.
    fn as_gpu_slice_mut(slice: &mut [Self]) -> Option<&mut [Complex32]>;
}

impl SftGpuStorage for Complex32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_gpu(self) -> Complex32 {
        self
    }

    fn from_gpu(value: Complex32) -> Self {
        value
    }

    fn as_gpu_slice(slice: &[Self]) -> Option<&[Complex32]> {
        Some(slice)
    }

    fn as_gpu_slice_mut(slice: &mut [Self]) -> Option<&mut [Complex32]> {
        Some(slice)
    }
}

impl SftGpuStorage for [f16; 2] {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_gpu(self) -> Complex32 {
        Complex32::new(self[0].to_f32(), self[1].to_f32())
    }

    fn from_gpu(value: Complex32) -> Self {
        [f16::from_f32(value.re), f16::from_f32(value.im)]
    }

    fn as_gpu_slice(_: &[Self]) -> Option<&[Complex32]> {
        None
    }

    fn as_gpu_slice_mut(_: &mut [Self]) -> Option<&mut [Complex32]> {
        None
    }
}
