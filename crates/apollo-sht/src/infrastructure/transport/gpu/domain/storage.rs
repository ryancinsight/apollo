//! Concrete storage admitted by the SHT accelerator boundary.

use apollo_fft::{f16, PrecisionProfile};
use eunomia::Complex32;

mod sealed {
    pub trait Sealed {}

    impl Sealed for eunomia::Complex32 {}
    impl Sealed for [apollo_fft::f16; 2] {}
}

/// Complex storage admitted by the concrete `Complex32` SHT kernels.
///
/// Native [`Complex32`] values transfer directly. Reduced `[f16; 2]` values
/// convert through Mnemosyne scratch. CPU [`eunomia::Complex64`] values are
/// deliberately excluded so typed GPU APIs cannot silently narrow them.
///
/// ```compile_fail
/// use apollo_sht::ShtGpuStorage;
/// use eunomia::Complex64;
///
/// fn require_gpu_storage<T: ShtGpuStorage>() {}
/// require_gpu_storage::<Complex64>();
/// ```
pub trait ShtGpuStorage: Copy + Send + Sync + 'static + sealed::Sealed {
    /// Precision profile required by this representation.
    const PROFILE: PrecisionProfile;

    /// Convert host storage into concrete accelerator storage.
    fn to_gpu(self) -> Complex32;

    /// Convert concrete accelerator storage back into this representation.
    fn from_gpu(value: Complex32) -> Self;

    /// Borrow storage as concrete accelerator values when layouts match.
    fn as_gpu_slice(slice: &[Self]) -> Option<&[Complex32]>;
}

impl ShtGpuStorage for Complex32 {
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
}

impl ShtGpuStorage for [f16; 2] {
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
}
