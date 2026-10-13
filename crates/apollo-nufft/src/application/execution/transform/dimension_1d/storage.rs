//! Typed complex storage views for 1D NUFFT paths.

use apollo_fft::{CpuStorage, PrecisionProfile, F16};
use eunomia::{Complex32, Complex64};

/// Complex storage admitted by typed NUFFT paths.
///
/// The conversion ladder comes from the shared [`CpuStorage`] vocabulary, and
/// this trait adds only the reinterpret views the device conversion helpers
/// use to skip a copy when the host layout already matches an accelerator
/// element. One storage type carries two truthful profiles, one per tier:
/// [`CpuStorage::PROFILE`] describes the widening arithmetic the CPU plans
/// execute, and [`DEVICE_PROFILE`] describes the native arithmetic the
/// accelerator typed paths execute over `Complex32` device buffers.
pub trait NufftComplexStorage: CpuStorage<Complex64> {
    /// Precision profile the accelerator typed paths advertise.
    ///
    /// The device element is `Complex32`, so the compute precision is the
    /// native one for each storage type: `f32` for `Complex32` and
    /// `[F16; 2]` (which widen from `F16`), and `f32` for `Complex64`
    /// (which narrows to `Complex32`).
    const DEVICE_PROFILE: PrecisionProfile;

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
    const DEVICE_PROFILE: PrecisionProfile = PrecisionProfile::F64_STORAGE_F32_COMPUTE;

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
    const DEVICE_PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    #[inline]
    fn as_c32_slice(slice: &[Self]) -> Option<&[Complex32]> {
        Some(slice)
    }

    #[inline]
    fn as_c32_slice_mut(slice: &mut [Self]) -> Option<&mut [Complex32]> {
        Some(slice)
    }
}

impl NufftComplexStorage for [F16; 2] {
    const DEVICE_PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_fft::{ComputePrecision, PrecisionProfile, StoragePrecision};

    /// The two tiers must disagree where the arithmetic disagrees.
    ///
    /// `CpuStorage::PROFILE` describes the widening arithmetic the CPU
    /// plans execute; `DEVICE_PROFILE` describes the native arithmetic the
    /// accelerator typed paths execute over `Complex32` device buffers.
    /// Conflating them silently rejects a caller whose profile matches the
    /// device, which the GPU-gated typed verification cannot observe on a
    /// host without an adapter.
    #[test]
    fn device_profiles_record_native_arithmetic_not_cpu_widening() {
        assert_eq!(
            <[F16; 2] as NufftComplexStorage>::DEVICE_PROFILE,
            PrecisionProfile::MIXED_PRECISION_F16_F32
        );
        assert_eq!(
            <Complex32 as NufftComplexStorage>::DEVICE_PROFILE,
            PrecisionProfile::LOW_PRECISION_F32
        );
        assert_eq!(
            <Complex64 as NufftComplexStorage>::DEVICE_PROFILE,
            PrecisionProfile::F64_STORAGE_F32_COMPUTE
        );

        // The CPU tier widens; the device tier does not.
        assert_eq!(
            <[F16; 2] as CpuStorage<Complex64>>::PROFILE,
            PrecisionProfile::F16_STORAGE_F64_COMPUTE
        );
        assert_eq!(
            <Complex32 as CpuStorage<Complex64>>::PROFILE,
            PrecisionProfile::F32_STORAGE_F64_COMPUTE
        );
    }

    /// The narrow device profile carries the narrowing that the typed
    /// path performs.
    #[test]
    fn narrow_device_profile_declares_f64_storage_over_f32_compute() {
        let profile = PrecisionProfile::F64_STORAGE_F32_COMPUTE;
        assert_eq!(profile.storage, StoragePrecision::F64);
        assert_eq!(profile.compute, ComputePrecision::F32);
    }
}
