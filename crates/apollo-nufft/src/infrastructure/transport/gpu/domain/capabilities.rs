//! NUFFT WGPU capability contracts.

use apollo_fft::PrecisionProfile;

/// Truthful WGPU NUFFT capability descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NufftWgpuCapabilities {
    /// Whether a WGPU adapter and device can be acquired.
    pub device_available: bool,
    /// Whether Type-1 1D NUFFT execution is implemented.
    pub supports_type1_1d: bool,
    /// Whether Type-2 1D NUFFT execution is implemented.
    pub supports_type2_1d: bool,
    /// Whether Type-1 3D NUFFT execution is implemented.
    pub supports_type1_3d: bool,
    /// Whether Type-2 3D NUFFT execution is implemented.
    pub supports_type2_3d: bool,
    /// Whether fast gridded Type-1 1D NUFFT execution is implemented.
    pub supports_fast_type1_1d: bool,
    /// Whether fast gridded Type-2 1D NUFFT execution is implemented.
    pub supports_fast_type2_1d: bool,
    /// Whether fast gridded Type-1 3D NUFFT execution is implemented.
    pub supports_fast_type1_3d: bool,
    /// Whether fast gridded Type-2 3D NUFFT execution is implemented.
    pub supports_fast_type2_3d: bool,
    /// Whether mixed-precision (f16/f32/f64) typed storage dispatch is supported.
    pub supports_mixed_precision: bool,
    /// Default precision profile for GPU execution.
    pub default_precision_profile: PrecisionProfile,
}

impl NufftWgpuCapabilities {
    /// Construct capabilities for the current crate state.
    #[must_use]
    pub const fn detected(device_available: bool) -> Self {
        Self {
            device_available,
            supports_type1_1d: false,
            supports_type2_1d: false,
            supports_type1_3d: false,
            supports_type2_3d: false,
            supports_fast_type1_1d: false,
            supports_fast_type2_1d: false,
            supports_fast_type1_3d: false,
            supports_fast_type2_3d: false,
            supports_mixed_precision: false,
            default_precision_profile: PrecisionProfile::LOW_PRECISION_F32,
        }
    }

    /// Construct capabilities for exact direct Type-1 GPU kernels.
    #[must_use]
    pub const fn direct_type1(device_available: bool) -> Self {
        Self {
            device_available,
            supports_type1_1d: device_available,
            supports_type2_1d: false,
            supports_type1_3d: device_available,
            supports_type2_3d: false,
            supports_fast_type1_1d: false,
            supports_fast_type2_1d: false,
            supports_fast_type1_3d: false,
            supports_fast_type2_3d: false,
            supports_mixed_precision: false,
            default_precision_profile: PrecisionProfile::LOW_PRECISION_F32,
        }
    }

    /// Construct capabilities for exact direct Type-1 and 1D Type-2 GPU kernels.
    #[must_use]
    pub const fn direct_type1_and_type2_1d(device_available: bool) -> Self {
        Self {
            device_available,
            supports_type1_1d: device_available,
            supports_type2_1d: device_available,
            supports_type1_3d: device_available,
            supports_type2_3d: false,
            supports_fast_type1_1d: false,
            supports_fast_type2_1d: false,
            supports_fast_type1_3d: false,
            supports_fast_type2_3d: false,
            supports_mixed_precision: false,
            default_precision_profile: PrecisionProfile::LOW_PRECISION_F32,
        }
    }

    /// Construct capabilities for all exact direct GPU kernels.
    #[must_use]
    pub const fn direct_all(device_available: bool) -> Self {
        Self {
            device_available,
            supports_type1_1d: device_available,
            supports_type2_1d: device_available,
            supports_type1_3d: device_available,
            supports_type2_3d: device_available,
            supports_fast_type1_1d: false,
            supports_fast_type2_1d: false,
            supports_fast_type1_3d: false,
            supports_fast_type2_3d: false,
            supports_mixed_precision: false,
            default_precision_profile: PrecisionProfile::LOW_PRECISION_F32,
        }
    }

    /// Construct capabilities for all direct kernels plus fast 1D gridded kernels.
    #[must_use]
    pub const fn direct_all_fast_1d(device_available: bool) -> Self {
        Self {
            device_available,
            supports_type1_1d: device_available,
            supports_type2_1d: device_available,
            supports_type1_3d: device_available,
            supports_type2_3d: device_available,
            supports_fast_type1_1d: device_available,
            supports_fast_type2_1d: device_available,
            supports_fast_type1_3d: false,
            supports_fast_type2_3d: false,
            supports_mixed_precision: false,
            default_precision_profile: PrecisionProfile::LOW_PRECISION_F32,
        }
    }

    /// Construct capabilities for all direct and all fast gridded kernels (full implementation).
    #[must_use]
    pub const fn direct_all_fast_all(device_available: bool) -> Self {
        Self {
            device_available,
            supports_type1_1d: device_available,
            supports_type2_1d: device_available,
            supports_type1_3d: device_available,
            supports_type2_3d: device_available,
            supports_fast_type1_1d: device_available,
            supports_fast_type2_1d: device_available,
            supports_fast_type1_3d: device_available,
            supports_fast_type2_3d: device_available,
            supports_mixed_precision: true,
            default_precision_profile: PrecisionProfile::LOW_PRECISION_F32,
        }
    }
}
