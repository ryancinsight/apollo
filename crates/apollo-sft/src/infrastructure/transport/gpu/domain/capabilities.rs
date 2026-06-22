//! WGPU capability contracts.

use apollo_fft::PrecisionProfile;

/// Truthful WGPU transform capability descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuCapabilities {
    /// Whether a WGPU adapter and device can be acquired.
    pub device_available: bool,
    /// Whether forward transform execution is implemented.
    pub supports_forward: bool,
    /// Whether inverse or adjoint transform execution is implemented.
    pub supports_inverse: bool,
    /// Whether mixed-precision (f16/f32/f64) typed storage dispatch is supported.
    pub supports_mixed_precision: bool,
    /// Default precision profile for GPU execution.
    pub default_precision_profile: PrecisionProfile,
}

impl WgpuCapabilities {
    /// Construct capabilities for the current crate state.
    #[must_use]
    pub const fn detected(device_available: bool) -> Self {
        Self {
            device_available,
            supports_forward: false,
            supports_inverse: false,
            supports_mixed_precision: false,
            default_precision_profile: PrecisionProfile::LOW_PRECISION_F32,
        }
    }

    /// Construct capabilities for a direct dense-spectrum SFT implementation.
    #[must_use]
    pub const fn direct_dense_spectrum(device_available: bool) -> Self {
        Self {
            device_available,
            supports_forward: device_available,
            supports_inverse: device_available,
            supports_mixed_precision: true,
            default_precision_profile: PrecisionProfile::LOW_PRECISION_F32,
        }
    }
}
