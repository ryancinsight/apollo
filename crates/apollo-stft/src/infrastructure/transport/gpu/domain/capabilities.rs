//! WGPU capability contracts for the STFT backend.

use apollo_fft::PrecisionProfile;

/// Truthful WGPU transform capability descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuCapabilities {
    /// Whether a WGPU adapter and device can be acquired.
    pub device_available: bool,
    /// Whether forward STFT execution is implemented.
    pub supports_forward: bool,
    /// Whether inverse STFT execution is implemented.
    pub supports_inverse: bool,
    /// Whether mixed-precision (f16/f32/f64) typed storage dispatch is supported.
    pub supports_mixed_precision: bool,
    /// Default precision profile for GPU execution.
    pub default_precision_profile: PrecisionProfile,
}

impl WgpuCapabilities {
    /// Construct capabilities reflecting zero-kernel state (both false).
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

    /// Construct capabilities for a forward-only implementation.
    ///
    /// `supports_forward` is true iff `device_available` is true.
    /// `supports_inverse` is always false (inverse STFT not implemented on GPU).
    #[must_use]
    pub const fn forward_only(device_available: bool) -> Self {
        Self {
            device_available,
            supports_forward: device_available,
            supports_inverse: false,
            supports_mixed_precision: true,
            default_precision_profile: PrecisionProfile::LOW_PRECISION_F32,
        }
    }

    /// Construct capabilities for a forward-and-inverse implementation.
    #[must_use]
    pub const fn forward_and_inverse(device_available: bool) -> Self {
        Self {
            device_available,
            supports_forward: device_available,
            supports_inverse: device_available,
            supports_mixed_precision: true,
            default_precision_profile: PrecisionProfile::LOW_PRECISION_F32,
        }
    }
}
