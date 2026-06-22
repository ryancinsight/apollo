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
    /// Whether GPU filtered backprojection (ramp filter + backprojection) is implemented.
    pub supports_filtered_backprojection: bool,
    /// Whether mixed-precision (f16/f32/f64) typed storage dispatch is supported.
    pub supports_mixed_precision: bool,
    /// Default precision profile for GPU execution.
    pub default_precision_profile: PrecisionProfile,
}

impl WgpuCapabilities {
    /// Construct capabilities for a boundary-only backend.
    #[must_use]
    pub const fn detected(device_available: bool) -> Self {
        Self {
            device_available,
            supports_forward: false,
            supports_inverse: false,
            supports_filtered_backprojection: false,
            supports_mixed_precision: false,
            default_precision_profile: PrecisionProfile::LOW_PRECISION_F32,
        }
    }

    /// Construct capabilities for a forward-only implementation.
    #[must_use]
    pub const fn forward_only(device_available: bool) -> Self {
        Self {
            device_available,
            supports_forward: device_available,
            supports_inverse: false,
            supports_filtered_backprojection: false,
            supports_mixed_precision: true,
            default_precision_profile: PrecisionProfile::LOW_PRECISION_F32,
        }
    }

    /// Construct capabilities for a forward-and-inverse (adjoint backprojection) implementation.
    #[must_use]
    pub const fn forward_and_inverse(device_available: bool) -> Self {
        Self {
            device_available,
            supports_forward: device_available,
            supports_inverse: device_available,
            supports_filtered_backprojection: false,
            supports_mixed_precision: true,
            default_precision_profile: PrecisionProfile::LOW_PRECISION_F32,
        }
    }

    /// Construct capabilities for an implementation with forward projection, adjoint
    /// backprojection, and ramp-filtered backprojection (FBP).
    ///
    /// ## Mathematical contract
    ///
    /// FBP applies the Ram-Lak ramp filter (Bracewell & Riddle 1967; Shepp & Logan 1974)
    /// to each projection row before adjoint backprojection, followed by π/angle_count
    /// normalization. This approximates the continuous FBP integral in the discrete setting.
    #[must_use]
    pub const fn forward_inverse_and_fbp(device_available: bool) -> Self {
        Self {
            device_available,
            supports_forward: device_available,
            supports_inverse: device_available,
            supports_filtered_backprojection: device_available,
            supports_mixed_precision: true,
            default_precision_profile: PrecisionProfile::LOW_PRECISION_F32,
        }
    }
}
