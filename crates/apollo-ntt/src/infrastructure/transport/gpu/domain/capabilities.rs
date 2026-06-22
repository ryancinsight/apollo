//! WGPU capability contracts for the NTT backend.
//!
//! The Number Theoretic Transform operates over exact residues in a prime field
//! F_m. There is no floating-point precision concept; all results are exact
//! modular integers bounded by the modulus `m`. The `PrecisionProfile` type
//! used by floating-point backends has no meaningful interpretation here and
//! is therefore absent from this capability descriptor.

/// Truthful WGPU transform capability descriptor for the NTT backend.
///
/// All fields reflect actual implementation state. No field is set to `true`
/// for a capability that is not implemented and verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuCapabilities {
    /// Whether a WGPU adapter and device can be acquired at runtime.
    pub device_available: bool,
    /// Whether forward NTT execution is implemented and verified.
    pub supports_forward: bool,
    /// Whether inverse NTT execution is implemented and verified.
    pub supports_inverse: bool,
    /// Whether mixed floating-point precision dispatch is supported.
    ///
    /// Always `false` for this backend. NTT is an exact integer transform;
    /// floating-point mixed-precision storage is architecturally unsupported
    /// by design (see gap_audit.md: "NTT-WGPU floating mixed precision is an
    /// architectural design contract, not a gap").
    pub supports_mixed_precision: bool,
    /// Whether exact narrowed `u32` residue storage dispatch is supported.
    ///
    /// `true` when a device is available. The `u32` residue surface is the
    /// canonical storage format for this WGPU backend because the default
    /// modulus 998244353 fits in `u32`, and WGSL 1.0 exposes only 32-bit
    /// integer types.
    pub supports_quantized_storage: bool,
}

impl WgpuCapabilities {
    /// Construct capabilities for a backend with no device present.
    ///
    /// All execution capability flags are `false`; only structural fields
    /// reflect the absence of a device.
    #[must_use]
    pub const fn detected(device_available: bool) -> Self {
        Self {
            device_available,
            supports_forward: false,
            supports_inverse: false,
            supports_mixed_precision: false,
            supports_quantized_storage: false,
        }
    }

    /// Construct capabilities for a fully operational forward and inverse NTT
    /// backend.
    ///
    /// When `device_available` is `false` all execution flags are `false`;
    /// when `true` they reflect the complete implemented capability set.
    #[must_use]
    pub const fn full(device_available: bool) -> Self {
        Self {
            device_available,
            supports_forward: device_available,
            supports_inverse: device_available,
            supports_mixed_precision: false,
            supports_quantized_storage: device_available,
        }
    }
}
