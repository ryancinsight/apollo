//! Shared WGPU error contracts for the Apollo WGPU crate family.
//!
//! Crate authors should use [`WgpuError`] and [`WgpuResult`] directly.
//! Device-acquisition failures are wrapped automatically via the
//! [`From<WgpuDeviceError>`] impl on [`WgpuError`].
//!
//! ```ignore
//! use apollo_wgpu_helpers::{WgpuError, WgpuResult};
//!
//! fn do_work() -> WgpuResult<()> {
//!     // Device errors propagate via `?`
//!     let device = WgpuDevice::try_default("apollo-mine-wgpu")?;
//!     Ok(())
//! }
//! ```

use thiserror::Error;

// ── Device-acquisition errors ────────────────────────────────────────────────

/// Result alias for device acquisition.
pub type WgpuDeviceResult<T> = Result<T, WgpuDeviceError>;

/// Errors that can occur during WGPU device acquisition.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum WgpuDeviceError {
    /// No suitable WGPU adapter could be found.
    #[error("wgpu adapter unavailable: {message}")]
    AdapterUnavailable {
        /// Adapter failure context.
        message: String,
    },

    /// A WGPU device could not be created from the adapter.
    #[error("wgpu device unavailable: {message}")]
    DeviceUnavailable {
        /// Device failure context.
        message: String,
    },
}

// ── Consolidated WGPU error ─────────────────────────────────────────────────

/// Result alias for WGPU operations.
pub type WgpuResult<T> = Result<T, WgpuError>;

/// Errors produced by WGPU backend operations across all Apollo transform
/// crates.
///
/// This enum consolidates the per-transform error variants into a single
/// shared type.  Transform-specific context (plan parameters, shape details)
/// is encoded in [`String`] message fields so that the enum remains
/// transform-agnostic while preserving full diagnostic information.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum WgpuError {
    // ── Device & infrastructure ──────────────────────────────────────
    /// WGPU device acquisition failed.
    #[error("wgpu device: {0}")]
    Device(#[from] WgpuDeviceError),

    /// Requested precision profile does not match the typed storage.
    #[error("precision profile does not match typed WGPU storage")]
    InvalidPrecisionProfile,

    /// Host readback from the staging buffer failed.
    #[error("wgpu buffer map failed: {message}")]
    BufferMapFailed {
        /// Mapping failure context.
        message: String,
    },

    /// Numerical execution is unsupported for the requested operation.
    #[error("{operation} is unsupported by the current WGPU capability set")]
    UnsupportedExecution {
        /// Requested operation name.
        operation: &'static str,
    },

    // ── Plan & length validation ─────────────────────────────────────
    /// Plan parameters, buffer lengths, or configuration are invalid.
    ///
    /// Transform-specific context (lengths, moduli, window sizes, etc.)
    /// is encoded in the message string.
    #[error("invalid plan: {message}")]
    InvalidPlan {
        /// Failure explanation including transform-specific parameter details.
        message: String,
    },

    /// Input or output length does not match the plan expectation.
    #[error("length mismatch: expected {expected}, got {actual}")]
    LengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length supplied.
        actual: usize,
    },

    // ── Shape & domain validation ────────────────────────────────────
    /// Multi-dimensional shape (image, sinogram, sample matrix, etc.)
    /// does not match the plan expectation.
    #[error("shape mismatch: {message}")]
    ShapeMismatch {
        /// Failure explanation including expected and actual dimensions.
        message: String,
    },

    /// Signal domain bounds are invalid.
    #[error("invalid signal domain: {message}")]
    InvalidSignalDomain {
        /// Failure explanation including domain bounds.
        message: String,
    },

    // ── Specialised ──────────────────────────────────────────────────
    /// Fractional transform order is NaN or infinite.
    #[error("fractional order must be finite")]
    NonFiniteOrder,

    /// Requested transform kind is not implemented on this backend.
    #[error("transform kind {kind} is unsupported by the current WGPU capability set")]
    UnsupportedKind {
        /// Unsupported transform kind.
        kind: &'static str,
    },

    /// Input signal is shorter than the minimum required length.
    #[error("input too short: required {min} samples, got {actual}")]
    InputTooShort {
        /// Minimum required signal length.
        min: usize,
        /// Actual signal length supplied.
        actual: usize,
    },
}
