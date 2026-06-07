//! Shared WGPU device-acquisition error type.
//!
//! Crate authors should wrap this in their domain-specific error enum:
//!
//! ```ignore
//! #[derive(Debug, Error, Clone, PartialEq, Eq)]
//! pub enum WgpuError {
//!     #[error("wGPU device: {0}")]
//!     Device(#[from] apollo_wgpu_helpers::WgpuDeviceError),
//!     // ... crate-specific variants
//! }
//! ```

use thiserror::Error;

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
