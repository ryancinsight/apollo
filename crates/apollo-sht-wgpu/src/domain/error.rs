//! WGPU error contracts.

use apollo_wgpu_helpers::WgpuDeviceError;
use thiserror::Error;

/// Result alias for WGPU operations.
pub type WgpuResult<T> = Result<T, WgpuError>;

/// Errors produced by WGPU backend operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum WgpuError {
    /// WGPU device acquisition failed.
    #[error("wgpu device: {0}")]
    Device(#[from] WgpuDeviceError),
    /// Plan parameters are invalid.
    #[error("invalid SHT WGPU plan: latitudes={latitudes}, longitudes={longitudes}, max_degree={max_degree}, reason={message}")]
    InvalidPlan {
        /// Latitude sample count.
        latitudes: usize,
        /// Longitude sample count.
        longitudes: usize,
        /// Maximum spherical harmonic degree.
        max_degree: usize,
        /// Failure explanation.
        message: &'static str,
    },
    /// Sample matrix shape does not match the plan.
    #[error("sample shape mismatch: expected ({expected_latitudes}, {expected_longitudes}), got ({actual_latitudes}, {actual_longitudes})")]
    SampleShapeMismatch {
        /// Expected latitude count.
        expected_latitudes: usize,
        /// Expected longitude count.
        expected_longitudes: usize,
        /// Actual latitude count.
        actual_latitudes: usize,
        /// Actual longitude count.
        actual_longitudes: usize,
    },
    /// Coefficient storage does not match the plan.
    #[error("coefficient shape mismatch: expected max_degree {expected}, got {actual}")]
    CoefficientShapeMismatch {
        /// Expected maximum degree.
        expected: usize,
        /// Actual maximum degree.
        actual: usize,
    },
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
    /// Requested precision profile does not match the typed storage.
    #[error("precision profile does not match typed SHT WGPU storage")]
    InvalidPrecisionProfile,
}
