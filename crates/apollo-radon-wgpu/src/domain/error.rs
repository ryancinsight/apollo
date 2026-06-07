//! WGPU error contracts.

use apollo_wgpu_helpers::WgpuDeviceError;
use thiserror::Error;

/// Result alias for WGPU operations.
pub type WgpuResult<T> = Result<T, WgpuError>;

/// Errors produced by WGPU backend operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum WgpuError {
    /// WGPU device acquisition failed.
    #[error("wgpu device: {0}")]
    Device(#[from] WgpuDeviceError),
    /// Plan geometry is invalid.
    #[error("invalid Radon WGPU plan: rows={rows}, cols={cols}, angles={angle_count}, detectors={detector_count}, spacing={detector_spacing}, reason={message}")]
    InvalidPlan {
        /// Image row count.
        rows: usize,
        /// Image column count.
        cols: usize,
        /// Angle count.
        angle_count: usize,
        /// Detector count.
        detector_count: usize,
        /// Detector spacing.
        detector_spacing: f64,
        /// Failure explanation.
        message: &'static str,
    },
    /// Image shape does not match the plan.
    #[error("image shape mismatch: expected {expected_rows}x{expected_cols}, got {actual_rows}x{actual_cols}")]
    ImageShapeMismatch {
        /// Expected rows.
        expected_rows: usize,
        /// Expected cols.
        expected_cols: usize,
        /// Actual rows.
        actual_rows: usize,
        /// Actual cols.
        actual_cols: usize,
    },
    /// Angle vector does not match the plan.
    #[error("angle count mismatch: expected {expected}, got {actual}")]
    AngleCountMismatch {
        /// Expected count.
        expected: usize,
        /// Actual count.
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
    /// Sinogram shape does not match the plan.
    #[error("sinogram shape mismatch: expected {expected_angles}\u{d7}{expected_detectors}, got {actual_angles}\u{d7}{actual_detectors}")]
    SinogramShapeMismatch {
        /// Expected angle count (rows).
        expected_angles: usize,
        /// Expected detector count (cols).
        expected_detectors: usize,
        /// Actual angle count.
        actual_angles: usize,
        /// Actual detector count.
        actual_detectors: usize,
    },
    /// Requested precision profile does not match the typed storage.
    #[error("precision profile does not match typed Radon WGPU storage")]
    InvalidPrecisionProfile,
}
