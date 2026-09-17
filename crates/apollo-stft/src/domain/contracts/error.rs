//! STFT contracts and capabilities.

use thiserror::Error;

/// Type alias for `Result<T, StftError>`.
pub type StftResult<T> = Result<T, StftError>;

/// Errors produced by STFT creation or execution.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum StftError {
    /// Frame length is zero.
    #[error("frame length must be > 0")]
    EmptyFrameLength,
    /// Hop size is zero.
    #[error("hop size must be > 0")]
    EmptyHopSize,
    /// Hop size exceeds frame length.
    #[error("hop size must be <= frame length")]
    HopExceedsFrame,
    /// Input is shorter than the frame length.
    #[error("input length must be >= frame length")]
    InputTooShort,
    /// Input length does not match the plan.
    #[error("input length mismatch")]
    LengthMismatch,
    /// The window length does not match the frame length.
    #[error("window length mismatch")]
    WindowLengthMismatch,
    /// Precision profile does not match the requested storage type.
    #[error("precision profile does not match storage type")]
    PrecisionMismatch,
}

/// Errors produced by [`estimate_peaks`](crate::estimate_peaks) for inputs it
/// cannot read.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PeakEstimationError {
    /// The spectrum has fewer than the three bins an estimate reads.
    #[error("a spectrum of {len} bins has fewer than the three an estimate reads")]
    FrameTooShort {
        /// The spectrum's length.
        len: usize,
    },
    /// The frame length is not exactly representable in the scalar, so bin
    /// positions would round.
    #[error("a frame of {len} bins is not exactly representable in the scalar")]
    FrameTooLong {
        /// The spectrum's length.
        len: usize,
    },
    /// A peak bin lies outside the spectrum.
    #[error("peak bin {bin} is outside a spectrum of {len} bins")]
    PeakOutOfRange {
        /// The offending bin.
        bin: usize,
        /// The spectrum's length.
        len: usize,
    },
    /// A peak bin is listed more than once.
    #[error("peak bin {bin} is listed more than once")]
    DuplicatePeak {
        /// The repeated bin.
        bin: usize,
    },
}
