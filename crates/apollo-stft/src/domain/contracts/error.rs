//! STFT contracts and capabilities.

use thiserror::Error;

/// Type alias for `Result<T, StftError>`.
pub type StftResult<T> = Result<T, StftError>;

/// Errors produced by STFT creation or execution.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
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
    /// A window parameter is out of range, or a supplied window value is not
    /// finite.
    #[error("window parameter out of range or window value not finite")]
    InvalidWindowParameter,
    /// Some sample receives window energy at most `ε` of the largest at the
    /// plan's hop. The inverse divides by that energy, and the relative error
    /// it leaves is bounded by `γ √(largest / weight)`, so below the floor the
    /// reconstruction is no longer held to six significant digits.
    #[error("the window does not overlap-add at this hop and length; the inverse is undefined")]
    WindowNotOverlapAdd,
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
    /// The frame is so long that a bin position near its end resolves no
    /// sub-bin offset in the scalar (`N ε ≥ ½`).
    #[error("a frame of {len} bins resolves no sub-bin offset in the scalar")]
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
    /// A peak bin and its mirror `N − bin` are both listed: for a real tone
    /// they hold the same tone.
    #[error("peak bins {mirror} and {bin} hold the same real tone")]
    MirrorPeak {
        /// The later of the two bins.
        bin: usize,
        /// Its mirror, listed earlier.
        mirror: usize,
    },
}
