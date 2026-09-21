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
    #[error("the window's overlap-add weight falls to at most eps of the largest at this hop and length; the inverse is too ill-conditioned there to be held to six significant digits")]
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

#[cfg(test)]
mod tests {
    use super::{PeakEstimationError, StftError};

    /// Every `StftError`, with the wildcard-free match that keeps the list
    /// complete: a variant added to the enum stops compiling here, rather
    /// than silently escaping the message check below.
    fn every_stft_error() -> Vec<StftError> {
        let all = vec![
            StftError::EmptyFrameLength,
            StftError::EmptyHopSize,
            StftError::HopExceedsFrame,
            StftError::InputTooShort,
            StftError::LengthMismatch,
            StftError::WindowLengthMismatch,
            StftError::PrecisionMismatch,
            StftError::InvalidWindowParameter,
            StftError::WindowNotOverlapAdd,
        ];
        for error in &all {
            match error {
                StftError::EmptyFrameLength
                | StftError::EmptyHopSize
                | StftError::HopExceedsFrame
                | StftError::InputTooShort
                | StftError::LengthMismatch
                | StftError::WindowLengthMismatch
                | StftError::PrecisionMismatch
                | StftError::InvalidWindowParameter
                | StftError::WindowNotOverlapAdd => {}
            }
        }
        all
    }

    /// Every `PeakEstimationError`, under the same rule.
    fn every_peak_error() -> Vec<PeakEstimationError> {
        let all = vec![
            PeakEstimationError::FrameTooShort { len: 2 },
            PeakEstimationError::FrameTooLong { len: 1 << 30 },
            PeakEstimationError::PeakOutOfRange { bin: 9, len: 4 },
            PeakEstimationError::DuplicatePeak { bin: 3 },
            PeakEstimationError::MirrorPeak { bin: 5, mirror: 3 },
        ];
        for error in &all {
            match error {
                PeakEstimationError::FrameTooShort { .. }
                | PeakEstimationError::FrameTooLong { .. }
                | PeakEstimationError::PeakOutOfRange { .. }
                | PeakEstimationError::DuplicatePeak { .. }
                | PeakEstimationError::MirrorPeak { .. } => {}
            }
        }
        all
    }

    /// A `#[error(...)]` literal written across source lines carries its own
    /// indentation into the message: `rustfmt` joins the lines and the
    /// continuation's spaces stay in the string. Nothing else catches it --
    /// fmt produces it, clippy does not read string contents, and the text
    /// reaches a caller through `Display` and through
    /// `WgpuError::InvalidPlan`. This crate's messages are one sentence of
    /// single-spaced prose, so a repeated space is that defect.
    #[test]
    fn no_error_message_carries_source_indentation() {
        let stft = every_stft_error();
        let peak = every_peak_error();
        let messages = stft
            .iter()
            .map(ToString::to_string)
            .chain(peak.iter().map(ToString::to_string));
        for message in messages {
            assert!(
                !message.contains("  "),
                "message carries a repeated space: {message:?}"
            );
            assert!(
                !message.contains('\n'),
                "message carries a newline: {message:?}"
            );
            assert!(!message.is_empty(), "a variant renders no message");
        }
    }
}
