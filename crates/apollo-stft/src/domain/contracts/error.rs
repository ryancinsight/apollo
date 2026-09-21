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
    use std::collections::BTreeSet;

    /// The newline this crate must never emit inside a message.
    const LINE_FEED: char = '\n';

    /// One index per variant, so the list below can be checked against the
    /// match rather than merely beside it.
    const STFT_VARIANTS: usize = 9;
    const PEAK_VARIANTS: usize = 5;

    /// Every `StftError`, kept complete in both directions.
    ///
    /// The wildcard-free match forces an arm when a variant is added. That
    /// alone is not enough: adding the arm is the minimal edit that clears
    /// the build, and it leaves the list short. So each arm carries a
    /// distinct index and the set of indices must be the full range — an
    /// omitted variant shrinks it, a repeated one shrinks it too.
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
        let seen: BTreeSet<usize> = all
            .iter()
            .map(|error| match error {
                StftError::EmptyFrameLength => 0,
                StftError::EmptyHopSize => 1,
                StftError::HopExceedsFrame => 2,
                StftError::InputTooShort => 3,
                StftError::LengthMismatch => 4,
                StftError::WindowLengthMismatch => 5,
                StftError::PrecisionMismatch => 6,
                StftError::InvalidWindowParameter => 7,
                StftError::WindowNotOverlapAdd => 8,
            })
            .collect();
        assert_eq!(
            seen,
            (0..STFT_VARIANTS).collect::<BTreeSet<usize>>(),
            "the list omits or repeats a variant the match names"
        );
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
        let seen: BTreeSet<usize> = all
            .iter()
            .map(|error| match error {
                PeakEstimationError::FrameTooShort { .. } => 0,
                PeakEstimationError::FrameTooLong { .. } => 1,
                PeakEstimationError::PeakOutOfRange { .. } => 2,
                PeakEstimationError::DuplicatePeak { .. } => 3,
                PeakEstimationError::MirrorPeak { .. } => 4,
            })
            .collect();
        assert_eq!(
            seen,
            (0..PEAK_VARIANTS).collect::<BTreeSet<usize>>(),
            "the list omits or repeats a variant the match names"
        );
        all
    }

    /// A `#[error(...)]` literal written across source lines carries its own
    /// indentation into the message: `rustfmt` joins the lines and the
    /// continuation's spaces stay in the string. Nothing else catches it —
    /// fmt produces it, clippy does not read string contents, and the text
    /// reaches a caller through `Display` and through
    /// `WgpuError::InvalidPlan`. This crate's messages are one sentence of
    /// single-spaced prose, so a repeated space is that defect.
    ///
    /// The enforcement is live only because this module sits in the defining
    /// crate, where `#[non_exhaustive]` does not apply; moved to `tests/` the
    /// matches above would need a `_` arm and would stop enforcing.
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
                !message.contains(LINE_FEED),
                "message carries a newline: {message:?}"
            );
            assert!(!message.is_empty(), "a variant renders no message");
        }
    }
}
