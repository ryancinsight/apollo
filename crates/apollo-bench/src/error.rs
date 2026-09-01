//! The failure a bench binary can report from `main`.
//!
//! Each bench previously carried its own two-variant aggregator of the mode
//! parser's error and the processor-binding error; the second occurrence
//! moved it here.

use std::fmt;

use crate::config::BenchmarkModeError;
use crate::measurement::ProcessorSelectionError;

/// Why a bench binary could not run: either its mode configuration or its
/// measurement-processor binding was rejected.
#[derive(Debug)]
pub enum BenchmarkError {
    /// The benchmark mode taken from the environment was invalid.
    Mode(BenchmarkModeError),
    /// No measurement processor could be bound.
    Processor(ProcessorSelectionError),
}

impl fmt::Display for BenchmarkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mode(error) => error.fmt(formatter),
            Self::Processor(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for BenchmarkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Mode(error) => Some(error),
            Self::Processor(error) => Some(error),
        }
    }
}

impl From<BenchmarkModeError> for BenchmarkError {
    fn from(error: BenchmarkModeError) -> Self {
        Self::Mode(error)
    }
}

impl From<ProcessorSelectionError> for BenchmarkError {
    fn from(error: ProcessorSelectionError) -> Self {
        Self::Processor(error)
    }
}
