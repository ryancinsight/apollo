use core::fmt::{self, Display, Formatter};
use std::io;
use std::num::ParseIntError;
use std::path::Path;
use thiserror::Error;

/// Reports an invalid or incomplete benchmark comparison.
///
/// Source traversal uses the type-erased `std::error::Error` contract only at
/// this cold diagnostic boundary.
#[derive(Debug, Error)]
#[error(transparent)]
pub struct ComparisonError {
    kind: ErrorKind,
}

impl ComparisonError {
    pub(super) fn read_directory(path: &Path, source: io::Error) -> Self {
        ErrorKind::ReadDirectory {
            path: path.display().to_string(),
            source,
        }
        .into()
    }

    pub(super) fn read_entry(path: &Path, source: io::Error) -> Self {
        ErrorKind::ReadEntry {
            path: path.display().to_string(),
            source,
        }
        .into()
    }

    pub(super) fn read_report(path: &Path, source: csv::Error) -> Self {
        ErrorKind::ReadReport {
            path: path.display().to_string(),
            source,
        }
        .into()
    }

    pub(super) fn unexpected_header(path: &Path, actual: String) -> Self {
        ErrorKind::UnexpectedHeader {
            path: path.display().to_string(),
            actual,
        }
        .into()
    }

    pub(super) fn missing_field(path: &Path, row: u64, column: &'static str) -> Self {
        ErrorKind::MissingField {
            path: path.display().to_string(),
            row,
            column,
        }
        .into()
    }

    pub(super) fn invalid_integer(
        path: &Path,
        row: u64,
        column: &'static str,
        value: &str,
        source: ParseIntError,
    ) -> Self {
        ErrorKind::InvalidInteger {
            path: path.display().to_string(),
            row,
            column,
            value: value.to_owned(),
            source,
        }
        .into()
    }

    pub(super) fn duplicate_case(path: &Path, row: u64, case: &str) -> Self {
        ErrorKind::DuplicateCase {
            path: path.display().to_string(),
            row,
            case: case.to_owned(),
        }
        .into()
    }

    pub(super) fn empty_report(path: &Path) -> Self {
        ErrorKind::EmptyReport {
            path: path.display().to_string(),
        }
        .into()
    }

    pub(super) fn invalid_record(
        path: &Path,
        row: u64,
        case: &str,
        invariant: RecordInvariant,
    ) -> Self {
        ErrorKind::InvalidRecord {
            path: path.display().to_string(),
            row,
            case: case.to_owned(),
            invariant,
        }
        .into()
    }

    pub(super) fn no_reports(path: &Path) -> Self {
        ErrorKind::NoReports {
            path: path.display().to_string(),
        }
        .into()
    }

    pub(super) fn missing_candidate_report(report: &Path) -> Self {
        ErrorKind::MissingCandidateReport {
            report: report.display().to_string(),
        }
        .into()
    }

    pub(super) fn missing_baseline_report(report: &Path) -> Self {
        ErrorKind::MissingBaselineReport {
            report: report.display().to_string(),
        }
        .into()
    }

    pub(super) fn missing_candidate_case(report: &Path, case: &str) -> Self {
        ErrorKind::MissingCandidateCase {
            report: report.display().to_string(),
            case: case.to_owned(),
        }
        .into()
    }

    pub(super) fn missing_baseline_case(report: &Path, case: &str) -> Self {
        ErrorKind::MissingBaselineCase {
            report: report.display().to_string(),
            case: case.to_owned(),
        }
        .into()
    }

    pub(super) fn insufficient_familywise_evidence(
        report: &Path,
        case: &str,
        sample_count: usize,
        family_size: usize,
    ) -> Self {
        ErrorKind::InsufficientFamilywiseEvidence {
            report: report.display().to_string(),
            case: case.to_owned(),
            sample_count,
            family_size,
        }
        .into()
    }

    pub(super) fn missing_candidate_first_case(report: &Path, case: &str) -> Self {
        ErrorKind::MissingCandidateFirstCase {
            report: report.display().to_string(),
            case: case.to_owned(),
        }
        .into()
    }

    pub(super) fn missing_baseline_first_case(report: &Path, case: &str) -> Self {
        ErrorKind::MissingBaselineFirstCase {
            report: report.display().to_string(),
            case: case.to_owned(),
        }
        .into()
    }
}

impl From<ErrorKind> for ComparisonError {
    fn from(kind: ErrorKind) -> Self {
        Self { kind }
    }
}

#[derive(Debug, Error)]
enum ErrorKind {
    #[error("cannot read benchmark directory {path}")]
    ReadDirectory {
        path: String,
        #[source]
        source: io::Error,
    },
    #[error("cannot inspect benchmark entry {path}")]
    ReadEntry {
        path: String,
        #[source]
        source: io::Error,
    },
    #[error("cannot parse benchmark report {path}")]
    ReadReport {
        path: String,
        #[source]
        source: csv::Error,
    },
    #[error("benchmark report {path} has unsupported header `{actual}`")]
    UnexpectedHeader { path: String, actual: String },
    #[error("benchmark report {path} row {row} omits `{column}`")]
    MissingField {
        path: String,
        row: u64,
        column: &'static str,
    },
    #[error("benchmark report {path} row {row} has invalid `{column}` value `{value}`")]
    InvalidInteger {
        path: String,
        row: u64,
        column: &'static str,
        value: String,
        #[source]
        source: ParseIntError,
    },
    #[error("benchmark report {path} row {row} duplicates case `{case}`")]
    DuplicateCase {
        path: String,
        row: u64,
        case: String,
    },
    #[error("benchmark report {path} contains no benchmark cases")]
    EmptyReport { path: String },
    #[error("benchmark report {path} row {row} case `{case}` violates {invariant}")]
    InvalidRecord {
        path: String,
        row: u64,
        case: String,
        invariant: RecordInvariant,
    },
    #[error("benchmark directory {path} contains no CSV reports")]
    NoReports { path: String },
    #[error("candidate benchmark directory omits report {report}")]
    MissingCandidateReport { report: String },
    #[error("baseline benchmark directory omits report {report}")]
    MissingBaselineReport { report: String },
    #[error("candidate report {report} omits baseline case `{case}`")]
    MissingCandidateCase { report: String, case: String },
    #[error("baseline report {report} omits candidate case `{case}`")]
    MissingBaselineCase { report: String, case: String },
    #[error(
        "report {report} case `{case}` has {sample_count} samples, insufficient for 5% family-wise error across {family_size} cases"
    )]
    InsufficientFamilywiseEvidence {
        report: String,
        case: String,
        sample_count: usize,
        family_size: usize,
    },
    #[error("candidate-first evidence omits baseline-first report {report} case `{case}`")]
    MissingCandidateFirstCase { report: String, case: String },
    #[error("baseline-first evidence omits candidate-first report {report} case `{case}`")]
    MissingBaselineFirstCase { report: String, case: String },
}

#[derive(Clone, Copy, Debug)]
pub(super) enum RecordInvariant {
    EmptyCase,
    ZeroSamples,
    SamplesNotOrdered,
    MinimumMismatch,
    MedianMismatch,
    UnsupportedSampleCount,
    MedianIntervalMismatch,
    ZeroIterations,
}

impl Display for RecordInvariant {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::EmptyCase => "a non-empty case label",
            Self::ZeroSamples => "non-empty ordered_samples_ns",
            Self::SamplesNotOrdered => "nondecreasing ordered_samples_ns",
            Self::MinimumMismatch => "min_ns == first ordered sample",
            Self::MedianMismatch => "median_ns == the ordered-sample median",
            Self::UnsupportedSampleCount => "1..=100 ordered samples",
            Self::MedianIntervalMismatch => {
                "median interval columns derived from ordered_samples_ns"
            }
            Self::ZeroIterations => "iterations_per_sample > 0",
        })
    }
}
