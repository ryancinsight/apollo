mod counterbalanced;
mod discovery;
mod error;
mod report;

pub use counterbalanced::{
    compare_counterbalanced_report_directories, CounterbalancedBenchmarkRegression,
    CounterbalancedComparisonSummary, IntervalSeparation,
};
pub use error::ComparisonError;

use report::ReportRecord;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const REQUIRED_CONFIDENCE_PARTS_PER_MILLION: u32 = 950_000;

/// Identifies one benchmark whose candidate median interval is wholly slower
/// than its baseline interval.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BenchmarkRegression {
    report: PathBuf,
    case: String,
    baseline_upper_nanoseconds: u128,
    candidate_lower_nanoseconds: u128,
}

impl BenchmarkRegression {
    /// Returns the report path relative to the compared directories.
    #[must_use]
    pub fn report(&self) -> &Path {
        &self.report
    }

    /// Returns the benchmark case label.
    #[must_use]
    pub fn case(&self) -> &str {
        &self.case
    }

    /// Returns the baseline median interval's upper bound.
    #[must_use]
    pub const fn baseline_upper_nanoseconds(&self) -> u128 {
        self.baseline_upper_nanoseconds
    }

    /// Returns the candidate median interval's lower bound.
    #[must_use]
    pub const fn candidate_lower_nanoseconds(&self) -> u128 {
        self.candidate_lower_nanoseconds
    }
}

/// Summarizes an exact base/head benchmark report comparison.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComparisonSummary {
    compared_reports: usize,
    compared_cases: usize,
    regressions: Vec<BenchmarkRegression>,
    compared_keys: BTreeSet<BenchmarkKey>,
}

impl ComparisonSummary {
    /// Returns the number of paired CSV reports.
    #[must_use]
    pub const fn compared_reports(&self) -> usize {
        self.compared_reports
    }

    /// Returns the number of paired benchmark cases.
    #[must_use]
    pub const fn compared_cases(&self) -> usize {
        self.compared_cases
    }

    /// Returns cases whose 95% median intervals prove a slowdown.
    #[must_use]
    pub fn regressions(&self) -> &[BenchmarkRegression] {
        &self.regressions
    }

    /// Returns whether no statistically supported slowdown was found.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.regressions.is_empty()
    }
}

/// Compares recursively discovered Apollo CSV reports from independent base
/// and candidate benchmark runs.
///
/// Report paths and case labels must match exactly. Each report must carry at
/// least a 95% distribution-free median interval. A case regresses only when
/// the candidate interval's lower bound exceeds the baseline interval's upper
/// bound.
///
/// # Errors
///
/// Returns [`ComparisonError`] when either directory is unreadable, a report
/// is malformed, report or case sets differ, or an interval has insufficient
/// confidence.
///
/// # Examples
///
/// ```
/// use apollo_bench::compare_report_directories;
/// use std::fs;
///
/// let sequence = std::time::SystemTime::now()
///     .duration_since(std::time::UNIX_EPOCH)?
///     .as_nanos();
/// let root = std::env::temp_dir()
///     .join(format!("apollo-bench-doc-comparison-{sequence}"));
/// let baseline = root.join("baseline");
/// let candidate = root.join("candidate");
/// fs::create_dir_all(&baseline)?;
/// fs::create_dir_all(&candidate)?;
/// let report = concat!(
///     "case,min_ns,median_ns,median_lower_ns,median_upper_ns,",
///     "median_confidence_ppm,samples,iterations_per_sample\n",
///     "fft/forward/256,80,100,90,110,964799,100,4\n"
/// );
/// fs::write(baseline.join("fft.csv"), report)?;
/// fs::write(candidate.join("fft.csv"), report)?;
///
/// let summary = compare_report_directories(&baseline, &candidate)?;
/// assert!(summary.passed());
/// assert_eq!(summary.compared_cases(), 1);
///
/// fs::remove_dir_all(root)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compare_report_directories(
    baseline_directory: impl AsRef<Path>,
    candidate_directory: impl AsRef<Path>,
) -> Result<ComparisonSummary, ComparisonError> {
    let baseline_reports = discovery::discover_reports(baseline_directory.as_ref())?;
    let candidate_reports = discovery::discover_reports(candidate_directory.as_ref())?;

    for report in baseline_reports.keys() {
        if !candidate_reports.contains_key(report) {
            return Err(ComparisonError::missing_candidate_report(report));
        }
    }
    for report in candidate_reports.keys() {
        if !baseline_reports.contains_key(report) {
            return Err(ComparisonError::missing_baseline_report(report));
        }
    }

    let mut compared_cases = 0_usize;
    let mut regressions = Vec::new();
    let mut compared_keys = BTreeSet::new();
    for (relative_path, baseline_path) in &baseline_reports {
        let candidate_path = candidate_reports
            .get(relative_path)
            .expect("invariant: paired report sets were validated");
        let baseline = report::read_report(baseline_path)?;
        let candidate = report::read_report(candidate_path)?;
        validate_case_sets(relative_path, &baseline, &candidate)?;

        for (case, baseline_record) in baseline {
            let candidate_record = candidate
                .get(&case)
                .expect("invariant: paired case sets were validated");
            validate_confidence(relative_path, &case, &baseline_record)?;
            validate_confidence(relative_path, &case, candidate_record)?;
            compared_cases += 1;
            compared_keys.insert(BenchmarkKey {
                report: relative_path.clone(),
                case: case.clone(),
            });

            if candidate_record.lower_nanoseconds > baseline_record.upper_nanoseconds {
                regressions.push(BenchmarkRegression {
                    report: relative_path.clone(),
                    case,
                    baseline_upper_nanoseconds: baseline_record.upper_nanoseconds,
                    candidate_lower_nanoseconds: candidate_record.lower_nanoseconds,
                });
            }
        }
    }

    Ok(ComparisonSummary {
        compared_reports: baseline_reports.len(),
        compared_cases,
        regressions,
        compared_keys,
    })
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct BenchmarkKey {
    report: PathBuf,
    case: String,
}

fn validate_case_sets(
    report: &Path,
    baseline: &std::collections::BTreeMap<String, ReportRecord>,
    candidate: &std::collections::BTreeMap<String, ReportRecord>,
) -> Result<(), ComparisonError> {
    for case in baseline.keys() {
        if !candidate.contains_key(case) {
            return Err(ComparisonError::missing_candidate_case(report, case));
        }
    }
    for case in candidate.keys() {
        if !baseline.contains_key(case) {
            return Err(ComparisonError::missing_baseline_case(report, case));
        }
    }
    Ok(())
}

fn validate_confidence(
    report: &Path,
    case: &str,
    record: &ReportRecord,
) -> Result<(), ComparisonError> {
    if record.confidence_parts_per_million < REQUIRED_CONFIDENCE_PARTS_PER_MILLION {
        return Err(ComparisonError::insufficient_confidence(
            report,
            case,
            record.confidence_parts_per_million,
        ));
    }
    Ok(())
}
