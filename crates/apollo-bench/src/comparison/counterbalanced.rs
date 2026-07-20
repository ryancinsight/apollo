use super::{
    compare_report_directories, BenchmarkKey, BenchmarkRegression, ComparisonError,
    ComparisonSummary,
};
use std::collections::BTreeMap;
use std::path::Path;

/// Records one execution order's disjoint median interval bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntervalSeparation {
    baseline_upper_nanoseconds: u128,
    candidate_lower_nanoseconds: u128,
}

impl IntervalSeparation {
    /// Returns the baseline median interval's upper bound.
    #[must_use]
    pub const fn baseline_upper_nanoseconds(self) -> u128 {
        self.baseline_upper_nanoseconds
    }

    /// Returns the candidate median interval's lower bound.
    #[must_use]
    pub const fn candidate_lower_nanoseconds(self) -> u128 {
        self.candidate_lower_nanoseconds
    }
}

/// Identifies a slowdown supported in both counterbalanced execution orders.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CounterbalancedBenchmarkRegression {
    report: std::path::PathBuf,
    case: String,
    baseline_first: IntervalSeparation,
    candidate_first: IntervalSeparation,
}

impl CounterbalancedBenchmarkRegression {
    /// Returns the report path relative to each compared directory.
    #[must_use]
    pub fn report(&self) -> &Path {
        &self.report
    }

    /// Returns the benchmark case label.
    #[must_use]
    pub fn case(&self) -> &str {
        &self.case
    }

    /// Returns the interval separation when the baseline executed first.
    #[must_use]
    pub const fn baseline_first(&self) -> IntervalSeparation {
        self.baseline_first
    }

    /// Returns the interval separation when the candidate executed first.
    #[must_use]
    pub const fn candidate_first(&self) -> IntervalSeparation {
        self.candidate_first
    }
}

/// Summarizes two counterbalanced base/head report comparisons.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CounterbalancedComparisonSummary {
    compared_reports: usize,
    compared_cases: usize,
    regressions: Vec<CounterbalancedBenchmarkRegression>,
}

impl CounterbalancedComparisonSummary {
    /// Returns the number of report identities compared in both orders.
    #[must_use]
    pub const fn compared_reports(&self) -> usize {
        self.compared_reports
    }

    /// Returns the number of case identities compared in both orders.
    #[must_use]
    pub const fn compared_cases(&self) -> usize {
        self.compared_cases
    }

    /// Returns cases whose candidate interval is slower in both orders.
    #[must_use]
    pub fn regressions(&self) -> &[CounterbalancedBenchmarkRegression] {
        &self.regressions
    }

    /// Returns whether no counterbalanced slowdown was found.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.regressions.is_empty()
    }
}

/// Compares base/head reports in both execution orders to reject systematic
/// order drift.
///
/// The baseline-first pair must be measured baseline then candidate. The
/// candidate-first pair must reverse that order. All four report trees must
/// contain the same cases. A regression is returned only when the candidate
/// interval is wholly slower in both orderings.
///
/// # Errors
///
/// Returns [`ComparisonError`] for any malformed, incomplete, low-confidence,
/// or mismatched evidence tree.
///
/// # Examples
///
/// ```
/// use apollo_bench::compare_counterbalanced_report_directories;
/// use std::fs;
///
/// let sequence = std::time::SystemTime::now()
///     .duration_since(std::time::UNIX_EPOCH)?
///     .as_nanos();
/// let root = std::env::temp_dir()
///     .join(format!("apollo-bench-doc-counterbalanced-{sequence}"));
/// let report = concat!(
///     "case,min_ns,median_ns,median_lower_ns,median_upper_ns,",
///     "median_confidence_ppm,samples,iterations_per_sample\n",
///     "fft/forward/256,80,100,90,110,964799,100,4\n"
/// );
/// for directory in [
///     "baseline-first/baseline",
///     "baseline-first/candidate",
///     "candidate-first/baseline",
///     "candidate-first/candidate",
/// ] {
///     let directory = root.join(directory);
///     fs::create_dir_all(&directory)?;
///     fs::write(directory.join("fft.csv"), report)?;
/// }
///
/// let summary = compare_counterbalanced_report_directories(
///     root.join("baseline-first/baseline"),
///     root.join("baseline-first/candidate"),
///     root.join("candidate-first/baseline"),
///     root.join("candidate-first/candidate"),
/// )?;
/// assert!(summary.passed());
///
/// fs::remove_dir_all(root)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compare_counterbalanced_report_directories(
    baseline_first_baseline: impl AsRef<Path>,
    baseline_first_candidate: impl AsRef<Path>,
    candidate_first_baseline: impl AsRef<Path>,
    candidate_first_candidate: impl AsRef<Path>,
) -> Result<CounterbalancedComparisonSummary, ComparisonError> {
    let baseline_first =
        compare_report_directories(baseline_first_baseline, baseline_first_candidate)?;
    let candidate_first =
        compare_report_directories(candidate_first_baseline, candidate_first_candidate)?;
    validate_case_universe(&baseline_first, &candidate_first)?;

    let candidate_first_regressions = candidate_first
        .regressions
        .iter()
        .map(|regression| (key(regression), regression))
        .collect::<BTreeMap<_, _>>();
    let regressions =
        baseline_first
            .regressions
            .iter()
            .filter_map(|regression| {
                candidate_first_regressions.get(&key(regression)).map(
                    |candidate_first_regression| CounterbalancedBenchmarkRegression {
                        report: regression.report.clone(),
                        case: regression.case.clone(),
                        baseline_first: separation(regression),
                        candidate_first: separation(candidate_first_regression),
                    },
                )
            })
            .collect();

    Ok(CounterbalancedComparisonSummary {
        compared_reports: baseline_first.compared_reports,
        compared_cases: baseline_first.compared_cases,
        regressions,
    })
}

fn validate_case_universe(
    baseline_first: &ComparisonSummary,
    candidate_first: &ComparisonSummary,
) -> Result<(), ComparisonError> {
    if let Some(missing) = baseline_first
        .compared_keys
        .difference(&candidate_first.compared_keys)
        .next()
    {
        return Err(ComparisonError::missing_candidate_first_case(
            &missing.report,
            &missing.case,
        ));
    }
    if let Some(missing) = candidate_first
        .compared_keys
        .difference(&baseline_first.compared_keys)
        .next()
    {
        return Err(ComparisonError::missing_baseline_first_case(
            &missing.report,
            &missing.case,
        ));
    }
    Ok(())
}

fn key(regression: &BenchmarkRegression) -> BenchmarkKey {
    BenchmarkKey {
        report: regression.report.clone(),
        case: regression.case.clone(),
    }
}

const fn separation(regression: &BenchmarkRegression) -> IntervalSeparation {
    IntervalSeparation {
        baseline_upper_nanoseconds: regression.baseline_upper_nanoseconds,
        candidate_lower_nanoseconds: regression.candidate_lower_nanoseconds,
    }
}
