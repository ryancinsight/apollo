use super::counterbalanced::{compare_report_set, CounterbalancedReportSet};
use super::{
    BenchmarkKey, ComparisonError, CounterbalancedBenchmarkRegression,
    CounterbalancedComparisonSummary,
};
use std::collections::BTreeMap;
use std::path::Path;

/// Identifies a slowdown reproduced by two phase-reversed ABBA blocks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplicatedCounterbalancedBenchmarkRegression {
    first: CounterbalancedBenchmarkRegression,
    second: CounterbalancedBenchmarkRegression,
}

impl ReplicatedCounterbalancedBenchmarkRegression {
    /// Returns the report path relative to every evidence directory.
    #[must_use]
    pub fn report(&self) -> &Path {
        self.first.report()
    }

    /// Returns the benchmark case label.
    #[must_use]
    pub fn case(&self) -> &str {
        self.first.case()
    }

    /// Returns the first ABBA block's supported slowdown.
    #[must_use]
    pub const fn first_replication(&self) -> &CounterbalancedBenchmarkRegression {
        &self.first
    }

    /// Returns the phase-reversed ABBA block's supported slowdown.
    #[must_use]
    pub const fn second_replication(&self) -> &CounterbalancedBenchmarkRegression {
        &self.second
    }
}

/// Summarizes two phase-reversed counterbalanced measurement blocks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplicatedCounterbalancedComparisonSummary {
    compared_reports: usize,
    compared_cases: usize,
    regressions: Vec<ReplicatedCounterbalancedBenchmarkRegression>,
    spread_suppressed: Vec<String>,
}

impl ReplicatedCounterbalancedComparisonSummary {
    /// Returns the number of report identities compared in every block.
    #[must_use]
    pub const fn compared_reports(&self) -> usize {
        self.compared_reports
    }

    /// Returns the number of case identities compared in every block.
    #[must_use]
    pub const fn compared_cases(&self) -> usize {
        self.compared_cases
    }

    /// Returns slowdowns supported by all four execution-order comparisons.
    #[must_use]
    pub fn regressions(&self) -> &[ReplicatedCounterbalancedBenchmarkRegression] {
        &self.regressions
    }

    /// Returns cases that separated inside every run but not across them.
    ///
    /// These are the cases the evidence cannot decide: each block's samples
    /// support a slowdown, yet the blocks disagree by at least as much as the
    /// slowdown itself. A non-empty list means the gate's resolution is bounded
    /// by host stability rather than by the code under test, so it is reported
    /// instead of being silently folded into a pass.
    #[must_use]
    pub fn spread_suppressed_cases(&self) -> &[String] {
        &self.spread_suppressed
    }

    /// Returns whether no slowdown reproduced across both blocks.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.regressions.is_empty()
    }
}

/// Compares two counterbalanced blocks whose execution phases are reversed.
///
/// The caller executes the first set as ABBA and the second as BAAB. This
/// assigns baseline and candidate to equal sums of the eight period indices
/// and their squares, balancing exposure to constant, linear, and quadratic
/// period terms. A case regresses only when all four base/head comparisons
/// support it *and* the slowest candidate median bound still clears the fastest
/// baseline median bound across all four. The second requirement charges the
/// decision the between-run spread the replication measures; a per-run median
/// interval bounds only that run's sampling noise, so four of them can separate
/// together on a host whose regime shifts between runs.
///
/// # Errors
///
/// Returns [`ComparisonError`] for malformed, incomplete, low-confidence, or
/// mismatched evidence in either block.
///
/// # Examples
///
/// ```
/// use apollo_bench::{
///     compare_replicated_counterbalanced_report_directories,
///     CounterbalancedReportSet,
/// };
/// use std::fs;
///
/// let sequence = std::time::SystemTime::now()
///     .duration_since(std::time::UNIX_EPOCH)?
///     .as_nanos();
/// let root = std::env::temp_dir()
///     .join(format!("apollo-bench-doc-replicated-{sequence}"));
/// let samples = (1..=100)
///     .map(|sample| sample.to_string())
///     .collect::<Vec<_>>()
///     .join(";");
/// let report = format!(
///     "case,min_ns,median_ns,median_lower_ns,median_upper_ns,\
///      median_confidence_ppm,ordered_samples_ns,iterations_per_sample\n\
///      fft/forward/256,1,50,40,61,964799,{samples},4\n"
/// );
/// for replication in ["first", "second"] {
///     for order in ["baseline-first", "candidate-first"] {
///         for revision in ["baseline", "candidate"] {
///             let directory = root.join(replication).join(order).join(revision);
///             fs::create_dir_all(&directory)?;
///             fs::write(directory.join("fft.csv"), &report)?;
///         }
///     }
/// }
/// let reports = |replication: &str| {
///     CounterbalancedReportSet::new(
///         root.join(replication).join("baseline-first/baseline"),
///         root.join(replication).join("baseline-first/candidate"),
///         root.join(replication).join("candidate-first/baseline"),
///         root.join(replication).join("candidate-first/candidate"),
///     )
/// };
///
/// let summary = compare_replicated_counterbalanced_report_directories([
///     reports("first"),
///     reports("second"),
/// ])?;
/// assert!(summary.passed());
///
/// fs::remove_dir_all(root)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn compare_replicated_counterbalanced_report_directories(
    replications: [CounterbalancedReportSet; 2],
) -> Result<ReplicatedCounterbalancedComparisonSummary, ComparisonError> {
    let [first_reports, second_reports] = replications;
    let first = compare_report_set(&first_reports)?;
    let second = compare_report_set(&second_reports)?;
    validate_replication_universe(&first, &second)?;

    let second_regressions = second
        .regressions()
        .iter()
        .map(|regression| (key(regression), regression))
        .collect::<BTreeMap<_, _>>();
    let mut spread_suppressed = Vec::new();
    let regressions = first
        .regressions()
        .iter()
        .filter_map(|regression| {
            let second = second_regressions.get(&key(regression))?;
            if separated_across_replications(regression, second) {
                return Some(ReplicatedCounterbalancedBenchmarkRegression {
                    first: regression.clone(),
                    second: (*second).clone(),
                });
            }
            spread_suppressed.push(format!(
                "{}: {}",
                regression.report().display(),
                regression.case()
            ));
            None
        })
        .collect();

    Ok(ReplicatedCounterbalancedComparisonSummary {
        compared_reports: first.compared_reports(),
        compared_cases: first.compared_cases(),
        regressions,
        spread_suppressed,
    })
}

/// Requires a slowdown to clear the spread the replicated design measures.
///
/// Each block's [`IntervalSeparation`] comes from one run's samples, so it
/// bounds that run's sampling noise alone. Re-executing an unchanged binary can
/// move a case's median far beyond that interval when the kernel occupies a
/// different load-duration regime from run to run, and four such intervals can
/// then separate together without any code change. Charging the decision the
/// slowest candidate bound against the fastest baseline bound, across every
/// replication and execution order, spends the between-run evidence the four
/// blocks were executed to obtain instead of discarding it.
fn separated_across_replications(
    first: &CounterbalancedBenchmarkRegression,
    second: &CounterbalancedBenchmarkRegression,
) -> bool {
    let separations = [
        first.baseline_first(),
        first.candidate_first(),
        second.baseline_first(),
        second.candidate_first(),
    ];
    let slowest_baseline = separations
        .iter()
        .map(|separation| separation.baseline_upper_nanoseconds())
        .max();
    let fastest_candidate = separations
        .iter()
        .map(|separation| separation.candidate_lower_nanoseconds())
        .min();
    match (fastest_candidate, slowest_baseline) {
        (Some(candidate), Some(baseline)) => candidate > baseline,
        _ => false,
    }
}

fn validate_replication_universe(
    first: &CounterbalancedComparisonSummary,
    second: &CounterbalancedComparisonSummary,
) -> Result<(), ComparisonError> {
    if let Some(missing) = first.compared_keys.difference(&second.compared_keys).next() {
        return Err(ComparisonError::missing_second_replication_case(
            &missing.report,
            &missing.case,
        ));
    }
    if let Some(missing) = second.compared_keys.difference(&first.compared_keys).next() {
        return Err(ComparisonError::missing_first_replication_case(
            &missing.report,
            &missing.case,
        ));
    }
    Ok(())
}

fn key(regression: &CounterbalancedBenchmarkRegression) -> BenchmarkKey {
    BenchmarkKey {
        report: regression.report().to_path_buf(),
        case: regression.case().to_owned(),
    }
}
