//! Command-line comparison of independently generated Apollo benchmark reports.

use apollo_bench::{
    compare_counterbalanced_report_directories,
    compare_replicated_counterbalanced_report_directories, compare_report_directories,
    ComparisonSummary, CounterbalancedComparisonSummary, CounterbalancedReportSet,
    ReplicatedCounterbalancedComparisonSummary,
};
use std::collections::BTreeMap;
use std::env;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    match parse_command().and_then(execute) {
        Ok(CommandSummary::Single(summary)) => report_single(&summary),
        Ok(CommandSummary::Counterbalanced(summary)) => report_counterbalanced(&summary),
        Ok(CommandSummary::ReplicatedCounterbalanced(summary)) => {
            report_replicated_counterbalanced(&summary)
        }
        Err(error) => {
            eprintln!("{error}");
            eprintln!(
                "usage: apollo-bench-compare <compare|compare-counterbalanced|compare-replicated-counterbalanced> [directory options]"
            );
            ExitCode::FAILURE
        }
    }
}

enum Command {
    Compare {
        baseline: PathBuf,
        candidate: PathBuf,
    },
    CompareCounterbalanced {
        baseline_first_baseline: PathBuf,
        baseline_first_candidate: PathBuf,
        candidate_first_baseline: PathBuf,
        candidate_first_candidate: PathBuf,
    },
    CompareReplicatedCounterbalanced {
        first: CounterbalancedReportSet,
        second: CounterbalancedReportSet,
    },
}

enum CommandSummary {
    Single(ComparisonSummary),
    Counterbalanced(CounterbalancedComparisonSummary),
    ReplicatedCounterbalanced(ReplicatedCounterbalancedComparisonSummary),
}

fn execute(command: Command) -> Result<CommandSummary, String> {
    match command {
        Command::Compare {
            baseline,
            candidate,
        } => compare_report_directories(baseline, candidate)
            .map(CommandSummary::Single)
            .map_err(|error| error.to_string()),
        Command::CompareCounterbalanced {
            baseline_first_baseline,
            baseline_first_candidate,
            candidate_first_baseline,
            candidate_first_candidate,
        } => compare_counterbalanced_report_directories(
            baseline_first_baseline,
            baseline_first_candidate,
            candidate_first_baseline,
            candidate_first_candidate,
        )
        .map(CommandSummary::Counterbalanced)
        .map_err(|error| error.to_string()),
        Command::CompareReplicatedCounterbalanced { first, second } => {
            compare_replicated_counterbalanced_report_directories([first, second])
                .map(CommandSummary::ReplicatedCounterbalanced)
                .map_err(|error| error.to_string())
        }
    }
}

fn report_single(summary: &ComparisonSummary) -> ExitCode {
    if summary.passed() {
        println!(
            "compared {} cases across {} reports; no supported regression",
            summary.compared_cases(),
            summary.compared_reports()
        );
        return ExitCode::SUCCESS;
    }

    for regression in summary.regressions() {
        eprintln!(
            "{}: {} candidate lower bound {} ns exceeds baseline upper bound {} ns",
            regression.report().display(),
            regression.case(),
            regression.candidate_lower_nanoseconds(),
            regression.baseline_upper_nanoseconds()
        );
    }
    ExitCode::FAILURE
}

fn report_counterbalanced(summary: &CounterbalancedComparisonSummary) -> ExitCode {
    if summary.passed() {
        println!(
            "counterbalanced {} cases across {} reports; no supported regression",
            summary.compared_cases(),
            summary.compared_reports()
        );
        return ExitCode::SUCCESS;
    }

    for regression in summary.regressions() {
        let baseline_first = regression.baseline_first();
        let candidate_first = regression.candidate_first();
        eprintln!(
            "{}: {} candidate is slower in both orders: baseline-first {} ns > {} ns; candidate-first {} ns > {} ns",
            regression.report().display(),
            regression.case(),
            baseline_first.candidate_lower_nanoseconds(),
            baseline_first.baseline_upper_nanoseconds(),
            candidate_first.candidate_lower_nanoseconds(),
            candidate_first.baseline_upper_nanoseconds()
        );
    }
    ExitCode::FAILURE
}

/// Reports cases the evidence could not decide.
///
/// Suppressed cases separated inside every run yet disagreed between runs by at
/// least the size of the slowdown. Printing them keeps a host whose spread
/// exceeds the effect visible, rather than letting it read as a clean pass.
fn report_spread_suppressed(summary: &ReplicatedCounterbalancedComparisonSummary) {
    let suppressed = summary.spread_suppressed_cases();
    if suppressed.is_empty() {
        return;
    }
    println!(
        "note: {} case(s) separated within runs but not across replications; \
         the between-run spread equals or exceeds the effect, so this run cannot \
         decide them:",
        suppressed.len()
    );
    for case in suppressed {
        println!("  {case}");
    }
}

fn report_replicated_counterbalanced(
    summary: &ReplicatedCounterbalancedComparisonSummary,
) -> ExitCode {
    if summary.passed() {
        println!(
            "replicated counterbalanced {} cases across {} reports; no supported regression",
            summary.compared_cases(),
            summary.compared_reports()
        );
        report_spread_suppressed(summary);
        return ExitCode::SUCCESS;
    }

    for regression in summary.regressions() {
        let first = regression.first_replication();
        let second = regression.second_replication();
        eprintln!(
            "{}: {} candidate is slower in all four comparisons: first baseline-first {} ns > {} ns; first candidate-first {} ns > {} ns; second baseline-first {} ns > {} ns; second candidate-first {} ns > {} ns",
            regression.report().display(),
            regression.case(),
            first.baseline_first().candidate_lower_nanoseconds(),
            first.baseline_first().baseline_upper_nanoseconds(),
            first.candidate_first().candidate_lower_nanoseconds(),
            first.candidate_first().baseline_upper_nanoseconds(),
            second.baseline_first().candidate_lower_nanoseconds(),
            second.baseline_first().baseline_upper_nanoseconds(),
            second.candidate_first().candidate_lower_nanoseconds(),
            second.candidate_first().baseline_upper_nanoseconds()
        );
    }
    ExitCode::FAILURE
}

fn parse_command() -> Result<Command, String> {
    let mut arguments = env::args_os();
    let _program = arguments.next();
    let command = arguments
        .next()
        .ok_or_else(|| "a comparison command is required".to_owned())?;
    let mut options = parse_options(arguments)?;

    if command == "compare" {
        let baseline = take_required(&mut options, "--baseline-directory")?;
        let candidate = take_required(&mut options, "--candidate-directory")?;
        reject_remaining(options)?;
        return Ok(Command::Compare {
            baseline,
            candidate,
        });
    }
    if command == "compare-counterbalanced" {
        let baseline_first_baseline =
            take_required(&mut options, "--baseline-first-baseline-directory")?;
        let baseline_first_candidate =
            take_required(&mut options, "--baseline-first-candidate-directory")?;
        let candidate_first_baseline =
            take_required(&mut options, "--candidate-first-baseline-directory")?;
        let candidate_first_candidate =
            take_required(&mut options, "--candidate-first-candidate-directory")?;
        reject_remaining(options)?;
        return Ok(Command::CompareCounterbalanced {
            baseline_first_baseline,
            baseline_first_candidate,
            candidate_first_baseline,
            candidate_first_candidate,
        });
    }
    if command == "compare-replicated-counterbalanced" {
        let first = take_counterbalanced_set(&mut options, "first")?;
        let second = take_counterbalanced_set(&mut options, "second")?;
        reject_remaining(options)?;
        return Ok(Command::CompareReplicatedCounterbalanced { first, second });
    }

    Err(format!(
        "unsupported command `{}`",
        command.to_string_lossy()
    ))
}

fn take_counterbalanced_set(
    options: &mut BTreeMap<OsString, PathBuf>,
    replication: &str,
) -> Result<CounterbalancedReportSet, String> {
    Ok(CounterbalancedReportSet::new(
        take_required_owned(
            options,
            format!("--{replication}-baseline-first-baseline-directory"),
        )?,
        take_required_owned(
            options,
            format!("--{replication}-baseline-first-candidate-directory"),
        )?,
        take_required_owned(
            options,
            format!("--{replication}-candidate-first-baseline-directory"),
        )?,
        take_required_owned(
            options,
            format!("--{replication}-candidate-first-candidate-directory"),
        )?,
    ))
}

fn parse_options(
    mut arguments: impl Iterator<Item = OsString>,
) -> Result<BTreeMap<OsString, PathBuf>, String> {
    let mut options = BTreeMap::new();
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for `{}`", flag.to_string_lossy()))?;
        if options.insert(flag.clone(), PathBuf::from(value)).is_some() {
            return Err(format!("duplicate argument `{}`", flag.to_string_lossy()));
        }
    }
    Ok(options)
}

fn take_required(
    options: &mut BTreeMap<OsString, PathBuf>,
    flag: &'static str,
) -> Result<PathBuf, String> {
    options
        .remove(OsStr::new(flag))
        .ok_or_else(|| format!("`{flag}` is required"))
}

fn take_required_owned(
    options: &mut BTreeMap<OsString, PathBuf>,
    flag: String,
) -> Result<PathBuf, String> {
    options
        .remove(OsStr::new(&flag))
        .ok_or_else(|| format!("`{flag}` is required"))
}

fn reject_remaining(options: BTreeMap<OsString, PathBuf>) -> Result<(), String> {
    if let Some((flag, _)) = options.first_key_value() {
        return Err(format!("unsupported argument `{}`", flag.to_string_lossy()));
    }
    Ok(())
}
