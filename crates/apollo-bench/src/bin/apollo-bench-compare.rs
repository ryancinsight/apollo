//! Command-line comparison of independently generated Apollo benchmark reports.

use apollo_bench::{
    compare_counterbalanced_report_directories, compare_report_directories, ComparisonSummary,
    CounterbalancedComparisonSummary,
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
        Err(error) => {
            eprintln!("{error}");
            eprintln!(
                "usage: apollo-bench-compare <compare|compare-counterbalanced> [directory options]"
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
}

enum CommandSummary {
    Single(ComparisonSummary),
    Counterbalanced(CounterbalancedComparisonSummary),
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

    Err(format!(
        "unsupported command `{}`",
        command.to_string_lossy()
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

fn reject_remaining(options: BTreeMap<OsString, PathBuf>) -> Result<(), String> {
    if let Some((flag, _)) = options.first_key_value() {
        return Err(format!("unsupported argument `{}`", flag.to_string_lossy()));
    }
    Ok(())
}
