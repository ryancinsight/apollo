//! Command-line comparison of independently generated Apollo benchmark reports.

use apollo_bench::compare_report_directories;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    match parse_directories().and_then(|(baseline, candidate)| {
        compare_report_directories(baseline, candidate).map_err(|error| error.to_string())
    }) {
        Ok(summary) if summary.passed() => {
            println!(
                "compared {} cases across {} reports; no supported regression",
                summary.compared_cases(),
                summary.compared_reports()
            );
            ExitCode::SUCCESS
        }
        Ok(summary) => {
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
        Err(error) => {
            eprintln!("{error}");
            eprintln!(
                "usage: apollo-bench-compare compare --baseline-directory <path> --candidate-directory <path>"
            );
            ExitCode::FAILURE
        }
    }
}

fn parse_directories() -> Result<(PathBuf, PathBuf), String> {
    let mut arguments = env::args_os();
    let _program = arguments.next();
    if arguments.next().as_deref() != Some(std::ffi::OsStr::new("compare")) {
        return Err("the first argument must be `compare`".to_owned());
    }

    let mut baseline = None;
    let mut candidate = None;
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for `{}`", flag.to_string_lossy()))?;
        if flag == "--baseline-directory" {
            baseline = Some(PathBuf::from(value));
        } else if flag == "--candidate-directory" {
            candidate = Some(PathBuf::from(value));
        } else {
            return Err(format!("unsupported argument `{}`", flag.to_string_lossy()));
        }
    }

    let baseline = baseline.ok_or_else(|| "`--baseline-directory` is required".to_owned())?;
    let candidate = candidate.ok_or_else(|| "`--candidate-directory` is required".to_owned())?;
    Ok((baseline, candidate))
}
