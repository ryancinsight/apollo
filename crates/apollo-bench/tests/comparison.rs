//! Value-semantic integration coverage for Apollo benchmark report comparison.

use apollo_bench::{compare_counterbalanced_report_directories, compare_report_directories};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

const HEADER: &str = "case,min_ns,median_ns,median_lower_ns,median_upper_ns,median_confidence_ppm,samples,iterations_per_sample\n";
static FIXTURE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let sequence = FIXTURE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "apollo-bench-comparison-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("fixture root creation must succeed");
        Self { root }
    }

    fn write(&self, relative: &str, body: &str) {
        let path = self.root.join(relative);
        fs::create_dir_all(
            path.parent()
                .expect("invariant: every fixture report has a parent"),
        )
        .expect("fixture report parent creation must succeed");
        fs::write(path, format!("{HEADER}{body}")).expect("fixture report write must succeed");
    }

    fn directory(&self, relative: &str) -> PathBuf {
        self.root.join(relative)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if self.root.exists() {
            fs::remove_dir_all(&self.root).expect("fixture cleanup must succeed");
        }
    }
}

#[test]
fn disjoint_slower_interval_is_a_regression() {
    let fixture = Fixture::new();
    fixture.write("base/fft/kernel.csv", "n256,80,100,90,110,964799,100,4\n");
    fixture.write(
        "candidate/fft/kernel.csv",
        "n256,100,120,111,130,964799,100,4\n",
    );

    let summary =
        compare_report_directories(fixture.directory("base"), fixture.directory("candidate"))
            .expect("valid paired reports must compare");

    assert_eq!(summary.compared_reports(), 1);
    assert_eq!(summary.compared_cases(), 1);
    assert_eq!(summary.regressions().len(), 1);
    assert_eq!(
        summary.regressions()[0].report(),
        Path::new("fft/kernel.csv")
    );
    assert_eq!(summary.regressions()[0].case(), "n256");
    assert_eq!(summary.regressions()[0].baseline_upper_nanoseconds(), 110);
    assert_eq!(summary.regressions()[0].candidate_lower_nanoseconds(), 111);
}

#[test]
fn overlapping_intervals_do_not_claim_a_regression() {
    let fixture = Fixture::new();
    fixture.write("base/kernel.csv", "n256,80,105,90,120,964799,100,4\n");
    fixture.write(
        "candidate/kernel.csv",
        "n256,100,125,115,140,964799,100,4\n",
    );

    let summary =
        compare_report_directories(fixture.directory("base"), fixture.directory("candidate"))
            .expect("valid paired reports must compare");

    assert!(summary.passed());
    assert_eq!(summary.compared_cases(), 1);
}

#[test]
fn missing_candidate_case_fails_closed() {
    let fixture = Fixture::new();
    fixture.write(
        "base/kernel.csv",
        "n256,80,105,90,120,964799,100,4\nn512,160,210,180,240,964799,100,4\n",
    );
    fixture.write("candidate/kernel.csv", "n256,80,105,90,120,964799,100,4\n");

    let error =
        compare_report_directories(fixture.directory("base"), fixture.directory("candidate"))
            .expect_err("an unpaired case must fail closed");

    assert!(error.to_string().contains("omits baseline case `n512`"));
}

#[test]
fn quoted_case_labels_round_trip_through_the_csv_parser() {
    let fixture = Fixture::new();
    fixture.write(
        "base/kernel.csv",
        "\"fft,forward/n256\",80,105,90,120,964799,100,4\n",
    );
    fixture.write(
        "candidate/kernel.csv",
        "\"fft,forward/n256\",80,105,90,120,964799,100,4\n",
    );

    let summary =
        compare_report_directories(fixture.directory("base"), fixture.directory("candidate"))
            .expect("quoted labels must retain their case identity");

    assert!(summary.passed());
    assert_eq!(summary.compared_cases(), 1);
}

#[test]
fn substandard_interval_confidence_fails_closed() {
    let fixture = Fixture::new();
    fixture.write("base/kernel.csv", "n256,80,105,90,120,937500,5,4\n");
    fixture.write("candidate/kernel.csv", "n256,80,105,90,120,937500,5,4\n");

    let error =
        compare_report_directories(fixture.directory("base"), fixture.directory("candidate"))
            .expect_err("less than 95 percent confidence must fail closed");

    assert!(error.to_string().contains("below the required 950000 ppm"));
}

#[test]
fn command_reports_the_compared_evidence_count() {
    let fixture = Fixture::new();
    fixture.write("base/kernel.csv", "n256,80,105,90,120,964799,100,4\n");
    fixture.write("candidate/kernel.csv", "n256,80,105,90,120,964799,100,4\n");

    let output = Command::new(env!("CARGO_BIN_EXE_apollo-bench-compare"))
        .arg("compare")
        .arg("--baseline-directory")
        .arg(fixture.directory("base"))
        .arg("--candidate-directory")
        .arg(fixture.directory("candidate"))
        .output()
        .expect("comparison command must execute");

    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).expect("command output must be UTF-8"),
        "compared 1 cases across 1 reports; no supported regression\n"
    );
    assert!(output.stderr.is_empty());
}

#[test]
fn empty_reports_fail_closed() {
    let fixture = Fixture::new();
    fixture.write("base/kernel.csv", "");
    fixture.write("candidate/kernel.csv", "");

    let error =
        compare_report_directories(fixture.directory("base"), fixture.directory("candidate"))
            .expect_err("a header-only report supplies no evidence");

    assert!(error.to_string().contains("contains no benchmark cases"));
}

#[test]
fn one_order_only_slowdown_is_rejected_as_order_drift() {
    let fixture = Fixture::new();
    fixture.write(
        "baseline-first/baseline/kernel.csv",
        "n256,80,100,90,110,964799,100,4\n",
    );
    fixture.write(
        "baseline-first/candidate/kernel.csv",
        "n256,100,120,111,130,964799,100,4\n",
    );
    fixture.write(
        "candidate-first/baseline/kernel.csv",
        "n256,100,120,110,130,964799,100,4\n",
    );
    fixture.write(
        "candidate-first/candidate/kernel.csv",
        "n256,80,105,95,115,964799,100,4\n",
    );

    let summary = compare_counterbalanced_report_directories(
        fixture.directory("baseline-first/baseline"),
        fixture.directory("baseline-first/candidate"),
        fixture.directory("candidate-first/baseline"),
        fixture.directory("candidate-first/candidate"),
    )
    .expect("counterbalanced report sets must compare");

    assert!(summary.passed());
    assert_eq!(summary.compared_cases(), 1);
}

#[test]
fn slowdown_in_both_orders_is_a_counterbalanced_regression() {
    let fixture = Fixture::new();
    for order in ["baseline-first", "candidate-first"] {
        fixture.write(
            &format!("{order}/baseline/kernel.csv"),
            "n256,80,100,90,110,964799,100,4\n",
        );
        fixture.write(
            &format!("{order}/candidate/kernel.csv"),
            "n256,100,120,111,130,964799,100,4\n",
        );
    }

    let summary = compare_counterbalanced_report_directories(
        fixture.directory("baseline-first/baseline"),
        fixture.directory("baseline-first/candidate"),
        fixture.directory("candidate-first/baseline"),
        fixture.directory("candidate-first/candidate"),
    )
    .expect("counterbalanced report sets must compare");

    assert_eq!(summary.regressions().len(), 1);
    let regression = &summary.regressions()[0];
    assert_eq!(regression.case(), "n256");
    assert_eq!(
        regression.baseline_first().baseline_upper_nanoseconds(),
        110
    );
    assert_eq!(
        regression.candidate_first().candidate_lower_nanoseconds(),
        111
    );
}

#[test]
fn counterbalanced_command_reports_unique_evidence_count() {
    let fixture = Fixture::new();
    for order in ["baseline-first", "candidate-first"] {
        fixture.write(
            &format!("{order}/baseline/kernel.csv"),
            "n256,80,100,90,110,964799,100,4\n",
        );
        fixture.write(
            &format!("{order}/candidate/kernel.csv"),
            "n256,80,100,90,110,964799,100,4\n",
        );
    }

    let output = Command::new(env!("CARGO_BIN_EXE_apollo-bench-compare"))
        .arg("compare-counterbalanced")
        .arg("--baseline-first-baseline-directory")
        .arg(fixture.directory("baseline-first/baseline"))
        .arg("--baseline-first-candidate-directory")
        .arg(fixture.directory("baseline-first/candidate"))
        .arg("--candidate-first-baseline-directory")
        .arg(fixture.directory("candidate-first/baseline"))
        .arg("--candidate-first-candidate-directory")
        .arg(fixture.directory("candidate-first/candidate"))
        .output()
        .expect("counterbalanced comparison command must execute");

    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).expect("command output must be UTF-8"),
        "counterbalanced 1 cases across 1 reports; no supported regression\n"
    );
    assert!(output.stderr.is_empty());
}

#[test]
fn counterbalanced_case_universes_must_match() {
    let fixture = Fixture::new();
    fixture.write(
        "baseline-first/baseline/kernel.csv",
        "n256,80,100,90,110,964799,100,4\n",
    );
    fixture.write(
        "baseline-first/candidate/kernel.csv",
        "n256,80,100,90,110,964799,100,4\n",
    );
    fixture.write(
        "candidate-first/baseline/kernel.csv",
        "n512,80,100,90,110,964799,100,4\n",
    );
    fixture.write(
        "candidate-first/candidate/kernel.csv",
        "n512,80,100,90,110,964799,100,4\n",
    );

    let error = compare_counterbalanced_report_directories(
        fixture.directory("baseline-first/baseline"),
        fixture.directory("baseline-first/candidate"),
        fixture.directory("candidate-first/baseline"),
        fixture.directory("candidate-first/candidate"),
    )
    .expect_err("counterbalanced case identities must match");

    assert!(error
        .to_string()
        .contains("candidate-first evidence omits baseline-first"));
}
