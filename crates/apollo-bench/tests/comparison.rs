//! Value-semantic integration coverage for Apollo benchmark report comparison.

#[path = "comparison/support.rs"]
mod support;

use apollo_bench::{
    compare_counterbalanced_report_directories,
    compare_replicated_counterbalanced_report_directories, compare_report_directories,
};
use std::path::Path;
use std::process::Command;
use support::{report_row, report_set, standard_row, write_counterbalanced_case, Fixture};

#[test]
fn disjoint_slower_interval_is_a_regression() {
    let fixture = Fixture::new();
    fixture.write("base/fft/kernel.csv", &[standard_row("n256", 1)]);
    fixture.write("candidate/fft/kernel.csv", &[standard_row("n256", 1_000)]);

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
    assert_eq!(summary.regressions()[0].baseline_upper_nanoseconds(), 62);
    assert_eq!(
        summary.regressions()[0].candidate_lower_nanoseconds(),
        1_038
    );
}

#[test]
fn overlapping_intervals_do_not_claim_a_regression() {
    let fixture = Fixture::new();
    fixture.write("base/kernel.csv", &[standard_row("n256", 1)]);
    fixture.write("candidate/kernel.csv", &[standard_row("n256", 20)]);

    let summary =
        compare_report_directories(fixture.directory("base"), fixture.directory("candidate"))
            .expect("valid paired reports must compare");

    assert!(summary.passed());
    assert_eq!(summary.compared_cases(), 1);
}

#[test]
fn familywise_interval_rejects_a_per_case_false_positive() {
    let fixture = Fixture::new();
    fixture.write("base/kernel.csv", &[standard_row("n256", 1)]);
    fixture.write("candidate/kernel.csv", &[standard_row("n256", 25)]);

    let uncorrected =
        compare_report_directories(fixture.directory("base"), fixture.directory("candidate"))
            .expect("one-case reports must compare");
    assert_eq!(uncorrected.regressions().len(), 1);

    fixture.write(
        "base/kernel.csv",
        &[standard_row("n256", 1), standard_row("n512", 1_000)],
    );
    fixture.write(
        "candidate/kernel.csv",
        &[standard_row("n256", 25), standard_row("n512", 1_000)],
    );

    let corrected =
        compare_report_directories(fixture.directory("base"), fixture.directory("candidate"))
            .expect("two-case reports must compare");
    assert!(corrected.passed());
    assert_eq!(corrected.compared_cases(), 2);
}

#[test]
fn missing_candidate_case_fails_closed() {
    let fixture = Fixture::new();
    fixture.write(
        "base/kernel.csv",
        &[standard_row("n256", 1), standard_row("n512", 1_000)],
    );
    fixture.write("candidate/kernel.csv", &[standard_row("n256", 1)]);

    let error =
        compare_report_directories(fixture.directory("base"), fixture.directory("candidate"))
            .expect_err("an unpaired case must fail closed");

    assert!(error.to_string().contains("omits baseline case `n512`"));
}

#[test]
fn quoted_case_labels_round_trip_through_the_csv_parser() {
    let fixture = Fixture::new();
    let row = standard_row("fft,forward/n256", 1);
    fixture.write("base/kernel.csv", std::slice::from_ref(&row));
    fixture.write("candidate/kernel.csv", &[row]);

    let summary =
        compare_report_directories(fixture.directory("base"), fixture.directory("candidate"))
            .expect("quoted labels must retain their case identity");

    assert!(summary.passed());
    assert_eq!(summary.compared_cases(), 1);
}

#[test]
fn insufficient_familywise_sample_count_fails_closed() {
    let fixture = Fixture::new();
    let first = report_row("n256", &[1, 2, 3, 4, 5, 6]);
    let second = report_row("n512", &[10, 11, 12, 13, 14, 15]);
    for directory in ["base", "candidate"] {
        fixture.write(
            &format!("{directory}/kernel.csv"),
            &[first.clone(), second.clone()],
        );
    }

    let error =
        compare_report_directories(fixture.directory("base"), fixture.directory("candidate"))
            .expect_err("six samples cannot control family-wise error over two cases");

    assert!(error
        .to_string()
        .contains("6 samples, insufficient for 5% family-wise error across 2 cases"));
}

#[test]
fn unordered_raw_evidence_fails_closed() {
    let fixture = Fixture::new();
    let row = report_row("n256", &[1, 3, 2, 4, 5, 6]);
    fixture.write("base/kernel.csv", std::slice::from_ref(&row));
    fixture.write("candidate/kernel.csv", &[row]);

    let error =
        compare_report_directories(fixture.directory("base"), fixture.directory("candidate"))
            .expect_err("unordered observations cannot establish order statistics");

    assert!(error
        .to_string()
        .contains("violates nondecreasing ordered_samples_ns"));
}

#[test]
fn command_reports_the_compared_evidence_count() {
    let fixture = Fixture::new();
    let row = standard_row("n256", 1);
    fixture.write("base/kernel.csv", std::slice::from_ref(&row));
    fixture.write("candidate/kernel.csv", &[row]);

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
    fixture.write("base/kernel.csv", &[]);
    fixture.write("candidate/kernel.csv", &[]);

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
        &[standard_row("n256", 1)],
    );
    fixture.write(
        "baseline-first/candidate/kernel.csv",
        &[standard_row("n256", 1_000)],
    );
    fixture.write(
        "candidate-first/baseline/kernel.csv",
        &[standard_row("n256", 1_000)],
    );
    fixture.write(
        "candidate-first/candidate/kernel.csv",
        &[standard_row("n256", 1)],
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
            &[standard_row("n256", 1)],
        );
        fixture.write(
            &format!("{order}/candidate/kernel.csv"),
            &[standard_row("n256", 1_000)],
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
    assert_eq!(regression.baseline_first().baseline_upper_nanoseconds(), 62);
    assert_eq!(
        regression.candidate_first().candidate_lower_nanoseconds(),
        1_038
    );
}

#[test]
fn counterbalanced_command_reports_unique_evidence_count() {
    let fixture = Fixture::new();
    let row = standard_row("n256", 1);
    for order in ["baseline-first", "candidate-first"] {
        fixture.write(
            &format!("{order}/baseline/kernel.csv"),
            std::slice::from_ref(&row),
        );
        fixture.write(
            &format!("{order}/candidate/kernel.csv"),
            std::slice::from_ref(&row),
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
        &[standard_row("n256", 1)],
    );
    fixture.write(
        "baseline-first/candidate/kernel.csv",
        &[standard_row("n256", 1)],
    );
    fixture.write(
        "candidate-first/baseline/kernel.csv",
        &[standard_row("n512", 1)],
    );
    fixture.write(
        "candidate-first/candidate/kernel.csv",
        &[standard_row("n512", 1)],
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

#[test]
fn phase_reversed_replication_rejects_one_block_slowdown() {
    let fixture = Fixture::new();
    write_counterbalanced_case(&fixture, "first", "n256", 1, 1_000);
    write_counterbalanced_case(&fixture, "second", "n256", 1_000, 1);

    let summary = compare_replicated_counterbalanced_report_directories([
        report_set(&fixture, "first"),
        report_set(&fixture, "second"),
    ])
    .expect("both replicated evidence blocks must compare");

    assert!(summary.passed());
    assert_eq!(summary.compared_cases(), 1);
}

#[test]
fn slowdown_must_reproduce_across_both_counterbalanced_blocks() {
    let fixture = Fixture::new();
    for replication in ["first", "second"] {
        write_counterbalanced_case(&fixture, replication, "n256", 1, 1_000);
    }

    let summary = compare_replicated_counterbalanced_report_directories([
        report_set(&fixture, "first"),
        report_set(&fixture, "second"),
    ])
    .expect("both replicated evidence blocks must compare");

    assert_eq!(summary.regressions().len(), 1);
    let regression = &summary.regressions()[0];
    assert_eq!(regression.case(), "n256");
    assert_eq!(
        regression
            .first_replication()
            .baseline_first()
            .baseline_upper_nanoseconds(),
        62
    );
    assert_eq!(
        regression
            .second_replication()
            .candidate_first()
            .candidate_lower_nanoseconds(),
        1_038
    );
}

#[test]
fn replicated_case_universes_must_match() {
    let fixture = Fixture::new();
    write_counterbalanced_case(&fixture, "first", "n256", 1, 1);
    write_counterbalanced_case(&fixture, "second", "n512", 1, 1);

    let error = compare_replicated_counterbalanced_report_directories([
        report_set(&fixture, "first"),
        report_set(&fixture, "second"),
    ])
    .expect_err("replications must contain identical case identities");

    assert!(error
        .to_string()
        .contains("second counterbalanced replication omits"));
}

#[test]
fn replicated_command_reports_unique_evidence_count() {
    let fixture = Fixture::new();
    for replication in ["first", "second"] {
        write_counterbalanced_case(&fixture, replication, "n256", 1, 1);
    }

    let output = Command::new(env!("CARGO_BIN_EXE_apollo-bench-compare"))
        .arg("compare-replicated-counterbalanced")
        .arg("--first-baseline-first-baseline-directory")
        .arg(fixture.directory("first/baseline-first/baseline"))
        .arg("--first-baseline-first-candidate-directory")
        .arg(fixture.directory("first/baseline-first/candidate"))
        .arg("--first-candidate-first-baseline-directory")
        .arg(fixture.directory("first/candidate-first/baseline"))
        .arg("--first-candidate-first-candidate-directory")
        .arg(fixture.directory("first/candidate-first/candidate"))
        .arg("--second-baseline-first-baseline-directory")
        .arg(fixture.directory("second/baseline-first/baseline"))
        .arg("--second-baseline-first-candidate-directory")
        .arg(fixture.directory("second/baseline-first/candidate"))
        .arg("--second-candidate-first-baseline-directory")
        .arg(fixture.directory("second/candidate-first/baseline"))
        .arg("--second-candidate-first-candidate-directory")
        .arg(fixture.directory("second/candidate-first/candidate"))
        .output()
        .expect("replicated counterbalanced command must execute");

    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).expect("command output must be UTF-8"),
        "replicated counterbalanced 1 cases across 1 reports; no supported regression\n"
    );
    assert!(output.stderr.is_empty());
}
