use apollo_bench::CounterbalancedReportSet;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

const HEADER: &str = "case,min_ns,median_ns,median_lower_ns,median_upper_ns,median_confidence_ppm,ordered_samples_ns,iterations_per_sample\n";
static FIXTURE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(super) struct Fixture {
    root: PathBuf,
}

impl Fixture {
    pub(super) fn new() -> Self {
        let sequence = FIXTURE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "apollo-bench-comparison-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("fixture root creation must succeed");
        Self { root }
    }

    pub(super) fn write(&self, relative: &str, rows: &[String]) {
        let path = self.root.join(relative);
        fs::create_dir_all(
            path.parent()
                .expect("invariant: every fixture report has a parent"),
        )
        .expect("fixture report parent creation must succeed");
        fs::write(path, format!("{HEADER}{}", rows.concat()))
            .expect("fixture report write must succeed");
    }

    pub(super) fn directory(&self, relative: &str) -> PathBuf {
        self.root.join(relative)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.root) {
            if std::thread::panicking() {
                return;
            }
            panic!("fixture cleanup failed at {}: {error}", self.root.display());
        }
    }
}

pub(super) fn standard_row(case: &str, start: u128) -> String {
    report_row(case, &(start..start + 100).collect::<Vec<_>>())
}

pub(super) fn report_row(case: &str, samples: &[u128]) -> String {
    let (interval_rank, confidence) = match samples.len() {
        6 => (1, 968_750),
        100 => (40, 964_799),
        count => panic!("fixture does not define the one-case interval for {count} samples"),
    };
    let minimum = samples[0];
    let median = samples[(samples.len() - 1) / 2]
        + (samples[samples.len() / 2] - samples[(samples.len() - 1) / 2]) / 2;
    let lower = samples[interval_rank - 1];
    let upper = samples[samples.len() - interval_rank];
    let samples = samples
        .iter()
        .map(u128::to_string)
        .collect::<Vec<_>>()
        .join(";");
    format!(
        "{},{minimum},{median},{lower},{upper},{confidence},{samples},4\n",
        csv_field(case)
    )
}

pub(super) fn write_counterbalanced_case(
    fixture: &Fixture,
    replication: &str,
    case: &str,
    baseline_start: u128,
    candidate_start: u128,
) {
    for order in ["baseline-first", "candidate-first"] {
        fixture.write(
            &format!("{replication}/{order}/baseline/kernel.csv"),
            &[standard_row(case, baseline_start)],
        );
        fixture.write(
            &format!("{replication}/{order}/candidate/kernel.csv"),
            &[standard_row(case, candidate_start)],
        );
    }
}

pub(super) fn report_set(fixture: &Fixture, replication: &str) -> CounterbalancedReportSet {
    CounterbalancedReportSet::new(
        fixture.directory(&format!("{replication}/baseline-first/baseline")),
        fixture.directory(&format!("{replication}/baseline-first/candidate")),
        fixture.directory(&format!("{replication}/candidate-first/baseline")),
        fixture.directory(&format!("{replication}/candidate-first/candidate")),
    )
}

fn csv_field(value: &str) -> String {
    if !value.contains([',', '"', '\n', '\r']) {
        return value.to_owned();
    }
    format!("\"{}\"", value.replace('"', "\"\""))
}
