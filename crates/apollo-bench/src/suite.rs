use crate::case::BenchmarkCase;
use crate::config::BenchmarkConfig;
use crate::measurement;
use crate::report::{self, BenchmarkRecord};

/// Executes benchmark cases and retains their normalized measurements.
#[derive(Debug)]
pub struct BenchmarkSuite {
    config: BenchmarkConfig,
    records: Vec<BenchmarkRecord>,
}

impl BenchmarkSuite {
    /// Creates a suite using the supplied timing configuration.
    #[must_use]
    pub const fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            records: Vec::new(),
        }
    }

    /// Measures one production closure with the suite configuration.
    pub fn run(&mut self, case: BenchmarkCase, operation: impl FnMut()) {
        self.run_with_config(self.config, case, operation);
    }

    /// Measures one production closure with an explicit case configuration.
    pub fn run_with_config(
        &mut self,
        config: BenchmarkConfig,
        case: BenchmarkCase,
        operation: impl FnMut(),
    ) {
        let summary = measurement::measure(config, operation);
        self.records.push(BenchmarkRecord::new(case, summary));
    }

    /// Returns records in the same order that their closures executed.
    #[must_use]
    pub fn records(&self) -> &[BenchmarkRecord] {
        &self.records
    }

    /// Renders the stable CSV report for the completed suite.
    #[must_use]
    pub fn report(&self) -> String {
        report::render(&self.records)
    }

    /// Writes the completed suite report to standard output.
    pub fn emit(&self) {
        print!("{}", self.report());
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new(BenchmarkConfig::standard())
    }
}

#[cfg(test)]
mod tests {
    use super::BenchmarkSuite;
    use crate::{BenchmarkCase, BenchmarkConfig};
    use std::cell::Cell;
    use std::time::Duration;

    #[test]
    fn suite_measures_the_supplied_closure_and_reports_its_case() {
        let config =
            BenchmarkConfig::try_with_budgets(Duration::from_nanos(1), Duration::from_nanos(1))
                .expect("invariant: non-zero literal durations");
        let executions = Cell::new(0_u32);
        let mut suite = BenchmarkSuite::new(config);
        suite.run(BenchmarkCase::new("suite", "increment", 1), || {
            executions.set(executions.get() + 1);
        });

        let record = &suite.records()[0];
        assert!(executions.get() >= 100);
        assert_eq!(record.sample_count(), 100);
        assert_eq!(record.iterations_per_sample(), 1);
        assert_eq!(
            suite.report(),
            format!(
                "case,min_ns,median_ns,median_lower_ns,median_upper_ns,median_confidence_ppm,ordered_samples_ns,iterations_per_sample\nsuite/increment/1,{},{},{},{},{},{},1\n",
                record.minimum_nanoseconds(),
                record.median_nanoseconds(),
                record.median_lower_nanoseconds(),
                record.median_upper_nanoseconds(),
                record.median_confidence_parts_per_million(),
                record
                    .ordered_samples_nanoseconds()
                    .iter()
                    .map(u128::to_string)
                    .collect::<Vec<_>>()
                    .join(";")
            )
        );
    }
}
