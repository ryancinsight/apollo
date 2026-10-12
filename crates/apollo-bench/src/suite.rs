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

    /// Measures two closures as an interleaved pair sharing one window.
    ///
    /// Sequential per-arm windows alias with package-turbo oscillation,
    /// so cross-arm ratios swing with turbo state instead of code (#639).
    /// The pair warms up together and alternates timed sample blocks, so
    /// both records' samples draw from the same turbo trajectory and
    /// their medians are directly comparable. Records push in (a, b)
    /// order; the pairing is a sampling property, not a schema change.
    pub fn run_paired_with_config(
        &mut self,
        config: BenchmarkConfig,
        case_a: BenchmarkCase,
        operation_a: impl FnMut(),
        case_b: BenchmarkCase,
        operation_b: impl FnMut(),
    ) {
        let (summary_a, summary_b) = measurement::measure_paired(config, operation_a, operation_b);
        self.records.push(BenchmarkRecord::new(case_a, summary_a));
        self.records.push(BenchmarkRecord::new(case_b, summary_b));
    }

    /// Measures one closure over inputs prepared outside the timed region.
    ///
    /// Use this wherever the operation is in-place and each iteration needs a
    /// fresh input: [`BenchmarkSuite::run`] would time the reset alongside the
    /// operation, and when two arms reset buffers of different element widths
    /// that difference is charged to the operation and distorts the ratio
    /// between them.
    pub fn run_batched<T>(
        &mut self,
        case: BenchmarkCase,
        setup: impl FnMut() -> T,
        operation: impl FnMut(&mut T),
    ) {
        let summary = measurement::measure_batched(self.config, setup, operation);
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
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new(BenchmarkConfig::standard())
    }
}

#[cfg(test)]
mod tests {
    use super::BenchmarkSuite;
    use crate::{BenchmarkCase, BenchmarkConfig, BenchmarkMode};
    use std::cell::Cell;
    use std::time::Duration;

    #[test]
    fn batched_run_excludes_setup_from_the_reported_measurement() {
        let config =
            BenchmarkConfig::try_with_budgets(Duration::from_nanos(1), Duration::from_nanos(1))
                .expect("invariant: non-zero literal durations");
        let setups = Cell::new(0_u32);
        let operations = Cell::new(0_u32);
        let mut suite = BenchmarkSuite::new(config);

        suite.run_batched(
            BenchmarkCase::new("core", "batched", 8),
            || {
                setups.set(setups.get() + 1);
                vec![0_u64; 8]
            },
            |input| {
                operations.set(operations.get() + 1);
                input[0] = input[0].wrapping_add(1);
            },
        );

        assert_eq!(suite.records().len(), 1, "one case yields one record");
        assert!(
            setups.get() >= operations.get(),
            "every timed operation must have had its input built first:              {} setups against {} operations",
            setups.get(),
            operations.get()
        );
        assert!(
            operations.get() > 0,
            "the batched operation must actually execute"
        );
    }

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
                "case,min_ps,median_ps,median_lower_ps,median_upper_ps,median_confidence_ppm,ordered_samples_ps,iterations_per_sample\nsuite/increment/1,{},{},{},{},{},{},1\n",
                record.minimum_picoseconds(),
                record.median_picoseconds(),
                record.median_lower_picoseconds(),
                record.median_upper_picoseconds(),
                record.median_confidence_parts_per_million(),
                record
                    .ordered_samples_picoseconds()
                    .iter()
                    .map(u128::to_string)
                    .collect::<Vec<_>>()
                    .join(";")
            )
        );
    }

    #[test]
    fn paired_run_alternates_arms_with_a_counterbalanced_lead() {
        use std::cell::RefCell;
        let config =
            BenchmarkConfig::try_with_budgets(Duration::from_nanos(1), Duration::from_nanos(1))
                .expect("invariant: non-zero literal durations");
        let log = RefCell::new(Vec::new());
        let mut suite = BenchmarkSuite::new(config);
        suite.run_paired_with_config(
            config,
            BenchmarkCase::new("suite", "arm_a", 1),
            || log.borrow_mut().push(0_u8),
            BenchmarkCase::new("suite", "arm_b", 1),
            || log.borrow_mut().push(1_u8),
        );

        assert_eq!(suite.records().len(), 2, "one pair yields two records");
        for record in suite.records() {
            assert_eq!(record.sample_count(), 100);
            assert_eq!(record.iterations_per_sample(), 1);
        }
        let log = log.borrow();
        let mut count_a = 0usize;
        for &mark in log.iter() {
            if mark == 0 {
                count_a += 1;
            }
        }
        let count_b = log.len() - count_a;
        assert_eq!(
            count_a, count_b,
            "warm-up pairs plus one block per arm per sample keep the arms equal"
        );
        // Warm-up length is timing-dependent, but the 100 timed samples at
        // one iteration per block are exactly 200 trailing marks in A,B,B,A
        // counterbalanced order.
        let tail = &log[log.len() - 200..];
        let expected: Vec<u8> = (0..100)
            .flat_map(|index| {
                if index % 2 == 0 {
                    [0_u8, 1_u8]
                } else {
                    [1_u8, 0_u8]
                }
            })
            .collect();
        assert_eq!(tail, expected.as_slice());
    }

    #[test]
    fn paired_smoke_executes_each_arm_once() {
        let config = BenchmarkMode::Smoke.apply(BenchmarkConfig::standard());
        let executions_a = Cell::new(0_u32);
        let executions_b = Cell::new(0_u32);
        let mut suite = BenchmarkSuite::new(config);
        suite.run_paired_with_config(
            config,
            BenchmarkCase::new("suite", "smoke_a", 1),
            || executions_a.set(executions_a.get() + 1),
            BenchmarkCase::new("suite", "smoke_b", 1),
            || executions_b.set(executions_b.get() + 1),
        );

        assert_eq!(suite.records().len(), 2);
        assert_eq!(executions_a.get(), 1);
        assert_eq!(executions_b.get(), 1);
        for record in suite.records() {
            assert_eq!(record.sample_count(), 1);
        }
    }

    #[test]
    fn smoke_executes_each_production_closure_once() {
        let config = BenchmarkMode::Smoke.apply(BenchmarkConfig::standard());
        let executions = Cell::new(0_u32);
        let mut suite = BenchmarkSuite::new(config);
        suite.run(BenchmarkCase::new("suite", "smoke", 1), || {
            executions.set(executions.get() + 1);
        });

        let record = &suite.records()[0];
        assert_eq!(executions.get(), 1);
        assert_eq!(record.sample_count(), 1);
        assert_eq!(record.iterations_per_sample(), 1);
        assert_eq!(record.median_confidence_parts_per_million(), 0);
    }
}
