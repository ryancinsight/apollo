use core::fmt::{self, Display, Formatter};
use core::num::NonZeroUsize;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::time::Duration;

const SAMPLE_COUNT: NonZeroUsize =
    NonZeroUsize::new(100).expect("invariant: Apollo's fixed benchmark sample count is non-zero");
const SMOKE_SAMPLE_COUNT: NonZeroUsize =
    NonZeroUsize::new(1).expect("invariant: smoke executes one observation");
const BENCHMARK_MODE_ENVIRONMENT: &str = "APOLLO_BENCH_MODE";
const REGRESSION_WARM_UP: Duration = Duration::from_millis(100);
const REGRESSION_MEASUREMENT: Duration = Duration::from_millis(400);
const SWEEP_WARM_UP: Duration = Duration::from_millis(15);
const SWEEP_MEASUREMENT: Duration = Duration::from_millis(75);

/// Selects full measurement or bounded executable smoke verification.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum BenchmarkMode {
    /// Execute the configured warm-up and measurement budgets.
    #[default]
    Measurement,
    /// Execute every case once without warm-up or statistical inference.
    Smoke,
}

impl BenchmarkMode {
    /// Reads the benchmark mode from `APOLLO_BENCH_MODE`.
    ///
    /// An absent variable selects [`Self::Measurement`]. The accepted explicit
    /// values are `measurement` and `smoke`.
    ///
    /// # Errors
    ///
    /// Returns an error when the variable contains any other value.
    pub fn from_environment() -> Result<Self, BenchmarkModeError> {
        match std::env::var_os(BENCHMARK_MODE_ENVIRONMENT) {
            None => Ok(Self::Measurement),
            Some(value) => Self::try_from_value(&value),
        }
    }

    /// Applies this execution mode without changing the measurement budgets.
    #[must_use]
    pub const fn apply(self, measurement: BenchmarkConfig) -> BenchmarkConfig {
        let sample_count = match self {
            Self::Measurement => SAMPLE_COUNT,
            Self::Smoke => SMOKE_SAMPLE_COUNT,
        };
        BenchmarkConfig {
            warm_up: measurement.warm_up,
            measurement: measurement.measurement,
            sample_count,
            mode: self,
        }
    }

    fn try_from_value(value: &OsStr) -> Result<Self, BenchmarkModeError> {
        if value == OsStr::new("measurement") {
            Ok(Self::Measurement)
        } else if value == OsStr::new("smoke") {
            Ok(Self::Smoke)
        } else {
            Err(BenchmarkModeError {
                value: value.to_os_string(),
            })
        }
    }
}

/// Reports an invalid `APOLLO_BENCH_MODE` value.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct BenchmarkModeError {
    value: OsString,
}

impl Display for BenchmarkModeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{BENCHMARK_MODE_ENVIRONMENT} must be `measurement` or `smoke`, got {}",
            self.value.display()
        )
    }
}

impl Error for BenchmarkModeError {}

/// Controls warm-up and sampling for one benchmark closure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BenchmarkConfig {
    warm_up: Duration,
    measurement: Duration,
    sample_count: NonZeroUsize,
    mode: BenchmarkMode,
}

impl BenchmarkConfig {
    /// Creates the 3-second warm-up, 5-second, 100-sample default contract.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            warm_up: Duration::from_secs(3),
            measurement: Duration::from_secs(5),
            sample_count: SAMPLE_COUNT,
            mode: BenchmarkMode::Measurement,
        }
    }

    /// Creates the bounded 100-millisecond warm-up and 400-millisecond
    /// measurement contract used by regression executables.
    #[must_use]
    pub const fn regression() -> Self {
        Self {
            warm_up: REGRESSION_WARM_UP,
            measurement: REGRESSION_MEASUREMENT,
            sample_count: SAMPLE_COUNT,
            mode: BenchmarkMode::Measurement,
        }
    }

    /// Creates the bounded 15-millisecond warm-up and 75-millisecond
    /// measurement contract used by wide sweeps that must complete inside a
    /// test harness budget.
    ///
    /// A sweep measuring dozens of lengths against several engines on both
    /// core classes exceeds any committed per-test timeout at the regression
    /// budgets: the per-case cost is analytic (`warm_up + measurement`), so
    /// the sweep's wall time is the case count times that constant and only
    /// the constant can shrink. The 90-millisecond case budget keeps the
    /// pinned small-sizes sweep under the 30-second harness timeout while the
    /// 100-sample estimator and both calibration paths stay untouched; a
    /// reported figure from this contract carries wider intervals, which a
    /// whole-sweep reading absorbs rather than a per-case verdict.
    #[must_use]
    pub const fn sweep() -> Self {
        Self {
            warm_up: SWEEP_WARM_UP,
            measurement: SWEEP_MEASUREMENT,
            sample_count: SAMPLE_COUNT,
            mode: BenchmarkMode::Measurement,
        }
    }

    /// Creates a configuration with the established non-zero 100-sample estimator.
    ///
    /// # Errors
    ///
    /// Returns an error if either duration is zero.
    pub fn try_with_budgets(
        warm_up: Duration,
        measurement: Duration,
    ) -> Result<Self, BenchmarkConfigError> {
        if warm_up.is_zero() {
            return Err(BenchmarkConfigError::ZeroWarmUp);
        }
        if measurement.is_zero() {
            return Err(BenchmarkConfigError::ZeroMeasurement);
        }

        Ok(Self {
            warm_up,
            measurement,
            sample_count: SAMPLE_COUNT,
            mode: BenchmarkMode::Measurement,
        })
    }

    pub(crate) const fn warm_up(self) -> Duration {
        self.warm_up
    }

    pub(crate) const fn measurement(self) -> Duration {
        self.measurement
    }

    pub(crate) const fn sample_count(self) -> NonZeroUsize {
        self.sample_count
    }

    pub(crate) const fn mode(self) -> BenchmarkMode {
        self.mode
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self::standard()
    }
}

/// Reports an invalid timing budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BenchmarkConfigError {
    /// A warm-up budget must execute at least one elapsed-time interval.
    ZeroWarmUp,
    /// A measurement budget must allocate positive sampling time.
    ZeroMeasurement,
}

impl Display for BenchmarkConfigError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroWarmUp => formatter.write_str("benchmark warm-up duration must be non-zero"),
            Self::ZeroMeasurement => {
                formatter.write_str("benchmark measurement duration must be non-zero")
            }
        }
    }
}

impl Error for BenchmarkConfigError {}

#[cfg(test)]
mod tests {
    use super::{
        BenchmarkConfig, BenchmarkConfigError, BenchmarkMode, BenchmarkModeError, SAMPLE_COUNT,
    };
    use std::ffi::{OsStr, OsString};
    use std::time::Duration;

    #[test]
    fn zero_budgets_are_distinct_typed_errors() {
        assert_eq!(
            BenchmarkConfig::try_with_budgets(Duration::ZERO, Duration::from_nanos(1)),
            Err(BenchmarkConfigError::ZeroWarmUp)
        );
        assert_eq!(
            BenchmarkConfig::try_with_budgets(Duration::from_nanos(1), Duration::ZERO),
            Err(BenchmarkConfigError::ZeroMeasurement)
        );
    }

    #[test]
    fn benchmark_mode_accepts_only_the_two_documented_values() {
        assert_eq!(
            BenchmarkMode::try_from_value(OsStr::new("measurement")),
            Ok(BenchmarkMode::Measurement)
        );
        assert_eq!(
            BenchmarkMode::try_from_value(OsStr::new("smoke")),
            Ok(BenchmarkMode::Smoke)
        );
        assert_eq!(
            BenchmarkMode::try_from_value(OsStr::new("quick")),
            Err(BenchmarkModeError {
                value: OsString::from("quick")
            })
        );
    }

    #[test]
    fn smoke_mode_executes_one_sample_without_mutating_budgets() {
        let measurement = BenchmarkConfig::standard();
        let smoke = BenchmarkMode::Smoke.apply(measurement);

        assert_eq!(smoke.warm_up(), measurement.warm_up());
        assert_eq!(smoke.measurement(), measurement.measurement());
        assert_eq!(smoke.sample_count().get(), 1);
        assert_eq!(smoke.mode(), BenchmarkMode::Smoke);
    }

    #[test]
    fn measurement_mode_restores_the_canonical_sample_count() {
        let standard = BenchmarkConfig::standard();
        let smoke = BenchmarkMode::Smoke.apply(standard);
        let measurement = BenchmarkMode::Measurement.apply(smoke);

        assert_eq!(measurement.warm_up(), standard.warm_up());
        assert_eq!(measurement.measurement(), standard.measurement());
        assert_eq!(measurement.sample_count().get(), SAMPLE_COUNT.get());
        assert_eq!(measurement.mode(), BenchmarkMode::Measurement);
    }

    #[test]
    fn regression_config_has_the_derived_per_case_budget() {
        let regression = BenchmarkConfig::regression();

        assert_eq!(regression.warm_up(), Duration::from_millis(100));
        assert_eq!(regression.measurement(), Duration::from_millis(400));
        assert_eq!(regression.sample_count().get(), 100);
    }

    #[test]
    fn sweep_config_has_the_derived_per_case_budget() {
        let sweep = BenchmarkConfig::sweep();

        assert_eq!(sweep.warm_up(), Duration::from_millis(15));
        assert_eq!(sweep.measurement(), Duration::from_millis(75));
        assert_eq!(sweep.sample_count().get(), 100);
        assert_eq!(sweep.mode(), BenchmarkMode::Measurement);
    }
}
