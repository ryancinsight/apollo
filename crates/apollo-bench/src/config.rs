use core::fmt::{self, Display, Formatter};
use core::num::NonZeroUsize;
use std::error::Error;
use std::time::Duration;

const SAMPLE_COUNT: NonZeroUsize =
    NonZeroUsize::new(100).expect("invariant: Apollo's fixed benchmark sample count is non-zero");

/// Controls warm-up and sampling for one benchmark closure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BenchmarkConfig {
    warm_up: Duration,
    measurement: Duration,
    sample_count: NonZeroUsize,
}

impl BenchmarkConfig {
    /// Creates the 3-second warm-up, 5-second, 100-sample default contract.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            warm_up: Duration::from_secs(3),
            measurement: Duration::from_secs(5),
            sample_count: SAMPLE_COUNT,
        }
    }

    /// Creates a configuration with the established 100-sample estimator.
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
    use super::{BenchmarkConfig, BenchmarkConfigError};
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
}
