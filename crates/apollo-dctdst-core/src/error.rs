use core::fmt;

/// Failure to construct a direct transform plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanError {
    /// A transform requires at least one sample.
    EmptyLength,
}

impl fmt::Display for PlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLength => formatter.write_str("transform length must be greater than zero"),
        }
    }
}

impl core::error::Error for PlanError {}
