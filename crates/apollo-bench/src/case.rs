use core::fmt::{self, Display, Formatter};

/// Identifies one measured production closure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BenchmarkCase {
    group: String,
    operation: String,
    parameter: String,
}

impl BenchmarkCase {
    /// Creates a case with a group, operation, and parameter label.
    #[must_use]
    pub fn new(
        group: impl Into<String>,
        operation: impl Into<String>,
        parameter: impl Display,
    ) -> Self {
        Self {
            group: group.into(),
            operation: operation.into(),
            parameter: parameter.to_string(),
        }
    }
}

impl Display for BenchmarkCase {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}/{}/{}",
            self.group, self.operation, self.parameter
        )
    }
}

#[cfg(test)]
mod tests {
    use super::BenchmarkCase;

    #[test]
    fn case_preserves_each_label_component() {
        assert_eq!(
            BenchmarkCase::new("fft", "forward", 256).to_string(),
            "fft/forward/256"
        );
    }
}
