//! Measurement-processor selection and binding shared by the benches.
//!
//! An unpinned process measures a scheduler blend of two core classes: the
//! EcoQoS probe saw 2000 calls land on all 24 logical processors, roughly half
//! on each class, with the 90th percentile at 2.4x the median
//! (`ATLAS-APOLLO-CENSUS-UNPINNED-BLEND-2026-09-01`). Every bench therefore
//! binds its measurement thread to one processor chosen by *queried* class and
//! prints which, so the class is produced by the run and not asserted by a
//! comment.
//!
//! Selection, in order:
//! 1. [`PROCESSOR_VAR`] names a logical processor: the operator's override.
//! 2. The most performant class the platform reports, taking its second
//!    member in index order — processor 0 is the conventional interrupt and
//!    DPC target — or its only member. The same rule the pinned lib-test
//!    probes apply, so bench and probe agree on the representative.
//! 3. The platform reports no classes: the processor the thread already runs
//!    on, labelled unclassified. Absence stays typed; no class is invented.
//!
//! A platform without exact binding yields an unpinned measurement that says
//! so — hermes binds on Windows only today, so Linux runs report unpinned
//! (`HS-PROCESSOR-BINDING-LINUX-2026-09-01`). A binding that fails on a
//! platform that supports it, or lands elsewhere than requested, is an error:
//! a pinned number from an unpinned thread is the defect this module exists
//! to remove. An operator override that cannot be honoured is likewise an
//! error, because exact pinning was asked for.

use std::env::VarError;
use std::fmt;
use std::num::ParseIntError;

use hermes_simd::{ProcessorBinding, ProcessorBindingError, ProcessorIndex};
use themis::{CpuTopology, EfficiencyClass};

/// Environment variable naming the logical processor to measure on.
pub const PROCESSOR_VAR: &str = "APOLLO_BENCH_PROCESSOR";

/// Why a given processor was chosen; part of every report header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessorSelection {
    /// [`PROCESSOR_VAR`] named it.
    Environment,
    /// Second member of the most performant reported efficiency class.
    HighestClass,
    /// No classes were reported; the current processor was kept.
    CurrentUnclassified,
}

/// A measurement thread bound to one processor, restored when dropped.
///
/// Hold it for the whole timed region. Dropping it releases the binding.
pub struct MeasurementProcessor {
    _binding: Option<ProcessorBinding>,
    processor: Option<ProcessorIndex>,
    class_label: Option<&'static str>,
    selection: Option<ProcessorSelection>,
}

impl MeasurementProcessor {
    /// The bound processor, or `None` when the platform cannot bind.
    #[must_use]
    pub fn processor(&self) -> Option<ProcessorIndex> {
        self.processor
    }

    /// The queried class label of the bound processor, when reported.
    #[must_use]
    pub fn class_label(&self) -> Option<&'static str> {
        self.class_label
    }

    /// How the processor was selected, or `None` when unpinned.
    #[must_use]
    pub fn selection(&self) -> Option<ProcessorSelection> {
        self.selection
    }

    /// Whether measurements run on one fixed processor.
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        self.processor.is_some()
    }

    /// One line for the report header stating exactly what was measured.
    #[must_use]
    pub fn describe(&self) -> String {
        let (Some(processor), Some(selection)) = (self.processor, self.selection) else {
            return "exact processor binding is unsupported; measurements are unpinned".to_owned();
        };
        let how = match selection {
            ProcessorSelection::Environment => "named by APOLLO_BENCH_PROCESSOR",
            ProcessorSelection::HighestClass => "second member of the highest efficiency class",
            ProcessorSelection::CurrentUnclassified => {
                "current processor; the platform reports no efficiency classes"
            }
        };
        match self.class_label {
            Some(label) => format!(
                "bound to logical processor {} ({label} class; {how})",
                processor.get()
            ),
            None => format!("bound to logical processor {} ({how})", processor.get()),
        }
    }
}

/// Why no measurement processor could be bound.
#[derive(Debug)]
pub enum ProcessorSelectionError {
    /// [`PROCESSOR_VAR`] was set but unreadable.
    Environment(VarError),
    /// [`PROCESSOR_VAR`] did not parse as a logical processor index.
    InvalidIndex(ParseIntError),
    /// Binding, or querying the current processor, failed.
    Binding(ProcessorBindingError),
    /// The binding succeeded but the thread landed elsewhere.
    Mismatch {
        /// The processor that was requested.
        requested: ProcessorIndex,
        /// The processor the thread was observed on afterwards.
        observed: ProcessorIndex,
    },
}

impl fmt::Display for ProcessorSelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Environment(error) => write!(formatter, "read {PROCESSOR_VAR}: {error}"),
            Self::InvalidIndex(error) => {
                write!(
                    formatter,
                    "{PROCESSOR_VAR} is not a processor index: {error}"
                )
            }
            Self::Binding(error) => write!(formatter, "bind measurement processor: {error}"),
            Self::Mismatch {
                requested,
                observed,
            } => write!(
                formatter,
                "requested logical processor {} but the thread runs on {}",
                requested.get(),
                observed.get()
            ),
        }
    }
}

impl std::error::Error for ProcessorSelectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Environment(error) => Some(error),
            Self::InvalidIndex(error) => Some(error),
            Self::Binding(error) => Some(error),
            Self::Mismatch { .. } => None,
        }
    }
}

/// Bind the calling thread to the measurement processor selected as the
/// module documentation describes, and verify the binding took.
///
/// # Errors
///
/// Returns [`ProcessorSelectionError`] when [`PROCESSOR_VAR`] is unreadable or
/// malformed, when binding fails on a platform that supports it, or when the
/// thread is observed on a processor other than the one requested.
pub fn bind_measurement_processor() -> Result<MeasurementProcessor, ProcessorSelectionError> {
    let topology = CpuTopology::detect();
    let class_count = topology
        .as_ref()
        .and_then(CpuTopology::efficiency_class_count);
    let label_of = |processor: ProcessorIndex| {
        let topology = topology.as_ref()?;
        let class = topology.processor_efficiency_class(processor.get())?;
        Some(class_label(class, class_count?))
    };

    let requested = match std::env::var(PROCESSOR_VAR) {
        Ok(value) => {
            let index = value
                .parse()
                .map_err(ProcessorSelectionError::InvalidIndex)?;
            Some((ProcessorIndex::new(index), ProcessorSelection::Environment))
        }
        Err(VarError::NotPresent) => match topology.as_ref().and_then(highest_class_member) {
            Some(processor) => Some((
                ProcessorIndex::new(processor),
                ProcessorSelection::HighestClass,
            )),
            None => match ProcessorIndex::current() {
                Ok(current) => Some((current, ProcessorSelection::CurrentUnclassified)),
                Err(ProcessorBindingError::UnsupportedPlatform) => None,
                Err(error) => return Err(ProcessorSelectionError::Binding(error)),
            },
        },
        Err(error) => return Err(ProcessorSelectionError::Environment(error)),
    };

    let Some((requested, selection)) = requested else {
        return Ok(MeasurementProcessor {
            _binding: None,
            processor: None,
            class_label: None,
            selection: None,
        });
    };

    let binding = match ProcessorBinding::bind(requested) {
        Ok(binding) => binding,
        // No binding backend on this platform (hermes binds on Windows only
        // today). A selected processor that cannot be bound is typed absence:
        // report an unpinned run rather than fail it. An operator override is
        // different: exact pinning was asked for, so its absence is an error.
        Err(ProcessorBindingError::UnsupportedPlatform)
            if selection != ProcessorSelection::Environment =>
        {
            return Ok(MeasurementProcessor {
                _binding: None,
                processor: None,
                class_label: None,
                selection: None,
            });
        }
        Err(error) => return Err(ProcessorSelectionError::Binding(error)),
    };
    // Let the scheduler act on the new mask before checking where we run.
    std::thread::yield_now();
    let observed = ProcessorIndex::current().map_err(ProcessorSelectionError::Binding)?;
    if observed != requested {
        return Err(ProcessorSelectionError::Mismatch {
            requested,
            observed,
        });
    }

    Ok(MeasurementProcessor {
        _binding: Some(binding),
        class_label: label_of(observed),
        processor: Some(observed),
        selection: Some(selection),
    })
}

/// The representative of the most performant reported class: its second
/// member in index order, or its only member. `None` when the platform
/// reports no classes.
fn highest_class_member(topology: &CpuTopology) -> Option<u32> {
    let highest = topology.highest_efficiency_class()?;
    representative(topology.processors_in_efficiency_class(highest)?)
}

/// Second member in iteration order, else the first, else `None`.
///
/// Skipping the first member avoids processor 0 — the conventional interrupt
/// and DPC target — by a rule that applies to every class alike rather than
/// by special-casing one host's indices.
fn representative(mut members: impl Iterator<Item = u32>) -> Option<u32> {
    let first = members.next();
    members.next().or(first)
}

/// Spell a class for a report header, given how many classes were reported.
fn class_label(class: EfficiencyClass, class_count: usize) -> &'static str {
    let rank = usize::from(class.rank());
    if class_count <= 1 {
        "uniform"
    } else if rank + 1 == class_count {
        "performance"
    } else if rank == 0 {
        "efficiency"
    } else {
        "intermediate"
    }
}

#[cfg(test)]
mod tests {
    use super::{class_label, representative};
    use themis::EfficiencyClass;

    #[test]
    fn the_second_member_represents_a_class_with_several() {
        // The developer host's performance set; processor 0 is skipped.
        assert_eq!(
            representative([0, 1, 10, 11, 12, 13, 22, 23].into_iter()),
            Some(1)
        );
        assert_eq!(representative([2, 3].into_iter()), Some(3));
    }

    #[test]
    fn a_lone_member_represents_its_class() {
        assert_eq!(representative([7].into_iter()), Some(7));
    }

    #[test]
    fn an_empty_class_has_no_representative() {
        assert_eq!(representative(std::iter::empty()), None);
    }

    #[test]
    fn labels_follow_rank_and_class_count() {
        let lowest = EfficiencyClass::LOWEST;
        let next = EfficiencyClass::new(1);
        let top = EfficiencyClass::new(2);
        assert_eq!(class_label(lowest, 1), "uniform");
        assert_eq!(class_label(lowest, 2), "efficiency");
        assert_eq!(class_label(next, 2), "performance");
        assert_eq!(class_label(lowest, 3), "efficiency");
        assert_eq!(class_label(next, 3), "intermediate");
        assert_eq!(class_label(top, 3), "performance");
    }
}
