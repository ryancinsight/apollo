//! Measurement processor selection over the themis topology.
//!
//! Core class is queried, never assumed (ADR 0043). The query itself belongs
//! to themis, which owns `CpuTopology` and reports a per-processor
//! [`EfficiencyClass`] with the absence discipline this instrument needs:
//! `None` when the platform said nothing, `Some(1)` class for a homogeneous
//! host, `Some(n > 1)` for a hybrid one. What remains here is apollo's own
//! measurement policy — which processor represents each class, and how the
//! choice is printed beside the numbers it produced — the input hermes ADR
//! 021 explicitly leaves to the caller ("Processor selection remains an Apollo
//! measurement-policy input; Hermes does not choose a core class").
//!
//! The instruments exist because they previously asserted a property of the
//! host they never measured:
//!
//! ```text
//! // Logical 0..8 are P-cores and 8..24 E-cores on the Core Ultra 9 285K.
//! for cpu in [2u32, 12] { ... let core = if landed < 8 { "P" } else { "E" };
//! ```
//!
//! The comment is false on the host it names: the performance set there is
//! `{0, 1, 10, 11, 12, 13, 22, 23}` — mask `0xc03c03` — so cpu 2 is an
//! efficiency core and cpu 12 a performance core, and every table comparing
//! them carried inverted headers. Nothing here restates that mask; the census
//! is printed by the run that produced the numbers.

use hermes_simd::ProcessorIndex;
use std::fmt::Write as _;
use std::sync::OnceLock;
use themis::{CpuTopology, EfficiencyClass};

/// Spells out a class rank, so a transcribed table header carries its meaning.
///
/// themis ranks are dense ordinals within one snapshot, not a two-valued flag:
/// the highest rank is the performance tier and rank 0 the most efficient one.
/// A host reporting a single class is uniform — a reported result, distinct
/// from absence — and is labelled as such rather than as a tier it has no
/// counterpart for.
fn label(class: EfficiencyClass, class_count: usize) -> &'static str {
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

/// A processor and the class the platform assigns it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeasurementCore {
    processor: ProcessorIndex,
    class: EfficiencyClass,
    label: &'static str,
}

impl MeasurementCore {
    /// The processor to bind.
    pub(crate) const fn processor(self) -> ProcessorIndex {
        self.processor
    }

    /// The queried class of [`Self::processor`].
    pub(crate) const fn class(self) -> EfficiencyClass {
        self.class
    }

    /// The spelled-out class label for a table header.
    pub(crate) const fn label(self) -> &'static str {
        self.label
    }
}

/// The processors the pinned probes measure on, at most one per selected class.
#[derive(Debug)]
pub(crate) struct Selection {
    cores: Vec<MeasurementCore>,
    census: Vec<(u32, EfficiencyClass)>,
    class_count: usize,
}

impl Selection {
    /// The chosen processors, most performant class first.
    pub(crate) fn cores(&self) -> &[MeasurementCore] {
        &self.cores
    }

    /// The chosen performance core, for probes that measure on one core.
    ///
    /// Single-core probes previously pinned to a literal `2`, an efficiency
    /// core on this host — including one whose name asserted the opposite.
    pub(crate) fn performance(&self) -> Option<MeasurementCore> {
        let highest = self.cores.iter().map(|core| core.class).max()?;
        self.cores
            .iter()
            .copied()
            .find(|core| core.class == highest)
    }

    /// The queried class of every processor, marking the ones selected.
    ///
    /// Printed by every probe ahead of its table, so the axis is produced by
    /// the run that produced the numbers instead of by a comment that can rot.
    pub(crate) fn describe(&self) -> String {
        let mut out = format!(
            "processor class census (themis, {} class{}):\n",
            self.class_count,
            if self.class_count == 1 { "" } else { "es" }
        );
        for (processor, class) in &self.census {
            let chosen = self
                .cores
                .iter()
                .any(|core| core.processor.get() == *processor);
            let _ = writeln!(
                out,
                "  cpu {processor:<2} rank {:<2} {:<12}{}",
                class.rank(),
                label(*class, self.class_count),
                if chosen { "  <- selected" } else { "" }
            );
        }
        out
    }
}

/// The cached selection for this test binary.
///
/// `None` when themis reports no processor class information, in which case a
/// class-labelled comparison is not measurable and the caller skips rather
/// than emitting a table with invented headers. A homogeneous host is *not*
/// absence: it reports one class, and the instrument measures on it.
pub(crate) fn selected() -> Option<&'static Selection> {
    static SELECTION: OnceLock<Option<Selection>> = OnceLock::new();
    SELECTION.get_or_init(build).as_ref()
}

fn build() -> Option<Selection> {
    let topology = CpuTopology::detect()?;
    // The absence oracle for the whole efficiency surface: `None` here means
    // the platform did not report, and every accessor below would be absent
    // with it.
    let class_count = topology.efficiency_class_count()?;
    let highest = topology.highest_efficiency_class()?;

    let census: Vec<(u32, EfficiencyClass)> = topology
        .efficiency_classes()?
        .iter()
        .enumerate()
        .filter_map(|(processor, class)| u32::try_from(processor).ok().map(|id| (id, *class)))
        .collect();

    // The instrument compares the extremes of the reported range: the most
    // performant tier, and — only when the host reports more than one — the
    // most efficient. A uniform host yields a single arm rather than an
    // invented second one.
    let mut arms = vec![highest];
    if class_count > 1 {
        arms.push(EfficiencyClass::LOWEST);
    }

    // The second processor of each class, in index order. Skipping the first
    // avoids processor 0 — the conventional Windows interrupt and DPC target —
    // by a rule that applies to every arm rather than by special-casing one
    // host's indices. Falls back to the only member when a class has one.
    let mut cores = Vec::new();
    for class in arms {
        let mut members = topology.processors_in_efficiency_class(class)?;
        let first = members.next();
        if let Some(processor) = members.next().or(first) {
            cores.push(MeasurementCore {
                processor: ProcessorIndex::new(processor),
                class,
                label: label(class, class_count),
            });
        }
    }

    Some(Selection {
        cores,
        census,
        class_count,
    })
}

#[cfg(test)]
mod tests {
    use super::{selected, Selection};

    /// The selection must be drawn from the queried census, never assumed: a
    /// selected processor carries the class the census reports for it.
    #[test]
    fn selection_is_drawn_from_the_queried_census() {
        let Some(selection) = selected() else {
            return;
        };
        assert!(
            selection.cores().len() <= 2,
            "at most one representative per compared class"
        );
        let census = selection.describe();
        for core in selection.cores() {
            let expected = format!(
                "cpu {:<2} rank {:<2} {:<12}  <- selected",
                core.processor().get(),
                core.class().rank(),
                core.label()
            );
            assert!(
                census.contains(&expected),
                "selected cpu {} must appear in the census as {}",
                core.processor().get(),
                core.label()
            );
        }
    }

    /// The performance representative is the most performant class selected,
    /// so a single-core probe never lands on an efficiency core.
    #[test]
    fn the_performance_core_carries_the_highest_selected_class() {
        let Some(selection) = selected() else {
            return;
        };
        let Some(performance) = Selection::performance(selection) else {
            return;
        };
        for core in selection.cores() {
            assert!(
                core.class() <= performance.class(),
                "cpu {} ranks above the selected performance core",
                core.processor().get()
            );
        }
    }
}
