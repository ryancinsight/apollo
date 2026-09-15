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
use themis::{CoreId, CpuTopology, EfficiencyClass};

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
    census: Vec<(u32, EfficiencyClass, Option<CoreId>)>,
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
        for (processor, class, core) in &self.census {
            let chosen = self
                .cores
                .iter()
                .any(|core| core.processor.get() == *processor);
            let _ = writeln!(
                out,
                "  {}",
                census_line(*processor, *class, self.class_count, *core, chosen)
            );
        }
        out
    }

    /// Every processor the platform assigned to `class`, from the census.
    ///
    /// Windows-only with the schedule instruments that consume it: the
    /// process-affinity regime they impose has no counterpart on the other
    /// hosts, so elsewhere this and its two companions below have no caller.
    ///
    /// The probes that bind one representative core need one processor; the
    /// schedule instruments need the whole set, and this is the structured
    /// answer — the census itself, not a re-parse of the printed form.
    #[cfg(target_os = "windows")]
    pub(crate) fn processors_in_class(
        &self,
        class: EfficiencyClass,
    ) -> impl Iterator<Item = u32> + '_ {
        self.census
            .iter()
            .filter(move |(_, candidate, _)| *candidate == class)
            .map(|(processor, _, _)| *processor)
    }

    /// The class of the selected performance representative, for probes that
    /// must enumerate that class' full membership.
    #[cfg(target_os = "windows")]
    pub(crate) fn performance_class(&self) -> Option<EfficiencyClass> {
        self.performance().map(|core| core.class())
    }

    /// The most efficient class on a host reporting more than one, so the
    /// schedule instruments can enumerate it. `None` on a uniform host —
    /// there is no second set to restrict to.
    #[cfg(target_os = "windows")]
    pub(crate) fn efficiency_class(&self) -> Option<EfficiencyClass> {
        if self.class_count <= 1 {
            return None;
        }
        self.census.iter().map(|(_, class, _)| *class).min()
    }
}

/// The cached selection for this test binary.
///
/// `None` when themis reports no processor class information, in which case a
/// class-labelled comparison is not measurable and the caller skips rather
/// than emitting a table with invented headers. A homogeneous host is *not*
/// absence: it reports one class, and the instrument measures on it.
///
/// Also `None` in an unoptimized build, which is the same absence for the same
/// reason: the numbers exist but mean nothing about the shipped code. A debug
/// build measured this crate's N = 16 route at 242 ns where release reads
/// 10.6 ns, and N = 64 at 15 us where release reads 48 ns — the gap is not a
/// constant factor, so debug figures reverse verdicts rather than merely
/// scaling them. Every pinned probe reaches its host through this function, so
/// declining here is what keeps one forgotten `--release` from producing a
/// table that looks exactly like a measurement.
pub(crate) fn selected() -> Option<&'static Selection> {
    if cfg!(debug_assertions) {
        return None;
    }
    static SELECTION: OnceLock<Option<Selection>> = OnceLock::new();
    SELECTION.get_or_init(build).as_ref()
}

/// One census row: the processor, its rank and label, its core when the
/// platform reports one, and the selection mark.
///
/// The core column is what shows two arms, or an arm and a busy peer, sharing
/// one execution core on a host with SMT.
fn census_line(
    processor: u32,
    class: EfficiencyClass,
    class_count: usize,
    core: Option<CoreId>,
    chosen: bool,
) -> String {
    let core = core.map_or(String::new(), |core| format!(" core {:<3}", core.get()));
    format!(
        "cpu {processor:<2} rank {:<2} {:<12}{core}{}",
        class.rank(),
        label(class, class_count),
        if chosen { "  <- selected" } else { "" }
    )
}

/// The processor an arm binds out of one class's `members`, in index order.
///
/// With a sibling table (`core_of` answers), the first member that is not
/// processor 0, shares no execution core with processor 0 — the conventional
/// Windows interrupt and DPC target — and shares none with a core an earlier
/// arm already holds (`taken`). Without one, the second member: skipping the
/// first avoids processor 0 by a rule that applies to every arm rather than
/// by special-casing one host's indices. Either way a class with no clear
/// member still yields its second (or only) member, since one arm beats none
/// and the census prints the core it landed on.
fn choose<C: Copy + PartialEq>(
    members: impl Iterator<Item = u32>,
    core_of: impl Fn(u32) -> Option<C>,
    taken: &[C],
) -> Option<u32> {
    let members: Vec<u32> = members.collect();
    let fallback = members.get(1).or(members.first()).copied();
    let Some(interrupt_core) = core_of(0) else {
        return fallback;
    };
    members
        .iter()
        .copied()
        .find(|&processor| {
            processor != 0
                && core_of(processor)
                    .is_some_and(|core| core != interrupt_core && !taken.contains(&core))
        })
        .or(fallback)
}

fn build() -> Option<Selection> {
    let topology = CpuTopology::detect()?;
    // The absence oracle for the whole efficiency surface: `None` here means
    // the platform did not report, and every accessor below would be absent
    // with it.
    let class_count = topology.efficiency_class_count()?;
    let highest = topology.highest_efficiency_class()?;

    // The sibling table is optional in its own right: a host that reports
    // classes but no cores selects by the index rule below and prints no
    // core column.
    let core_of = |processor: u32| topology.smt().and_then(|view| view.core_of(processor));
    let census: Vec<(u32, EfficiencyClass, Option<CoreId>)> = topology
        .efficiency_classes()?
        .iter()
        .enumerate()
        .filter_map(|(processor, class)| {
            u32::try_from(processor)
                .ok()
                .map(|id| (id, *class, core_of(id)))
        })
        .collect();

    // The instrument compares the extremes of the reported range: the most
    // performant tier, and — only when the host reports more than one — the
    // most efficient. A uniform host yields a single arm rather than an
    // invented second one.
    let mut arms = vec![highest];
    if class_count > 1 {
        arms.push(EfficiencyClass::LOWEST);
    }

    let mut cores = Vec::new();
    let mut taken = Vec::new();
    for class in arms {
        let members = topology.processors_in_efficiency_class(class)?;
        if let Some(processor) = choose(members, core_of, &taken) {
            taken.extend(core_of(processor));
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
    use super::{choose, selected, Selection};
    use themis::CpuTopology;

    /// Eight processors as four two-thread cores: `[0, 0, 1, 1, 2, 2, 3, 3]`.
    fn paired_core(processor: u32) -> Option<u32> {
        (processor < 8).then_some(processor / 2)
    }

    /// With a sibling table, an arm skips processor 0 and its sibling and lands
    /// on the next core; the second arm then skips the core the first holds.
    #[test]
    fn siblings_reported_arms_avoid_the_interrupt_core_and_each_other() {
        let first = choose([0, 1, 2, 3].into_iter(), paired_core, &[]);
        assert_eq!(first, Some(2));
        let taken = [paired_core(2).expect("core of 2")];
        let second = choose([2, 3, 4, 5].into_iter(), paired_core, &taken);
        assert_eq!(second, Some(4));
    }

    /// Without a sibling table the second member is the arm, or the only one.
    #[test]
    fn without_a_sibling_table_the_second_member_is_chosen() {
        let absent = |_: u32| -> Option<u32> { None };
        assert_eq!(choose([4, 5, 6].into_iter(), absent, &[]), Some(5));
        assert_eq!(choose([7].into_iter(), absent, &[]), Some(7));
        assert_eq!(choose(core::iter::empty(), absent, &[]), None);
    }

    /// A class confined to processor 0's core still yields an arm — its second
    /// member — rather than none.
    #[test]
    fn a_class_confined_to_the_interrupt_core_still_yields_its_second_member() {
        assert_eq!(choose([0, 1].into_iter(), paired_core, &[]), Some(1));
        assert_eq!(choose([1].into_iter(), paired_core, &[]), Some(1));
    }

    /// On this host, whatever it reports: with a sibling table the selected
    /// processors hold distinct cores and none holds processor 0's; without
    /// one the selection is the index rule.
    #[test]
    fn the_host_selection_holds_distinct_cores_clear_of_the_interrupt_core() {
        let Some(selection) = selected() else {
            return;
        };
        let Some(topology) = CpuTopology::detect() else {
            return;
        };
        let Some(view) = topology.smt() else {
            return;
        };
        let cores: Vec<_> = selection
            .cores()
            .iter()
            .map(|core| {
                view.core_of(core.processor().get())
                    .expect("selected processor is in the snapshot")
            })
            .collect();
        let interrupt = view.core_of(0).expect("processor 0 is in the snapshot");
        for (index, core) in cores.iter().enumerate() {
            assert_ne!(*core, interrupt, "arm {index} shares processor 0's core");
            assert!(
                !cores[..index].contains(core),
                "arm {index} shares a core with an earlier arm"
            );
        }
    }

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
            let reported = selection
                .census
                .iter()
                .find(|(processor, _, _)| *processor == core.processor().get())
                .and_then(|(_, _, reported)| *reported);
            let expected = super::census_line(
                core.processor().get(),
                core.class(),
                selection.class_count,
                reported,
                true,
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
