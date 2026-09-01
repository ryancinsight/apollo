//! Measurement processor selection by queried core class.
//!
//! **Interim.** The canonical home for processor topology is themis, which
//! owns `CpuTopology` (NUMA nodes, cache levels) and already queries
//! `GetLogicalProcessorInformationEx`. Core efficiency class is the dimension
//! it does not yet expose, which is why hardcoded core indices have been
//! hand-rolled per repository. This module is `cfg(test)` instrument support
//! only; it is deleted, not adapted, once `themis-topology` lands
//! (`ATLAS-APOLLO-CORE-CLASS-UPSTREAM-2026-09-01`, ADR 0043).
//!
//! It exists because the pinned probes previously asserted a property of the
//! host they never measured:
//!
//! ```text
//! // Logical 0..8 are P-cores and 8..24 E-cores on the Core Ultra 9 285K.
//! for cpu in [2u32, 12] { ... let core = if landed < 8 { "P" } else { "E" };
//! ```
//!
//! The comment is false on the host it names. Windows reports the performance
//! set as `{0, 1, 10, 11, 12, 13, 22, 23}` — mask `0xc03c03` — so cpu 2 is an
//! efficiency core and cpu 12 is a performance core. Both probe arms were
//! mislabelled, and every table comparing them carried inverted headers.

use hermes_simd::ProcessorIndex;
use std::fmt::Write as _;
use std::sync::OnceLock;

/// Hybrid core class, as reported by the operating system.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CoreClass {
    /// Lower `EfficiencyClass`: greater efficiency, less performance.
    Efficiency,
    /// Higher `EfficiencyClass`: greater performance, less efficiency.
    Performance,
}

impl CoreClass {
    /// Spelled out, so a transcribed table header carries its own meaning.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Performance => "performance",
            Self::Efficiency => "efficiency",
        }
    }
}

/// A processor and the class the operating system assigns it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeasurementCore {
    processor: ProcessorIndex,
    class: CoreClass,
}

impl MeasurementCore {
    /// The processor to bind.
    pub(crate) const fn processor(self) -> ProcessorIndex {
        self.processor
    }

    /// The queried class of [`Self::processor`].
    pub(crate) const fn class(self) -> CoreClass {
        self.class
    }
}

/// The processors the pinned probes measure on, at most one per class.
#[derive(Debug)]
pub(crate) struct Selection {
    cores: Vec<MeasurementCore>,
    census: Vec<(u32, CoreClass)>,
}

impl Selection {
    /// The chosen processors, performance class first.
    pub(crate) fn cores(&self) -> &[MeasurementCore] {
        &self.cores
    }

    /// The chosen performance core, for probes that measure on one core.
    ///
    /// Single-core probes previously pinned to a literal `2`, an efficiency
    /// core — including one whose name asserted the opposite.
    pub(crate) fn performance(&self) -> Option<MeasurementCore> {
        self.cores
            .iter()
            .copied()
            .find(|core| core.class == CoreClass::Performance)
    }

    /// The queried class of every processor, marking the ones selected.
    ///
    /// Printed by every probe ahead of its table, so the axis is produced by
    /// the run that produced the numbers instead of by a comment that can rot.
    pub(crate) fn describe(&self) -> String {
        let mut out = String::from("processor class census (queried):\n");
        for (processor, class) in &self.census {
            let chosen = self
                .cores
                .iter()
                .any(|core| core.processor.get() == *processor);
            let _ = writeln!(
                out,
                "  cpu {processor:<2} {:<11}{}",
                class.label(),
                if chosen { "  <- selected" } else { "" }
            );
        }
        out
    }
}

/// The cached selection for this test binary.
///
/// `None` when the platform reports no processor class information, in which
/// case a class-labelled comparison is not measurable and the caller skips
/// rather than emitting a table with invented headers.
pub(crate) fn selected() -> Option<&'static Selection> {
    static SELECTION: OnceLock<Option<Selection>> = OnceLock::new();
    SELECTION.get_or_init(build).as_ref()
}

fn build() -> Option<Selection> {
    let census = platform::processor_classes()?;

    // The second processor of each class, in index order. Skipping the first
    // avoids processor 0 — the conventional Windows interrupt and DPC target —
    // by a rule that applies to both arms rather than by special-casing one
    // host's indices. Falls back to the only member when a class has one.
    let mut cores = Vec::new();
    for class in [CoreClass::Performance, CoreClass::Efficiency] {
        let mut members = census
            .iter()
            .filter(|(_, candidate)| *candidate == class)
            .map(|&(processor, _)| processor);
        let first = members.next();
        if let Some(processor) = members.next().or(first) {
            cores.push(MeasurementCore {
                processor: ProcessorIndex::new(processor),
                class,
            });
        }
    }

    Some(Selection { cores, census })
}

#[cfg(windows)]
mod platform {
    use super::CoreClass;

    /// `RelationProcessorCore`, the only relationship this query needs.
    const RELATION_PROCESSOR_CORE: u32 = 0;
    /// `ERROR_INSUFFICIENT_BUFFER`, the expected failure of the sizing call.
    const ERROR_INSUFFICIENT_BUFFER: u32 = 122;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct GroupAffinity {
        mask: usize,
        group: u16,
        reserved: [u16; 3],
    }

    #[repr(C)]
    struct ProcessorRelationship {
        flags: u8,
        efficiency_class: u8,
        reserved: [u8; 20],
        group_count: u16,
        group_mask: [GroupAffinity; 1],
    }

    #[repr(C)]
    struct LogicalProcessorInformationEx {
        relationship: u32,
        size: u32,
        processor: ProcessorRelationship,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetLogicalProcessorInformationEx(
            relationship: u32,
            buffer: *mut u8,
            returned_length: *mut u32,
        ) -> i32;
        fn GetLastError() -> u32;
    }

    /// Every logical processor paired with its queried class, index-ordered.
    ///
    /// `EfficiencyClass` is an ordinal, not a two-valued flag: the maximum
    /// observed value is the performance tier and everything below it is an
    /// efficiency tier. A non-hybrid host reports one value, so every core is
    /// reported as performance class and the caller finds no efficiency arm.
    pub(super) fn processor_classes() -> Option<Vec<(u32, CoreClass)>> {
        let mut len: u32 = 0;
        // SAFETY: the documented sizing call — a null buffer with a valid
        // out-pointer for the required length. It is expected to fail with
        // ERROR_INSUFFICIENT_BUFFER after writing `len`; no buffer is read.
        let sized = unsafe {
            GetLogicalProcessorInformationEx(
                RELATION_PROCESSOR_CORE,
                core::ptr::null_mut(),
                &raw mut len,
            )
        };
        // SAFETY: no preconditions; reads this thread's last-error slot, which
        // the call above just set.
        if sized != 0 || unsafe { GetLastError() } != ERROR_INSUFFICIENT_BUFFER || len == 0 {
            return None;
        }

        // Records are read out of this buffer as `LogicalProcessorInformationEx`,
        // whose alignment is 8 (from `GroupAffinity`'s `usize`), so the
        // allocation is made through a correctly aligned element type rather
        // than a `Vec<u8>`.
        let words = (len as usize).div_ceil(size_of::<u64>());
        let mut buffer = vec![0u64; words];
        let mut written = len;
        // SAFETY: `buffer` is writable for at least `len` bytes and `written`
        // is a valid out-pointer. The call fills the buffer with a packed
        // sequence of variable-length records and sets `written` to the bytes
        // it used.
        let filled = unsafe {
            GetLogicalProcessorInformationEx(
                RELATION_PROCESSOR_CORE,
                buffer.as_mut_ptr().cast::<u8>(),
                &raw mut written,
            )
        };
        if filled == 0 {
            return None;
        }

        let bytes = (written as usize).min(words * size_of::<u64>());
        let base = buffer.as_ptr().cast::<u8>();
        let mut offset = 0usize;
        let mut classes: Vec<(u32, u8)> = Vec::new();

        while offset + size_of::<LogicalProcessorInformationEx>() <= bytes {
            // SAFETY: `offset + size_of::<_>() <= bytes` bounds a whole header
            // inside the region the OS filled, and the pointer is 8-byte
            // aligned because `base` comes from a `u64` allocation and every
            // record size is a multiple of 8.
            let record = unsafe { &*base.add(offset).cast::<LogicalProcessorInformationEx>() };
            let size = record.size as usize;
            if size == 0 || offset + size > bytes {
                break;
            }
            if record.relationship == RELATION_PROCESSOR_CORE {
                let efficiency_class = record.processor.efficiency_class;
                let group_count = record.processor.group_count as usize;
                let masks =
                    core::ptr::addr_of!(record.processor.group_mask).cast::<GroupAffinity>();
                for index in 0..group_count {
                    // SAFETY: `group_count` is the OS-reported length of the
                    // trailing `group_mask` array and `size`, checked above,
                    // covers the whole record including it.
                    let affinity = unsafe { *masks.add(index) };
                    let group = u32::from(affinity.group);
                    for bit in 0..usize::BITS {
                        if affinity.mask & (1usize << bit) != 0 {
                            classes.push((group * 64 + bit, efficiency_class));
                        }
                    }
                }
            }
            offset += size;
        }

        let peak = classes.iter().map(|&(_, class)| class).max()?;
        classes.sort_unstable_by_key(|&(processor, _)| processor);
        Some(
            classes
                .into_iter()
                .map(|(processor, class)| {
                    let class = if class == peak {
                        CoreClass::Performance
                    } else {
                        CoreClass::Efficiency
                    };
                    (processor, class)
                })
                .collect(),
        )
    }
}

#[cfg(not(windows))]
mod platform {
    use super::CoreClass;

    /// Processor class discovery is Windows-only, matching the pinned probes.
    pub(super) fn processor_classes() -> Option<Vec<(u32, CoreClass)>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::selected;

    /// The selection must be drawn from the queried census, never assumed: a
    /// selected processor carries the class the census reports for it.
    #[test]
    fn selection_is_drawn_from_the_queried_census() {
        let Some(selection) = selected() else {
            return;
        };
        assert!(
            selection.cores().len() <= 2,
            "at most one representative per class"
        );
        let census = selection.describe();
        for core in selection.cores() {
            let expected = format!(
                "cpu {:<2} {:<11}  <- selected",
                core.processor().get(),
                core.class().label()
            );
            assert!(
                census.contains(&expected),
                "selected cpu {} must appear in the census as {}",
                core.processor().get(),
                core.class().label()
            );
        }
    }
}
