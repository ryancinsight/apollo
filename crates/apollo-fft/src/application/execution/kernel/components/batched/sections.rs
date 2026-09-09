//! Per-pass cycle attribution for planar test probes.
//!
//! Totals accumulate in fixed per-thread storage so the recorder does not
//! perturb allocation probes, including a worker's first transform. Only
//! [`take`] allocates the report consumed outside the measured kernel.
//!
//! The route's sections are the top-level labels; each stage set's sweeps
//! record beneath it as `t1..t3` (time-decimated) and `f1..f3`
//! (frequency-decimated), so a stage set's cost separates into the sweeps
//! that carry a seam and those that do not. [`is_sweep`] tells the two
//! levels apart for a report that must not count a sweep twice.

use std::cell::RefCell;

const SECTION_LABELS: [&str; 12] = [
    "deint",
    "stages1",
    "transpose",
    "stages2",
    "reint",
    "combine",
    "t1",
    "t2",
    "t3",
    "f1",
    "f2",
    "f3",
];
const SECTION_COUNT: usize = SECTION_LABELS.len();

/// Whether `label` names a sweep nested inside a stage-set section.
pub(crate) fn is_sweep(label: &str) -> bool {
    super::TIME_SWEEPS.contains(&label) || super::FREQUENCY_SWEEPS.contains(&label)
}

#[derive(Clone, Copy)]
struct SectionTotal {
    label: &'static str,
    cycles: u64,
    passes: u64,
}

thread_local! {
    static TOTALS: RefCell<[Option<SectionTotal>; SECTION_COUNT]> =
        const { RefCell::new([None; SECTION_COUNT]) };
}

/// Adds one pass's cycle count to its label's running total.
pub(crate) fn record(label: &'static str, cycles: u64) {
    assert!(
        SECTION_LABELS.contains(&label),
        "unknown planar section label: {label}"
    );
    TOTALS.with_borrow_mut(|totals| {
        if let Some(entry) = totals
            .iter_mut()
            .flatten()
            .find(|entry| entry.label == label)
        {
            entry.cycles += cycles;
            entry.passes += 1;
        } else {
            let slot = totals
                .iter_mut()
                .find(|entry| entry.is_none())
                .expect("invariant: the planar driver records at most twelve section labels");
            *slot = Some(SectionTotal {
                label,
                cycles,
                passes: 1,
            });
        }
    });
}

/// Drains accumulated totals as `(label, cycles, passes)` in first-seen order.
pub(crate) fn take() -> Vec<(&'static str, u64, u64)> {
    TOTALS
        .with_borrow_mut(|totals| std::mem::replace(totals, [None; SECTION_COUNT]))
        .into_iter()
        .flatten()
        .map(
            |SectionTotal {
                 label,
                 cycles,
                 passes,
             }| (label, cycles, passes),
        )
        .collect()
}

#[test]
fn section_totals_preserve_order_and_reset() {
    for (index, label) in SECTION_LABELS.into_iter().enumerate() {
        record(
            label,
            u64::try_from(index).expect("section index fits u64") + 1,
        );
    }
    record("transpose", 11);
    assert_eq!(
        take(),
        vec![
            ("deint", 1, 1),
            ("stages1", 2, 1),
            ("transpose", 14, 2),
            ("stages2", 4, 1),
            ("reint", 5, 1),
            ("combine", 6, 1),
            ("t1", 7, 1),
            ("t2", 8, 1),
            ("t3", 9, 1),
            ("f1", 10, 1),
            ("f2", 11, 1),
            ("f3", 12, 1),
        ]
    );
    record("reint", 7);
    assert_eq!(take(), vec![("reint", 7, 1)]);
}
