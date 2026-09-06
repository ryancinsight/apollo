//! Per-pass cycle attribution for planar test probes.
//!
//! Totals accumulate in fixed per-thread storage so the recorder does not
//! perturb allocation probes, including a worker's first transform. Only
//! [`take`] allocates the report consumed outside the measured kernel.

use std::cell::RefCell;

const SECTION_LABELS: [&str; 6] = [
    "deint",
    "stages1",
    "transpose",
    "stages2",
    "reint",
    "combine",
];
const SECTION_COUNT: usize = SECTION_LABELS.len();

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
                .expect("invariant: the planar driver records at most six section labels");
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
        ]
    );
    record("reint", 7);
    assert_eq!(take(), vec![("reint", 7, 1)]);
}
