//! Allocation-free phase observation, absent from production builds.
//!
//! Only the calling thread records time. Row phases include Moirai submission,
//! execution and synchronous join; they do not report worker CPU time or
//! isolate scheduler overhead. Inactive probes never read the clock.

use std::cell::{Cell, RefCell};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub(super) enum Phase {
    FirstTranspose,
    FirstRows,
    MultiplyTranspose,
    SecondRows,
    FinalTranspose,
}

impl Phase {
    const ALL: [Self; 5] = [
        Self::FirstTranspose,
        Self::FirstRows,
        Self::MultiplyTranspose,
        Self::SecondRows,
        Self::FinalTranspose,
    ];

    const fn index(self) -> usize {
        match self {
            Self::FirstTranspose => 0,
            Self::FirstRows => 1,
            Self::MultiplyTranspose => 2,
            Self::SecondRows => 3,
            Self::FinalTranspose => 4,
        }
    }

    pub(super) fn start(self) -> Span {
        Span {
            phase: self,
            started: ACTIVE.with(Cell::get).then(Instant::now),
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Total {
    elapsed: Duration,
    calls: u64,
}

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static TOTALS: RefCell<[Total; Phase::ALL.len()]> = const {
        RefCell::new([Total { elapsed: Duration::ZERO, calls: 0 }; Phase::ALL.len()])
    };
}

pub(super) struct Span {
    phase: Phase,
    started: Option<Instant>,
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some(started) = self.started {
            let elapsed = started.elapsed();
            TOTALS.with_borrow_mut(|totals| {
                let total = &mut totals[self.phase.index()];
                total.elapsed += elapsed;
                total.calls += 1;
            });
        }
    }
}

struct Capture;

impl Drop for Capture {
    fn drop(&mut self) {
        ACTIVE.set(false);
    }
}

fn capture<R>(operation: impl FnOnce() -> R) -> (R, [Total; Phase::ALL.len()], Duration) {
    assert!(
        !ACTIVE.replace(true),
        "phase captures cannot nest on one thread"
    );
    let capture = Capture;
    TOTALS.with_borrow_mut(|totals| *totals = [Total::default(); Phase::ALL.len()]);
    let start = Instant::now();
    let result = operation();
    let elapsed = start.elapsed();
    drop(capture);
    let totals = TOTALS.with_borrow(|totals| *totals);
    (result, totals, elapsed)
}

mod tests;
