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
            started: (ACTIVE.get() == Some(Mode::Timing)).then(Instant::now),
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Total {
    elapsed: Duration,
    calls: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Timing,
    Buffers,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Buffers {
    data_address: usize,
    scratch_address: usize,
    data_bytes: usize,
    scratch_bytes: usize,
    rows: usize,
    columns: usize,
    data_row_bytes: usize,
    scratch_row_bytes: usize,
}

thread_local! {
    static ACTIVE: Cell<Option<Mode>> = const { Cell::new(None) };
    static BUFFERS: Cell<Option<Buffers>> = const { Cell::new(None) };
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

impl Capture {
    fn start(mode: Mode) -> Self {
        assert!(ACTIVE.get().is_none(), "captures cannot nest on one thread");
        ACTIVE.set(Some(mode));
        Self
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        ACTIVE.set(None);
    }
}

fn capture<R>(operation: impl FnOnce() -> R) -> (R, [Total; Phase::ALL.len()], Duration) {
    let capture = Capture::start(Mode::Timing);
    TOTALS.with_borrow_mut(|totals| *totals = [Total::default(); Phase::ALL.len()]);
    let start = Instant::now();
    let result = operation();
    let elapsed = start.elapsed();
    drop(capture);
    let totals = TOTALS.with_borrow(|totals| *totals);
    (result, totals, elapsed)
}

// Geometry is captured only during the unmeasured prewarm. Addresses describe
// these borrowed slices; they carry no ownership or physical-cache mapping.
pub(super) fn observe_buffers<T>(data: &[T], scratch: &[T], rows: usize, columns: usize) {
    if ACTIVE.get() != Some(Mode::Buffers) {
        return;
    }
    let observed = Buffers {
        data_address: data.as_ptr().addr(),
        scratch_address: scratch.as_ptr().addr(),
        data_bytes: size_of_val(data),
        scratch_bytes: size_of_val(scratch),
        rows,
        columns,
        data_row_bytes: columns
            .checked_mul(size_of::<T>())
            .expect("row byte extent fits storage"),
        scratch_row_bytes: rows
            .checked_mul(size_of::<T>())
            .expect("row byte extent fits storage"),
    };
    assert!(
        BUFFERS.replace(Some(observed)).is_none(),
        "one decomposition must supply buffer geometry"
    );
}

fn capture_buffers<R>(operation: impl FnOnce() -> R) -> (R, Buffers) {
    let capture = Capture::start(Mode::Buffers);
    BUFFERS.set(None);
    let result = operation();
    drop(capture);
    (
        result,
        BUFFERS
            .take()
            .expect("one decomposition must supply buffer geometry"),
    )
}

mod tests;
