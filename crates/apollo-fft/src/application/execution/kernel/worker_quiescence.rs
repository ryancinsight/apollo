//! Deterministic worker-idle and allocation observation for retained-footprint probes.

use std::cell::Cell;
use std::sync::{Condvar, Mutex, OnceLock};
use std::thread::ThreadId;
use std::time::Duration;

const MAX_OBSERVED_WORKERS: usize = 128;
const QUIESCENCE_TIMEOUT: Duration = Duration::from_secs(5);

struct Observation {
    ids: [Option<ThreadId>; MAX_OBSERVED_WORKERS],
    required_phase: [u64; MAX_OBSERVED_WORKERS],
    seen_phase: [u64; MAX_OBSERVED_WORKERS],
    post_release_capacity: [usize; MAX_OBSERVED_WORKERS],
    len: usize,
    phase: u64,
}

impl Observation {
    const fn new() -> Self {
        Self {
            ids: [None; MAX_OBSERVED_WORKERS],
            required_phase: [0; MAX_OBSERVED_WORKERS],
            seen_phase: [0; MAX_OBSERVED_WORKERS],
            post_release_capacity: [0; MAX_OBSERVED_WORKERS],
            len: 0,
            phase: 0,
        }
    }

    fn index_of(&self, id: ThreadId) -> Option<usize> {
        self.ids[..self.len]
            .iter()
            .position(|candidate| *candidate == Some(id))
    }

    fn index_or_insert(&mut self, id: ThreadId) -> usize {
        if let Some(index) = self.index_of(id) {
            return index;
        }
        assert!(
            self.len < MAX_OBSERVED_WORKERS,
            "worker-id observation exceeded its fixed capacity"
        );
        let index = self.len;
        self.ids[index] = Some(id);
        self.len += 1;
        index
    }

    fn phase_complete(&self) -> bool {
        let (required_workers, complete_workers) = self.phase_progress();
        required_workers != 0 && required_workers == complete_workers
    }

    fn phase_progress(&self) -> (usize, usize) {
        let mut required_workers = 0;
        let mut complete_workers = 0;
        for index in 0..self.len {
            if self.required_phase[index] == self.phase {
                required_workers += 1;
                if self.seen_phase[index] >= self.phase {
                    complete_workers += 1;
                }
            }
        }
        (required_workers, complete_workers)
    }
}

static OBSERVATION: OnceLock<(Mutex<Observation>, Condvar)> = OnceLock::new();

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn observation() -> &'static (Mutex<Observation>, Condvar) {
    OBSERVATION.get_or_init(|| (Mutex::new(Observation::new()), Condvar::new()))
}

fn is_moirai_worker() -> bool {
    std::thread::current()
        .name()
        .is_some_and(|name| name.starts_with("moirai-worker-"))
}

pub(crate) fn record_allocation() {
    ALLOCATIONS.with(|count| count.set(count.get().saturating_add(1)));
}

pub(crate) fn allocation_count() -> usize {
    ALLOCATIONS.with(Cell::get)
}

pub(crate) fn arm() {
    let (lock, _) = observation();
    let mut guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    *guard = Observation::new();
}

pub(crate) fn record_worker() {
    if !is_moirai_worker() {
        return;
    }
    let id = std::thread::current().id();
    let (lock, _) = observation();
    let mut guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let index = guard.index_or_insert(id);
    guard.required_phase[index] = guard.phase;
}

pub(crate) fn begin_phase() {
    let (lock, _) = observation();
    let mut guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.phase = guard
        .phase
        .checked_add(1)
        .expect("invariant: one retained-footprint probe phase fits u64");
}

pub(crate) fn observe_idle() {
    if !is_moirai_worker() {
        return;
    }
    let id = std::thread::current().id();
    let post_release_capacity = crate::thread_local_scratch_capacity();
    let (lock, condvar) = observation();
    let mut guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(index) = guard.index_of(id) else {
        return;
    };
    if guard.required_phase[index] == guard.phase {
        guard.seen_phase[index] = guard.phase;
        guard.post_release_capacity[index] = post_release_capacity;
        condvar.notify_all();
    }
}

pub(crate) fn wait_for_phase() {
    let (lock, condvar) = observation();
    let guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let (guard, timeout) = condvar
        .wait_timeout_while(guard, QUIESCENCE_TIMEOUT, |state| !state.phase_complete())
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (required_workers, complete_workers) = guard.phase_progress();
    assert!(
        guard.phase_complete(),
        "worker idle hooks observed {complete_workers}/{required_workers} phase-{} owners across {} threads before {:?}",
        guard.phase,
        guard.len,
        timeout
    );
    let residual_capacity: usize = (0..guard.len)
        .filter(|&index| guard.required_phase[index] == guard.phase)
        .map(|index| guard.post_release_capacity[index])
        .sum();
    assert_eq!(
        residual_capacity, 0,
        "worker idle reclamation left {residual_capacity} scratch elements resident"
    );
}
