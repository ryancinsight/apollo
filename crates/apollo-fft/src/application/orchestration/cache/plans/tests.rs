//! The plan caches' bounds and clearing (ADR 0068). nextest runs each test
//! in its own process, so the process-wide tables start empty in each.

use super::{
    clear_plan_caches, PlanCacheProvider, SharedPlans, LOCAL_CAPACITY, SHARED_1D_PRECISE,
    SHARED_CAPACITY,
};
use crate::domain::metadata::shape::Shape1D;
use std::cell::Cell;
use std::sync::{mpsc, Arc};

fn plan(n: usize) -> Arc<crate::FftPlan1D<f64>> {
    <f64 as PlanCacheProvider>::get_1d_plan(Shape1D::new(n).expect("non-zero"))
}

#[test]
fn a_repeated_shape_returns_the_same_plan() {
    let first = plan(96);
    let _other = plan(97);
    assert!(Arc::ptr_eq(&first, &plan(96)), "ring hit");
    for n in 98..98 + LOCAL_CAPACITY {
        let _ = plan(n);
    }
    assert!(
        Arc::ptr_eq(&first, &plan(96)),
        "shared-table hit past the ring"
    );
}

#[test]
fn the_shared_table_holds_at_most_its_capacity() {
    for n in 1..=4 * SHARED_CAPACITY {
        let _ = plan(n);
    }
    assert_eq!(SHARED_1D_PRECISE.len(), SHARED_CAPACITY);
}

#[test]
fn the_least_recently_used_plan_is_the_one_evicted() {
    let shared = SharedPlans::<usize, usize>::new();
    let builds = Cell::new(0usize);
    let get = |key: usize| {
        shared.get_or_build(key, || {
            builds.set(builds.get() + 1);
            key
        })
    };
    for key in 0..SHARED_CAPACITY {
        get(key);
    }
    get(0);
    get(SHARED_CAPACITY);
    assert_eq!(shared.len(), SHARED_CAPACITY);
    let before = builds.get();
    get(0);
    assert_eq!(builds.get(), before, "key 0 was used last and must survive");
    get(1);
    assert_eq!(
        builds.get(),
        before + 1,
        "key 1 was the least recent and was evicted"
    );
}

#[test]
fn clearing_releases_plans_no_caller_holds() {
    let held = plan(1000);
    let released = Arc::downgrade(&plan(1024));
    assert!(
        released.upgrade().is_some(),
        "the caches keep an unheld plan"
    );
    clear_plan_caches();
    assert!(released.upgrade().is_none(), "cleared caches keep nothing");
    assert_eq!(SHARED_1D_PRECISE.len(), 0);
    let rebuilt = plan(1000);
    assert!(
        !Arc::ptr_eq(&held, &rebuilt),
        "a held plan is not re-cached"
    );
}

#[test]
fn another_threads_ring_empties_on_its_next_lookup() {
    let (to_worker, worker_inbox) = mpsc::sync_channel::<()>(0);
    let (to_main, main_inbox) =
        mpsc::sync_channel::<Option<std::sync::Weak<crate::FftPlan1D<f64>>>>(0);
    let worker = std::thread::spawn(move || {
        to_main
            .send(Some(Arc::downgrade(&plan(512))))
            .expect("main receives the plan");
        worker_inbox.recv().expect("main signals after clearing");
        // The first lookup after the clear sees the new epoch and empties
        // this thread's ring, releasing its only reference.
        let _ = plan(256);
        to_main
            .send(None)
            .expect("main receives the lookup's completion");
        // Alive until main has checked, so thread exit cannot be what
        // released the plan.
        worker_inbox.recv().expect("main signals the end");
    });
    let released = main_inbox
        .recv()
        .expect("worker sends")
        .expect("the first message carries the plan");
    clear_plan_caches();
    assert!(
        released.upgrade().is_some(),
        "the worker's ring holds it until the worker looks up again"
    );
    to_worker.send(()).expect("worker receives");
    assert!(main_inbox.recv().expect("worker reports").is_none());
    assert!(
        released.upgrade().is_none(),
        "the ring emptied on its next lookup, the thread still alive"
    );
    to_worker.send(()).expect("worker receives the end");
    worker.join().expect("worker completes");
}

static TEST_SHARED: SharedPlans<usize, usize> = SharedPlans::new();

thread_local! {
    static TEST_LOCAL: std::cell::RefCell<super::LocalPlans<usize, usize>> =
        const { std::cell::RefCell::new(super::LocalPlans::new()) };
}

/// Looks `key` up through the test ring and table, counting builds.
fn counted(key: usize, builds: &std::sync::atomic::AtomicUsize) -> Arc<usize> {
    super::lookup(&TEST_LOCAL, &TEST_SHARED, key, || {
        builds.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        key
    })
}

#[test]
fn a_ring_hit_does_not_consult_the_shared_table() {
    let builds = std::sync::atomic::AtomicUsize::new(0);
    let first = counted(7, &builds);
    // Emptying the table without an epoch bump leaves only the ring able to
    // serve the shape.
    TEST_SHARED.clear();
    assert!(
        Arc::ptr_eq(&first, &counted(7, &builds)),
        "served from the ring"
    );
    assert_eq!(builds.load(std::sync::atomic::Ordering::Relaxed), 1);
}

#[test]
fn a_ring_hit_moves_to_the_front() {
    let builds = std::sync::atomic::AtomicUsize::new(0);
    let a = counted(1, &builds);
    for key in 2..=LOCAL_CAPACITY {
        let _ = counted(key, &builds);
    }
    // Hitting 1 makes 2 the least recent; one more shape pushes 2 out.
    let _ = counted(1, &builds);
    let _ = counted(LOCAL_CAPACITY + 1, &builds);
    TEST_SHARED.clear();
    let before = builds.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        Arc::ptr_eq(&a, &counted(1, &builds)),
        "1 stayed in the ring"
    );
    let _ = counted(2, &builds);
    assert_eq!(
        builds.load(std::sync::atomic::Ordering::Relaxed),
        before + 1,
        "2 left the ring and the table, so it was rebuilt"
    );
}

#[test]
fn ring_hits_keep_a_plan_recent_in_the_shared_table() {
    let builds = std::sync::atomic::AtomicUsize::new(0);
    let hot = counted(0, &builds);
    // Every shape after 0 is a miss, and 0 is used from the ring between
    // them, so it is the most recent plan throughout.
    for key in 1..=SHARED_CAPACITY {
        let _ = counted(key, &builds);
        let _ = counted(0, &builds);
    }
    assert_eq!(TEST_SHARED.len(), SHARED_CAPACITY);
    // Another thread, with its own empty ring, must find 0 in the table.
    let (hot_again, rebuilt) = std::thread::spawn(move || {
        let builds = std::sync::atomic::AtomicUsize::new(0);
        let plan = counted(0, &builds);
        (plan, builds.load(std::sync::atomic::Ordering::Relaxed))
    })
    .join()
    .expect("the other thread completes");
    assert_eq!(rebuilt, 0, "the hot plan was evicted");
    assert!(Arc::ptr_eq(&hot, &hot_again));
}

#[test]
fn concurrent_misses_build_a_shape_once() {
    const THREADS: usize = 8;
    let shared = SharedPlans::<usize, usize>::new();
    for key in 0..32 {
        let builds = std::sync::atomic::AtomicUsize::new(0);
        let start = std::sync::Barrier::new(THREADS);
        std::thread::scope(|scope| {
            for _ in 0..THREADS {
                scope.spawn(|| {
                    start.wait();
                    let _ = shared.get_or_build(key, || {
                        builds.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        // Work long enough that the other threads' read
                        // checks all miss before the write lock is released.
                        (0..20_000).fold(key, |acc, i| acc.wrapping_add(i))
                    });
                });
            }
        });
        assert_eq!(
            builds.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "key {key}"
        );
    }
}

#[test]
fn clearing_from_a_thread_local_destructor_does_not_abort() {
    struct ClearsOnDrop;
    impl Drop for ClearsOnDrop {
        fn drop(&mut self) {
            clear_plan_caches();
        }
    }
    thread_local! {
        static GUARD: std::cell::RefCell<Option<ClearsOnDrop>> =
            const { std::cell::RefCell::new(None) };
    }
    std::thread::spawn(|| {
        // Registered before the ring, so destroyed after it: the clear then
        // runs with this thread's ring already gone.
        GUARD.with_borrow_mut(|guard| *guard = Some(ClearsOnDrop));
        let _ = plan(64);
    })
    .join()
    .expect("thread teardown with a clear in a destructor completes");
}
