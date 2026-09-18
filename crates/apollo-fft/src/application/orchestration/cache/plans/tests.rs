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
