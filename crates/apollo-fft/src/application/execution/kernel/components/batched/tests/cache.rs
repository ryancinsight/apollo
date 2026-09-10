//! The plan and fold caches: one table per length and direction, shared
//! across threads.

use super::super::BatchedPlanCache;

#[test]
fn plans_are_cached_per_length_and_direction() {
    let a = <f64 as BatchedPlanCache>::cached_plan::<false>(64);
    let b = <f64 as BatchedPlanCache>::cached_plan::<false>(64);
    assert!(
        std::sync::Arc::ptr_eq(&a, &b),
        "a repeated request must reuse the cached plan rather than rebuild it"
    );
    let inv = <f64 as BatchedPlanCache>::cached_plan::<true>(64);
    assert!(
        !std::sync::Arc::ptr_eq(&a, &inv),
        "forward and inverse plans carry conjugate twiddles and must not share"
    );
}

#[test]
fn batched_plans_and_planes_are_shared_across_threads() {
    // `FourStepFold` owns two-level tables, `m (F + m / F)` pairs, and the
    // caches
    // are keyed per thread; if a thread that misses builds its own table
    // instead of taking the shared one, that storage exists once per thread
    // that touches the length and nothing evicts it.
    //
    // Both handles are spawned before either is joined, and both `Arc`s stay
    // live across the comparison: a per-thread cache dies with its thread, so
    // comparing raw addresses of dropped allocations could match by reuse.
    // `Arc::ptr_eq` on two live handles cannot.
    const LEN: usize = 1 << 12;
    const HALF: usize = 1 << 6;

    let plan_handles: Vec<_> = (0..2)
        .map(|_| std::thread::spawn(|| <f64 as BatchedPlanCache>::cached_plan::<false>(LEN)))
        .collect();
    let plans: Vec<_> = plan_handles
        .into_iter()
        .map(|handle| handle.join().expect("plan builder thread must not panic"))
        .collect();
    assert!(
        std::sync::Arc::ptr_eq(&plans[0], &plans[1]),
        "each thread built its own {LEN}-point batched plan"
    );

    let planes_handles: Vec<_> = (0..2)
        .map(|_| {
            std::thread::spawn(|| {
                <f64 as BatchedPlanCache>::cached_four_step_fold::<false>(LEN, HALF, HALF)
            })
        })
        .collect();
    let planes: Vec<_> = planes_handles
        .into_iter()
        .map(|handle| handle.join().expect("planes builder thread must not panic"))
        .collect();
    assert!(
        std::sync::Arc::ptr_eq(&planes[0], &planes[1]),
        "each thread built its own {LEN}-point four-step planes, duplicating          {} bytes of plane storage per thread",
        2 * HALF * HALF * core::mem::size_of::<f64>()
    );
}
