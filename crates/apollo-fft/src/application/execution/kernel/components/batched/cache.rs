//! Shared immutable tables and bounded worker-local handles.

use super::{BatchedPlan, FourStepFold};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use hermes_simd::LaneScalar;
use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

/// A worker needs one active length per direction. Tables themselves remain
/// process-shared; changing lengths replaces only a handle, without growing
/// a per-worker map or allocating when a later submission uses a new worker.
struct TableCache<T> {
    directions: RefCell<[Option<TableEntry<T>>; 2]>,
}

struct TableEntry<T> {
    length: usize,
    table: Arc<T>,
}

impl<T> TableCache<T> {
    const fn new() -> Self {
        Self {
            directions: RefCell::new([None, None]),
        }
    }

    fn get<const INVERSE: bool>(&self, length: usize) -> Option<Arc<T>> {
        self.directions.borrow()[usize::from(INVERSE)]
            .as_ref()
            .filter(|entry| entry.length == length)
            .map(|entry| Arc::clone(&entry.table))
    }

    fn insert<const INVERSE: bool>(&self, length: usize, table: &Arc<T>) {
        self.directions.borrow_mut()[usize::from(INVERSE)] = Some(TableEntry {
            length,
            table: Arc::clone(table),
        });
    }
}

/// Process-wide plan storage behind the bounded per-thread handles.
///
/// A `BatchedPlan` owns `len - 1` twiddle pairs, O(16 sqrt(n)) bytes, and a
/// `FourStepFold` its two-level tables, O(16 sqrt(n) (F + sqrt(n) / F)). The thread-local handles below are the lock-free fast path, but a
/// miss used to *build* a private table, so retention multiplied by the worker
/// count of whatever executor drives the transform. A miss now takes the shared
/// table and replaces one directional handle, so every thread converges on
/// one allocation without growing its own table index.
type GlobalPlanCache<T> = LazyLock<RwLock<HashMap<(usize, bool), Arc<BatchedPlan<T>>>>>;
type GlobalFoldCache<T> = LazyLock<RwLock<HashMap<(usize, bool), Arc<FourStepFold<T>>>>>;

static PLAN_GLOBAL_F64: GlobalPlanCache<f64> = LazyLock::new(|| RwLock::new(HashMap::new()));
static PLAN_GLOBAL_F32: GlobalPlanCache<f32> = LazyLock::new(|| RwLock::new(HashMap::new()));
static FOLD_GLOBAL_F64: GlobalFoldCache<f64> = LazyLock::new(|| RwLock::new(HashMap::new()));
static FOLD_GLOBAL_F32: GlobalFoldCache<f32> = LazyLock::new(|| RwLock::new(HashMap::new()));

thread_local! {
    static PLAN_CACHE_F64: TableCache<BatchedPlan<f64>> = const { TableCache::new() };
    static PLAN_CACHE_F32: TableCache<BatchedPlan<f32>> = const { TableCache::new() };
    static FOLD_CACHE_F64: TableCache<FourStepFold<f64>> = const { TableCache::new() };
    static FOLD_CACHE_F32: TableCache<FourStepFold<f32>> = const { TableCache::new() };
}

/// Scalars whose batched plans are cached per thread.
pub(crate) trait BatchedPlanCache:
    MixedRadixScalar + LaneScalar + super::radix::Lane + eunomia::layout::Pod + Sized
{
    fn cached_plan<const INVERSE: bool>(len: usize) -> Arc<BatchedPlan<Self>>;
    fn cached_four_step_fold<const INVERSE: bool>(
        n: usize,
        rows: usize,
        cols: usize,
    ) -> Arc<FourStepFold<Self>>;
}

macro_rules! impl_plan_cache {
    ($t:ty, $cache:ident, $planes:ident, $plan_global:ident, $planes_global:ident) => {
        impl BatchedPlanCache for $t {
            fn cached_plan<const INVERSE: bool>(len: usize) -> Arc<BatchedPlan<Self>> {
                // The miss path is outlined and cold so the thread-local hit --
                // the path repeated same-length transforms take -- stays small enough to
                // inline into the caller. Folding the shared-map lookup inline
                // here regressed `fft_kernel_strategy/generic_selector` at 64 and
                // 256 in all four counterbalanced comparisons.
                #[cold]
                #[inline(never)]
                fn miss<const INVERSE: bool>(
                    key: (usize, bool),
                    len: usize,
                ) -> Arc<BatchedPlan<$t>> {
                    // Drop the read guard before the write path: a guard held in
                    // an `if let` scrutinee outlives the `else` arm, and this lock
                    // is not reentrant.
                    let shared = $plan_global.read().get(&key).cloned();
                    if let Some(plan) = shared {
                        return plan;
                    }
                    // Re-check under the write lock and build there: these tables
                    // are O(16n), so blocking a concurrent misser costs less than
                    // letting it build a duplicate the map discards.
                    let mut guard = $plan_global.write();
                    Arc::clone(
                        guard
                            .entry(key)
                            .or_insert_with(|| Arc::new(BatchedPlan::<$t>::new::<INVERSE>(len))),
                    )
                }

                $cache.with(|c| {
                    let key = (len, INVERSE);
                    if let Some(plan) = c.get::<INVERSE>(len) {
                        return plan;
                    }
                    let plan = miss::<INVERSE>(key, len);
                    c.insert::<INVERSE>(len, &plan);
                    plan
                })
            }

            fn cached_four_step_fold<const INVERSE: bool>(
                n: usize,
                rows: usize,
                cols: usize,
            ) -> Arc<FourStepFold<Self>> {
                #[cold]
                #[inline(never)]
                fn miss<const INVERSE: bool>(
                    key: (usize, bool),
                    n: usize,
                    rows: usize,
                    cols: usize,
                ) -> Arc<FourStepFold<$t>> {
                    let shared = $planes_global.read().get(&key).cloned();
                    if let Some(planes) = shared {
                        return planes;
                    }
                    let mut guard = $planes_global.write();
                    Arc::clone(guard.entry(key).or_insert_with(|| {
                        Arc::new(FourStepFold::<$t>::new::<INVERSE>(
                            n,
                            rows,
                            cols,
                            super::LaneOrder::for_batch::<$t>(cols),
                        ))
                    }))
                }

                $planes.with(|c| {
                    let key = (n, INVERSE);
                    if let Some(planes) = c.get::<INVERSE>(n) {
                        return planes;
                    }
                    let planes = miss::<INVERSE>(key, n, rows, cols);
                    c.insert::<INVERSE>(n, &planes);
                    planes
                })
            }
        }
    };
}

impl_plan_cache!(
    f64,
    PLAN_CACHE_F64,
    FOLD_CACHE_F64,
    PLAN_GLOBAL_F64,
    FOLD_GLOBAL_F64
);
impl_plan_cache!(
    f32,
    PLAN_CACHE_F32,
    FOLD_CACHE_F32,
    PLAN_GLOBAL_F32,
    FOLD_GLOBAL_F32
);
