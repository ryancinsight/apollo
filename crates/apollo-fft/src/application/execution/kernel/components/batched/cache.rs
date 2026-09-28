//! Weak indexes for operation-owned immutable tables.

use super::fold::FourStepFold;
use super::plan::BatchedPlan;
use crate::application::execution::kernel::components::winograd::ShortWinogradScalar;
use hermes_simd::LaneScalar;
use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Weak};

/// A worker indexes one active length per direction. Changing lengths replaces
/// only a weak handle, so worker-local state neither retains tables nor grows.
struct TableCache<T> {
    directions: RefCell<[Option<TableEntry<T>>; 2]>,
}

struct TableEntry<T> {
    length: usize,
    table: Weak<T>,
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
            .and_then(|entry| entry.table.upgrade())
    }

    fn insert<const INVERSE: bool>(&self, length: usize, table: &Arc<T>) {
        self.directions.borrow_mut()[usize::from(INVERSE)] = Some(TableEntry {
            length,
            table: Arc::downgrade(table),
        });
    }
}

/// A worker's one active fold table.
///
/// The fold no longer varies by direction -- the inverse pass conjugates
/// the shared forward table in place ([`super::seams::FoldDirection`])
/// instead of a second cached copy (`APOLLO-MEM-INVERSE-CONJUGATE`) -- so
/// this holds a single slot rather than [`TableCache`]'s per-direction
/// pair.
struct FoldTableCache<T> {
    entry: RefCell<Option<TableEntry<T>>>,
}

impl<T> FoldTableCache<T> {
    const fn new() -> Self {
        Self {
            entry: RefCell::new(None),
        }
    }

    fn get(&self, length: usize) -> Option<Arc<T>> {
        self.entry
            .borrow()
            .as_ref()
            .filter(|entry| entry.length == length)
            .and_then(|entry| entry.table.upgrade())
    }

    fn insert(&self, length: usize, table: &Arc<T>) {
        *self.entry.borrow_mut() = Some(TableEntry {
            length,
            table: Arc::downgrade(table),
        });
    }
}

/// Process-wide weak indexes behind the bounded per-thread indexes.
///
/// A `BatchedPlan` owns `len - 1` twiddle pairs, O(16 sqrt(n)) bytes, and a
/// `FourStepFold` its two-level tables, O(16 sqrt(n) (F + sqrt(n) / F)),
/// one table per length rather than one per (length, direction): the fold
/// no longer varies by direction, so its map is keyed on length alone
/// (`APOLLO-MEM-INVERSE-CONJUGATE`). The thread-local handles below are the
/// lock-free fast path, but a miss used to *build* a private table, so
/// retention multiplied by the worker count of whatever executor drives the
/// transform. A miss now takes a live operation owner's shared table and
/// replaces one weak handle, so every thread converges on one allocation.
/// Dead weak entries are pruned on a miss, bounding metadata by live owners.
type GlobalPlanCache<T> = LazyLock<RwLock<HashMap<(usize, bool), Weak<BatchedPlan<T>>>>>;
type GlobalFoldCache<T> = LazyLock<RwLock<HashMap<usize, Weak<FourStepFold<T>>>>>;

static PLAN_GLOBAL_F64: GlobalPlanCache<f64> = LazyLock::new(|| RwLock::new(HashMap::new()));
static PLAN_GLOBAL_F32: GlobalPlanCache<f32> = LazyLock::new(|| RwLock::new(HashMap::new()));
static FOLD_GLOBAL_F64: GlobalFoldCache<f64> = LazyLock::new(|| RwLock::new(HashMap::new()));
static FOLD_GLOBAL_F32: GlobalFoldCache<f32> = LazyLock::new(|| RwLock::new(HashMap::new()));

thread_local! {
    static PLAN_CACHE_F64: TableCache<BatchedPlan<f64>> = const { TableCache::new() };
    static PLAN_CACHE_F32: TableCache<BatchedPlan<f32>> = const { TableCache::new() };
    static FOLD_CACHE_F64: FoldTableCache<FourStepFold<f64>> = const { FoldTableCache::new() };
    static FOLD_CACHE_F32: FoldTableCache<FourStepFold<f32>> = const { FoldTableCache::new() };
}

/// Scalar seam for acquiring reusable batched plan tables.
pub trait BatchedPlanCache:
    ShortWinogradScalar + LaneScalar + super::lane::Lane + eunomia::layout::Pod + Sized
{
    fn cached_plan<const INVERSE: bool>(len: usize) -> Arc<BatchedPlan<Self>>;
    /// The forward four-step fold table for `n` (`rows x cols` planes); the
    /// inverse direction conjugates it in the pass rather than requesting a
    /// second table (`super::seams::FoldDirection`).
    fn cached_four_step_fold(n: usize, rows: usize, cols: usize) -> Arc<FourStepFold<Self>>;
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
                    let shared = $plan_global.read().get(&key).and_then(Weak::upgrade);
                    if let Some(plan) = shared {
                        return plan;
                    }
                    // Re-check under the write lock and build there: these tables
                    // are O(16n), so blocking a concurrent misser costs less than
                    // letting it build a duplicate the map discards.
                    let mut guard = $plan_global.write();
                    if let Some(plan) = guard.get(&key).and_then(Weak::upgrade) {
                        return plan;
                    }
                    guard.retain(|_, plan| plan.strong_count() != 0);
                    let plan = Arc::new(BatchedPlan::<$t>::new::<INVERSE>(len));
                    guard.insert(key, Arc::downgrade(&plan));
                    plan
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

            fn cached_four_step_fold(
                n: usize,
                rows: usize,
                cols: usize,
            ) -> Arc<FourStepFold<Self>> {
                #[cold]
                #[inline(never)]
                fn miss(n: usize, rows: usize, cols: usize) -> Arc<FourStepFold<$t>> {
                    let shared = $planes_global.read().get(&n).and_then(Weak::upgrade);
                    if let Some(planes) = shared {
                        return planes;
                    }
                    let mut guard = $planes_global.write();
                    if let Some(planes) = guard.get(&n).and_then(Weak::upgrade) {
                        return planes;
                    }
                    guard.retain(|_, planes| planes.strong_count() != 0);
                    let planes = Arc::new(FourStepFold::<$t>::new(
                        n,
                        rows,
                        cols,
                        super::lane_order::LaneOrder::for_batch::<$t>(cols),
                    ));
                    guard.insert(n, Arc::downgrade(&planes));
                    planes
                }

                $planes.with(|c| {
                    if let Some(planes) = c.get(n) {
                        return planes;
                    }
                    let planes = miss(n, rows, cols);
                    c.insert(n, &planes);
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

#[cfg(test)]
pub(crate) fn retained_bytes_f64() -> usize {
    let plans = PLAN_GLOBAL_F64
        .read()
        .values()
        .filter_map(Weak::upgrade)
        .map(|plan| plan.retained_bytes())
        .sum::<usize>();
    plans
        + FOLD_GLOBAL_F64
            .read()
            .values()
            .filter_map(Weak::upgrade)
            .map(|fold| fold.retained_bytes())
            .sum::<usize>()
}
impl_plan_cache!(
    f32,
    PLAN_CACHE_F32,
    FOLD_CACHE_F32,
    PLAN_GLOBAL_F32,
    FOLD_GLOBAL_F32
);
