//! Bounded, clearable caches of reusable FFT plans (ADR 0068).
//!
//! Every shape a process transforms used to add a plan to a process-wide
//! map and to each calling thread's map, for the life of the process. Each
//! cache is now bounded: a shared table of at most `SHARED_CAPACITY` (64)
//! plans per scalar and dimension, evicting the least recently used, and a
//! per-thread ring of the `LOCAL_CAPACITY` (4) most recently used, which keeps
//! a repeated shape on a lock-free, allocation-free path.
//! [`clear_plan_caches`] empties both.

use crate::application::execution::plan::fft::dimension_1d::FftPlan1D;
use crate::application::execution::plan::fft::dimension_2d::FftPlan2D;
use crate::application::execution::plan::fft::dimension_3d::FftPlan3D;
use crate::application::execution::plan::fft::real_storage::RealFftData;
use crate::domain::metadata::shape::{Shape1D, Shape2D, Shape3D};
use eunomia::F16;
use parking_lot::RwLock;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::LocalKey;

/// Plans each shared cache keeps: one per distinct shape, the least
/// recently used evicted past this. A policy bound rather than a derived
/// one; it caps retained memory at this many times the largest plan of the
/// scalar and dimension (ADR 0068), and a workload cycling through more
/// shapes rebuilds rather than grows.
const SHARED_CAPACITY: usize = 64;

/// Plans each thread keeps in its most-recently-used ring. A transform that
/// alternates a few shapes (a forward and an inverse length, the axes of a
/// plane) finds them here without touching the shared table's lock.
const LOCAL_CAPACITY: usize = 4;

/// Bumped by [`clear_plan_caches`]; a thread's ring holding an older value
/// is stale and empties on its next lookup.
static EPOCH: AtomicU64 = AtomicU64::new(0);

/// Zero-cost cache resolution trait for real storage types.
pub trait PlanCacheProvider: RealFftData {
    /// Retrieve or instantiate a generic 1D plan.
    fn get_1d_plan(shape: Shape1D) -> Arc<FftPlan1D<Self::PlanScalar>>;
    /// Retrieve or instantiate a generic 2D plan.
    fn get_2d_plan(shape: Shape2D) -> Arc<FftPlan2D<Self::PlanScalar>>;
    /// Retrieve or instantiate a generic 3D plan.
    fn get_3d_plan(shape: Shape3D) -> Arc<FftPlan3D<Self::PlanScalar>>;
}

/// Releases every plan the crate's plan caches hold.
///
/// The shared tables empty at once, and so does the calling thread's ring;
/// another thread's ring empties on that thread's next plan lookup, so a
/// thread that never transforms again keeps at most `LOCAL_CAPACITY` (4)
/// plans per cache until it exits. A plan a caller still holds stays alive
/// with that caller. Plans are rebuilt on their next use.
pub fn clear_plan_caches() {
    SHARED_1D_PRECISE.clear();
    SHARED_2D_PRECISE.clear();
    SHARED_3D_PRECISE.clear();
    SHARED_1D_REDUCED.clear();
    SHARED_2D_REDUCED.clear();
    SHARED_3D_REDUCED.clear();
    // No other access is ordered against this one: a thread reading the
    // old value keeps plans that are still valid, only for longer.
    EPOCH.fetch_add(1, Ordering::Relaxed);
    LOCAL_1D_PRECISE.with_borrow_mut(LocalPlans::clear);
    LOCAL_2D_PRECISE.with_borrow_mut(LocalPlans::clear);
    LOCAL_3D_PRECISE.with_borrow_mut(LocalPlans::clear);
    LOCAL_1D_REDUCED.with_borrow_mut(LocalPlans::clear);
    LOCAL_2D_REDUCED.with_borrow_mut(LocalPlans::clear);
    LOCAL_3D_REDUCED.with_borrow_mut(LocalPlans::clear);
}

/// One cached plan and the tick of its last use.
struct Entry<K, P> {
    key: K,
    plan: Arc<P>,
    last_used: AtomicU64,
}

/// The process-wide table of one scalar and dimension, least recently used
/// evicted past `SHARED_CAPACITY` (64).
struct SharedPlans<K, P> {
    entries: RwLock<Vec<Entry<K, P>>>,
    tick: AtomicU64,
}

impl<K: Copy + Eq, P> SharedPlans<K, P> {
    const fn new() -> Self {
        Self {
            entries: parking_lot::const_rwlock(Vec::new()),
            tick: AtomicU64::new(0),
        }
    }

    /// The next recency stamp. Stamps only order uses for eviction, so no
    /// access synchronizes through them.
    fn stamp(&self) -> u64 {
        self.tick.fetch_add(1, Ordering::Relaxed)
    }

    /// The plan for `key`, built by `build` under the write lock on a miss
    /// so concurrent callers build it once.
    fn get_or_build(&self, key: K, build: impl FnOnce() -> P) -> Arc<P> {
        if let Some(entry) = self.entries.read().iter().find(|entry| entry.key == key) {
            entry.last_used.store(self.stamp(), Ordering::Relaxed);
            return Arc::clone(&entry.plan);
        }
        let mut entries = self.entries.write();
        if let Some(entry) = entries.iter().find(|entry| entry.key == key) {
            entry.last_used.store(self.stamp(), Ordering::Relaxed);
            return Arc::clone(&entry.plan);
        }
        if entries.len() >= SHARED_CAPACITY {
            let oldest = entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used.load(Ordering::Relaxed))
                .map(|(index, _)| index);
            if let Some(oldest) = oldest {
                entries.swap_remove(oldest);
            }
        }
        let plan = Arc::new(build());
        entries.push(Entry {
            key,
            plan: Arc::clone(&plan),
            last_used: AtomicU64::new(self.stamp()),
        });
        plan
    }

    fn clear(&self) {
        self.entries.write().clear();
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.read().len()
    }
}

/// One thread's most-recently-used plans, most recent first, valid for the
/// [`EPOCH`] it records.
struct LocalPlans<K, P> {
    epoch: u64,
    ring: [Option<(K, Arc<P>)>; LOCAL_CAPACITY],
}

impl<K: Copy + Eq, P> LocalPlans<K, P> {
    const fn new() -> Self {
        Self {
            epoch: 0,
            ring: [const { None }; LOCAL_CAPACITY],
        }
    }

    fn clear(&mut self) {
        self.ring = [const { None }; LOCAL_CAPACITY];
    }

    /// The plan for `key` if the ring holds it, moved to the front.
    fn get(&mut self, key: K) -> Option<Arc<P>> {
        let epoch = EPOCH.load(Ordering::Relaxed);
        if self.epoch != epoch {
            self.clear();
            self.epoch = epoch;
            return None;
        }
        let at = self
            .ring
            .iter()
            .position(|slot| slot.as_ref().is_some_and(|(held, _)| *held == key))?;
        self.ring[..=at].rotate_right(1);
        self.ring[0].as_ref().map(|(_, plan)| Arc::clone(plan))
    }

    /// Records `plan` as the most recent, dropping the least recent.
    fn put(&mut self, key: K, plan: &Arc<P>) {
        self.ring.rotate_right(1);
        self.ring[0] = Some((key, Arc::clone(plan)));
    }
}

/// The ring, then the shared table, then a build.
fn lookup<K: Copy + Eq + 'static, P: 'static>(
    local: &'static LocalKey<RefCell<LocalPlans<K, P>>>,
    shared: &SharedPlans<K, P>,
    key: K,
    build: impl FnOnce() -> P,
) -> Arc<P> {
    if let Some(plan) = local.with_borrow_mut(|ring| ring.get(key)) {
        return plan;
    }
    let plan = shared.get_or_build(key, build);
    local.with_borrow_mut(|ring| ring.put(key, &plan));
    plan
}

static SHARED_1D_PRECISE: SharedPlans<usize, FftPlan1D<f64>> = SharedPlans::new();
static SHARED_2D_PRECISE: SharedPlans<(usize, usize), FftPlan2D<f64>> = SharedPlans::new();
static SHARED_3D_PRECISE: SharedPlans<(usize, usize, usize), FftPlan3D<f64>> = SharedPlans::new();
static SHARED_1D_REDUCED: SharedPlans<usize, FftPlan1D<f32>> = SharedPlans::new();
static SHARED_2D_REDUCED: SharedPlans<(usize, usize), FftPlan2D<f32>> = SharedPlans::new();
static SHARED_3D_REDUCED: SharedPlans<(usize, usize, usize), FftPlan3D<f32>> = SharedPlans::new();

thread_local! {
    static LOCAL_1D_PRECISE: RefCell<LocalPlans<usize, FftPlan1D<f64>>> =
        const { RefCell::new(LocalPlans::new()) };
    static LOCAL_2D_PRECISE: RefCell<LocalPlans<(usize, usize), FftPlan2D<f64>>> =
        const { RefCell::new(LocalPlans::new()) };
    static LOCAL_3D_PRECISE: RefCell<LocalPlans<(usize, usize, usize), FftPlan3D<f64>>> =
        const { RefCell::new(LocalPlans::new()) };
    static LOCAL_1D_REDUCED: RefCell<LocalPlans<usize, FftPlan1D<f32>>> =
        const { RefCell::new(LocalPlans::new()) };
    static LOCAL_2D_REDUCED: RefCell<LocalPlans<(usize, usize), FftPlan2D<f32>>> =
        const { RefCell::new(LocalPlans::new()) };
    static LOCAL_3D_REDUCED: RefCell<LocalPlans<(usize, usize, usize), FftPlan3D<f32>>> =
        const { RefCell::new(LocalPlans::new()) };
}

impl PlanCacheProvider for f64 {
    #[inline]
    fn get_1d_plan(shape: Shape1D) -> Arc<FftPlan1D<Self::PlanScalar>> {
        lookup(&LOCAL_1D_PRECISE, &SHARED_1D_PRECISE, shape.n(), || {
            FftPlan1D::new(shape)
        })
    }

    #[inline]
    fn get_2d_plan(shape: Shape2D) -> Arc<FftPlan2D<Self::PlanScalar>> {
        lookup(
            &LOCAL_2D_PRECISE,
            &SHARED_2D_PRECISE,
            (shape.nx(), shape.ny()),
            || FftPlan2D::new(shape),
        )
    }

    #[inline]
    fn get_3d_plan(shape: Shape3D) -> Arc<FftPlan3D<Self::PlanScalar>> {
        let key = (shape.nx(), shape.ny(), shape.nz());
        lookup(&LOCAL_3D_PRECISE, &SHARED_3D_PRECISE, key, || {
            FftPlan3D::new(shape)
        })
    }
}

impl PlanCacheProvider for f32 {
    #[inline]
    fn get_1d_plan(shape: Shape1D) -> Arc<FftPlan1D<Self::PlanScalar>> {
        lookup(&LOCAL_1D_REDUCED, &SHARED_1D_REDUCED, shape.n(), || {
            FftPlan1D::new(shape)
        })
    }

    #[inline]
    fn get_2d_plan(shape: Shape2D) -> Arc<FftPlan2D<Self::PlanScalar>> {
        lookup(
            &LOCAL_2D_REDUCED,
            &SHARED_2D_REDUCED,
            (shape.nx(), shape.ny()),
            || FftPlan2D::new(shape),
        )
    }

    #[inline]
    fn get_3d_plan(shape: Shape3D) -> Arc<FftPlan3D<Self::PlanScalar>> {
        let key = (shape.nx(), shape.ny(), shape.nz());
        lookup(&LOCAL_3D_REDUCED, &SHARED_3D_REDUCED, key, || {
            FftPlan3D::new(shape)
        })
    }
}

impl PlanCacheProvider for F16 {
    #[inline]
    fn get_1d_plan(shape: Shape1D) -> Arc<FftPlan1D<Self::PlanScalar>> {
        <f32 as PlanCacheProvider>::get_1d_plan(shape)
    }

    #[inline]
    fn get_2d_plan(shape: Shape2D) -> Arc<FftPlan2D<Self::PlanScalar>> {
        <f32 as PlanCacheProvider>::get_2d_plan(shape)
    }

    #[inline]
    fn get_3d_plan(shape: Shape3D) -> Arc<FftPlan3D<Self::PlanScalar>> {
        <f32 as PlanCacheProvider>::get_3d_plan(shape)
    }
}

#[cfg(test)]
mod tests;
