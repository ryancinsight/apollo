//! Shared cache implementations for reusable FFT plans.

use crate::application::execution::plan::fft::dimension_1d::FftPlan1D;
use crate::application::execution::plan::fft::dimension_2d::FftPlan2D;
use crate::application::execution::plan::fft::dimension_3d::FftPlan3D;
use crate::application::execution::plan::fft::real_storage::RealFftData;
use crate::domain::metadata::shape::{Shape1D, Shape2D, Shape3D};
use half::f16;
use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

type Plan2Key = (usize, usize);
type Plan3Key = (usize, usize, usize);
type Plan2Map<T> = HashMap<Plan2Key, Arc<FftPlan2D<T>>>;
type Plan3Map<T> = HashMap<Plan3Key, Arc<FftPlan3D<T>>>;
type SharedPlan2Map<T> = RwLock<Plan2Map<T>>;
type SharedPlan3Map<T> = RwLock<Plan3Map<T>>;

/// Zero-cost cache resolution trait for real storage types.
pub trait PlanCacheProvider: RealFftData {
    /// Retrieve or instantiate a generic 1D plan.
    fn get_1d_plan(shape: Shape1D) -> Arc<FftPlan1D<Self::PlanScalar>>;
    /// Retrieve or instantiate a generic 2D plan.
    fn get_2d_plan(shape: Shape2D) -> Arc<FftPlan2D<Self::PlanScalar>>;
    /// Retrieve or instantiate a generic 3D plan.
    fn get_3d_plan(shape: Shape3D) -> Arc<FftPlan3D<Self::PlanScalar>>;
}

impl PlanCacheProvider for f64 {
    #[inline]
    fn get_1d_plan(shape: Shape1D) -> Arc<FftPlan1D<Self::PlanScalar>> {
        thread_local! {
            static LAST_PLAN: RefCell<Option<(usize, Arc<FftPlan1D<f64>>)>> = RefCell::new(None);
            static TLS_CACHE: RefCell<HashMap<usize, Arc<FftPlan1D<f64>>>> = RefCell::new(HashMap::new());
        }
        static GLOBAL_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<FftPlan1D<f64>>>>> =
            std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

        let key = shape.n;
        if let Some(plan) = LAST_PLAN.with(|last| {
            let borrow = last.borrow();
            if let Some((n, plan)) = &*borrow {
                if *n == key {
                    return Some(Arc::clone(plan));
                }
            }
            None
        }) {
            return plan;
        }

        if let Some(plan) = TLS_CACHE.with(|cache| cache.borrow().get(&key).map(Arc::clone)) {
            LAST_PLAN.with(|last| *last.borrow_mut() = Some((key, Arc::clone(&plan))));
            return plan;
        }

        if let Some(plan) = GLOBAL_CACHE.read().get(&key) {
            let plan_clone = Arc::clone(plan);
            TLS_CACHE.with(|cache| cache.borrow_mut().insert(key, Arc::clone(&plan_clone)));
            LAST_PLAN.with(|last| *last.borrow_mut() = Some((key, Arc::clone(&plan_clone))));
            return plan_clone;
        }

        let mut guard = GLOBAL_CACHE.write();
        if let Some(plan) = guard.get(&key) {
            let plan_clone = Arc::clone(plan);
            TLS_CACHE.with(|cache| cache.borrow_mut().insert(key, Arc::clone(&plan_clone)));
            LAST_PLAN.with(|last| *last.borrow_mut() = Some((key, Arc::clone(&plan_clone))));
            return plan_clone;
        }

        let plan = Arc::new(FftPlan1D::<f64>::new(shape));
        guard.insert(key, Arc::clone(&plan));
        TLS_CACHE.with(|cache| cache.borrow_mut().insert(key, Arc::clone(&plan)));
        LAST_PLAN.with(|last| *last.borrow_mut() = Some((key, Arc::clone(&plan))));
        plan
    }

    #[inline]
    fn get_2d_plan(shape: Shape2D) -> Arc<FftPlan2D<Self::PlanScalar>> {
        thread_local! {
            static TLS_CACHE: RefCell<Plan2Map<f64>> =
                RefCell::new(HashMap::new());
        }
        static GLOBAL_CACHE: std::sync::LazyLock<SharedPlan2Map<f64>> =
            std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

        let key = (shape.nx, shape.ny);
        if let Some(plan) = TLS_CACHE.with(|cache| cache.borrow().get(&key).map(Arc::clone)) {
            return plan;
        }

        if let Some(plan) = GLOBAL_CACHE.read().get(&key) {
            TLS_CACHE.with(|cache| cache.borrow_mut().insert(key, Arc::clone(plan)));
            return Arc::clone(plan);
        }

        let mut guard = GLOBAL_CACHE.write();
        if let Some(plan) = guard.get(&key) {
            TLS_CACHE.with(|cache| cache.borrow_mut().insert(key, Arc::clone(plan)));
            return Arc::clone(plan);
        }

        let plan = Arc::new(FftPlan2D::<f64>::new(shape));
        guard.insert(key, Arc::clone(&plan));
        TLS_CACHE.with(|cache| cache.borrow_mut().insert(key, Arc::clone(&plan)));
        plan
    }

    #[inline]
    fn get_3d_plan(shape: Shape3D) -> Arc<FftPlan3D<Self::PlanScalar>> {
        thread_local! {
            static TLS_CACHE: RefCell<Plan3Map<f64>> =
                RefCell::new(HashMap::new());
        }
        static GLOBAL_CACHE: std::sync::LazyLock<SharedPlan3Map<f64>> =
            std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

        let key = (shape.nx, shape.ny, shape.nz);
        if let Some(plan) = TLS_CACHE.with(|cache| cache.borrow().get(&key).map(Arc::clone)) {
            return plan;
        }

        if let Some(plan) = GLOBAL_CACHE.read().get(&key) {
            TLS_CACHE.with(|cache| cache.borrow_mut().insert(key, Arc::clone(plan)));
            return Arc::clone(plan);
        }

        let mut guard = GLOBAL_CACHE.write();
        if let Some(plan) = guard.get(&key) {
            TLS_CACHE.with(|cache| cache.borrow_mut().insert(key, Arc::clone(plan)));
            return Arc::clone(plan);
        }

        let plan = Arc::new(FftPlan3D::<f64>::new(shape));
        guard.insert(key, Arc::clone(&plan));
        TLS_CACHE.with(|cache| cache.borrow_mut().insert(key, Arc::clone(&plan)));
        plan
    }
}

impl PlanCacheProvider for f32 {
    #[inline]
    fn get_1d_plan(shape: Shape1D) -> Arc<FftPlan1D<Self::PlanScalar>> {
        thread_local! {
            static LAST_PLAN: RefCell<Option<(usize, Arc<FftPlan1D<f32>>)>> = RefCell::new(None);
            static TLS_CACHE: RefCell<HashMap<usize, Arc<FftPlan1D<f32>>>> = RefCell::new(HashMap::new());
        }
        static GLOBAL_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Arc<FftPlan1D<f32>>>>> =
            std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

        let key = shape.n;
        if let Some(plan) = LAST_PLAN.with(|last| {
            let borrow = last.borrow();
            if let Some((n, plan)) = &*borrow {
                if *n == key {
                    return Some(Arc::clone(plan));
                }
            }
            None
        }) {
            return plan;
        }

        if let Some(plan) = TLS_CACHE.with(|cache| cache.borrow().get(&key).map(Arc::clone)) {
            LAST_PLAN.with(|last| *last.borrow_mut() = Some((key, Arc::clone(&plan))));
            return plan;
        }

        if let Some(plan) = GLOBAL_CACHE.read().get(&key) {
            let plan_clone = Arc::clone(plan);
            TLS_CACHE.with(|cache| cache.borrow_mut().insert(key, Arc::clone(&plan_clone)));
            LAST_PLAN.with(|last| *last.borrow_mut() = Some((key, Arc::clone(&plan_clone))));
            return plan_clone;
        }

        let mut guard = GLOBAL_CACHE.write();
        if let Some(plan) = guard.get(&key) {
            let plan_clone = Arc::clone(plan);
            TLS_CACHE.with(|cache| cache.borrow_mut().insert(key, Arc::clone(&plan_clone)));
            LAST_PLAN.with(|last| *last.borrow_mut() = Some((key, Arc::clone(&plan_clone))));
            return plan_clone;
        }

        let plan = Arc::new(FftPlan1D::<f32>::new(shape));
        guard.insert(key, Arc::clone(&plan));
        TLS_CACHE.with(|cache| cache.borrow_mut().insert(key, Arc::clone(&plan)));
        LAST_PLAN.with(|last| *last.borrow_mut() = Some((key, Arc::clone(&plan))));
        plan
    }

    #[inline]
    fn get_2d_plan(shape: Shape2D) -> Arc<FftPlan2D<Self::PlanScalar>> {
        thread_local! {
            static TLS_CACHE: RefCell<Plan2Map<f32>> =
                RefCell::new(HashMap::new());
        }
        static GLOBAL_CACHE: std::sync::LazyLock<SharedPlan2Map<f32>> =
            std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

        let key = (shape.nx, shape.ny);
        if let Some(plan) = TLS_CACHE.with(|cache| cache.borrow().get(&key).map(Arc::clone)) {
            return plan;
        }

        if let Some(plan) = GLOBAL_CACHE.read().get(&key) {
            TLS_CACHE.with(|cache| cache.borrow_mut().insert(key, Arc::clone(plan)));
            return Arc::clone(plan);
        }

        let mut guard = GLOBAL_CACHE.write();
        if let Some(plan) = guard.get(&key) {
            TLS_CACHE.with(|cache| cache.borrow_mut().insert(key, Arc::clone(plan)));
            return Arc::clone(plan);
        }

        let plan = Arc::new(FftPlan2D::<f32>::new(shape));
        guard.insert(key, Arc::clone(&plan));
        TLS_CACHE.with(|cache| cache.borrow_mut().insert(key, Arc::clone(&plan)));
        plan
    }

    #[inline]
    fn get_3d_plan(shape: Shape3D) -> Arc<FftPlan3D<Self::PlanScalar>> {
        thread_local! {
            static TLS_CACHE: RefCell<Plan3Map<f32>> =
                RefCell::new(HashMap::new());
        }
        static GLOBAL_CACHE: std::sync::LazyLock<SharedPlan3Map<f32>> =
            std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

        let key = (shape.nx, shape.ny, shape.nz);
        if let Some(plan) = TLS_CACHE.with(|cache| cache.borrow().get(&key).map(Arc::clone)) {
            return plan;
        }

        if let Some(plan) = GLOBAL_CACHE.read().get(&key) {
            TLS_CACHE.with(|cache| cache.borrow_mut().insert(key, Arc::clone(plan)));
            return Arc::clone(plan);
        }

        let mut guard = GLOBAL_CACHE.write();
        if let Some(plan) = guard.get(&key) {
            TLS_CACHE.with(|cache| cache.borrow_mut().insert(key, Arc::clone(plan)));
            return Arc::clone(plan);
        }

        let plan = Arc::new(FftPlan3D::<f32>::new(shape));
        guard.insert(key, Arc::clone(&plan));
        TLS_CACHE.with(|cache| cache.borrow_mut().insert(key, Arc::clone(&plan)));
        plan
    }
}

impl PlanCacheProvider for f16 {
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
