//! Precomputed Bluestein plan structs and their caching infrastructure.
//!
//! Separates plan construction (chirp/b_m synthesis) from execution (forward/inverse).
//! Plans are stored in a two-level cache: a thread-local `HashMap` for contention-free
//! hot-path lookup and a global `RwLock<HashMap>` for cross-thread promotion.

use super::mixed_radix;
use super::pointwise::{fill_and_mul_from_input, fill_and_mul_from_input_conj};
use super::pointwise::{mul_pointwise_with_twiddle, mul_pointwise_with_twiddle_inverse_kernel};
use super::zero_fill;
use num_complex::{Complex32, Complex64};
use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// ── Global plan caches ────────────────────────────────────────────────────────

static BLUESTEIN_PLAN_CACHE_64: std::sync::LazyLock<RwLock<HashMap<usize, Arc<BluesteinPlan64>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static BLUESTEIN_PLAN_CACHE_32: std::sync::LazyLock<RwLock<HashMap<usize, Arc<BluesteinPlan32>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

// ── Thread-local plan caches ──────────────────────────────────────────────────

thread_local! {
    static BLUESTEIN_PLAN_TL_64: RefCell<HashMap<usize, Arc<BluesteinPlan64>>> =
        RefCell::new(HashMap::new());
    static BLUESTEIN_PLAN_TL_32: RefCell<HashMap<usize, Arc<BluesteinPlan32>>> =
        RefCell::new(HashMap::new());
    pub(super) static BLUESTEIN_SCRATCH_64: RefCell<Vec<Complex64>> =
        const { RefCell::new(Vec::new()) };
    pub(super) static BLUESTEIN_SCRATCH_32: RefCell<Vec<Complex32>> =
        const { RefCell::new(Vec::new()) };
}

// ── Plan structs ──────────────────────────────────────────────────────────────

/// Precomputed context for arbitrary-length Bluestein chirp-Z transform (f64).
///
/// # Chirp filter semantics
///
/// - `chirp[k]` = `exp(-πi k²/N)` — forward chirp factor.
/// - `b_m` = FFT of filter `b[m] = chirp[m].conj()` = `exp(+πi m²/N)`,
///   used for the forward-transform convolution step.
/// - `b_m` is pre-transformed once during plan construction with mixed-radix twiddle
///   caches and reused on every transform.
#[derive(Clone, Debug)]
pub struct BluesteinPlan64 {
    n: usize,
    m: usize,
    chirp: Vec<Complex64>,
    b_m: Vec<Complex64>,
}

impl BluesteinPlan64 {
    /// Build a reusable plan for a non-power-of-two length `n` transform.
    ///
    /// Twiddle tables are not stored; `mixed_radix` caches them per-radix.
    pub fn new(n: usize) -> Self {
        let m = (2 * n.saturating_sub(1).max(1)).next_power_of_two();
        let chirp: Vec<Complex64> = {
            let mut v = Vec::with_capacity(n);
            unsafe { v.set_len(n) };
            for k in 0..n {
                let angle = -std::f64::consts::PI * (k * k) as f64 / n as f64;
                unsafe { *v.get_unchecked_mut(k) = Complex64::new(angle.cos(), angle.sin()) };
            }
            v
        };

        let mut b_m = Vec::with_capacity(m);
        unsafe { b_m.set_len(m) };
        b_m[0] = Complex64::new(1.0, 0.0);
        for k in 1..n {
            let bk = chirp[k].conj();
            b_m[k] = bk;
            b_m[m - k] = bk;
        }
        let gap_end = m + 1 - n;
        if n < gap_end {
            zero_fill(&mut b_m[n..gap_end]);
        }
        mixed_radix::forward_inplace_64_with_twiddles(&mut b_m, None);

        Self { n, m, chirp, b_m }
    }

    /// Padded convolution size `M = next_power_of_two(2N - 1)`.
    #[inline]
    pub fn m(&self) -> usize {
        self.m
    }

    /// Forward transform using a pre-allocated scratch slice of length `M`.
    pub fn forward_with_scratch(&self, data: &mut [Complex64], scratch_a: &mut [Complex64]) {
        assert_eq!(data.len(), self.n);
        assert_eq!(scratch_a.len(), self.m);
        {
            let (head, tail) = scratch_a.split_at_mut(self.n);
            fill_and_mul_from_input(head, data, &self.chirp);
            if !tail.is_empty() {
                zero_fill(tail);
            }
        }
        mixed_radix::forward_inplace_64_with_twiddles(scratch_a, None);
        mul_pointwise_with_twiddle(scratch_a, &self.b_m);
        mixed_radix::inverse_inplace_64_with_twiddles(scratch_a, None);
        let scratch_head = &scratch_a[..self.n];
        fill_and_mul_from_input(data, scratch_head, &self.chirp);
    }

    /// Unnormalized inverse transform using a pre-allocated scratch slice.
    pub fn inverse_unnorm_with_scratch(&self, data: &mut [Complex64], scratch_a: &mut [Complex64]) {
        assert_eq!(data.len(), self.n);
        assert_eq!(scratch_a.len(), self.m);
        {
            let (head, tail) = scratch_a.split_at_mut(self.n);
            fill_and_mul_from_input_conj(head, data, &self.chirp);
            if !tail.is_empty() {
                zero_fill(tail);
            }
        }
        mixed_radix::forward_inplace_64_with_twiddles(scratch_a, None);
        mul_pointwise_with_twiddle_inverse_kernel(scratch_a, &self.b_m);
        mixed_radix::inverse_inplace_64_with_twiddles(scratch_a, None);
        let scratch_head = &scratch_a[..self.n];
        fill_and_mul_from_input_conj(data, scratch_head, &self.chirp);
    }
}

// ── BluesteinPlan32 ───────────────────────────────────────────────────────────

/// Precomputed context for arbitrary-length Bluestein chirp-Z transform (f32).
#[derive(Clone, Debug)]
pub struct BluesteinPlan32 {
    n: usize,
    m: usize,
    chirp: Vec<Complex32>,
    b_m: Vec<Complex32>,
}

impl BluesteinPlan32 {
    /// Build a reusable plan for a non-power-of-two length `n` transform.
    pub fn new(n: usize) -> Self {
        let m = (2 * n.saturating_sub(1).max(1)).next_power_of_two();
        let chirp: Vec<Complex32> = {
            let mut v = Vec::with_capacity(n);
            unsafe { v.set_len(n) };
            for k in 0..n {
                let angle = -(std::f64::consts::PI * (k as f64 * k as f64) / n as f64) as f32;
                unsafe { *v.get_unchecked_mut(k) = Complex32::new(angle.cos(), angle.sin()) };
            }
            v
        };

        let mut b_m = Vec::with_capacity(m);
        unsafe { b_m.set_len(m) };
        b_m[0] = Complex32::new(1.0, 0.0);
        for k in 1..n {
            let bk = chirp[k].conj();
            b_m[k] = bk;
            b_m[m - k] = bk;
        }
        let gap_end = m + 1 - n;
        if n < gap_end {
            zero_fill(&mut b_m[n..gap_end]);
        }
        mixed_radix::forward_inplace_32_with_twiddles(&mut b_m, None);

        Self { n, m, chirp, b_m }
    }

    /// Padded convolution size `M = next_power_of_two(2N - 1)`.
    #[inline]
    pub fn m(&self) -> usize {
        self.m
    }

    /// Forward transform using a pre-allocated scratch slice of length `M`.
    pub fn forward_with_scratch(&self, data: &mut [Complex32], scratch_a: &mut [Complex32]) {
        assert_eq!(data.len(), self.n);
        assert_eq!(scratch_a.len(), self.m);
        {
            let (head, tail) = scratch_a.split_at_mut(self.n);
            fill_and_mul_from_input(head, data, &self.chirp);
            if !tail.is_empty() {
                zero_fill(tail);
            }
        }
        mixed_radix::forward_inplace_32_with_twiddles(scratch_a, None);
        mul_pointwise_with_twiddle(scratch_a, &self.b_m);
        mixed_radix::inverse_inplace_32_with_twiddles(scratch_a, None);
        let scratch_head = &scratch_a[..self.n];
        fill_and_mul_from_input(data, scratch_head, &self.chirp);
    }

    /// Unnormalized inverse transform using a pre-allocated scratch slice.
    pub fn inverse_unnorm_with_scratch(&self, data: &mut [Complex32], scratch_a: &mut [Complex32]) {
        assert_eq!(data.len(), self.n);
        assert_eq!(scratch_a.len(), self.m);
        {
            let (head, tail) = scratch_a.split_at_mut(self.n);
            fill_and_mul_from_input_conj(head, data, &self.chirp);
            if !tail.is_empty() {
                zero_fill(tail);
            }
        }
        mixed_radix::forward_inplace_32_with_twiddles(scratch_a, None);
        mul_pointwise_with_twiddle_inverse_kernel(scratch_a, &self.b_m);
        mixed_radix::inverse_inplace_32_with_twiddles(scratch_a, None);
        let scratch_head = &scratch_a[..self.n];
        fill_and_mul_from_input_conj(data, scratch_head, &self.chirp);
    }
}

// ── Two-level plan caching ────────────────────────────────────────────────────

#[inline]
pub(crate) fn cached_plan64(n: usize) -> Arc<BluesteinPlan64> {
    BLUESTEIN_PLAN_TL_64.with(|plan_map| {
        if let Some(plan) = plan_map.borrow().get(&n).cloned() {
            return plan;
        }
        let plan = {
            let maybe_cached = BLUESTEIN_PLAN_CACHE_64.read().get(&n).cloned();
            if let Some(plan) = maybe_cached {
                plan
            } else {
                let new_plan: Arc<BluesteinPlan64> = Arc::new(BluesteinPlan64::new(n));
                BLUESTEIN_PLAN_CACHE_64
                    .write()
                    .entry(n)
                    .or_insert_with(|| Arc::clone(&new_plan))
                    .clone()
            }
        };
        plan_map.borrow_mut().insert(n, Arc::clone(&plan));
        plan
    })
}

#[inline]
pub(crate) fn cached_plan32(n: usize) -> Arc<BluesteinPlan32> {
    BLUESTEIN_PLAN_TL_32.with(|plan_map| {
        if let Some(plan) = plan_map.borrow().get(&n).cloned() {
            return plan;
        }
        let plan = {
            let maybe_cached = BLUESTEIN_PLAN_CACHE_32.read().get(&n).cloned();
            if let Some(plan) = maybe_cached {
                plan
            } else {
                let new_plan: Arc<BluesteinPlan32> = Arc::new(BluesteinPlan32::new(n));
                BLUESTEIN_PLAN_CACHE_32
                    .write()
                    .entry(n)
                    .or_insert_with(|| Arc::clone(&new_plan))
                    .clone()
            }
        };
        plan_map.borrow_mut().insert(n, Arc::clone(&plan));
        plan
    })
}

// ── Scratch accessors ─────────────────────────────────────────────────────────

#[inline]
pub(crate) fn with_scratch64<R, F>(m: usize, f: F) -> R
where
    F: FnOnce(&mut [Complex64]) -> R,
{
    BLUESTEIN_SCRATCH_64.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        if scratch.len() < m {
            scratch.resize(m, Complex64::default());
        }
        f(&mut scratch[..m])
    })
}

#[inline]
pub(crate) fn with_scratch32<R, F>(m: usize, f: F) -> R
where
    F: FnOnce(&mut [Complex32]) -> R,
{
    BLUESTEIN_SCRATCH_32.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        if scratch.len() < m {
            scratch.resize(m, Complex32::default());
        }
        f(&mut scratch[..m])
    })
}
