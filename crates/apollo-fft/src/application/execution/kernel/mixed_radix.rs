//! Mixed-radix strategy facade.
//!
//! Routes in-place FFTs to the best available radix kernel for the given length:
//! - Power-of-8  → `radix8` (Winograd DFT-8 inner butterfly, optional rayon).
//! - Power-of-4  → `radix4` (radix-4 DIT butterfly).
//! - Other PoT   → `radix2` (iterative Cooley-Tukey, stages 1-4 use
//!                 compile-time constants to avoid trig calls).
//! - Arbitrary   → Bluestein chirp-Z (no-alloc on the hot path when the
//!                 caller supplies `Some(twiddles)`).
//!
//! ## SSOT principle
//!
//! The inner dispatch tree (`is_power_of_eight` → `radix8`, `is_power_of_four`
//! → `radix4`, `is_power_of_two` → `radix2`, else → `bluestein`) is expressed
//! **once** via the `pow2_dispatch!` macro and reused for every precision ×
//! operation combination.  Adding a new precision requires one new set of
//! 6 one-liner public wrappers; no algorithm body needs to change.
//!
//! ## Precision strategy
//!
//! | Variant | Element type | Notes |
//! |---------|--------------|-------|
//! | `_64`   | `Complex64`  | Native f64 throughout. |
//! | `_32`   | `Complex32`  | Native f32 throughout. |
//! | `_f16`  | `Cf16`       | Each sub-kernel promotes to f32 internally; non-PoT sizes use `run_f16_via_f32` + Bluestein-f32. |

use super::f16_bridge::run_f16_via_f32;
use super::radix2_f16::Cf16;
use super::radix_shape::{is_power_of_eight, is_power_of_four};
use super::{bluestein, radix2, radix2_f16, radix4, radix8};
use num_complex::{Complex32, Complex64};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// ── Global backing caches (cross-thread sharing, written once per size) ───────

static TWIDDLE_FWD_64_CACHE: Lazy<RwLock<HashMap<usize, Arc<[Complex64]>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_INV_64_CACHE: Lazy<RwLock<HashMap<usize, Arc<[Complex64]>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_FWD_32_CACHE: Lazy<RwLock<HashMap<usize, Arc<[Complex32]>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_INV_32_CACHE: Lazy<RwLock<HashMap<usize, Arc<[Complex32]>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_FWD_F16_CACHE: Lazy<RwLock<HashMap<usize, Arc<[Cf16]>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));
static TWIDDLE_INV_F16_CACHE: Lazy<RwLock<HashMap<usize, Arc<[Cf16]>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

// ── Thread-local fast-path caches (zero locking on the hot path after warmup) ─
//
// On a cache hit the hot path is:
//   TW_FWD_64.with(|c| c.borrow().get(&n).cloned()) — no atomic, no lock.
// On a miss, the global RwLock is consulted or the table is built once, then
// the Arc is stored into the thread-local HashMap for all future calls.

thread_local! {
    static TL_FWD_64: RefCell<HashMap<usize, Arc<[Complex64]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_INV_64: RefCell<HashMap<usize, Arc<[Complex64]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_FWD_32: RefCell<HashMap<usize, Arc<[Complex32]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_INV_32: RefCell<HashMap<usize, Arc<[Complex32]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_FWD_F16: RefCell<HashMap<usize, Arc<[Cf16]>>> =
        RefCell::new(HashMap::with_capacity(8));
    static TL_INV_F16: RefCell<HashMap<usize, Arc<[Cf16]>>> =
        RefCell::new(HashMap::with_capacity(8));
}

/// Retrieves the Arc from the thread-local map or falls back to global.
/// `build_fn` is called at most once per (thread, size) pair.
#[inline]
fn tl_cached<T: Clone>(
    tl: &'static std::thread::LocalKey<RefCell<HashMap<usize, Arc<[T]>>>>,
    global: &'static Lazy<RwLock<HashMap<usize, Arc<[T]>>>>,
    n: usize,
    build_fn: impl FnOnce(usize) -> Vec<T>,
) -> Arc<[T]> {
    // Fast path: hit in this thread's HashMap — no lock or atomic.
    if let Some(tw) = tl.with(|c| c.borrow().get(&n).cloned()) {
        return tw;
    }
    // Slow path (once per thread per size): read or write the global cache.
    // NOTE: The read guard MUST be dropped (via let binding with `;`) before
    // calling global.write() to avoid a same-thread read→write deadlock with
    // parking_lot::RwLock, which does not allow upgrading a read lock on the
    // same thread.
    let tw = {
        let maybe_cached = global.read().get(&n).cloned(); // ReadGuard drops here at `;`
        if let Some(tw) = maybe_cached {
            tw
        } else {
            let new_tw: Arc<[T]> = Arc::from(build_fn(n));
            global
                .write()
                .entry(n)
                .or_insert_with(|| Arc::clone(&new_tw))
                .clone()
        }
    };
    tl.with(|c| c.borrow_mut().insert(n, Arc::clone(&tw)));
    tw
}

#[inline]
pub(crate) fn cached_twiddle_fwd_64(n: usize) -> Arc<[Complex64]> {
    tl_cached(
        &TL_FWD_64,
        &TWIDDLE_FWD_64_CACHE,
        n,
        radix2::build_forward_twiddle_table_64,
    )
}

#[inline]
pub(crate) fn cached_twiddle_inv_64(n: usize) -> Arc<[Complex64]> {
    tl_cached(
        &TL_INV_64,
        &TWIDDLE_INV_64_CACHE,
        n,
        radix2::build_inverse_twiddle_table_64,
    )
}

#[inline]
pub(crate) fn cached_twiddle_fwd_32(n: usize) -> Arc<[Complex32]> {
    tl_cached(
        &TL_FWD_32,
        &TWIDDLE_FWD_32_CACHE,
        n,
        radix2::build_forward_twiddle_table_32,
    )
}

#[inline]
pub(crate) fn cached_twiddle_inv_32(n: usize) -> Arc<[Complex32]> {
    tl_cached(
        &TL_INV_32,
        &TWIDDLE_INV_32_CACHE,
        n,
        radix2::build_inverse_twiddle_table_32,
    )
}

#[inline]
pub(crate) fn cached_twiddle_fwd_f16(n: usize) -> Arc<[Cf16]> {
    tl_cached(
        &TL_FWD_F16,
        &TWIDDLE_FWD_F16_CACHE,
        n,
        radix2_f16::build_forward_twiddle_table_f16,
    )
}

#[inline]
pub(crate) fn cached_twiddle_inv_f16(n: usize) -> Arc<[Cf16]> {
    tl_cached(
        &TL_INV_F16,
        &TWIDDLE_INV_F16_CACHE,
        n,
        radix2_f16::build_inverse_twiddle_table_f16,
    )
}

// ── SSOT dispatch macro ───────────────────────────────────────────────────────

/// Emit the dispatch body for a `_with_twiddles` function that is already
/// inside `if data.len().is_power_of_two()`.
///
/// Parameters:
/// - `$data`      — the `&mut [T]` expression
/// - `$twiddles`  — the `Option<&[T]>` expression
/// - `cache`      — thread-local twiddle-cache lookup (`fn(usize) -> Arc<[T]>`)
/// - `r8`, `r4`, `r2` — `with_twiddles_fn` paths (single function each)
///
/// When `$twiddles` is `None`, the cache is consulted (O(1) HashMap lookup
/// on the hot path after the first per-thread per-size warm-up, no locks).
/// The previous `_no` (twiddle-rebuilding) variants are no longer needed.
macro_rules! pow2_dispatch {
    (
        $data:expr, $twiddles:expr,
        cache = $cache:path,
        r8 = $r8_tw:path,
        r4 = $r4_tw:path,
        r2 = $r2_tw:path
    ) => {{
        if is_power_of_eight($data.len()) {
            if let Some(tw) = $twiddles {
                $r8_tw($data, tw);
            } else {
                $r8_tw($data, &$cache($data.len()));
            }
        } else if is_power_of_four($data.len()) {
            if let Some(tw) = $twiddles {
                $r4_tw($data, tw);
            } else {
                $r4_tw($data, &$cache($data.len()));
            }
        } else if let Some(tw) = $twiddles {
            $r2_tw($data, tw);
        } else {
            $r2_tw($data, &$cache($data.len()));
        }
    }};
}

macro_rules! pow2_dispatch_no_r8 {
    (
        $data:expr, $twiddles:expr,
        cache = $cache:path,
        r4 = $r4_tw:path,
        r2 = $r2_tw:path
    ) => {{
        if is_power_of_four($data.len()) {
            if let Some(tw) = $twiddles {
                $r4_tw($data, tw);
            } else {
                $r4_tw($data, &$cache($data.len()));
            }
        } else if let Some(tw) = $twiddles {
            $r2_tw($data, tw);
        } else {
            $r2_tw($data, &$cache($data.len()));
        }
    }};
}

// ── f64 ───────────────────────────────────────────────────────────────────────

/// In-place forward FFT (unnormalized, f64) with optional precomputed twiddles.
#[inline]
pub fn forward_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: Option<&[Complex64]>) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        pow2_dispatch!(
            data, twiddles,
            cache = cached_twiddle_fwd_64,
            r8 = radix8::forward_inplace_64_with_twiddles,
            r4 = radix4::forward_inplace_64_with_twiddles,
            r2 = radix2::forward_inplace_64_with_twiddles
        );
    } else {
        bluestein::forward_inplace_64(data);
    }
}

/// In-place inverse FFT (unnormalized, f64) with optional precomputed twiddles.
#[inline]
pub fn inverse_inplace_unnorm_64_with_twiddles(
    data: &mut [Complex64],
    twiddles: Option<&[Complex64]>,
) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        pow2_dispatch!(
            data, twiddles,
            cache = cached_twiddle_inv_64,
            r8 = radix8::inverse_inplace_unnorm_64_with_twiddles,
            r4 = radix4::inverse_inplace_unnorm_64_with_twiddles,
            r2 = radix2::inverse_inplace_unnorm_64_with_twiddles
        );
    } else {
        bluestein::inverse_inplace_unnorm_64(data);
    }
}

/// In-place inverse FFT normalized by 1/N (f64) with optional precomputed twiddles.
#[inline]
pub fn inverse_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: Option<&[Complex64]>) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        pow2_dispatch!(
            data, twiddles,
            cache = cached_twiddle_inv_64,
            r8 = radix8::inverse_inplace_64_with_twiddles,
            r4 = radix4::inverse_inplace_64_with_twiddles,
            r2 = radix2::inverse_inplace_64_with_twiddles
        );
    } else {
        bluestein::inverse_inplace_64(data);
    }
}

/// In-place forward FFT (unnormalized, f64).
pub fn forward_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_fwd_64(data.len());
        forward_inplace_64_with_twiddles(data, Some(tw.as_ref()));
    } else {
        forward_inplace_64_with_twiddles(data, None);
    }
}

/// In-place inverse FFT (unnormalized, f64).
pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_inv_64(data.len());
        inverse_inplace_unnorm_64_with_twiddles(data, Some(tw.as_ref()));
    } else {
        inverse_inplace_unnorm_64_with_twiddles(data, None);
    }
}

/// In-place inverse FFT normalized by 1/N (f64).
pub fn inverse_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_inv_64(data.len());
        inverse_inplace_64_with_twiddles(data, Some(tw.as_ref()));
    } else {
        inverse_inplace_64_with_twiddles(data, None);
    }
}

// ── f32 ───────────────────────────────────────────────────────────────────────

/// In-place forward FFT (unnormalized, f32) with optional precomputed twiddles.
#[inline]
pub fn forward_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: Option<&[Complex32]>) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        pow2_dispatch!(
            data, twiddles,
            cache = cached_twiddle_fwd_32,
            r8 = radix8::forward_inplace_32_with_twiddles,
            r4 = radix4::forward_inplace_32_with_twiddles,
            r2 = radix2::forward_inplace_32_with_twiddles
        );
    } else {
        bluestein::forward_inplace_32(data);
    }
}

/// In-place inverse FFT (unnormalized, f32) with optional precomputed twiddles.
#[inline]
pub fn inverse_inplace_unnorm_32_with_twiddles(
    data: &mut [Complex32],
    twiddles: Option<&[Complex32]>,
) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        pow2_dispatch!(
            data, twiddles,
            cache = cached_twiddle_inv_32,
            r8 = radix8::inverse_inplace_unnorm_32_with_twiddles,
            r4 = radix4::inverse_inplace_unnorm_32_with_twiddles,
            r2 = radix2::inverse_inplace_unnorm_32_with_twiddles
        );
    } else {
        bluestein::inverse_inplace_unnorm_32(data);
    }
}

/// In-place inverse FFT normalized by 1/N (f32) with optional precomputed twiddles.
#[inline]
pub fn inverse_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: Option<&[Complex32]>) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        pow2_dispatch!(
            data, twiddles,
            cache = cached_twiddle_inv_32,
            r8 = radix8::inverse_inplace_32_with_twiddles,
            r4 = radix4::inverse_inplace_32_with_twiddles,
            r2 = radix2::inverse_inplace_32_with_twiddles
        );
    } else {
        bluestein::inverse_inplace_32(data);
    }
}

/// In-place forward FFT (unnormalized, f32).
pub fn forward_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_fwd_32(data.len());
        forward_inplace_32_with_twiddles(data, Some(tw.as_ref()));
    } else {
        forward_inplace_32_with_twiddles(data, None);
    }
}

/// In-place inverse FFT (unnormalized, f32).
pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_inv_32(data.len());
        inverse_inplace_unnorm_32_with_twiddles(data, Some(tw.as_ref()));
    } else {
        inverse_inplace_unnorm_32_with_twiddles(data, None);
    }
}

/// In-place inverse FFT normalized by 1/N (f32).
pub fn inverse_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_inv_32(data.len());
        inverse_inplace_32_with_twiddles(data, Some(tw.as_ref()));
    } else {
        inverse_inplace_32_with_twiddles(data, None);
    }
}

// ── f16 ───────────────────────────────────────────────────────────────────────

/// In-place forward FFT (unnormalized, f16 storage) with optional precomputed twiddles.
///
/// Non-PoT lengths promote to f32, run Bluestein-f32, and demote via
/// `run_f16_via_f32`. This is the only allocation site for f16 non-PoT sizes.
#[inline]
pub fn forward_inplace_f16_with_twiddles(data: &mut [Cf16], twiddles: Option<&[Cf16]>) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        pow2_dispatch_no_r8!(
            data, twiddles,
            cache = cached_twiddle_fwd_f16,
            r4 = radix4::forward_inplace_f16_with_twiddles,
            r2 = radix2_f16::forward_inplace_f16_with_twiddles
        );
    } else {
        run_f16_via_f32(data, bluestein::forward_inplace_32);
    }
}

/// In-place inverse FFT (unnormalized, f16 storage) with optional precomputed twiddles.
#[inline]
pub fn inverse_inplace_unnorm_f16_with_twiddles(data: &mut [Cf16], twiddles: Option<&[Cf16]>) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        pow2_dispatch_no_r8!(
            data, twiddles,
            cache = cached_twiddle_inv_f16,
            r4 = radix4::inverse_inplace_unnorm_f16_with_twiddles,
            r2 = radix2_f16::inverse_inplace_unnorm_f16_with_twiddles
        );
    } else {
        run_f16_via_f32(data, bluestein::inverse_inplace_unnorm_32);
    }
}

/// In-place inverse FFT normalized by 1/N (f16 storage) with optional precomputed twiddles.
#[inline]
pub fn inverse_inplace_f16_with_twiddles(data: &mut [Cf16], twiddles: Option<&[Cf16]>) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        pow2_dispatch_no_r8!(
            data, twiddles,
            cache = cached_twiddle_inv_f16,
            r4 = radix4::inverse_inplace_f16_with_twiddles,
            r2 = radix2_f16::inverse_inplace_f16_with_twiddles
        );
    } else {
        run_f16_via_f32(data, bluestein::inverse_inplace_32);
    }
}

/// In-place forward FFT (unnormalized, f16 storage).
pub fn forward_inplace_f16(data: &mut [Cf16]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_fwd_f16(data.len());
        forward_inplace_f16_with_twiddles(data, Some(tw.as_ref()));
    } else {
        forward_inplace_f16_with_twiddles(data, None);
    }
}

/// In-place inverse FFT (unnormalized, f16 storage).
pub fn inverse_inplace_unnorm_f16(data: &mut [Cf16]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_inv_f16(data.len());
        inverse_inplace_unnorm_f16_with_twiddles(data, Some(tw.as_ref()));
    } else {
        inverse_inplace_unnorm_f16_with_twiddles(data, None);
    }
}

/// In-place inverse FFT normalized by 1/N (f16 storage).
pub fn inverse_inplace_f16(data: &mut [Cf16]) {
    if data.len() <= 1 {
        return;
    }
    if data.len().is_power_of_two() {
        let tw = cached_twiddle_inv_f16(data.len());
        inverse_inplace_f16_with_twiddles(data, Some(tw.as_ref()));
    } else {
        inverse_inplace_f16_with_twiddles(data, None);
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::max_abs_err_64;
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward_64, dft_inverse_64};

    #[test]
    fn mixed_forward_n32_matches_direct() {
        let n = 32usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.29).sin(), (k as f64 * 0.17).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "mixed-radix forward mismatch err={err:.2e}");
    }

    #[test]
    fn mixed_inverse_unnorm_n32_matches_direct() {
        let n = 32usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.19).cos(), (k as f64 * 0.07).sin()))
            .collect();
        let mut got = input.clone();
        inverse_inplace_unnorm_64(&mut got);
        let expected = dft_inverse_64(&input)
            .into_iter()
            .map(|x| x * n as f64)
            .collect::<Vec<_>>();
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "mixed-radix inverse mismatch err={err:.2e}");
    }
}
