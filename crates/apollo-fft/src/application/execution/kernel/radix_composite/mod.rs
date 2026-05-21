//! Mixed-radix Stockham autosort FFT for 2/3/5/7-smooth composite lengths.
//!
//! ## Algorithm — out-of-place Stockham ping-pong
//!
//! Given factorization N = r₀ · r₁ · … · r_{L-1} (innermost radix r₀ first),
//! with `prev_len_s = r₀ · … · r_{s-1}` and `groups_s = N / (r_s · prev_len_s)`:
//!
//! For stage s (reading from `src`, writing to `dst`):
//!   For each group index `b ∈ 0..groups_s` and offset `j ∈ 0..prev_len_s`:
//!     1. Gather r_s elements from `src` at stride `groups_s · prev_len_s`:
//!        `x[k] = src[k · groups_s · prev_len_s + b · prev_len_s + j]`
//!     2. Apply inter-stage twiddle `W_{stage_len_s}^{j·k}` for k = 1..r_s.
//!     3. Apply DFT-r_s butterfly.
//!     4. Write to contiguous block in `dst`:
//!        `dst[b · stage_len_s + j + k · prev_len_s] = result[k]`
//!
//! Alternate `src`/`dst` each stage (ping-pong). The strided gather at stage 0
//! (`prev_len=1`, stride=`groups`) implicitly performs the mixed-radix
//! digit-reversal, so no standalone permutation pass is needed.
//!
//! ## Correctness proof sketch
//!
//! The output layout invariant after stage s is: `dst` contains `groups_s`
//! contiguous blocks of `stage_len_s` elements each, where within each block
//! the partial DFT is in natural frequency order. At stage 0 this is trivially
//! satisfied by the strided gather (each size-r₀ butterfly receives inputs whose
//! index spacing equals the groups count, reproducing digit-reversal implicitly).
//! Inductively, the invariant is preserved by each stage's scatter pattern. QED.
//!
//! ## Complexity and allocation
//!
//! O(N log N) time. Scratch is an N-element thread-local ping-pong buffer reused
//! across calls. Twiddle tables are cached by the exact radix decomposition and
//! transform direction. If L is odd the final result is in `scratch`; a single
//! `data.copy_from_slice` brings it back. If L is even the result lands directly
//! in `data`.
//!
//! ## Supported radix set
//!
//! {2, 3, 4, 5, 7, 8}. N must have no prime factor outside {2, 3, 5, 7};
//! sizes with other prime factors fall back to Bluestein chirp-Z.
//!
//! ## References
//!
//! - Cooley, J.W. & Tukey, J.W. (1965). An algorithm for the machine
//!   calculation of complex Fourier series. *Math. Comp.* 19, 297–301.
//! - Glassman, A.J. (1970). A generalization of the Fast Fourier Transform.
//!   *IEEE Trans. Comput.* C-19(2), 105–116. (Stockham autosort, mixed-radix.)

#![allow(clippy::uninit_vec)]

pub(super) mod stage;

#[cfg(test)]
mod tests;

use num_complex::Complex;

use std::cell::RefCell;

use std::sync::Arc;

use super::radix_stage::normalize_inplace;

use super::winograd::{apply_twiddle_impl, WinogradScalar};

use stage::{stockham_stage, use_parallel_stage};

#[derive(Clone)]
struct CompositeTwiddleEntry<C> {
    radices: Arc<[usize]>,
    twiddles: Arc<[C]>,
    offsets: Arc<[usize]>,
}

pub trait CompositeCache: WinogradScalar {
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex<Self>]) -> R) -> R;
    fn cached_twiddles(inverse: bool, radices: &[usize]) -> (Arc<[Complex<Self>]>, Arc<[usize]>);
}

thread_local! {
    static TL_SCRATCH_64: RefCell<Vec<num_complex::Complex64>> = const { RefCell::new(Vec::new()) };
    static TL_SCRATCH_32: RefCell<Vec<num_complex::Complex32>> = const { RefCell::new(Vec::new()) };

    static TL_TWIDDLES_FWD_64: RefCell<Vec<CompositeTwiddleEntry<num_complex::Complex64>>> = const { RefCell::new(Vec::new()) };
    static TL_TWIDDLES_INV_64: RefCell<Vec<CompositeTwiddleEntry<num_complex::Complex64>>> = const { RefCell::new(Vec::new()) };

    static TL_TWIDDLES_FWD_32: RefCell<Vec<CompositeTwiddleEntry<num_complex::Complex32>>> = const { RefCell::new(Vec::new()) };
    static TL_TWIDDLES_INV_32: RefCell<Vec<CompositeTwiddleEntry<num_complex::Complex32>>> = const { RefCell::new(Vec::new()) };
}

fn build_composite_twiddles<F: WinogradScalar>(
    inverse: bool,
    radices: &[usize],
) -> (Vec<Complex<F>>, Vec<usize>) {
    let sign: f64 = if inverse { 1.0 } else { -1.0 };
    let total_twiddles: usize = radices
        .iter()
        .scan(1usize, |p, &r| {
            let out = *p;
            *p *= r;
            Some(out)
        })
        .sum();
    let mut all_twiddles = Vec::with_capacity(total_twiddles);
    // SAFETY: `Complex<F>` is plain numeric storage for the sealed f32/f64
    // implementors of `WinogradScalar`; every slot is overwritten below.
    unsafe { all_twiddles.set_len(total_twiddles) };
    let mut stage_offsets = Vec::with_capacity(radices.len());
    // SAFETY: `usize` has no drop glue and every slot is overwritten below.
    unsafe { stage_offsets.set_len(radices.len()) };

    let one = Complex::new(F::cast_f64(1.0), F::cast_f64(0.0));
    let mut prev_len = 1usize;
    let mut tw_idx = 0;
    let mut offset_idx = 0;
    for &r in radices {
        let stage_len = prev_len * r;
        unsafe { *stage_offsets.get_unchecked_mut(offset_idx) = tw_idx };
        offset_idx += 1;
        let base_angle = sign * std::f64::consts::TAU / stage_len as f64;
        let w_base = Complex::new(F::cast_f64(base_angle.cos()), F::cast_f64(base_angle.sin()));
        let mut tw = one;
        for _ in 0..prev_len {
            unsafe { *all_twiddles.get_unchecked_mut(tw_idx) = tw };
            tw_idx += 1;
            tw = apply_twiddle_impl(tw, w_base);
        }
        prev_len = stage_len;
    }
    debug_assert_eq!(tw_idx, total_twiddles);
    debug_assert_eq!(offset_idx, radices.len());
    (all_twiddles, stage_offsets)
}

impl CompositeCache for f64 {
    #[inline]
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex<Self>]) -> R) -> R {
        TL_SCRATCH_64.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            if scratch.len() < n {
                // The Stockham kernel writes every slot before reading it.
                // Skipping zero-init matches the identical pattern in mixed_radix.rs.
                let cur = scratch.len();
                scratch.reserve(n.saturating_sub(cur));
                // SAFETY: Complex<f64> is plain-data (no Drop); the kernel
                // overwrites every element before reading.
                unsafe { scratch.set_len(n) };
            }
            f(&mut scratch[..n])
        })
    }

    #[inline]
    fn cached_twiddles(inverse: bool, radices: &[usize]) -> (Arc<[Complex<Self>]>, Arc<[usize]>) {
        let tl = if inverse {
            &TL_TWIDDLES_INV_64
        } else {
            &TL_TWIDDLES_FWD_64
        };
        if let Some(cached) = tl.with(|cache| {
            cache
                .borrow()
                .iter()
                .find(|entry| entry.radices.as_ref() == radices)
                .map(|entry| (Arc::clone(&entry.twiddles), Arc::clone(&entry.offsets)))
        }) {
            return cached;
        }
        let (tw, offsets) = build_composite_twiddles::<f64>(inverse, radices);
        let tw = Arc::from(tw.into_boxed_slice());
        let offsets = Arc::from(offsets.into_boxed_slice());
        tl.with(|c| {
            c.borrow_mut().push(CompositeTwiddleEntry {
                radices: Arc::from(radices),
                twiddles: Arc::clone(&tw),
                offsets: Arc::clone(&offsets),
            });
        });
        (tw, offsets)
    }
}

impl CompositeCache for f32 {
    #[inline]
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex<Self>]) -> R) -> R {
        TL_SCRATCH_32.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            if scratch.len() < n {
                // Same rationale as f64: Stockham overwrites before reading.
                let cur = scratch.len();
                scratch.reserve(n.saturating_sub(cur));
                // SAFETY: Complex<f32> is plain-data (no Drop); the kernel
                // overwrites every element before reading.
                unsafe { scratch.set_len(n) };
            }
            f(&mut scratch[..n])
        })
    }

    #[inline]
    fn cached_twiddles(inverse: bool, radices: &[usize]) -> (Arc<[Complex<Self>]>, Arc<[usize]>) {
        let tl = if inverse {
            &TL_TWIDDLES_INV_32
        } else {
            &TL_TWIDDLES_FWD_32
        };
        if let Some(cached) = tl.with(|cache| {
            cache
                .borrow()
                .iter()
                .find(|entry| entry.radices.as_ref() == radices)
                .map(|entry| (Arc::clone(&entry.twiddles), Arc::clone(&entry.offsets)))
        }) {
            return cached;
        }
        let (tw, offsets) = build_composite_twiddles::<f32>(inverse, radices);
        let tw = Arc::from(tw.into_boxed_slice());
        let offsets = Arc::from(offsets.into_boxed_slice());
        tl.with(|c| {
            c.borrow_mut().push(CompositeTwiddleEntry {
                radices: Arc::from(radices),
                twiddles: Arc::clone(&tw),
                offsets: Arc::clone(&offsets),
            });
        });
        (tw, offsets)
    }
}

/// In-place forward FFT (unnormalized) for 2/3/5/7-smooth composite N.
///
/// Uses the out-of-place Stockham autosort formulation with reusable
/// thread-local scratch and cached twiddles.
#[inline]
pub fn forward_inplace_with_radices<F: CompositeCache>(data: &mut [Complex<F>], radices: &[usize]) {
    composite_core_with_radices(data, false, radices);
}

/// In-place inverse FFT (unnormalized) for 2/3/5/7-smooth composite N.
///
/// Uses the out-of-place Stockham autosort formulation with reusable
/// thread-local scratch and cached twiddles.
#[inline]
pub fn inverse_inplace_unnorm_with_radices<F: CompositeCache>(
    data: &mut [Complex<F>],
    radices: &[usize],
) {
    composite_core_with_radices(data, true, radices);
}

/// In-place inverse FFT normalized by 1/N for 2/3/5/7-smooth composite N.
///
/// Equivalent to `inverse_inplace_unnorm_with_radices` followed by `* (1/N)`.
#[inline]
pub fn inverse_inplace_with_radices<F: CompositeCache>(data: &mut [Complex<F>], radices: &[usize]) {
    composite_core_with_radices(data, true, radices);
    normalize_inplace(data, F::cast_f64(1.0 / data.len() as f64));
}

/// Out-of-place Stockham ping-pong kernel for mixed-radix FFT.
///
/// Eliminates the digit-reversal permutation pass by absorbing it into the
/// strided-read pattern of the first butterfly stage.
///
/// # Addressing
///
/// For stage s with radix `r`, `prev_len`, `groups = N / (r * prev_len)`,
/// `stride = groups * prev_len`:
///
/// - Read:  `src[k * stride + b * prev_len + j]`  for k = 0..r
/// - Write: `dst[b * stage_len + j + k * prev_len]` for k = 0..r
///
/// At stage 0 (`prev_len=1`, `stride=groups=N/r`) the read indices are
/// `0, groups, 2*groups, ..., (r-1)*groups` shifted by `b` — exactly the
/// mixed-radix digit-reversal scatter.
fn composite_core_with_radices<F: CompositeCache>(
    data: &mut [Complex<F>],
    inverse: bool,
    radices: &[usize],
) {
    let n = data.len();
    if n <= 1 || radices.is_empty() {
        return;
    }
    debug_assert_eq!(radices.iter().product::<usize>(), n);
    debug_assert!(radices.iter().all(|r| [2usize, 3, 4, 5, 7, 8].contains(r)));

    let (all_twiddles, stage_offsets) = F::cached_twiddles(inverse, radices);

    F::with_scratch(n, |scratch| {
        let mut src_is_data = true;
        let mut prev_len = 1usize;

        for (stage_idx, &r) in radices.iter().enumerate() {
            let stage_len = prev_len * r;
            let groups = n / stage_len;
            let offset = stage_offsets[stage_idx];
            let stage_twiddles = &all_twiddles[offset..offset + prev_len];
            let use_parallel = use_parallel_stage(n, stage_len, groups);

            if src_is_data {
                stockham_stage(
                    data,
                    scratch,
                    r,
                    prev_len,
                    groups,
                    stage_len,
                    stage_twiddles,
                    inverse,
                    use_parallel,
                );
            } else {
                stockham_stage(
                    scratch,
                    data,
                    r,
                    prev_len,
                    groups,
                    stage_len,
                    stage_twiddles,
                    inverse,
                    use_parallel,
                );
            }

            src_is_data = !src_is_data;
            prev_len = stage_len;
        }

        if !src_is_data {
            data.copy_from_slice(scratch);
        }
    });
}
