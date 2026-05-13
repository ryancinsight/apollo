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
use num_complex::Complex;

use rayon::prelude::*;

use std::cell::RefCell;

use std::sync::Arc;

use super::radix_stage::normalize_inplace;

use super::tuning::RADIX_PARALLEL_CHUNK_THRESHOLD;

use super::winograd::{
    apply_twiddle_impl, dft2_impl, dft3_impl, dft4_impl, dft5_impl, dft7_impl, dft8_impl,
    WinogradScalar,
};

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

// ── inner butterfly dispatchers ───────────────────────────────────────────────

#[inline]
fn apply_dft_r_impl<F: WinogradScalar>(data: &mut [Complex<F>], r: usize, inverse: bool) {
    match r {
        2 => {
            let (lo, hi) = data.split_at_mut(1);
            dft2_impl(&mut lo[0], &mut hi[0]);
        }
        3 => {
            let mut b = [data[0], data[1], data[2]];
            dft3_impl(&mut b, inverse);
            data[..3].copy_from_slice(&b);
        }
        4 => {
            let mut b = [data[0], data[1], data[2], data[3]];
            dft4_impl(&mut b, inverse);
            data[..4].copy_from_slice(&b);
        }
        5 => {
            let mut b = [data[0], data[1], data[2], data[3], data[4]];
            dft5_impl(&mut b, inverse);
            data[..5].copy_from_slice(&b);
        }
        7 => {
            dft7_impl(&mut data[..7], inverse);
        }
        8 => {
            let mut b = [
                data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
            ];
            dft8_impl(&mut b, inverse);
            data[..8].copy_from_slice(&b);
        }
        _ => unreachable!("unsupported radix {r}"),
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
            let use_parallel =
                n >= RADIX_PARALLEL_CHUNK_THRESHOLD && stage_len >= 512 && groups >= 4;

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

/// Single out-of-place Stockham butterfly stage.
///
/// Reads from `src` with stride `groups * prev_len`, writes to `dst` in
/// contiguous `stage_len`-element blocks. Safe to call with `src == data` and
/// `dst == scratch` or vice-versa — the two slices must not alias.
#[inline]
fn stockham_stage<F: WinogradScalar>(
    src: &[Complex<F>],
    dst: &mut [Complex<F>],
    r: usize,
    prev_len: usize,
    groups: usize,
    stage_len: usize,
    stage_twiddles: &[Complex<F>],
    inverse: bool,
    use_parallel: bool,
) {
    let stride = groups * prev_len;
    if use_parallel {
        // Parallel over output blocks (b index). `src` is shared read-only.
        dst.par_chunks_mut(stage_len)
            .enumerate()
            .for_each(|(b, dst_block)| {
                let mut buf = [Complex::new(F::cast_f64(0.0), F::cast_f64(0.0)); 8];
                let src_base = b * prev_len;
                stockham_block(
                    src,
                    dst_block,
                    r,
                    prev_len,
                    stride,
                    stage_twiddles,
                    inverse,
                    src_base,
                    &mut buf,
                );
            });
    } else {
        let mut buf = [Complex::new(F::cast_f64(0.0), F::cast_f64(0.0)); 8];
        for b in 0..groups {
            let src_base = b * prev_len;
            let dst_block = &mut dst[b * stage_len..(b + 1) * stage_len];
            stockham_block(
                src,
                dst_block,
                r,
                prev_len,
                stride,
                stage_twiddles,
                inverse,
                src_base,
                &mut buf,
            );
        }
    }
}

/// Process one output block `b` of size `stage_len` for a single Stockham stage.
///
/// j=0 fast path: W^0 = 1 — gather, DFT-r, scatter with no multiply.
/// j>0 path: recurrence-based twiddle application.
#[inline]
fn stockham_block<F: WinogradScalar>(
    src: &[Complex<F>],
    dst_block: &mut [Complex<F>],
    r: usize,
    prev_len: usize,
    stride: usize,
    stage_twiddles: &[Complex<F>],
    inverse: bool,
    src_base: usize,
    buf: &mut [Complex<F>; 8],
) {
    // ── j = 0: W^0 = 1, no multiply ─────────────────────────────────────────
    for k in 0..r {
        // SAFETY: stride * k + src_base < n by loop invariant on b and stage layout.
        buf[k] = *unsafe { src.get_unchecked(k * stride + src_base) };
    }
    apply_dft_r_impl(&mut buf[..r], r, inverse);
    for k in 0..r {
        // SAFETY: k * prev_len < stage_len = prev_len * r.
        *unsafe { dst_block.get_unchecked_mut(k * prev_len) } = buf[k];
    }

    // ── j = 1..prev_len: with twiddle ────────────────────────────────────────
    for j in 1..prev_len {
        for k in 0..r {
            buf[k] = *unsafe { src.get_unchecked(k * stride + src_base + j) };
        }
        let base_tw = *unsafe { stage_twiddles.get_unchecked(j) };
        let mut tw_k = base_tw;
        for k in 1..r {
            buf[k] = apply_twiddle_impl(buf[k], tw_k);
            if k + 1 < r {
                tw_k = apply_twiddle_impl(tw_k, base_tw);
            }
        }
        apply_dft_r_impl(&mut buf[..r], r, inverse);
        for k in 0..r {
            *unsafe { dst_block.get_unchecked_mut(j + k * prev_len) } = buf[k];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
    use crate::application::execution::kernel::radix_shape::factorize_composite;
    use num_complex::{Complex32, Complex64};

    fn max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0f64, f64::max)
    }

    fn forward_inplace_64(data: &mut [Complex64]) {
        let radices = factorize_composite(data.len()).expect("test length must be 2/3/5/7-smooth");
        forward_inplace_with_radices(data, &radices);
    }

    fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
        let radices = factorize_composite(data.len()).expect("test length must be 2/3/5/7-smooth");
        inverse_inplace_unnorm_with_radices(data, &radices);
    }

    fn forward_inplace_32(data: &mut [Complex32]) {
        let radices = factorize_composite(data.len()).expect("test length must be 2/3/5/7-smooth");
        forward_inplace_with_radices(data, &radices);
    }

    // ── factorize_composite ───────────────────────────────────────────────────

    #[test]
    fn factorize_supported_sizes() {
        for &n in &[
            3usize,
            5,
            6,
            7,
            9,
            10,
            12,
            14,
            15,
            18,
            21,
            24,
            25,
            28,
            35,
            42,
            48,
            49,
            50,
            56,
            63,
            70,
            75,
            98,
            100,
            120,
            125,
            147,
            150,
            192,
            200,
            210,
            240,
            245,
            250,
            294,
            300,
            343,
            3430,
            375,
            384,
            392,
            450,
            500,
            588,
            600,
            686,
            700,
            750,
            784,
            864,
            900,
            980,
            1000,
            1200,
            1400,
            1470,
            1500,
            1960,
            2000,
            2400,
            2500,
            2940,
            3000,
            3430 * 2,
            3430 * 3,
            4000,
            4500,
            5000,
            6000,
            7000,
            7500,
            10000,
        ] {
            let result = factorize_composite(n);
            assert!(result.is_some(), "factorize_composite({n}) returned None");
            let radices = result.unwrap();
            assert_eq!(
                radices.iter().product::<usize>(),
                n,
                "factorize_composite({n}): product mismatch"
            );
            for &r in &radices {
                assert!(
                    [2, 3, 4, 5, 7, 8].contains(&r),
                    "factorize_composite({n}): unsupported radix {r}"
                );
            }
        }
    }

    #[test]
    fn factorize_pow2_returns_none() {
        for exp in 1..=20u32 {
            let n = 1usize << exp;
            assert!(
                factorize_composite(n).is_none(),
                "factorize_composite({n}) should be None for pure-PoT"
            );
        }
    }

    #[test]
    fn factorize_non_smooth_returns_none() {
        for &n in &[
            11usize, 13, 17, 19, 22, 23, 26, 29, 31, 33, 34, 38, 46, 58, 121, 143,
        ] {
            assert!(
                factorize_composite(n).is_none(),
                "factorize_composite({n}) should be None (has prime > 7)"
            );
        }
    }

    // ── forward + roundtrip correctness ──────────────────────────────────────

    fn check_forward(n: usize, tol: f64) {
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.37).sin(), (k as f64 * 0.19).cos()))
            .collect();
        let expected = dft_forward(&input);
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let err = max_err(&got, &expected);
        assert!(
            err < tol,
            "forward N={n}: max_err={err:.2e} (tol={tol:.2e})"
        );
    }

    fn check_roundtrip(n: usize, tol: f64) {
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.53).cos(), (k as f64 * 0.27).sin()))
            .collect();
        let mut buf = input.clone();
        forward_inplace_64(&mut buf);
        inverse_inplace_unnorm_64(&mut buf);
        let recovered: Vec<Complex64> = buf.iter().map(|x| x / n as f64).collect();
        let err = max_err(&recovered, &input);
        assert!(
            err < tol,
            "roundtrip N={n}: max_err={err:.2e} (tol={tol:.2e})"
        );
    }

    fn check_inverse(n: usize, tol: f64) {
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.61).cos(), (k as f64 * 0.43).sin()))
            .collect();
        let expected_unnorm: Vec<Complex64> = dft_inverse(&input)
            .into_iter()
            .map(|x| x * n as f64)
            .collect();
        let mut got = input.clone();
        inverse_inplace_unnorm_64(&mut got);
        let err = max_err(&got, &expected_unnorm);
        assert!(
            err < tol,
            "inverse N={n}: max_err={err:.2e} (tol={tol:.2e})"
        );
    }

    #[test]
    fn forward_n7() {
        check_forward(7, 1e-13);
    }
    #[test]
    fn forward_n3() {
        check_forward(3, 1e-13);
    }
    #[test]
    fn forward_n5() {
        check_forward(5, 1e-13);
    }
    #[test]
    fn forward_n9() {
        check_forward(9, 1e-12);
    }
    #[test]
    fn forward_n15() {
        check_forward(15, 1e-12);
    }
    #[test]
    fn forward_n25() {
        check_forward(25, 1e-12);
    }
    #[test]
    fn forward_n6() {
        check_forward(6, 1e-13);
    }
    #[test]
    fn forward_n10() {
        check_forward(10, 1e-12);
    }
    #[test]
    fn forward_n14() {
        check_forward(14, 1e-12);
    }
    #[test]
    fn forward_n21() {
        check_forward(21, 1e-11);
    }

    #[test]
    fn forward_n100() {
        check_forward(100, 1e-11);
    }
    #[test]
    fn forward_n1000() {
        check_forward(1000, 1e-9);
    }
    #[test]
    fn forward_n10000() {
        check_forward(10000, 1e-8);
    }

    #[test]
    fn forward_n12() {
        check_forward(12, 1e-13);
    }

    #[test]
    fn twiddle_cache_distinguishes_radix_order_for_same_length() {
        let input: Vec<Complex64> = (0..12)
            .map(|i| Complex64::new((i as f64 * 0.37).sin(), (i as f64 * 0.11).cos()))
            .collect();
        let expected = dft_forward(&input);

        let mut radix_3_4 = input.clone();
        forward_inplace_with_radices(&mut radix_3_4, &[3, 4]);
        assert!(
            max_err(&radix_3_4, &expected) < 1e-12,
            "radix [3,4] cache path must match direct DFT"
        );

        let mut radix_4_3 = input;
        forward_inplace_with_radices(&mut radix_4_3, &[4, 3]);
        assert!(
            max_err(&radix_4_3, &expected) < 1e-12,
            "radix [4,3] cache path must not reuse [3,4] twiddles"
        );
    }

    #[test]
    fn forward_n24() {
        check_forward(24, 1e-12);
    }
    #[test]
    fn forward_n48() {
        check_forward(48, 1e-12);
    }
    #[test]
    fn forward_n192() {
        check_forward(192, 1e-11);
    }
    #[test]
    fn forward_n384() {
        check_forward(384, 1e-10);
    }

    #[test]
    fn roundtrip_n100() {
        check_roundtrip(100, 1e-12);
    }
    #[test]
    fn roundtrip_n1000() {
        check_roundtrip(1000, 1e-11);
    }
    #[test]
    fn roundtrip_n14() {
        check_roundtrip(14, 1e-12);
    }
    #[test]
    fn roundtrip_n10000() {
        check_roundtrip(10000, 1e-10);
    }

    #[test]
    fn inverse_n14() {
        check_inverse(14, 1e-12);
    }
    #[test]
    fn inverse_n100() {
        check_inverse(100, 1e-11);
    }
    #[test]
    fn inverse_n1000() {
        check_inverse(1000, 1e-10);
    }

    #[test]
    fn forward_dc_n100() {
        let mut buf = vec![Complex64::new(1.0, 0.0); 100];
        forward_inplace_64(&mut buf);
        assert!((buf[0] - Complex64::new(100.0, 0.0)).norm() < 1e-10);
        for x in &buf[1..] {
            assert!(x.norm() < 1e-10, "non-zero bin: {:?}", x);
        }
    }

    #[test]
    fn forward_dc_n1000() {
        let mut buf = vec![Complex64::new(1.0, 0.0); 1000];
        forward_inplace_64(&mut buf);
        assert!((buf[0] - Complex64::new(1000.0, 0.0)).norm() < 1e-9);
        for x in &buf[1..] {
            assert!(x.norm() < 1e-9, "non-zero bin: {:?}", x);
        }
    }

    #[test]
    fn forward_f32_n100_matches_f64_reference() {
        let input: Vec<Complex64> = (0..100usize)
            .map(|k| Complex64::new((k as f64 * 0.29).sin(), (k as f64 * 0.47).cos()))
            .collect();
        let expected = dft_forward(&input);
        let mut buf: Vec<Complex32> = input
            .iter()
            .map(|x| Complex32::new(x.re as f32, x.im as f32))
            .collect();
        forward_inplace_32(&mut buf);
        let got: Vec<Complex64> = buf
            .iter()
            .map(|x| Complex64::new(x.re as f64, x.im as f64))
            .collect();
        let err = max_err(&got, &expected);
        assert!(err < 1e-4, "f32 forward N=100 max_err={err:.2e}");
    }

    #[test]
    fn forward_f32_n1000_matches_f64_reference() {
        let input: Vec<Complex64> = (0..1000usize)
            .map(|k| Complex64::new((k as f64 * 0.13).sin(), (k as f64 * 0.31).cos()))
            .collect();
        let expected = dft_forward(&input);
        let mut buf: Vec<Complex32> = input
            .iter()
            .map(|x| Complex32::new(x.re as f32, x.im as f32))
            .collect();
        forward_inplace_32(&mut buf);
        let got: Vec<Complex64> = buf
            .iter()
            .map(|x| Complex64::new(x.re as f64, x.im as f64))
            .collect();
        let err = max_err(&got, &expected);
        assert!(err < 2e-3, "f32 forward N=1000 max_err={err:.2e}");
    }
}

