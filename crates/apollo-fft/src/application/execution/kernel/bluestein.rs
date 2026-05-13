#![allow(clippy::uninit_vec)]
//! Bluestein chirp-Z FFT for arbitrary-length DFT.
//!
//! ## Mathematical derivation
//!
//! Using the Bluestein identity kn = [-(k-n)^2 + k^2 + n^2] / 2:
//! X[k] = chirp[k] * (a_M circular_conv b_M)[k]
//! where chirp[k] = exp(-pi*i*k^2/N), a_M[n] = x[n]*chirp[n] (zero-padded to M),
//! b_M is the filter exp(+pi*i*m^2/N) arranged for circular convolution,
//! and M = next_pow2(2N-1).

#![allow(clippy::empty_line_after_doc_comments)]

use super::mixed_radix;

use super::radix_stage::{normalize_inplace_c32, normalize_inplace_c64};
use num_complex::{Complex32, Complex64};
use parking_lot::RwLock;
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::Arc;
use std::sync::OnceLock;

/// Newtype wrapper that makes a raw mutable pointer `Send + Sync`.
///
/// # Safety
/// The caller must guarantee that the pointer's backing storage lives for the
/// duration of any parallel section and that non-overlapping sub-slices are
/// handed out to different threads (enforced by the chunk decomposition below).

static BLUESTEIN_PLAN_CACHE_64: std::sync::LazyLock<RwLock<HashMap<usize, Arc<BluesteinPlan64>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
static BLUESTEIN_PLAN_CACHE_32: std::sync::LazyLock<RwLock<HashMap<usize, Arc<BluesteinPlan32>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

thread_local! {
    static BLUESTEIN_PLAN_TL_64: RefCell<HashMap<usize, Arc<BluesteinPlan64>>> =
        RefCell::new(HashMap::new());
    static BLUESTEIN_PLAN_TL_32: RefCell<HashMap<usize, Arc<BluesteinPlan32>>> =
        RefCell::new(HashMap::new());
    static BLUESTEIN_SCRATCH_64: RefCell<Vec<Complex64>> = const { RefCell::new(Vec::new()) };
    static BLUESTEIN_SCRATCH_32: RefCell<Vec<Complex32>> = const { RefCell::new(Vec::new()) };
}

const BLUESTEIN_PARALLEL_POINTWISE_THRESHOLD: usize = 65_536;
const BLUESTEIN_PARALLEL_POINTWISE_CHUNK: usize = 4_096;
const BLUESTEIN_SIMD_POINTWISE_MIN: usize = 8;

#[inline]
#[cfg(target_arch = "x86_64")]
fn has_avx_fma() -> bool {
    static HAS_AVX_FMA: OnceLock<bool> = OnceLock::new();
    *HAS_AVX_FMA.get_or_init(|| {
        std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma")
    })
}

#[inline]
#[cfg(not(target_arch = "x86_64"))]
const fn has_avx_fma() -> bool {
    false
}

#[inline]
fn zero_fill_complex64(dst: &mut [Complex64]) {
    if dst.is_empty() {
        return;
    }
    // Safety: `Complex64` is a plain two-`f64` POD type (no drop glue).
    unsafe {
        std::ptr::write_bytes(
            dst.as_mut_ptr().cast::<u8>(),
            0,
            dst.len() * size_of::<Complex64>(),
        );
    }
}

#[inline]
fn zero_fill_complex32(dst: &mut [Complex32]) {
    if dst.is_empty() {
        return;
    }
    // Safety: `Complex32` is a plain two-`f32` POD type (no drop glue).
    unsafe {
        std::ptr::write_bytes(
            dst.as_mut_ptr().cast::<u8>(),
            0,
            dst.len() * size_of::<Complex32>(),
        );
    }
}

#[inline]
fn par_fill_and_mul_from_input_64(
    dst: &mut [Complex64],
    input: &[Complex64],
    factors: &[Complex64],
) {
    let use_avx = has_avx_fma();
    if use_avx {
        dst.par_chunks_mut(BLUESTEIN_PARALLEL_POINTWISE_CHUNK)
            .zip(input.par_chunks(BLUESTEIN_PARALLEL_POINTWISE_CHUNK))
            .zip(factors.par_chunks(BLUESTEIN_PARALLEL_POINTWISE_CHUNK))
            .for_each(|((dst_chunk, input_chunk), factor_chunk)| {
                if dst_chunk.len() >= BLUESTEIN_SIMD_POINTWISE_MIN {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        mul_complex_pointwise_64_avx_from_input(
                            dst_chunk,
                            input_chunk,
                            factor_chunk,
                        );
                    }
                } else {
                    for ((out, in_v), factor) in dst_chunk
                        .iter_mut()
                        .zip(input_chunk.iter())
                        .zip(factor_chunk.iter())
                    {
                        *out = *in_v * *factor;
                    }
                }
            });
    } else {
        dst.par_iter_mut()
            .zip(input.par_iter().zip(factors.par_iter()))
            .for_each(|(out, (in_v, factor))| *out = *in_v * *factor);
    }
}

#[inline]
fn par_fill_and_mul_from_input_64_conj(
    dst: &mut [Complex64],
    input: &[Complex64],
    factors: &[Complex64],
) {
    let use_avx = has_avx_fma();
    if use_avx {
        dst.par_chunks_mut(BLUESTEIN_PARALLEL_POINTWISE_CHUNK)
            .zip(input.par_chunks(BLUESTEIN_PARALLEL_POINTWISE_CHUNK))
            .zip(factors.par_chunks(BLUESTEIN_PARALLEL_POINTWISE_CHUNK))
            .for_each(|((dst_chunk, input_chunk), factor_chunk)| {
                if dst_chunk.len() >= BLUESTEIN_SIMD_POINTWISE_MIN {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        mul_complex_pointwise_64_avx_from_input_conj(
                            dst_chunk,
                            input_chunk,
                            factor_chunk,
                        );
                    }
                } else {
                    for ((out, in_v), factor) in dst_chunk
                        .iter_mut()
                        .zip(input_chunk.iter())
                        .zip(factor_chunk.iter())
                    {
                        let re = in_v.re * factor.re + in_v.im * factor.im;
                        let im = in_v.im * factor.re - in_v.re * factor.im;
                        *out = Complex64::new(re, im);
                    }
                }
            });
    } else {
        dst.par_iter_mut()
            .zip(input.par_iter().zip(factors.par_iter()))
            .for_each(|(out, (in_v, factor))| {
                let re = in_v.re * factor.re + in_v.im * factor.im;
                let im = in_v.im * factor.re - in_v.re * factor.im;
                *out = Complex64::new(re, im);
            });
    }
}

#[inline]
fn par_fill_and_mul_from_input_32(
    dst: &mut [Complex32],
    input: &[Complex32],
    factors: &[Complex32],
) {
    let use_avx = has_avx_fma();
    if use_avx {
        dst.par_chunks_mut(BLUESTEIN_PARALLEL_POINTWISE_CHUNK)
            .zip(input.par_chunks(BLUESTEIN_PARALLEL_POINTWISE_CHUNK))
            .zip(factors.par_chunks(BLUESTEIN_PARALLEL_POINTWISE_CHUNK))
            .for_each(|((dst_chunk, input_chunk), factor_chunk)| {
                if dst_chunk.len() >= BLUESTEIN_SIMD_POINTWISE_MIN {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        mul_complex_pointwise_32_avx_from_input(
                            dst_chunk,
                            input_chunk,
                            factor_chunk,
                        );
                    }
                } else {
                    for ((out, in_v), factor) in dst_chunk
                        .iter_mut()
                        .zip(input_chunk.iter())
                        .zip(factor_chunk.iter())
                    {
                        *out = *in_v * *factor;
                    }
                }
            });
    } else {
        dst.par_iter_mut()
            .zip(input.par_iter().zip(factors.par_iter()))
            .for_each(|(out, (in_v, factor))| *out = *in_v * *factor);
    }
}

#[inline]
fn par_fill_and_mul_from_input_32_conj(
    dst: &mut [Complex32],
    input: &[Complex32],
    factors: &[Complex32],
) {
    let use_avx = has_avx_fma();
    if use_avx {
        dst.par_chunks_mut(BLUESTEIN_PARALLEL_POINTWISE_CHUNK)
            .zip(input.par_chunks(BLUESTEIN_PARALLEL_POINTWISE_CHUNK))
            .zip(factors.par_chunks(BLUESTEIN_PARALLEL_POINTWISE_CHUNK))
            .for_each(|((dst_chunk, input_chunk), factor_chunk)| {
                if dst_chunk.len() >= BLUESTEIN_SIMD_POINTWISE_MIN {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        mul_complex_pointwise_32_avx_from_input_conj(
                            dst_chunk,
                            input_chunk,
                            factor_chunk,
                        );
                    }
                } else {
                    for ((out, in_v), factor) in dst_chunk
                        .iter_mut()
                        .zip(input_chunk.iter())
                        .zip(factor_chunk.iter())
                    {
                        let re = in_v.re * factor.re + in_v.im * factor.im;
                        let im = in_v.im * factor.re - in_v.re * factor.im;
                        *out = Complex32::new(re, im);
                    }
                }
            });
    } else {
        dst.par_iter_mut()
            .zip(input.par_iter().zip(factors.par_iter()))
            .for_each(|(out, (in_v, factor))| {
                let re = in_v.re * factor.re + in_v.im * factor.im;
                let im = in_v.im * factor.re - in_v.re * factor.im;
                *out = Complex32::new(re, im);
            });
    }
}

#[inline]
fn par_mul_pointwise_64_inplace(dst: &mut [Complex64], twiddle: &[Complex64]) {
    let use_avx = has_avx_fma();
    if use_avx {
        dst.par_chunks_mut(BLUESTEIN_PARALLEL_POINTWISE_CHUNK)
            .zip(twiddle.par_chunks(BLUESTEIN_PARALLEL_POINTWISE_CHUNK))
            .for_each(|(dst_chunk, factor_chunk)| {
                if dst_chunk.len() >= BLUESTEIN_SIMD_POINTWISE_MIN {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        mul_complex_pointwise_64_avx_inplace(dst_chunk, factor_chunk);
                    }
                } else {
                    for (out, factor) in dst_chunk.iter_mut().zip(factor_chunk.iter()) {
                        *out *= *factor;
                    }
                }
            });
    } else {
        dst.par_iter_mut()
            .zip(twiddle.par_iter())
            .for_each(|(out, factor)| *out *= *factor);
    }
}

#[inline]
fn par_mul_pointwise_32_inplace(dst: &mut [Complex32], twiddle: &[Complex32]) {
    let use_avx = has_avx_fma();
    if use_avx {
        dst.par_chunks_mut(BLUESTEIN_PARALLEL_POINTWISE_CHUNK)
            .zip(twiddle.par_chunks(BLUESTEIN_PARALLEL_POINTWISE_CHUNK))
            .for_each(|(dst_chunk, factor_chunk)| {
                if dst_chunk.len() >= BLUESTEIN_SIMD_POINTWISE_MIN {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        mul_complex_pointwise_32_avx_inplace(dst_chunk, factor_chunk);
                    }
                } else {
                    for (out, factor) in dst_chunk.iter_mut().zip(factor_chunk.iter()) {
                        *out *= *factor;
                    }
                }
            });
    } else {
        dst.par_iter_mut()
            .zip(twiddle.par_iter())
            .for_each(|(out, factor)| *out *= *factor);
    }
}

#[inline]
fn par_mul_pointwise_64_inplace_inverse_tail(
    dst: &mut [Complex64],
    twiddle: &[Complex64],
    base_index: usize,
) {
    if dst.is_empty() {
        return;
    }
    let full_len = twiddle.len();
    debug_assert!(base_index < full_len);
    debug_assert!(base_index + dst.len() <= full_len);
    let use_avx = has_avx_fma();
    // par_chunks_mut decomposes `dst` into non-overlapping mutable sub-slices,
    // avoiding raw pointer aliasing and eliminating the need for unsafe Send/Sync impls.
    dst.par_chunks_mut(BLUESTEIN_PARALLEL_POINTWISE_CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let chunk_start = chunk_idx * BLUESTEIN_PARALLEL_POINTWISE_CHUNK;
            let chunk_len = chunk.len();
            let factor_base = full_len - (base_index + chunk_start);
            if use_avx && chunk_len >= BLUESTEIN_SIMD_POINTWISE_MIN {
                unsafe {
                    mul_complex_pointwise_64_avx_inplace_inverse_chunk(chunk, twiddle, factor_base);
                }
            } else {
                for (offset, out) in chunk.iter_mut().enumerate() {
                    let factor = twiddle[full_len - (base_index + chunk_start + offset)];
                    let re = out.re * factor.re + out.im * factor.im;
                    let im = out.im * factor.re - out.re * factor.im;
                    *out = Complex64::new(re, im);
                }
            }
        });
}

#[inline]
fn par_mul_pointwise_32_inplace_inverse_tail(
    dst: &mut [Complex32],
    twiddle: &[Complex32],
    base_index: usize,
) {
    if dst.is_empty() {
        return;
    }
    let full_len = twiddle.len();
    debug_assert!(base_index < full_len);
    debug_assert!(base_index + dst.len() <= full_len);
    let use_avx = has_avx_fma();
    dst.par_chunks_mut(BLUESTEIN_PARALLEL_POINTWISE_CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let chunk_start = chunk_idx * BLUESTEIN_PARALLEL_POINTWISE_CHUNK;
            let chunk_len = chunk.len();
            let factor_base = full_len - (base_index + chunk_start);
            if use_avx && chunk_len >= BLUESTEIN_SIMD_POINTWISE_MIN {
                unsafe {
                    mul_complex_pointwise_32_avx_inplace_inverse_chunk(chunk, twiddle, factor_base);
                }
            } else {
                for (offset, out) in chunk.iter_mut().enumerate() {
                    let factor = twiddle[full_len - (base_index + chunk_start + offset)];
                    let re = out.re * factor.re + out.im * factor.im;
                    let im = out.im * factor.re - out.re * factor.im;
                    *out = Complex32::new(re, im);
                }
            }
        });
}

#[inline]
fn fill_and_mul_from_input_64(dst: &mut [Complex64], input: &[Complex64], factors: &[Complex64]) {
    debug_assert_eq!(dst.len(), input.len());
    debug_assert_eq!(dst.len(), factors.len());
    if dst.len() >= BLUESTEIN_PARALLEL_POINTWISE_THRESHOLD {
        par_fill_and_mul_from_input_64(dst, input, factors);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= BLUESTEIN_SIMD_POINTWISE_MIN && has_avx_fma() {
        unsafe {
            mul_complex_pointwise_64_avx_from_input(dst, input, factors);
            return;
        }
    }
    for ((out, in_v), factor) in dst.iter_mut().zip(input.iter()).zip(factors.iter()) {
        *out = *in_v * *factor;
    }
}

#[inline]
fn fill_and_mul_from_input_64_conj(
    dst: &mut [Complex64],
    input: &[Complex64],
    factors: &[Complex64],
) {
    debug_assert_eq!(dst.len(), input.len());
    debug_assert_eq!(dst.len(), factors.len());
    if dst.is_empty() {
        return;
    }
    if dst.len() >= BLUESTEIN_PARALLEL_POINTWISE_THRESHOLD {
        par_fill_and_mul_from_input_64_conj(dst, input, factors);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= BLUESTEIN_SIMD_POINTWISE_MIN && has_avx_fma() {
        unsafe {
            mul_complex_pointwise_64_avx_from_input_conj(dst, input, factors);
            return;
        }
    }
    for (out, (in_v, factor)) in dst.iter_mut().zip(input.iter().zip(factors.iter())) {
        let re = in_v.re * factor.re + in_v.im * factor.im;
        let im = in_v.im * factor.re - in_v.re * factor.im;
        *out = Complex64::new(re, im);
    }
}

#[inline]
fn fill_and_mul_from_input_32(dst: &mut [Complex32], input: &[Complex32], factors: &[Complex32]) {
    debug_assert_eq!(dst.len(), input.len());
    debug_assert_eq!(dst.len(), factors.len());
    if dst.len() >= BLUESTEIN_PARALLEL_POINTWISE_THRESHOLD {
        par_fill_and_mul_from_input_32(dst, input, factors);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= BLUESTEIN_SIMD_POINTWISE_MIN && has_avx_fma() {
        unsafe {
            mul_complex_pointwise_32_avx_from_input(dst, input, factors);
            return;
        }
    }
    for ((out, in_v), factor) in dst.iter_mut().zip(input.iter()).zip(factors.iter()) {
        *out = *in_v * *factor;
    }
}

#[inline]
fn fill_and_mul_from_input_32_conj(
    dst: &mut [Complex32],
    input: &[Complex32],
    factors: &[Complex32],
) {
    debug_assert_eq!(dst.len(), input.len());
    debug_assert_eq!(dst.len(), factors.len());
    if dst.is_empty() {
        return;
    }
    if dst.len() >= BLUESTEIN_PARALLEL_POINTWISE_THRESHOLD {
        par_fill_and_mul_from_input_32_conj(dst, input, factors);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= BLUESTEIN_SIMD_POINTWISE_MIN && has_avx_fma() {
        unsafe {
            mul_complex_pointwise_32_avx_from_input_conj(dst, input, factors);
            return;
        }
    }
    for (out, (in_v, factor)) in dst.iter_mut().zip(input.iter().zip(factors.iter())) {
        let re = in_v.re * factor.re + in_v.im * factor.im;
        let im = in_v.im * factor.re - in_v.re * factor.im;
        *out = Complex32::new(re, im);
    }
}

#[inline]
fn mul_pointwise_64_with_twiddle(dst: &mut [Complex64], twiddle: &[Complex64]) {
    debug_assert_eq!(dst.len(), twiddle.len());
    if dst.is_empty() {
        return;
    }
    if dst.len() >= BLUESTEIN_PARALLEL_POINTWISE_THRESHOLD {
        par_mul_pointwise_64_inplace(dst, twiddle);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= BLUESTEIN_SIMD_POINTWISE_MIN && has_avx_fma() {
        unsafe {
            mul_complex_pointwise_64_avx_inplace(dst, twiddle);
            return;
        }
    }
    for (out, factor) in dst.iter_mut().zip(twiddle.iter()) {
        *out *= *factor;
    }
}

#[inline]
fn mul_pointwise_64_with_twiddle_inverse_kernel(dst: &mut [Complex64], twiddle: &[Complex64]) {
    debug_assert_eq!(dst.len(), twiddle.len());
    if dst.is_empty() {
        return;
    }
    if dst.len() >= BLUESTEIN_PARALLEL_POINTWISE_THRESHOLD {
        if has_avx_fma() && rayon::current_num_threads() == 1 {
            unsafe {
                mul_complex_pointwise_64_avx_inplace_inverse(dst, twiddle);
                return;
            }
        }
        let (head, tail) = dst.split_at_mut(1);
        if let Some(head) = head.first_mut() {
            let factor = twiddle[0];
            // k=0: multiply by B_m_inv[0] = b_m[0].conj() → conjugate multiply
            let re = head.re * factor.re + head.im * factor.im;
            let im = head.im * factor.re - head.re * factor.im;
            *head = Complex64::new(re, im);
        }
        par_mul_pointwise_64_inplace_inverse_tail(tail, twiddle, 1);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= BLUESTEIN_SIMD_POINTWISE_MIN && has_avx_fma() {
        unsafe {
            mul_complex_pointwise_64_avx_inplace_inverse(dst, twiddle);
            return;
        }
    }
    let (head, tail) = dst.split_at_mut(1);
    if let Some(head) = head.first_mut() {
        let factor = twiddle[0];
        // k=0: B_m_inv[0] = b_m[0].conj() → conjugate multiply
        let re = head.re * factor.re + head.im * factor.im;
        let im = head.im * factor.re - head.re * factor.im;
        *head = Complex64::new(re, im);
    }
    for (out, factor) in tail.iter_mut().zip(twiddle[1..].iter().rev()) {
        let re = out.re * factor.re + out.im * factor.im;
        let im = out.im * factor.re - out.re * factor.im;
        *out = Complex64::new(re, im);
    }
}

#[inline]
fn mul_pointwise_32_with_twiddle(dst: &mut [Complex32], twiddle: &[Complex32]) {
    debug_assert_eq!(dst.len(), twiddle.len());
    if dst.is_empty() {
        return;
    }
    if dst.len() >= BLUESTEIN_PARALLEL_POINTWISE_THRESHOLD {
        par_mul_pointwise_32_inplace(dst, twiddle);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= BLUESTEIN_SIMD_POINTWISE_MIN && has_avx_fma() {
        unsafe {
            mul_complex_pointwise_32_avx_inplace(dst, twiddle);
            return;
        }
    }
    for (out, factor) in dst.iter_mut().zip(twiddle.iter()) {
        *out *= *factor;
    }
}

#[inline]
fn mul_pointwise_32_with_twiddle_inverse_kernel(dst: &mut [Complex32], twiddle: &[Complex32]) {
    debug_assert_eq!(dst.len(), twiddle.len());
    if dst.is_empty() {
        return;
    }
    if dst.len() >= BLUESTEIN_PARALLEL_POINTWISE_THRESHOLD {
        if has_avx_fma() && rayon::current_num_threads() == 1 {
            unsafe {
                mul_complex_pointwise_32_avx_inplace_inverse(dst, twiddle);
                return;
            }
        }
        let (head, tail) = dst.split_at_mut(1);
        if let Some(head) = head.first_mut() {
            let factor = twiddle[0];
            // k=0: B_m_inv[0] = b_m[0].conj() → conjugate multiply
            let re = head.re * factor.re + head.im * factor.im;
            let im = head.im * factor.re - head.re * factor.im;
            *head = Complex32::new(re, im);
        }
        par_mul_pointwise_32_inplace_inverse_tail(tail, twiddle, 1);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= BLUESTEIN_SIMD_POINTWISE_MIN && has_avx_fma() {
        unsafe {
            mul_complex_pointwise_32_avx_inplace_inverse(dst, twiddle);
            return;
        }
    }
    let (head, tail) = dst.split_at_mut(1);
    if let Some(head) = head.first_mut() {
        let factor = twiddle[0];
        // k=0: B_m_inv[0] = b_m[0].conj() → conjugate multiply
        let re = head.re * factor.re + head.im * factor.im;
        let im = head.im * factor.re - head.re * factor.im;
        *head = Complex32::new(re, im);
    }
    for (out, factor) in tail.iter_mut().zip(twiddle[1..].iter().rev()) {
        let re = out.re * factor.re + out.im * factor.im;
        let im = out.im * factor.re - out.re * factor.im;
        *out = Complex32::new(re, im);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn mul_complex_pointwise_64_avx_from_input(
    dst: &mut [Complex64],
    input: &[Complex64],
    twiddle: &[Complex64],
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_storeu_pd,
        _mm256_unpackhi_pd, _mm256_unpacklo_pd,
    };
    debug_assert_eq!(dst.len(), input.len());
    debug_assert_eq!(dst.len(), twiddle.len());

    let count = dst.len();
    let dst_f = dst.as_mut_ptr().cast::<f64>();
    let in_f = input.as_ptr().cast::<f64>();
    let tw_f = twiddle.as_ptr().cast::<f64>();
    let batches = count / 2;
    for b in 0..batches {
        let f = b * 4;
        let x = _mm256_loadu_pd(in_f.add(f));
        let w = _mm256_loadu_pd(tw_f.add(f));
        let x_perm = _mm256_permute_pd(x, 5);
        let ac = _mm256_unpacklo_pd(w, w);
        let bd = _mm256_unpackhi_pd(w, w);
        let yw = _mm256_fmaddsub_pd(ac, x, _mm256_mul_pd(bd, x_perm));
        _mm256_storeu_pd(dst_f.add(f), yw);
    }
    let tail = batches * 2;
    for i in tail..count {
        dst[i] = input[i] * twiddle[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn mul_complex_pointwise_64_avx_from_input_conj(
    dst: &mut [Complex64],
    input: &[Complex64],
    twiddle: &[Complex64],
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_set1_pd,
        _mm256_storeu_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd,
    };
    debug_assert_eq!(dst.len(), input.len());
    debug_assert_eq!(dst.len(), twiddle.len());

    let count = dst.len();
    let dst_f = dst.as_mut_ptr().cast::<f64>();
    let in_f = input.as_ptr().cast::<f64>();
    let tw_f = twiddle.as_ptr().cast::<f64>();
    let batches = count / 2;
    let neg = _mm256_set1_pd(-1.0);
    for b in 0..batches {
        let f = b * 4;
        let x = _mm256_loadu_pd(in_f.add(f));
        let w = _mm256_loadu_pd(tw_f.add(f));
        let x_perm = _mm256_permute_pd(x, 5);
        let ac = _mm256_unpacklo_pd(w, w);
        let bd = _mm256_mul_pd(_mm256_unpackhi_pd(w, w), neg);
        let yw = _mm256_fmaddsub_pd(ac, x, _mm256_mul_pd(bd, x_perm));
        _mm256_storeu_pd(dst_f.add(f), yw);
    }
    let tail = batches * 2;
    for i in tail..count {
        let in_v = input[i];
        let factor = twiddle[i];
        let re = in_v.re * factor.re + in_v.im * factor.im;
        let im = in_v.im * factor.re - in_v.re * factor.im;
        dst[i] = Complex64::new(re, im);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn mul_complex_pointwise_64_avx_inplace(dst: &mut [Complex64], twiddle: &[Complex64]) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_storeu_pd,
        _mm256_unpackhi_pd, _mm256_unpacklo_pd,
    };
    debug_assert_eq!(dst.len(), twiddle.len());

    let count = dst.len();
    let dst_f = dst.as_mut_ptr().cast::<f64>();
    let tw_f = twiddle.as_ptr().cast::<f64>();
    let batches = count / 2;
    for b in 0..batches {
        let f = b * 4;
        let x = _mm256_loadu_pd(dst_f.add(f));
        let w = _mm256_loadu_pd(tw_f.add(f));
        let x_perm = _mm256_permute_pd(x, 5);
        let ac = _mm256_unpacklo_pd(w, w);
        let bd = _mm256_unpackhi_pd(w, w);
        let yw = _mm256_fmaddsub_pd(ac, x, _mm256_mul_pd(bd, x_perm));
        _mm256_storeu_pd(dst_f.add(f), yw);
    }
    let tail = batches * 2;
    for i in tail..count {
        dst[i] *= twiddle[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn mul_complex_pointwise_64_avx_inplace_inverse(
    dst: &mut [Complex64],
    twiddle: &[Complex64],
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_set1_pd,
        _mm256_setr_pd, _mm256_storeu_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd,
    };
    debug_assert_eq!(dst.len(), twiddle.len());

    let len = dst.len();
    if len == 0 {
        return;
    }
    let factor = twiddle[0];
    // k=0: B_m_inv[0] = b_m[0].conj() → conjugate multiply
    let re = dst[0].re * factor.re + dst[0].im * factor.im;
    let im = dst[0].im * factor.re - dst[0].re * factor.im;
    dst[0] = Complex64::new(re, im);
    let count = len - 1;
    if count == 0 {
        return;
    }
    let dst_f = dst.as_mut_ptr().cast::<f64>();
    let batches = count / 2;
    let neg = _mm256_set1_pd(-1.0);
    for b in 0..batches {
        let dst_offset = 2 + b * 4;
        let x = _mm256_loadu_pd(dst_f.add(dst_offset));
        let first = twiddle[len - 1 - 2 * b];
        let second = twiddle[len - 2 - 2 * b];
        let w = _mm256_setr_pd(first.re, first.im, second.re, second.im);
        let x_perm = _mm256_permute_pd(x, 5);
        let ac = _mm256_unpacklo_pd(w, w);
        let bd = _mm256_mul_pd(_mm256_unpackhi_pd(w, w), neg);
        let yw = _mm256_fmaddsub_pd(ac, x, _mm256_mul_pd(bd, x_perm));
        _mm256_storeu_pd(dst_f.add(dst_offset), yw);
    }
    if count % 2 == 1 {
        let out = &mut dst[len - 1];
        let factor = twiddle[1];
        let re = out.re * factor.re + out.im * factor.im;
        let im = out.im * factor.re - out.re * factor.im;
        *out = Complex64::new(re, im);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn mul_complex_pointwise_64_avx_inplace_inverse_chunk(
    dst: &mut [Complex64],
    twiddle: &[Complex64],
    factor_base: usize,
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_set1_pd,
        _mm256_setr_pd, _mm256_storeu_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd,
    };
    let count = dst.len();
    if count == 0 {
        return;
    }
    debug_assert!(factor_base < twiddle.len());
    debug_assert!(factor_base + 1 >= count);
    let dst_f = dst.as_mut_ptr().cast::<f64>();
    let neg = _mm256_set1_pd(-1.0);
    for b in 0..(count / 2) {
        let dst_offset = b * 4;
        let first = twiddle[factor_base - 2 * b];
        let second = twiddle[factor_base - (2 * b + 1)];
        let w = _mm256_setr_pd(first.re, first.im, second.re, second.im);
        let x = _mm256_loadu_pd(dst_f.add(dst_offset));
        let x_perm = _mm256_permute_pd(x, 5);
        let ac = _mm256_unpacklo_pd(w, w);
        let bd = _mm256_mul_pd(_mm256_unpackhi_pd(w, w), neg);
        let yw = _mm256_fmaddsub_pd(ac, x, _mm256_mul_pd(bd, x_perm));
        _mm256_storeu_pd(dst_f.add(dst_offset), yw);
    }
    if count % 2 == 1 {
        let out = &mut dst[count - 1];
        let factor = twiddle[factor_base - (count - 1)];
        let re = out.re * factor.re + out.im * factor.im;
        let im = out.im * factor.re - out.re * factor.im;
        *out = Complex64::new(re, im);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn mul_complex_pointwise_32_avx_from_input(
    dst: &mut [Complex32],
    input: &[Complex32],
    twiddle: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_ps, _mm256_loadu_ps, _mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_mul_ps,
        _mm256_permute_ps, _mm256_storeu_ps,
    };
    debug_assert_eq!(dst.len(), input.len());
    debug_assert_eq!(dst.len(), twiddle.len());

    let count = dst.len();
    let dst_f = dst.as_mut_ptr().cast::<f32>();
    let in_f = input.as_ptr().cast::<f32>();
    let tw_f = twiddle.as_ptr().cast::<f32>();
    let batches = count / 4;
    for b in 0..batches {
        let f = b * 8;
        let x = _mm256_loadu_ps(in_f.add(f));
        let w = _mm256_loadu_ps(tw_f.add(f));
        let w_re = _mm256_moveldup_ps(w);
        let w_im = _mm256_movehdup_ps(w);
        let x_perm = _mm256_permute_ps(x, 0xB1);
        let yw = _mm256_fmaddsub_ps(w_re, x, _mm256_mul_ps(w_im, x_perm));
        _mm256_storeu_ps(dst_f.add(f), yw);
    }
    let tail = batches * 4;
    for i in tail..count {
        dst[i] = input[i] * twiddle[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn mul_complex_pointwise_32_avx_from_input_conj(
    dst: &mut [Complex32],
    input: &[Complex32],
    twiddle: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_ps, _mm256_loadu_ps, _mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_mul_ps,
        _mm256_permute_ps, _mm256_set1_ps, _mm256_storeu_ps,
    };
    debug_assert_eq!(dst.len(), input.len());
    debug_assert_eq!(dst.len(), twiddle.len());

    let count = dst.len();
    let dst_f = dst.as_mut_ptr().cast::<f32>();
    let in_f = input.as_ptr().cast::<f32>();
    let tw_f = twiddle.as_ptr().cast::<f32>();
    let batches = count / 4;
    let neg = _mm256_set1_ps(-1.0);
    for b in 0..batches {
        let f = b * 8;
        let x = _mm256_loadu_ps(in_f.add(f));
        let w = _mm256_loadu_ps(tw_f.add(f));
        let w_re = _mm256_moveldup_ps(w);
        let w_im = _mm256_mul_ps(_mm256_movehdup_ps(w), neg);
        let x_perm = _mm256_permute_ps(x, 0xB1);
        let yw = _mm256_fmaddsub_ps(w_re, x, _mm256_mul_ps(w_im, x_perm));
        _mm256_storeu_ps(dst_f.add(f), yw);
    }
    let tail = batches * 4;
    for i in tail..count {
        let in_v = input[i];
        let factor = twiddle[i];
        let re = in_v.re * factor.re + in_v.im * factor.im;
        let im = in_v.im * factor.re - in_v.re * factor.im;
        dst[i] = Complex32::new(re, im);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn mul_complex_pointwise_32_avx_inplace_inverse(
    dst: &mut [Complex32],
    twiddle: &[Complex32],
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_ps, _mm256_loadu_ps, _mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_mul_ps,
        _mm256_permute_ps, _mm256_set1_ps, _mm256_setr_ps, _mm256_storeu_ps,
    };
    debug_assert_eq!(dst.len(), twiddle.len());

    let len = dst.len();
    if len == 0 {
        return;
    }
    let factor = twiddle[0];
    // k=0: B_m_inv[0] = b_m[0].conj() → conjugate multiply
    let re = dst[0].re * factor.re + dst[0].im * factor.im;
    let im = dst[0].im * factor.re - dst[0].re * factor.im;
    dst[0] = Complex32::new(re, im);
    let count = len - 1;
    if count == 0 {
        return;
    }
    let dst_f = dst.as_mut_ptr().cast::<f32>();
    let batches = count / 4;
    let neg = _mm256_set1_ps(-1.0);
    for b in 0..batches {
        let dst_offset = 2 + b * 8;
        let x = _mm256_loadu_ps(dst_f.add(dst_offset));
        let i0 = len - 1 - 4 * b;
        let i1 = len - 2 - 4 * b;
        let i2 = len - 3 - 4 * b;
        let i3 = len - 4 - 4 * b;
        let w0 = twiddle[i0];
        let w1 = twiddle[i1];
        let w2 = twiddle[i2];
        let w3 = twiddle[i3];
        let w = _mm256_setr_ps(w0.re, w0.im, w1.re, w1.im, w2.re, w2.im, w3.re, w3.im);
        let w_re = _mm256_moveldup_ps(w);
        let w_im = _mm256_mul_ps(_mm256_movehdup_ps(w), neg);
        let x_perm = _mm256_permute_ps(x, 0xB1);
        let yw = _mm256_fmaddsub_ps(w_re, x, _mm256_mul_ps(w_im, x_perm));
        _mm256_storeu_ps(dst_f.add(dst_offset), yw);
    }
    let rem = count % 4;
    if rem != 0 {
        let base = (count / 4) * 4 + 1;
        let batches = count / 4;
        for k in 0..rem {
            let out = &mut dst[base + k];
            let factor = twiddle[len - (4 * batches + k + 1)];
            let re = out.re * factor.re + out.im * factor.im;
            let im = out.im * factor.re - out.re * factor.im;
            *out = Complex32::new(re, im);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn mul_complex_pointwise_32_avx_inplace_inverse_chunk(
    dst: &mut [Complex32],
    twiddle: &[Complex32],
    factor_base: usize,
) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_ps, _mm256_loadu_ps, _mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_mul_ps,
        _mm256_permute_ps, _mm256_set1_ps, _mm256_setr_ps, _mm256_storeu_ps,
    };
    let count = dst.len();
    if count == 0 {
        return;
    }
    debug_assert!(factor_base < twiddle.len());
    debug_assert!(factor_base + 1 >= count);
    let dst_f = dst.as_mut_ptr().cast::<f32>();
    let neg = _mm256_set1_ps(-1.0);
    for b in 0..(count / 4) {
        let dst_offset = b * 8;
        let w0 = twiddle[factor_base - 4 * b];
        let w1 = twiddle[factor_base - (4 * b + 1)];
        let w2 = twiddle[factor_base - (4 * b + 2)];
        let w3 = twiddle[factor_base - (4 * b + 3)];
        let w = _mm256_setr_ps(w0.re, w0.im, w1.re, w1.im, w2.re, w2.im, w3.re, w3.im);
        let x = _mm256_loadu_ps(dst_f.add(dst_offset));
        let x_perm = _mm256_permute_ps(x, 0xB1);
        let w_re = _mm256_moveldup_ps(w);
        let w_im = _mm256_mul_ps(_mm256_movehdup_ps(w), neg);
        let yw = _mm256_fmaddsub_ps(w_re, x, _mm256_mul_ps(w_im, x_perm));
        _mm256_storeu_ps(dst_f.add(dst_offset), yw);
    }
    if count % 4 != 0 {
        let base = (count / 4) * 4;
        for k in 0..(count % 4) {
            let out = &mut dst[base + k];
            let factor = twiddle[factor_base - (base + k)];
            let re = out.re * factor.re + out.im * factor.im;
            let im = out.im * factor.re - out.re * factor.im;
            *out = Complex32::new(re, im);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn mul_complex_pointwise_32_avx_inplace(dst: &mut [Complex32], twiddle: &[Complex32]) {
    use std::arch::x86_64::{
        _mm256_fmaddsub_ps, _mm256_loadu_ps, _mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_mul_ps,
        _mm256_permute_ps, _mm256_storeu_ps,
    };
    debug_assert_eq!(dst.len(), twiddle.len());

    let count = dst.len();
    let dst_f = dst.as_mut_ptr().cast::<f32>();
    let tw_f = twiddle.as_ptr().cast::<f32>();
    let batches = count / 4;
    for b in 0..batches {
        let f = b * 8;
        let x = _mm256_loadu_ps(dst_f.add(f));
        let w = _mm256_loadu_ps(tw_f.add(f));
        let w_re = _mm256_moveldup_ps(w);
        let w_im = _mm256_movehdup_ps(w);
        let x_perm = _mm256_permute_ps(x, 0xB1);
        let yw = _mm256_fmaddsub_ps(w_re, x, _mm256_mul_ps(w_im, x_perm));
        _mm256_storeu_ps(dst_f.add(f), yw);
    }
    let tail = batches * 4;
    for i in tail..count {
        dst[i] *= twiddle[i];
    }
}

#[inline]
fn cached_plan64(n: usize) -> Arc<BluesteinPlan64> {
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
fn cached_plan32(n: usize) -> Arc<BluesteinPlan32> {
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

#[inline]
fn with_scratch64<R, F>(m: usize, f: F) -> R
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
fn with_scratch32<R, F>(m: usize, f: F) -> R
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

/// Precomputed context for arbitrary-length Bluestein chirp-Z transform.
/// Eliminates `O(N)` dynamic memory allocations per kernel evaluation.
///
/// # Chirp filter semantics
///
/// - `chirp[k]` = `exp(-πi k²/N)` — forward chirp factor
/// - `b_m` = FFT of filter `b[m] = chirp[m].conj()` = `exp(+πi m²/N)`,
///   used for the forward-transform convolution step
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
    /// Initialize a new Bluestein plan for length `n`.
    ///
    /// The M-point twiddle tables are NOT stored in the plan; `mixed_radix`
    /// builds and caches per-radix twiddle tables in its own thread-local store,
    /// keyed by size, ensuring the format matches the dispatched kernel (radix-2,
    /// radix-4, or radix-8) rather than the radix-2 format built by
    /// `radix2::build_forward_twiddle_table_64`.
    pub fn new(n: usize) -> Self {
        let m = (2 * n.saturating_sub(1).max(1)).next_power_of_two();
        let chirp: Vec<Complex64> = (0..n)
            .map(|k| {
                let angle = -std::f64::consts::PI * (k * k) as f64 / n as f64;
                Complex64::new(angle.cos(), angle.sin())
            })
            .collect();

        let mut b_m = Vec::with_capacity(m);
        // SAFETY: every value read by the subsequent FFT is initialized below:
        // DC is written, mirrored chirp entries are written, and the remaining
        // convolution gap is zero-filled before `b_m` is read.
        unsafe { b_m.set_len(m) };
        b_m[0] = Complex64::new(1.0, 0.0);
        for k in 1..n {
            let bk = chirp[k].conj();
            b_m[k] = bk;
            b_m[m - k] = bk;
        }
        let gap_end = m + 1 - n;
        if n < gap_end {
            zero_fill_complex64(&mut b_m[n..gap_end]);
        }
        // None: let mixed_radix use its own cached twiddle tables in the correct
        // format for the radix (8, 4, or 2) selected for size m.
        mixed_radix::forward_inplace_64_with_twiddles(&mut b_m, None);

        Self { n, m, chirp, b_m }
    }

    /// Retrieve the working padded dimension M.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Forward transform into data given a pre-allocated scratch sequence of length `M`.
    ///
    /// Uses the precomputed `b_m` filter and scratch reuse; avoids per-call chirp/twiddle
    /// synthesis overhead.
    pub fn forward_with_scratch(&self, data: &mut [Complex64], scratch_a: &mut [Complex64]) {
        assert_eq!(data.len(), self.n);
        assert_eq!(scratch_a.len(), self.m);

        {
            let (scratch_head, scratch_tail) = scratch_a.split_at_mut(self.n);
            fill_and_mul_from_input_64(scratch_head, data, &self.chirp);
            if !scratch_tail.is_empty() {
                zero_fill_complex64(scratch_tail);
            }
        } // scratch_head and scratch_tail borrows end here

        mixed_radix::forward_inplace_64_with_twiddles(scratch_a, None);
        mul_pointwise_64_with_twiddle(scratch_a, &self.b_m);
        mixed_radix::inverse_inplace_64_with_twiddles(scratch_a, None);

        let scratch_head = &scratch_a[..self.n];
        fill_and_mul_from_input_64(data, scratch_head, &self.chirp);
    }

    /// Inverse unnormalized transform.
    pub fn inverse_unnorm_with_scratch(&self, data: &mut [Complex64], scratch_a: &mut [Complex64]) {
        assert_eq!(data.len(), self.n);
        assert_eq!(scratch_a.len(), self.m);

        {
            let (scratch_head, scratch_tail) = scratch_a.split_at_mut(self.n);
            fill_and_mul_from_input_64_conj(scratch_head, data, &self.chirp);
            if !scratch_tail.is_empty() {
                zero_fill_complex64(scratch_tail);
            }
        } // scratch_head and scratch_tail borrows end here

        mixed_radix::forward_inplace_64_with_twiddles(scratch_a, None);
        mul_pointwise_64_with_twiddle_inverse_kernel(scratch_a, &self.b_m);
        mixed_radix::inverse_inplace_64_with_twiddles(scratch_a, None);

        let scratch_head = &scratch_a[..self.n];
        fill_and_mul_from_input_64_conj(data, scratch_head, &self.chirp);
    }
}

/// Cached per-length Bluestein context for `Complex32` transforms.
///
/// Stores chirp vectors and padded convolution kernels with the precomputed `b_m`
/// filter. The structure is cloneable via `Arc`
/// for lock-free sharing across transform invocations.
#[derive(Clone, Debug)]
pub struct BluesteinPlan32 {
    n: usize,
    m: usize,
    chirp: Vec<Complex32>,
    b_m: Vec<Complex32>,
}

impl BluesteinPlan32 {
    /// Build a reusable plan for a non-power-of-two length `n` transform.
    /// Twiddle tables are not stored; `mixed_radix` caches them per-radix.
    pub fn new(n: usize) -> Self {
        let m = (2 * n.saturating_sub(1).max(1)).next_power_of_two();
        let chirp: Vec<Complex32> = (0..n)
            .map(|k| {
                let angle = -(std::f64::consts::PI * (k as f64 * k as f64) / n as f64) as f32;
                Complex32::new(angle.cos(), angle.sin())
            })
            .collect();

        let mut b_m = Vec::with_capacity(m);
        // SAFETY: every value read by the subsequent FFT is initialized below:
        // DC is written, mirrored chirp entries are written, and the remaining
        // convolution gap is zero-filled before `b_m` is read.
        unsafe { b_m.set_len(m) };
        b_m[0] = Complex32::new(1.0, 0.0);
        for k in 1..n {
            let bk = chirp[k].conj();
            b_m[k] = bk;
            b_m[m - k] = bk;
        }
        let gap_end = m + 1 - n;
        if n < gap_end {
            zero_fill_complex32(&mut b_m[n..gap_end]);
        }
        mixed_radix::forward_inplace_32_with_twiddles(&mut b_m, None);

        Self { n, m, chirp, b_m }
    }

    /// Padded convolution size (next power-of-two of `2n - 1`).
    pub fn m(&self) -> usize {
        self.m
    }

    /// Forward transform using preallocated scratch and plan-owned metadata.
    pub fn forward_with_scratch(&self, data: &mut [Complex32], scratch_a: &mut [Complex32]) {
        assert_eq!(data.len(), self.n);
        assert_eq!(scratch_a.len(), self.m);

        {
            let (scratch_head, scratch_tail) = scratch_a.split_at_mut(self.n);
            fill_and_mul_from_input_32(scratch_head, data, &self.chirp);
            if !scratch_tail.is_empty() {
                zero_fill_complex32(scratch_tail);
            }
        }

        mixed_radix::forward_inplace_32_with_twiddles(scratch_a, None);
        mul_pointwise_32_with_twiddle(scratch_a, &self.b_m);
        mixed_radix::inverse_inplace_32_with_twiddles(scratch_a, None);

        let scratch_head = &scratch_a[..self.n];
        fill_and_mul_from_input_32(data, scratch_head, &self.chirp);
    }

    /// Forward inverse (unnormalized) transform using preallocated scratch.
    pub fn inverse_unnorm_with_scratch(&self, data: &mut [Complex32], scratch_a: &mut [Complex32]) {
        assert_eq!(data.len(), self.n);
        assert_eq!(scratch_a.len(), self.m);

        {
            let (scratch_head, scratch_tail) = scratch_a.split_at_mut(self.n);
            fill_and_mul_from_input_32_conj(scratch_head, data, &self.chirp);
            if !scratch_tail.is_empty() {
                zero_fill_complex32(scratch_tail);
            }
        }

        mixed_radix::forward_inplace_32_with_twiddles(scratch_a, None);
        mul_pointwise_32_with_twiddle_inverse_kernel(scratch_a, &self.b_m);
        mixed_radix::inverse_inplace_32_with_twiddles(scratch_a, None);

        let scratch_head = &scratch_a[..self.n];
        fill_and_mul_from_input_32_conj(data, scratch_head, &self.chirp);
    }
}

/// In-place forward Bluestein chirp-Z transform for `Complex64`.
///
/// Computes the exact discrete Fourier transform in $O(N \log N)$ time for arbitrary
/// and non-power-of-two lengths by zero-padding and using cyclic convolution.
/// When `data.len()` is a power-of-two, execution delegates to the radix-2 kernel.
pub fn forward_inplace_64(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        mixed_radix::forward_inplace_64(data);
        return;
    }
    let plan = cached_plan64(n);
    with_scratch64(plan.m(), |a_m| plan.forward_with_scratch(data, a_m));
}

/// In-place unnormalized inverse Bluestein chirp-Z transform for `Complex64`.
///
/// Computes the adjoint discrete Fourier transform without the $1/N$ scaling factor.
/// Employs conjugated chirp sequences while maintaining cyclic convolution guarantees.
pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        mixed_radix::inverse_inplace_unnorm_64(data);
        return;
    }
    let plan = cached_plan64(n);
    with_scratch64(plan.m(), |a_m| plan.inverse_unnorm_with_scratch(data, a_m));
}

/// In-place normalized inverse Bluestein chirp-Z transform for `Complex64`.
///
/// Computes the exact inverse discrete Fourier transform by evaluating the unnormalized
/// transformation and applying a $1/N$ scaling factor.
pub fn inverse_inplace_64(data: &mut [Complex64]) {
    inverse_inplace_unnorm_64(data);
    normalize_inplace_c64(data, 1.0 / data.len() as f64);
}

/// In-place forward Bluestein chirp-Z transform for `Complex32`.
///
/// Evaluates arbitrary-length DFT in-place on native `Complex32` with a cached
/// plan and reusable scratch.
pub fn forward_inplace_32(data: &mut [Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        mixed_radix::forward_inplace_32(data);
        return;
    }
    let plan = cached_plan32(n);
    with_scratch32(plan.m(), |a_m| plan.forward_with_scratch(data, a_m));
}

/// In-place unnormalized inverse Bluestein chirp-Z transform for `Complex32`.
///
/// Evaluates the adjoint arbitrary-length DFT without scaling on native `Complex32`
/// using cached kernels and reusable scratch.
pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        mixed_radix::inverse_inplace_unnorm_32(data);
        return;
    }
    let plan = cached_plan32(n);
    with_scratch32(plan.m(), |a_m| plan.inverse_unnorm_with_scratch(data, a_m));
}

/// In-place normalized inverse Bluestein chirp-Z transform for `Complex32`.
///
/// Sequentially computes the unnormalized inverse and scales by $1/N$.
pub fn inverse_inplace_32(data: &mut [Complex32]) {
    inverse_inplace_unnorm_32(data);
    normalize_inplace_c32(data, 1.0f32 / data.len() as f32);
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::max_abs_err_64 as max_abs_err;
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};

    fn sig(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|k| {
                let t = k as f64 / n as f64;
                Complex64::new(
                    (std::f64::consts::TAU * 3.0 * t).sin()
                        + 0.5 * (std::f64::consts::TAU * 7.0 * t).cos(),
                    0.25 * (std::f64::consts::TAU * 2.0 * t).sin(),
                )
            })
            .collect()
    }

    #[test]
    fn forward_matches_direct_for_n3() {
        let input = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let expected = dft_forward(&input);
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        assert!(max_abs_err(&got, &expected) < 1e-12);
    }

    #[test]
    fn forward_matches_direct_for_n5() {
        let input = sig(5);
        let expected = dft_forward(&input);
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        assert!(
            max_abs_err(&got, &expected) < 1e-12,
            "n=5 err={}",
            max_abs_err(&got, &expected)
        );
    }

    #[test]
    fn forward_matches_direct_for_n6() {
        let input = sig(6);
        let expected = dft_forward(&input);
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        assert!(
            max_abs_err(&got, &expected) < 1e-12,
            "n=6 err={}",
            max_abs_err(&got, &expected)
        );
    }

    #[test]
    fn forward_matches_direct_for_n7() {
        let input = sig(7);
        let expected = dft_forward(&input);
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        assert!(
            max_abs_err(&got, &expected) < 1e-12,
            "n=7 err={}",
            max_abs_err(&got, &expected)
        );
    }

    #[test]
    fn forward_matches_direct_for_n11() {
        let input = sig(11);
        let expected = dft_forward(&input);
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        assert!(
            max_abs_err(&got, &expected) < 1e-11,
            "n=11 err={}",
            max_abs_err(&got, &expected)
        );
    }

    #[test]
    fn roundtrip_for_non_power_of_two() {
        for &n in &[3usize, 5, 6, 7, 9, 11] {
            let input = sig(n);
            let mut data = input.clone();
            forward_inplace_64(&mut data);
            inverse_inplace_64(&mut data);
            let err = max_abs_err(&data, &input);
            assert!(err < 1e-11, "roundtrip n={n} err={err:.2e}");
        }
    }

    #[test]
    fn repeated_forward_same_input_same_output() {
        let input = sig(45);
        let mut a = input.clone();
        let mut b = input;
        forward_inplace_64(&mut a);
        forward_inplace_64(&mut b);
        let err = max_abs_err(&a, &b);
        assert!(err < 1e-12, "repeat determinism err={err:.2e}");
    }

    #[test]
    fn unnorm_inverse_differs_from_norm_by_n() {
        let n = 7usize;
        let input = sig(n);
        let mut spec = input.clone();
        forward_inplace_64(&mut spec);
        let mut unnorm = spec.clone();
        inverse_inplace_unnorm_64(&mut unnorm);
        let mut norm = spec.clone();
        inverse_inplace_64(&mut norm);
        let err = max_abs_err(
            &unnorm,
            &norm.iter().map(|x| x * n as f64).collect::<Vec<_>>(),
        );
        assert!(err < 1e-11, "unnorm/norm ratio failed n={n}: err={err:.2e}");
    }

    #[test]
    fn inverse_matches_direct_for_n5() {
        let input = sig(5);
        let expected = dft_inverse(&input);
        let mut got = input.clone();
        inverse_inplace_64(&mut got);
        assert!(
            max_abs_err(&got, &expected) < 1e-12,
            "inverse n=5 err={}",
            max_abs_err(&got, &expected)
        );
    }

    #[test]
    fn power_of_two_uses_stockham() {
        let n = 8usize;
        let input = sig(n);
        let mut bl = input.clone();
        forward_inplace_64(&mut bl);
        let mut st = input.clone();
        mixed_radix::forward_inplace_64(&mut st);
        assert!(max_abs_err(&bl, &st) < 1e-14, "bluestein vs stockham n=8");
    }
}
