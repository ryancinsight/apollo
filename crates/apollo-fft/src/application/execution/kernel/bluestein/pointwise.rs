//! Generic pointwise kernel operations for the Bluestein chirp-Z transform.
//!
//! All functions are monomorphized over `C: BluesteinScalar`, replacing the
//! former `_64`/`_32` type-suffixed duplicates.  The compiler emits machine code
//! identical to the former hand-written specializations through zero-cost
//! monomorphization; no vtable or dynamic dispatch is present.

#![allow(clippy::uninit_vec)]

use super::scalar::BluesteinScalar;
use rayon::prelude::*;

const BLUESTEIN_PARALLEL_POINTWISE_THRESHOLD: usize = 65_536;
const BLUESTEIN_PARALLEL_POINTWISE_CHUNK: usize = 4_096;

#[inline]
fn has_avx_fma() -> bool {
    super::has_avx_fma()
}

// ── Zero-fill ─────────────────────────────────────────────────────────────────

/// Zero-initialize `dst` via `write_bytes` — safe because `C: BluesteinScalar`
/// has no drop glue (it is `Copy`) and all-zero bytes form a valid `C` value.
#[inline]
pub(crate) fn zero_fill<C: BluesteinScalar>(dst: &mut [C]) {
    if dst.is_empty() {
        return;
    }
    // SAFETY: `C` is Copy + no drop glue; all-zero bytes are a valid `C`.
    unsafe {
        std::ptr::write_bytes(dst.as_mut_ptr().cast::<u8>(), 0, dst.len() * C::BYTE_SIZE);
    }
}

// ── fill_and_mul: dst[i] = input[i] * factors[i] ─────────────────────────────

#[inline]
pub(crate) fn fill_and_mul_from_input<C: BluesteinScalar>(
    dst: &mut [C],
    input: &[C],
    factors: &[C],
) {
    debug_assert_eq!(dst.len(), input.len());
    debug_assert_eq!(dst.len(), factors.len());
    if dst.len() >= BLUESTEIN_PARALLEL_POINTWISE_THRESHOLD {
        par_fill_and_mul_from_input(dst, input, factors);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= C::SIMD_MIN && has_avx_fma() {
        unsafe {
            C::avx_fill_mul(dst, input, factors);
            return;
        }
    }
    for ((out, in_v), factor) in dst.iter_mut().zip(input.iter()).zip(factors.iter()) {
        *out = *in_v * *factor;
    }
}

#[inline]
pub(crate) fn fill_and_mul_from_input_conj<C: BluesteinScalar>(
    dst: &mut [C],
    input: &[C],
    factors: &[C],
) {
    debug_assert_eq!(dst.len(), input.len());
    debug_assert_eq!(dst.len(), factors.len());
    if dst.is_empty() {
        return;
    }
    if dst.len() >= BLUESTEIN_PARALLEL_POINTWISE_THRESHOLD {
        par_fill_and_mul_from_input_conj(dst, input, factors);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= C::SIMD_MIN && has_avx_fma() {
        unsafe {
            C::avx_fill_mul_conj(dst, input, factors);
            return;
        }
    }
    for (out, (in_v, factor)) in dst.iter_mut().zip(input.iter().zip(factors.iter())) {
        *out = C::conj_mul(*in_v, *factor);
    }
}

// ── mul_pointwise_with_twiddle: dst[i] *= twiddle[i] ─────────────────────────

#[inline]
pub(crate) fn mul_pointwise_with_twiddle<C: BluesteinScalar>(dst: &mut [C], twiddle: &[C]) {
    debug_assert_eq!(dst.len(), twiddle.len());
    if dst.is_empty() {
        return;
    }
    if dst.len() >= BLUESTEIN_PARALLEL_POINTWISE_THRESHOLD {
        par_mul_pointwise_inplace(dst, twiddle);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= C::SIMD_MIN && has_avx_fma() {
        unsafe {
            C::avx_mul_inplace(dst, twiddle);
            return;
        }
    }
    for (out, factor) in dst.iter_mut().zip(twiddle.iter()) {
        *out *= *factor;
    }
}

/// Inverse twiddle kernel: element 0 is conjugate-multiplied by `twiddle[0]`;
/// elements `1..` are conjugate-multiplied against `twiddle` read in reverse.
#[inline]
pub(crate) fn mul_pointwise_with_twiddle_inverse_kernel<C: BluesteinScalar>(
    dst: &mut [C],
    twiddle: &[C],
) {
    debug_assert_eq!(dst.len(), twiddle.len());
    if dst.is_empty() {
        return;
    }
    if dst.len() >= BLUESTEIN_PARALLEL_POINTWISE_THRESHOLD {
        if has_avx_fma() && rayon::current_num_threads() == 1 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                C::avx_mul_inplace_inverse(dst, twiddle);
                return;
            }
        }
        let (head, tail) = dst.split_at_mut(1);
        if let Some(h) = head.first_mut() {
            *h = C::conj_mul(*h, twiddle[0]);
        }
        par_mul_pointwise_inplace_inverse_tail(tail, twiddle, 1);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if dst.len() >= C::SIMD_MIN && has_avx_fma() {
        unsafe {
            C::avx_mul_inplace_inverse(dst, twiddle);
            return;
        }
    }
    let (head, tail) = dst.split_at_mut(1);
    if let Some(h) = head.first_mut() {
        *h = C::conj_mul(*h, twiddle[0]);
    }
    for (out, factor) in tail.iter_mut().zip(twiddle[1..].iter().rev()) {
        *out = C::conj_mul(*out, *factor);
    }
}

// ── Parallel helpers ──────────────────────────────────────────────────────────

#[inline]
fn par_fill_and_mul_from_input<C: BluesteinScalar>(dst: &mut [C], input: &[C], factors: &[C]) {
    let use_avx = has_avx_fma();
    if use_avx {
        dst.par_chunks_mut(BLUESTEIN_PARALLEL_POINTWISE_CHUNK)
            .zip(input.par_chunks(BLUESTEIN_PARALLEL_POINTWISE_CHUNK))
            .zip(factors.par_chunks(BLUESTEIN_PARALLEL_POINTWISE_CHUNK))
            .for_each(|((dst_chunk, input_chunk), factor_chunk)| {
                if dst_chunk.len() >= C::SIMD_MIN {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        C::avx_fill_mul(dst_chunk, input_chunk, factor_chunk);
                    }
                    return;
                }
                for ((out, in_v), factor) in dst_chunk
                    .iter_mut()
                    .zip(input_chunk.iter())
                    .zip(factor_chunk.iter())
                {
                    *out = *in_v * *factor;
                }
            });
    } else {
        dst.par_iter_mut()
            .zip(input.par_iter().zip(factors.par_iter()))
            .for_each(|(out, (in_v, factor))| *out = *in_v * *factor);
    }
}

#[inline]
fn par_fill_and_mul_from_input_conj<C: BluesteinScalar>(dst: &mut [C], input: &[C], factors: &[C]) {
    let use_avx = has_avx_fma();
    if use_avx {
        dst.par_chunks_mut(BLUESTEIN_PARALLEL_POINTWISE_CHUNK)
            .zip(input.par_chunks(BLUESTEIN_PARALLEL_POINTWISE_CHUNK))
            .zip(factors.par_chunks(BLUESTEIN_PARALLEL_POINTWISE_CHUNK))
            .for_each(|((dst_chunk, input_chunk), factor_chunk)| {
                if dst_chunk.len() >= C::SIMD_MIN {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        C::avx_fill_mul_conj(dst_chunk, input_chunk, factor_chunk);
                    }
                    return;
                }
                for (out, (in_v, factor)) in dst_chunk
                    .iter_mut()
                    .zip(input_chunk.iter().zip(factor_chunk.iter()))
                {
                    *out = C::conj_mul(*in_v, *factor);
                }
            });
    } else {
        dst.par_iter_mut()
            .zip(input.par_iter().zip(factors.par_iter()))
            .for_each(|(out, (in_v, factor))| *out = C::conj_mul(*in_v, *factor));
    }
}

#[inline]
fn par_mul_pointwise_inplace<C: BluesteinScalar>(dst: &mut [C], twiddle: &[C]) {
    let use_avx = has_avx_fma();
    if use_avx {
        dst.par_chunks_mut(BLUESTEIN_PARALLEL_POINTWISE_CHUNK)
            .zip(twiddle.par_chunks(BLUESTEIN_PARALLEL_POINTWISE_CHUNK))
            .for_each(|(dst_chunk, factor_chunk)| {
                if dst_chunk.len() >= C::SIMD_MIN {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        C::avx_mul_inplace(dst_chunk, factor_chunk);
                    }
                    return;
                }
                for (out, factor) in dst_chunk.iter_mut().zip(factor_chunk.iter()) {
                    *out *= *factor;
                }
            });
    } else {
        dst.par_iter_mut()
            .zip(twiddle.par_iter())
            .for_each(|(out, factor)| *out *= *factor);
    }
}

#[inline]
fn par_mul_pointwise_inplace_inverse_tail<C: BluesteinScalar>(
    dst: &mut [C],
    twiddle: &[C],
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
            if use_avx && chunk_len >= C::SIMD_MIN {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    C::avx_mul_inplace_inverse_chunk(chunk, twiddle, factor_base);
                    return;
                }
            }
            for (offset, out) in chunk.iter_mut().enumerate() {
                let factor = twiddle[full_len - (base_index + chunk_start + offset)];
                *out = C::conj_mul(*out, factor);
            }
        });
}
