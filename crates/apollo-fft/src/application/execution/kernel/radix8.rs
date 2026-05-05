//! Radix-8 DIT FFT using Winograd DFT-8 inner butterflies.
//!
//! ## Algorithm
//!
//! For an N-point transform where N is a power of 8, we perform an iterative
//! DIT in stages of radix 8.  Each stage processes groups of length `len = 8^s`
//! (for stage s = 1, 2, …, log₈ N):
//!
//! 1. Apply the digit-reverse permutation (base-8) once up front.
//! 2. For each stage, partition the data into non-overlapping groups of `len`
//!    elements.  Within each group, apply inter-group twiddle factors
//!    `W_len^{k·j}` to the non-first sub-blocks, then call the Winograd DFT-8
//!    kernel on each set of 8 elements separated by stride `len/8`.
//!
//! The Winograd DFT-8 kernel requires only 4 real multiplications (the
//! `±√2/2` factors on the odd path) and 26 real additions, versus 64
//! multiplications for the generic DFT-matrix approach.
//!
//! ## References
//!
//! - Winograd, S. (1978). On computing the discrete Fourier transform.
//!   *Mathematics of Computation*, 32(141), 175-199.

use super::radix2_f16::Cf16;
use super::{radix2, winograd};
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
use std::arch::x86_64::{
    _mm256_fmaddsub_pd, _mm256_mul_pd, _mm256_permute_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd,
};

#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline(always)]
unsafe fn cmul2_r8(
    a: std::arch::x86_64::__m256d,
    b: std::arch::x86_64::__m256d,
) -> std::arch::x86_64::__m256d {
    let ar = _mm256_unpacklo_pd(a, a);
    let ai = _mm256_unpackhi_pd(a, a);
    let bsw = _mm256_permute_pd(b, 0b0101);
    _mm256_fmaddsub_pd(ar, b, _mm256_mul_pd(ai, bsw))
}

#[inline]
fn is_power_of_eight(n: usize) -> bool {
    n > 0 && n.is_power_of_two() && (n.trailing_zeros() % 3 == 0)
}

// ── digit-reverse permutation (base 8) ───────────────────────────────────────

#[inline]
fn reverse_base8(mut v: usize, digits: u32) -> usize {
    let mut r = 0;
    for _ in 0..digits {
        r = (r << 3) | (v & 7);
        v >>= 3;
    }
    r
}

fn digit_reverse_64(data: &mut [Complex64]) {
    let digits = data.len().trailing_zeros() / 3;
    for i in 0..data.len() {
        let j = reverse_base8(i, digits);
        if j > i {
            data.swap(i, j);
        }
    }
}

fn digit_reverse_32(data: &mut [Complex32]) {
    let digits = data.len().trailing_zeros() / 3;
    for i in 0..data.len() {
        let j = reverse_base8(i, digits);
        if j > i {
            data.swap(i, j);
        }
    }
}

// ── Rayon thresholds ──────────────────────────────────────────────────────────

/// Minimum `eighth` (j-count per chunk) to activate inner j-loop Rayon.
///
/// Set to a very large value to effectively disable parallelism until SIMD
/// single-threaded performance is optimized first.  Each j-iteration costs
/// ~100–150 ns; need ≥ 50k iters to justify ~5 µs Rayon overhead per stage.
const RAYON_PAR_J_MIN_R8: usize = 1 << 30;

/// Minimum `eighth` and chunk count for outer-chunk Rayon.
/// Disabled until single-threaded SIMD is the bottleneck.
const RAYON_CHUNK_J_MIN_R8: usize = 1 << 30;
const RAYON_CHUNK_COUNT_R8: usize = 4;

// ── Sequential per-chunk helpers ──────────────────────────────────────────────

/// Process one radix-8 chunk sequentially.
#[inline(always)]
fn process_r8_chunk_seq_64(
    chunk: &mut [Complex64],
    eighth: usize,
    stage: Option<&[Complex64]>,
    inverse: bool,
) {
    // j = 0: W_len^{p*0} = 1 for all p — no twiddle multiplication needed.
    {
        let mut buf = [Complex64::new(0.0, 0.0); 8];
        for p in 0..8 {
            buf[p] = chunk[p * eighth];
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
        unsafe { winograd::dft8_avx_fma_64(&mut buf, inverse); }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")))]
        winograd::dft8_64(&mut buf, inverse);
        for p in 0..8 {
            chunk[p * eighth] = buf[p];
        }
    }
    // j = 1..eighth: W_len^{p*j} requires actual twiddle multiplications.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    {
        // SIMD path: process pairs of consecutive j values simultaneously.
        // process_j_pair_simd_64 advances two twiddle recurrences in one __m256d
        // and applies both data mults with one cmul2_r8 per p — 2× throughput.
        let mut j = 1usize;
        while j + 1 < eighth {
            let (step_j, step_j1) = match stage {
                Some(s) => (s[j], s[j + 1]),
                None => {
                    let a = if inverse { std::f64::consts::TAU } else { -std::f64::consts::TAU };
                    let f = a / (eighth * 8) as f64;
                    (
                        Complex64::new((f * j as f64).cos(), (f * j as f64).sin()),
                        Complex64::new((f * (j + 1) as f64).cos(), (f * (j + 1) as f64).sin()),
                    )
                }
            };
            process_j_pair_simd_64(chunk, eighth, j, step_j, step_j1, inverse);
            j += 2;
        }
        // Scalar tail for odd eighth.
        if j < eighth {
            let step = match stage {
                Some(s) => s[j],
                None => {
                    let a = if inverse { std::f64::consts::TAU } else { -std::f64::consts::TAU };
                    let f = a / (eighth * 8) as f64;
                    Complex64::new((f * j as f64).cos(), (f * j as f64).sin())
                }
            };
            process_j_scalar_64(chunk, eighth, j, step, inverse);
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")))]
    {
        for j in 1..eighth {
            let step = match stage {
                Some(s) => s[j],
                None => {
                    let a = if inverse {
                        std::f64::consts::TAU * j as f64 / (eighth * 8) as f64
                    } else {
                        -std::f64::consts::TAU * j as f64 / (eighth * 8) as f64
                    };
                    Complex64::new(a.cos(), a.sin())
                }
            };
            process_j_scalar_64(chunk, eighth, j, step, inverse);
        }
    }
}

/// Process one j-position in the radix-8 chunk using scalar ops + dft8_avx_fma_64.
#[inline(always)]
fn process_j_scalar_64(
    chunk: &mut [Complex64],
    eighth: usize,
    j: usize,
    step: Complex64,
    inverse: bool,
) {
    let mut buf = [Complex64::new(0.0, 0.0); 8];
    buf[0] = chunk[j];
    let mut tw = step;
    for p in 1..8 {
        buf[p] = winograd::apply_twiddle_64(chunk[j + p * eighth], tw);
        tw = winograd::apply_twiddle_64(tw, step);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    unsafe { winograd::dft8_avx_fma_64(&mut buf, inverse); }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")))]
    winograd::dft8_64(&mut buf, inverse);
    for p in 0..8 {
        chunk[j + p * eighth] = buf[p];
    }
}

/// Process one j-position using SIMD twiddle application + dft8_avx_fma_64.
///
/// Twiddle recurrence: advances `tw_pair = [tw_j, tw_{j+1}]` with one `cmul2_r8`
/// per p-step, applying both twiddle mults to `[data[j+p*e], data[j+1+p*e]]`
/// in a single SIMD multiply.  Requires j+1 < eighth (pairs); handles remainder
/// in `process_j_scalar_64`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline(always)]
fn process_j_pair_simd_64(
    chunk: &mut [Complex64],
    eighth: usize,
    j: usize,          // j and j+1 are both processed
    step_j: Complex64,
    step_j1: Complex64,
    inverse: bool,
) {
    use std::arch::x86_64::{_mm256_loadu_pd, _mm256_storeu_pd};

    let mut buf_j = [Complex64::new(0.0, 0.0); 8];
    let mut buf_j1 = [Complex64::new(0.0, 0.0); 8];
    buf_j[0] = chunk[j];
    buf_j1[0] = chunk[j + 1];

    // Pack step_j and step_{j+1} into one __m256d for SIMD recurrence.
    let step_pair = unsafe {
        _mm256_loadu_pd([step_j.re, step_j.im, step_j1.re, step_j1.im].as_ptr())
    };
    // tw_pair starts at step_pair (twiddle^1).
    let mut tw_pair = step_pair;

    for p in 1..8usize {
        let base = j + p * eighth;
        // Load [data[j+p*e], data[j+1+p*e]] — contiguous in memory.
        let data_pair = unsafe { _mm256_loadu_pd(chunk.as_ptr().add(base) as *const f64) };
        // Apply twiddles: [data_j * tw_j, data_{j+1} * tw_{j+1}]
        let result = unsafe { cmul2_r8(data_pair, tw_pair) };
        // Store result back and extract scalar values for buf.
        let mut tmp = [0.0f64; 4];
        unsafe { _mm256_storeu_pd(tmp.as_mut_ptr(), result) };
        buf_j[p] = Complex64::new(tmp[0], tmp[1]);
        buf_j1[p] = Complex64::new(tmp[2], tmp[3]);
        // Advance both twiddle recurrences: tw_pair *= step_pair (elementwise complex).
        tw_pair = unsafe { cmul2_r8(tw_pair, step_pair) };
    }

    unsafe {
        winograd::dft8_avx_fma_64(&mut buf_j, inverse);
        winograd::dft8_avx_fma_64(&mut buf_j1, inverse);
    }

    for p in 0..8 {
        chunk[j + p * eighth] = buf_j[p];
        chunk[j + 1 + p * eighth] = buf_j1[p];
    }
}

/// Process one j-position in the radix-8 chunk for Complex32 (scalar twiddle + dft8 kernel).
#[inline(always)]
fn process_j_scalar_32(
    chunk: &mut [Complex32],
    eighth: usize,
    j: usize,
    step: Complex32,
    inverse: bool,
) {
    let mut buf = [Complex32::new(0.0, 0.0); 8];
    buf[0] = chunk[j];
    let mut tw = step;
    for p in 1..8 {
        buf[p] = winograd::apply_twiddle_32(chunk[j + p * eighth], tw);
        tw = winograd::apply_twiddle_32(tw, step);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    unsafe { winograd::dft8_avx_fma_32(&mut buf, inverse); }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")))]
    winograd::dft8_32(&mut buf, inverse);
    for p in 0..8 {
        chunk[j + p * eighth] = buf[p];
    }
}

/// Process a pair of adjacent j-positions for Complex32 using SIMD twiddle recurrence.
///
/// Packs `[tw_j, tw_{j+1}]` into one `__m256` (4 Complex32 = 8 f32) and advances
/// both twiddle recurrences simultaneously with `cmul4_32`.  Applies twiddles to
/// `[data[j+p*e], data[j+1+p*e]]` in one SIMD multiply per p.  2× throughput
/// vs scalar on the twiddle application; inner DFT-8 uses `dft8_avx_fma_32`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
#[inline(always)]
fn process_j_pair_simd_32(
    chunk: &mut [Complex32],
    eighth: usize,
    j: usize,
    step_j: Complex32,
    step_j1: Complex32,
    inverse: bool,
) {
    use std::arch::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};

    let mut buf_j  = [Complex32::new(0.0, 0.0); 8];
    let mut buf_j1 = [Complex32::new(0.0, 0.0); 8];
    buf_j[0]  = chunk[j];
    buf_j1[0] = chunk[j + 1];

    // Pack step_j and step_{j+1} into one __m256 (positions 0,1 and 2,3 within
    // the lower 128 bits; upper 128 bits are filled with the same pair for cmul4_32).
    // Layout: [step_j.re, step_j.im, step_j1.re, step_j1.im,
    //          step_j.re, step_j.im, step_j1.re, step_j1.im]
    let step_arr = [
        step_j.re, step_j.im, step_j1.re, step_j1.im,
        step_j.re, step_j.im, step_j1.re, step_j1.im,
    ];
    let step_pair = unsafe { _mm256_loadu_ps(step_arr.as_ptr()) };
    // tw_pair starts at step_pair (twiddle^1).
    let mut tw_pair = step_pair;

    for p in 1..8usize {
        let base = j + p * eighth;
        // Load [data[j+p*e], data[j+1+p*e]] into lower 128 bits and replicate to upper.
        // The two Complex32 are contiguous in memory.
        let v2 = [
            chunk[base].re, chunk[base].im,
            chunk[base + 1].re, chunk[base + 1].im,
            chunk[base].re, chunk[base].im,
            chunk[base + 1].re, chunk[base + 1].im,
        ];
        let data_pair = unsafe { _mm256_loadu_ps(v2.as_ptr()) };
        // Apply twiddles: cmul4_32 computes [d_j * tw_j, d_{j+1} * tw_{j+1}] × 2 (upper mirrors lower).
        let result = unsafe { winograd::cmul4_32(data_pair, tw_pair) };
        let mut tmp = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), result) };
        buf_j[p]  = Complex32::new(tmp[0], tmp[1]);
        buf_j1[p] = Complex32::new(tmp[2], tmp[3]);
        // Advance twiddle recurrence.
        tw_pair = unsafe { winograd::cmul4_32(tw_pair, step_pair) };
    }

    unsafe {
        winograd::dft8_avx_fma_32(&mut buf_j,  inverse);
        winograd::dft8_avx_fma_32(&mut buf_j1, inverse);
    }
    for p in 0..8 {
        chunk[j +     p * eighth] = buf_j[p];
        chunk[j + 1 + p * eighth] = buf_j1[p];
    }
}

#[inline(always)]
fn process_r8_chunk_seq_32(
    chunk: &mut [Complex32],
    eighth: usize,
    stage: Option<&[Complex32]>,
    inverse: bool,
) {
    // j = 0: W_len^{p*0} = 1 for all p — no twiddle multiplication needed.
    {
        let mut buf = [Complex32::new(0.0, 0.0); 8];
        for p in 0..8 {
            buf[p] = chunk[p * eighth];
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
        unsafe { winograd::dft8_avx_fma_32(&mut buf, inverse); }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")))]
        winograd::dft8_32(&mut buf, inverse);
        for p in 0..8 {
            chunk[p * eighth] = buf[p];
        }
    }
    // j = 1..eighth: W_len^{p*j} requires actual twiddle multiplications.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))]
    {
        let mut j = 1usize;
        while j + 1 < eighth {
            let (step_j, step_j1) = match stage {
                Some(s) => (s[j], s[j + 1]),
                None => {
                    let a = if inverse { std::f64::consts::TAU } else { -std::f64::consts::TAU };
                    let f = a / (eighth * 8) as f64;
                    (
                        Complex32::new((f * j as f64).cos() as f32, (f * j as f64).sin() as f32),
                        Complex32::new((f * (j + 1) as f64).cos() as f32, (f * (j + 1) as f64).sin() as f32),
                    )
                }
            };
            process_j_pair_simd_32(chunk, eighth, j, step_j, step_j1, inverse);
            j += 2;
        }
        if j < eighth {
            let step = match stage {
                Some(s) => s[j],
                None => {
                    let a = if inverse { std::f64::consts::TAU } else { -std::f64::consts::TAU };
                    let f = a / (eighth * 8) as f64;
                    Complex32::new((f * j as f64).cos() as f32, (f * j as f64).sin() as f32)
                }
            };
            process_j_scalar_32(chunk, eighth, j, step, inverse);
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma")))]
    {
        for j in 1..eighth {
            let step = match stage {
                Some(s) => s[j],
                None => {
                    let a = if inverse {
                        std::f64::consts::TAU * j as f64 / (eighth * 8) as f64
                    } else {
                        -std::f64::consts::TAU * j as f64 / (eighth * 8) as f64
                    };
                    Complex32::new(a.cos() as f32, a.sin() as f32)
                }
            };
            process_j_scalar_32(chunk, eighth, j, step, inverse);
        }
    }
}

// ── core Winograd-radix-8 in-place kernel ─────────────────────────────────────

fn winograd_r8_inplace_64(data: &mut [Complex64], twiddles: Option<&[Complex64]>, inverse: bool) {
    debug_assert!(is_power_of_eight(data.len()));
    if data.len() <= 1 {
        return;
    }
    digit_reverse_64(data);

    let n = data.len();
    let mut len = 8usize;
    while len <= n {
        let eighth = len >> 3;
        let half = len >> 1;
        let stage = twiddles.map(|t| &t[(half - 1)..(half - 1 + half)]);
        let num_chunks = n / len;

        if eighth >= RAYON_PAR_J_MIN_R8 {
            // Inner j-parallel: split each chunk into 8 disjoint sub-slices,
            // zip them as par_iter_mut so each j-iteration runs independently.
            // W_len^{p·j} are computed via the serial recurrence within each
            // j-closure (only 7 multiplications, not worth further parallelism).
            if let Some(st) = stage {
                let stage_slice = &st[..eighth];
                for chunk in data.chunks_exact_mut(len) {
                    let e = eighth;
                    let (g0, rest) = chunk.split_at_mut(e);
                    let (g1, rest) = rest.split_at_mut(e);
                    let (g2, rest) = rest.split_at_mut(e);
                    let (g3, rest) = rest.split_at_mut(e);
                    let (g4, rest) = rest.split_at_mut(e);
                    let (g5, rest) = rest.split_at_mut(e);
                    let (g6, g7) = rest.split_at_mut(e);
                    g0.par_iter_mut()
                        .zip(g1.par_iter_mut())
                        .zip(g2.par_iter_mut())
                        .zip(g3.par_iter_mut())
                        .zip(g4.par_iter_mut())
                        .zip(g5.par_iter_mut())
                        .zip(g6.par_iter_mut())
                        .zip(g7.par_iter_mut())
                        .zip(stage_slice.par_iter())
                        .for_each(
                            |(((((((  (a0, a1), a2), a3), a4), a5), a6), a7), step)| {
                                let mut buf = [Complex64::new(0.0, 0.0); 8];
                                buf[0] = *a0;
                                let srcs = [*a1, *a2, *a3, *a4, *a5, *a6, *a7];
                                let mut tw = *step;
                                for p in 1..8usize {
                                    buf[p] = winograd::apply_twiddle_64(srcs[p - 1], tw);
                                    tw = winograd::apply_twiddle_64(tw, *step);
                                }
                                winograd::dft8_64(&mut buf, inverse);
                                *a0 = buf[0];
                                *a1 = buf[1];
                                *a2 = buf[2];
                                *a3 = buf[3];
                                *a4 = buf[4];
                                *a5 = buf[5];
                                *a6 = buf[6];
                                *a7 = buf[7];
                            },
                        );
                }
            } else {
                // No twiddle table: fall back to sequential (only hits _no variants).
                for chunk in data.chunks_exact_mut(len) {
                    process_r8_chunk_seq_64(chunk, eighth, None, inverse);
                }
            }
        } else if eighth >= RAYON_CHUNK_J_MIN_R8 && num_chunks >= RAYON_CHUNK_COUNT_R8 {
            // Outer chunk-parallel: distribute independent chunks across Rayon threads.
            // Each thread processes one chunk's j-loop sequentially.
            data.par_chunks_mut(len).for_each(|chunk| {
                process_r8_chunk_seq_64(chunk, eighth, stage, inverse);
            });
        } else {
            // Sequential: small N or first stage.
            for chunk in data.chunks_exact_mut(len) {
                process_r8_chunk_seq_64(chunk, eighth, stage, inverse);
            }
        }

        len <<= 3;
    }
}

fn winograd_r8_inplace_32(data: &mut [Complex32], twiddles: Option<&[Complex32]>, inverse: bool) {
    debug_assert!(is_power_of_eight(data.len()));
    if data.len() <= 1 {
        return;
    }
    digit_reverse_32(data);

    let n = data.len();
    let mut len = 8usize;
    while len <= n {
        let eighth = len >> 3;
        let half = len >> 1;
        let stage = twiddles.map(|t| &t[(half - 1)..(half - 1 + half)]);
        let num_chunks = n / len;

        if eighth >= RAYON_PAR_J_MIN_R8 {
            if let Some(st) = stage {
                let stage_slice = &st[..eighth];
                for chunk in data.chunks_exact_mut(len) {
                    let e = eighth;
                    let (g0, rest) = chunk.split_at_mut(e);
                    let (g1, rest) = rest.split_at_mut(e);
                    let (g2, rest) = rest.split_at_mut(e);
                    let (g3, rest) = rest.split_at_mut(e);
                    let (g4, rest) = rest.split_at_mut(e);
                    let (g5, rest) = rest.split_at_mut(e);
                    let (g6, g7) = rest.split_at_mut(e);
                    g0.par_iter_mut()
                        .zip(g1.par_iter_mut())
                        .zip(g2.par_iter_mut())
                        .zip(g3.par_iter_mut())
                        .zip(g4.par_iter_mut())
                        .zip(g5.par_iter_mut())
                        .zip(g6.par_iter_mut())
                        .zip(g7.par_iter_mut())
                        .zip(stage_slice.par_iter())
                        .for_each(
                            |(((((((  (a0, a1), a2), a3), a4), a5), a6), a7), step)| {
                                let mut buf = [Complex32::new(0.0, 0.0); 8];
                                buf[0] = *a0;
                                let srcs = [*a1, *a2, *a3, *a4, *a5, *a6, *a7];
                                let mut tw = *step;
                                for p in 1..8usize {
                                    buf[p] = winograd::apply_twiddle_32(srcs[p - 1], tw);
                                    tw = winograd::apply_twiddle_32(tw, *step);
                                }
                                winograd::dft8_32(&mut buf, inverse);
                                *a0 = buf[0];
                                *a1 = buf[1];
                                *a2 = buf[2];
                                *a3 = buf[3];
                                *a4 = buf[4];
                                *a5 = buf[5];
                                *a6 = buf[6];
                                *a7 = buf[7];
                            },
                        );
                }
            } else {
                for chunk in data.chunks_exact_mut(len) {
                    process_r8_chunk_seq_32(chunk, eighth, None, inverse);
                }
            }
        } else if eighth >= RAYON_CHUNK_J_MIN_R8 && num_chunks >= RAYON_CHUNK_COUNT_R8 {
            data.par_chunks_mut(len).for_each(|chunk| {
                process_r8_chunk_seq_32(chunk, eighth, stage, inverse);
            });
        } else {
            for chunk in data.chunks_exact_mut(len) {
                process_r8_chunk_seq_32(chunk, eighth, stage, inverse);
            }
        }

        len <<= 3;
    }
}

// ── public API ────────────────────────────────────────────────────────────────

/// Forward FFT (unnormalized) for power-of-eight lengths using caller-provided twiddles.
#[inline]
pub fn forward_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    winograd_r8_inplace_64(data, Some(twiddles), false);
}

/// Inverse FFT (unnormalized) for power-of-eight lengths using caller-provided twiddles.
#[inline]
pub fn inverse_inplace_unnorm_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    winograd_r8_inplace_64(data, Some(twiddles), true);
}

/// Inverse FFT normalized by 1/N for power-of-eight lengths using caller-provided twiddles.
#[inline]
pub fn inverse_inplace_64_with_twiddles(data: &mut [Complex64], twiddles: &[Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    winograd_r8_inplace_64(data, Some(twiddles), true);
    let inv_n = 1.0 / data.len() as f64;
    for v in data.iter_mut() {
        *v *= inv_n;
    }
}

/// Forward FFT (unnormalized) for power-of-eight lengths.
pub fn forward_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    let twiddles = radix2::build_forward_twiddle_table_64(data.len());
    winograd_r8_inplace_64(data, Some(&twiddles), false);
}

/// Inverse FFT (unnormalized) for power-of-eight lengths.
pub fn inverse_inplace_unnorm_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_64(data.len());
    winograd_r8_inplace_64(data, Some(&twiddles), true);
}

/// Inverse FFT normalized by 1/N for power-of-eight lengths.
pub fn inverse_inplace_64(data: &mut [Complex64]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_64(data.len());
    winograd_r8_inplace_64(data, Some(&twiddles), true);
    let inv_n = 1.0 / data.len() as f64;
    for v in data.iter_mut() {
        *v *= inv_n;
    }
}

/// Forward FFT (unnormalized, f32) for power-of-eight lengths using caller-provided twiddles.
#[inline]
pub fn forward_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    winograd_r8_inplace_32(data, Some(twiddles), false);
}

/// Inverse FFT (unnormalized, f32) for power-of-eight lengths using caller-provided twiddles.
#[inline]
pub fn inverse_inplace_unnorm_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    winograd_r8_inplace_32(data, Some(twiddles), true);
}

/// Inverse FFT normalized by 1/N (f32) for power-of-eight lengths using caller-provided twiddles.
#[inline]
pub fn inverse_inplace_32_with_twiddles(data: &mut [Complex32], twiddles: &[Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    winograd_r8_inplace_32(data, Some(twiddles), true);
    let inv_n = 1.0f32 / data.len() as f32;
    for v in data.iter_mut() {
        *v *= inv_n;
    }
}

/// Forward FFT (unnormalized, f32) for power-of-eight lengths.
pub fn forward_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    let twiddles = radix2::build_forward_twiddle_table_32(data.len());
    winograd_r8_inplace_32(data, Some(&twiddles), false);
}

/// Inverse FFT (unnormalized, f32) for power-of-eight lengths.
pub fn inverse_inplace_unnorm_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_32(data.len());
    winograd_r8_inplace_32(data, Some(&twiddles), true);
}

/// Inverse FFT normalized by 1/N (f32) for power-of-eight lengths.
pub fn inverse_inplace_32(data: &mut [Complex32]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    let twiddles = radix2::build_inverse_twiddle_table_32(data.len());
    winograd_r8_inplace_32(data, Some(&twiddles), true);
    let inv_n = 1.0f32 / data.len() as f32;
    for v in data.iter_mut() {
        *v *= inv_n;
    }
}

// ── Cf16: promote to Complex32 via f16_bridge, run the corresponding f32 kernel ──

/// Forward FFT (unnormalized, f16 storage) for power-of-eight lengths using
/// caller-provided twiddles. Promotes to `Complex32` internally.
#[inline]
pub fn forward_inplace_f16_with_twiddles(data: &mut [Cf16], twiddles: &[Cf16]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    super::f16_bridge::run_f16_via_f32_with_twiddles(
        data,
        twiddles,
        forward_inplace_32_with_twiddles,
    );
}

/// Forward FFT (unnormalized, f16 storage) for power-of-eight lengths.
pub fn forward_inplace_f16(data: &mut [Cf16]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    super::f16_bridge::run_f16_via_f32(data, forward_inplace_32);
}

/// Inverse FFT (unnormalized, f16 storage) for power-of-eight lengths using
/// caller-provided twiddles. Promotes to `Complex32` internally.
#[inline]
pub fn inverse_inplace_unnorm_f16_with_twiddles(data: &mut [Cf16], twiddles: &[Cf16]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    super::f16_bridge::run_f16_via_f32_with_twiddles(
        data,
        twiddles,
        inverse_inplace_unnorm_32_with_twiddles,
    );
}

/// Inverse FFT (unnormalized, f16 storage) for power-of-eight lengths.
pub fn inverse_inplace_unnorm_f16(data: &mut [Cf16]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    super::f16_bridge::run_f16_via_f32(data, inverse_inplace_unnorm_32);
}

/// Inverse FFT normalized by 1/N (f16 storage) for power-of-eight lengths using
/// caller-provided twiddles. Promotes to `Complex32` internally.
#[inline]
pub fn inverse_inplace_f16_with_twiddles(data: &mut [Cf16], twiddles: &[Cf16]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    super::f16_bridge::run_f16_via_f32_with_twiddles(
        data,
        twiddles,
        inverse_inplace_32_with_twiddles,
    );
}

/// Inverse FFT normalized by 1/N (f16 storage) for power-of-eight lengths.
pub fn inverse_inplace_f16(data: &mut [Cf16]) {
    if data.len() <= 1 {
        return;
    }
    debug_assert!(is_power_of_eight(data.len()));
    super::f16_bridge::run_f16_via_f32(data, inverse_inplace_32);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::execution::kernel::direct::{dft_forward_64, dft_inverse_64};

    fn max_abs_err_64(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).norm())
            .fold(0.0f64, f64::max)
    }

    #[test]
    fn radix8_forward_n8_matches_direct() {
        let n = 8usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.31).sin(), (k as f64 * 0.17).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-11, "radix8 n=8 forward err={err:.2e}");
    }

    #[test]
    fn radix8_forward_n64_matches_direct() {
        let n = 64usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.13).sin(), (k as f64 * 0.09).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "radix8 n=64 forward err={err:.2e}");
    }

    #[test]
    fn radix8_forward_n512_matches_direct() {
        let n = 512usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.07).sin(), (k as f64 * 0.04).cos()))
            .collect();
        let mut got = input.clone();
        forward_inplace_64(&mut got);
        let expected = dft_forward_64(&input);
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-9, "radix8 n=512 forward err={err:.2e}");
    }

    #[test]
    fn radix8_inverse_unnorm_n64_matches_direct() {
        let n = 64usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.07).cos(), (k as f64 * 0.11).sin()))
            .collect();
        let mut got = input.clone();
        inverse_inplace_unnorm_64(&mut got);
        let expected = dft_inverse_64(&input)
            .into_iter()
            .map(|x| x * n as f64)
            .collect::<Vec<_>>();
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1e-10, "radix8 n=64 inverse unnorm err={err:.2e}");
    }

    #[test]
    fn radix8_roundtrip_n64() {
        let n = 64usize;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.19).sin(), -(k as f64 * 0.23).cos()))
            .collect();
        let mut buf = input.clone();
        forward_inplace_64(&mut buf);
        inverse_inplace_64(&mut buf);
        let err = max_abs_err_64(&buf, &input);
        assert!(err < 1e-10, "radix8 roundtrip err={err:.2e}");
    }
}
