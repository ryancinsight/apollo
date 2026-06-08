use num_complex::Complex;

use super::arity::dispatch_radix_stage;
use super::cache::CompositeCache;
use super::stockham_stage_fused_adaptive;
use crate::application::execution::kernel::mixed_radix::traits::ShortWinogradScalar;
use crate::application::execution::kernel::tuning::{
    FUSE_THRESHOLD, RADIX_PARALLEL_CHUNK_THRESHOLD,
};


/// Maximum number of stages that may be folded into one adaptive fused pass.
///
/// With FUSE_THRESHOLD = 65_536 and all-radix-4 after lowering, the worst case
/// is log_4(65_536) = 8 stages. 16 provides headroom for mixed radix sequences
/// while keeping the stack-allocated twiddle-pointer array bounded.
const MAX_FUSE_DEPTH: usize = 16;

pub(super) fn composite_core_with_radices<
    F: CompositeCache + ShortWinogradScalar,
    const INVERSE: bool,
>(
    data: &mut [Complex<F>],
    radices: &[usize],
    pointwise_spectrum: Option<&[Complex<F>]>,
) {
    let n = data.len();
    if n <= 1 || radices.is_empty() {
        return;
    }
    debug_assert_eq!(radices.iter().product::<usize>(), n);

    debug_assert!(
        !radices.windows(2).any(|w| w[0] == 2 && w[1] == 2),
        "composite radices must be lowered before execution"
    );

    let (all_twiddles, stage_offsets) = F::cached_twiddles::<INVERSE>(radices);

    // When n ≤ FUSE_THRESHOLD every stage is fused into one block (accumulated
    // radix product = n ≤ FUSE_THRESHOLD). Use the flat iterative Stockham path
    // instead of the recursive per-group `composite_fused_adaptive`:
    //
    // Benefit 1 — cache efficiency: the flat path reads and writes a single
    // n-element array per pass. The recursive path writes to O(depth) intermediate
    // `mid` buffers totalling ~1.5× n elements (e.g. ~31 K extra elements for
    // M=20736), pushing the working set from ~332 KB to ~829 KB (f64). The flat
    // path keeps the working set within L2 for M=20736.
    //
    // Benefit 2 — future AVX2: the flat outer loop allows a single per-stage AVX2
    // feature check (O(n_stages) overhead) instead of per-group (O(n/R) overhead).
    if n <= FUSE_THRESHOLD {
        let n_stages = radices.len();
        debug_assert!(n_stages <= MAX_FUSE_DEPTH);
        let mut twiddle_slices: [&[Complex<F>]; MAX_FUSE_DEPTH] = [&[]; MAX_FUSE_DEPTH];
        let mut stage_prev_len = 1usize;
        for i in 0..n_stages {
            let radix = radices[i];
            let offset = stage_offsets[i];
            let len = (radix - 1) * stage_prev_len;
            twiddle_slices[i] = &all_twiddles[offset..offset + len];
            stage_prev_len *= radix;
        }
        F::with_scratch(n, |scratch| {
            flat_stockham_fused::<F, INVERSE>(
                data,
                scratch,
                radices,
                &twiddle_slices[..n_stages],
                pointwise_spectrum,
            );
        });
        return;
    }

    // Multi-block path: n > FUSE_THRESHOLD. Stages are grouped into fused blocks
    // that each fit within FUSE_THRESHOLD; each block uses `stockham_stage_fused_adaptive`.
    F::with_scratch(n, |scratch| {
        let mut src_is_data = true;
        let mut prev_len = 1usize;
        let mut stage_idx = 0usize;

        while stage_idx < radices.len() {
            let mut fuse_end = stage_idx + 1;
            let mut fused_radix = radices[stage_idx];
            while fuse_end < radices.len() && (fuse_end - stage_idx) < MAX_FUSE_DEPTH {
                let next_radix = fused_radix * radices[fuse_end];
                if prev_len * next_radix > FUSE_THRESHOLD {
                    break;
                }
                fused_radix = next_radix;
                fuse_end += 1;
            }

            let stage_count = fuse_end - stage_idx;
            let mut twiddle_slices: [&[Complex<F>]; MAX_FUSE_DEPTH] = [&[]; MAX_FUSE_DEPTH];
            let mut stage_prev_len = prev_len;
            for i in 0..stage_count {
                let radix = radices[stage_idx + i];
                let offset = stage_offsets[stage_idx + i];
                let len = (radix - 1) * stage_prev_len;
                twiddle_slices[i] = &all_twiddles[offset..offset + len];
                stage_prev_len *= radix;
            }

            let fused_radices = &radices[stage_idx..fuse_end];
            let twiddles = &twiddle_slices[..stage_count];
            let pointwise = if fuse_end == radices.len() {
                pointwise_spectrum
            } else {
                None
            };
            let use_parallel = n >= RADIX_PARALLEL_CHUNK_THRESHOLD && stage_prev_len >= 512;

            if src_is_data {
                stockham_stage_fused_adaptive::<F, INVERSE>(
                    data,
                    scratch,
                    prev_len,
                    fused_radices,
                    twiddles,
                    pointwise,
                    use_parallel,
                );
            } else {
                stockham_stage_fused_adaptive::<F, INVERSE>(
                    scratch,
                    data,
                    prev_len,
                    fused_radices,
                    twiddles,
                    pointwise,
                    use_parallel,
                );
            }

            src_is_data = !src_is_data;
            prev_len = stage_prev_len;
            stage_idx = fuse_end;
        }

        if !src_is_data {
            data.copy_from_slice(scratch);
        }
    });
}

/// Flat iterative Stockham FFT for the fully-fused single-block case (n ≤ FUSE_THRESHOLD).
///
/// Replaces the recursive `composite_fused_adaptive` path. Performs `n_stages` sequential
/// passes over the data, each pass processing all `G_s = n / (r_s × P_s)` groups in a flat
/// outer loop. `dispatch_single_radix` is `#[inline]`, so the match and call frame
/// collapse into a tight per-group loop body with no call overhead.
///
/// ## Stockham addressing (pass s)
/// - `P_s = prev_len = Π_{i<s} r_i`  (accumulated stride before pass s)
/// - `G_s = n / (r_s × P_s)`         (group count for pass s)
/// - Read:  `src[j + g×P_s + k×(G_s×P_s)]`,  k∈[0,r_s), j∈[0,P_s)
/// - Write: `dst_block[j + k×P_s]`   where `dst_block = dst[g×r_s×P_s .. (g+1)×r_s×P_s]`
///
/// The last pass always has G=1 (all stages fused → `prev_len_last = n / r_last`), so the
/// pointwise spectrum (convolution) is applied exactly once to all n elements.
#[inline]
fn flat_stockham_fused<F: CompositeCache + ShortWinogradScalar, const INVERSE: bool>(
    data: &mut [Complex<F>],
    scratch: &mut [Complex<F>],
    radices: &[usize],
    twiddles: &[&[Complex<F>]],
    pointwise_spectrum: Option<&[Complex<F>]>,
) {
    let n = data.len();
    let n_stages = radices.len();
    let mut prev_len = 1usize;
    let mut src_is_data = true;

    for s in 0..n_stages {
        let r = radices[s];
        let stage_chunk = prev_len * r; // r_s × P_s: output block size per group
        let g_count = n / stage_chunk; // G_s: groups this pass
        let tw = twiddles[s];
        let is_last = s + 1 == n_stages;
        // pointwise is applied once on the last pass (g_count == 1 there, covers all n elements).
        let pointwise = if is_last { pointwise_spectrum } else { None };

        // Per-stage dispatch: try AVX2 for r=3 and r=4 (amortizes #[target_feature]
        // overhead across all g_count groups; O(1) overhead per stage vs O(g_count)).
        // Falls back to the scalar per-group loop when AVX2 is unavailable.
        let avx2_handled = match r {
            2 => {
                if src_is_data {
                    F::try_flat_pass_r2::<INVERSE>(
                        data,
                        scratch,
                        prev_len,
                        g_count,
                        stage_chunk,
                        tw,
                        pointwise,
                    )
                } else {
                    F::try_flat_pass_r2::<INVERSE>(
                        scratch,
                        data,
                        prev_len,
                        g_count,
                        stage_chunk,
                        tw,
                        pointwise,
                    )
                }
            }
            3 => {
                if src_is_data {
                    F::try_flat_pass_r3::<INVERSE>(
                        data,
                        scratch,
                        prev_len,
                        g_count,
                        stage_chunk,
                        tw,
                        pointwise,
                    )
                } else {
                    F::try_flat_pass_r3::<INVERSE>(
                        scratch,
                        data,
                        prev_len,
                        g_count,
                        stage_chunk,
                        tw,
                        pointwise,
                    )
                }
            }
            4 => {
                if src_is_data {
                    F::try_flat_pass_r4::<INVERSE>(
                        data,
                        scratch,
                        prev_len,
                        g_count,
                        stage_chunk,
                        tw,
                        pointwise,
                    )
                } else {
                    F::try_flat_pass_r4::<INVERSE>(
                        scratch,
                        data,
                        prev_len,
                        g_count,
                        stage_chunk,
                        tw,
                        pointwise,
                    )
                }
            }
            5 => {
                if src_is_data {
                    F::try_flat_pass_r5::<INVERSE>(
                        data,
                        scratch,
                        prev_len,
                        g_count,
                        stage_chunk,
                        tw,
                        pointwise,
                    )
                } else {
                    F::try_flat_pass_r5::<INVERSE>(
                        scratch,
                        data,
                        prev_len,
                        g_count,
                        stage_chunk,
                        tw,
                        pointwise,
                    )
                }
            }
            7 => {
                if src_is_data {
                    F::try_flat_pass_r7::<INVERSE>(
                        data,
                        scratch,
                        prev_len,
                        g_count,
                        stage_chunk,
                        tw,
                        pointwise,
                    )
                } else {
                    F::try_flat_pass_r7::<INVERSE>(
                        scratch,
                        data,
                        prev_len,
                        g_count,
                        stage_chunk,
                        tw,
                        pointwise,
                    )
                }
            }
            _ => false,
        };
        if avx2_handled {
            src_is_data = !src_is_data;
            prev_len = stage_chunk;
            continue;
        }

        if src_is_data {
            dispatch_radix_stage::<F, INVERSE>(data, scratch, prev_len, g_count, r, tw, pointwise);
        } else {
            dispatch_radix_stage::<F, INVERSE>(scratch, data, prev_len, g_count, r, tw, pointwise);
        }

        src_is_data = !src_is_data;
        prev_len = stage_chunk;
    }

    // If the final result landed in scratch, copy it back to data.
    if !src_is_data {
        data.copy_from_slice(scratch);
    }
}
