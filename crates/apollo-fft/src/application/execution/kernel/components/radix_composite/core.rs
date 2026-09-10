use eunomia::Complex;

use super::arity::dispatch_radix_stage;
use super::cache::CompositeCache;
use super::flat_pass::{FlatPassR2, FlatPassR3, FlatPassR4, FlatPassR5, FlatPassR7};
use super::stockham_stage_fused_adaptive;
use crate::application::execution::kernel::components::winograd::ShortWinogradScalar;
use crate::application::execution::kernel::tuning::FUSE_THRESHOLD;
use hermes_simd::{LaneKernel, Simd, SimdArch, SimdKernel};

/// Maximum number of stages that may be folded into one adaptive fused pass.
///
/// With FUSE_THRESHOLD = 65_536 and all-radix-4 after lowering, the worst case
/// is log_4(65_536) = 8 stages. 16 provides headroom for mixed radix sequences
/// while keeping the stack-allocated twiddle-pointer array bounded.
const MAX_FUSE_DEPTH: usize = 16;

pub(super) fn composite_core_with_radices<
    P: moirai::ExecutionPolicy,
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
    // Benefit 2: the flat outer loop dispatches one register-width kernel per
    // stage (O(n_stages) dispatches) instead of per group (O(n/R)).
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
            let dispatch_is_parallel = P::parallelize(n) && stage_prev_len >= 512;

            if src_is_data {
                if dispatch_is_parallel {
                    stockham_stage_fused_adaptive::<moirai::Parallel, F, INVERSE>(
                        data,
                        scratch,
                        prev_len,
                        fused_radices,
                        twiddles,
                        pointwise,
                    );
                } else {
                    stockham_stage_fused_adaptive::<moirai::Sequential, F, INVERSE>(
                        data,
                        scratch,
                        prev_len,
                        fused_radices,
                        twiddles,
                        pointwise,
                    );
                }
            } else {
                if dispatch_is_parallel {
                    stockham_stage_fused_adaptive::<moirai::Parallel, F, INVERSE>(
                        scratch,
                        data,
                        prev_len,
                        fused_radices,
                        twiddles,
                        pointwise,
                    );
                } else {
                    stockham_stage_fused_adaptive::<moirai::Sequential, F, INVERSE>(
                        scratch,
                        data,
                        prev_len,
                        fused_radices,
                        twiddles,
                        pointwise,
                    );
                }
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
/// Performs `n_stages` sequential passes over the data, each pass processing
/// all `G_s = n / (r_s × P_s)` groups. The register width is dispatched once
/// per transform: the stage loop runs inside the backend's frame as
/// [`FlatStockham`], each stage's flat pass a direct call, so the dispatch
/// (feature checks, call layers, the kernel frame) is paid once rather than
/// once per stage. A stage the width declines runs the scalar pass.
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
    hermes_simd::vectorize(FlatStockham::<F, INVERSE> {
        data,
        scratch,
        radices,
        twiddles,
        pointwise_spectrum,
    });
}

/// One transform's flat Stockham passes on the dispatched register width;
/// the result ends in `data`.
struct FlatStockham<'a, F, const INVERSE: bool> {
    data: &'a mut [Complex<F>],
    scratch: &'a mut [Complex<F>],
    radices: &'a [usize],
    /// Stage `s`'s twiddles, arm `k` at `(k - 1) * prev_len`.
    twiddles: &'a [&'a [Complex<F>]],
    pointwise_spectrum: Option<&'a [Complex<F>]>,
}

impl<F, const INVERSE: bool> LaneKernel<F> for FlatStockham<'_, F, INVERSE>
where
    F: CompositeCache + ShortWinogradScalar,
{
    type Output = ();

    #[expect(
        clippy::inline_always,
        reason = "the stage loop and every pass it calls must inline into the backend's target-feature frame"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<F>>(self, simd: Simd<F, A>) {
        let Self {
            data,
            scratch,
            radices,
            twiddles,
            pointwise_spectrum,
        } = self;
        let n = data.len();
        let n_stages = radices.len();
        let mut prev_len = 1usize;
        let mut src_is_data = true;

        for (s, (&r, &tw)) in radices.iter().zip(twiddles).enumerate() {
            let stage_chunk = prev_len * r; // r_s × P_s: output block size per group
            let g_count = n / stage_chunk; // G_s: groups this pass
                                           // The pointwise spectrum is applied once, on the last pass
                                           // (g_count == 1 there, covering all n elements).
            let pointwise = if s + 1 == n_stages {
                pointwise_spectrum
            } else {
                None
            };
            let (src, dst): (&[Complex<F>], &mut [Complex<F>]) = if src_is_data {
                (&*data, &mut *scratch)
            } else {
                (&*scratch, &mut *data)
            };

            let vector_handled = match r {
                // Below n = 64 the radix-2 pass exceeds the scalar radix-2
                // cost (measured at N = 32); tiny stages stay scalar.
                2 => {
                    n >= 64
                        && FlatPassR2 {
                            src,
                            dst: &mut *dst,
                            prev_len,
                            g_count,
                            stage_chunk,
                            tw,
                            pointwise,
                        }
                        .call(simd)
                }
                3 => FlatPassR3::<F, INVERSE> {
                    src,
                    dst: &mut *dst,
                    prev_len,
                    g_count,
                    stage_chunk,
                    tw,
                    pointwise,
                }
                .call(simd),
                4 => FlatPassR4::<F, INVERSE> {
                    src,
                    dst: &mut *dst,
                    prev_len,
                    g_count,
                    stage_chunk,
                    tw,
                    pointwise,
                }
                .call(simd),
                5 => FlatPassR5::<F, INVERSE> {
                    src,
                    dst: &mut *dst,
                    prev_len,
                    g_count,
                    stage_chunk,
                    tw,
                    pointwise,
                }
                .call(simd),
                7 => FlatPassR7::<F, INVERSE> {
                    src,
                    dst: &mut *dst,
                    prev_len,
                    g_count,
                    stage_chunk,
                    tw,
                    pointwise,
                }
                .call(simd),
                _ => false,
            };
            if !vector_handled {
                dispatch_radix_stage::<F, INVERSE>(src, dst, prev_len, g_count, r, tw, pointwise);
            }

            src_is_data = !src_is_data;
            prev_len = stage_chunk;
        }

        // If the final result landed in scratch, copy it back to data.
        if !src_is_data {
            data.copy_from_slice(scratch);
        }
    }
}
