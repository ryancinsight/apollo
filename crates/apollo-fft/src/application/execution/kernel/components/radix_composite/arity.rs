use super::cache::CompositeCache;
use crate::application::execution::kernel::components::winograd::apply_twiddle_impl;
use crate::application::execution::kernel::mixed_radix::traits::ShortWinogradScalar;
use num_complex::Complex;
use num_traits::Zero;

pub(super) trait FusedStage {
    fn compute_group<F: CompositeCache + ShortWinogradScalar, const INVERSE: bool>(
        src: &[Complex<F>],
        dst: &mut [Complex<F>],
        prev_len: usize,
        b_out: usize,
        groups_out: usize,
        twiddles: &[&[Complex<F>]],
        tw_idx: usize,
        pointwise: Option<&[Complex<F>]>,
    );
}

/// Dispatch the radix-R butterfly for a single column buffer.
/// Monomorphized per (F, R) at compile time; zero abstraction overhead.
///
/// Uses const R in match to enable LLVM to optimize each branch independently.
/// The match is resolved at compile time for monomorphized R, so this is zero-cost.
#[inline]
fn apply_dft_r<F: CompositeCache + ShortWinogradScalar, const R: usize, const INVERSE: bool>(
    buf: &mut [Complex<F>; R],
) {
    // Compile-time branch on const R — LLVM can inline and vectorize each case.
    match R {
        2 => {
            let ptr = buf.as_mut_ptr().cast::<[Complex<F>; 2]>();
            F::dft2(unsafe { &mut *ptr });
        }
        3 => {
            let ptr = buf.as_mut_ptr().cast::<[Complex<F>; 3]>();
            if INVERSE {
                F::dft3::<true>(unsafe { &mut *ptr });
            } else {
                F::dft3::<false>(unsafe { &mut *ptr });
            }
        }
        4 => {
            let ptr = buf.as_mut_ptr().cast::<[Complex<F>; 4]>();
            if INVERSE {
                F::dft4::<true>(unsafe { &mut *ptr });
            } else {
                F::dft4::<false>(unsafe { &mut *ptr });
            }
        }
        5 => {
            let ptr = buf.as_mut_ptr().cast::<[Complex<F>; 5]>();
            if INVERSE {
                F::dft5::<true>(unsafe { &mut *ptr });
            } else {
                F::dft5::<false>(unsafe { &mut *ptr });
            }
        }
        7 => {
            let ptr = buf.as_mut_ptr().cast::<[Complex<F>; 7]>();
            if INVERSE {
                F::dft7::<true>(unsafe { &mut *ptr });
            } else {
                F::dft7::<false>(unsafe { &mut *ptr });
            }
        }
        8 => {
            let ptr = buf.as_mut_ptr().cast::<[Complex<F>; 8]>();
            if INVERSE {
                F::dft8::<true>(unsafe { &mut *ptr });
            } else {
                F::dft8::<false>(unsafe { &mut *ptr });
            }
        }
        16 => {
            let ptr = buf.as_mut_ptr().cast::<[Complex<F>; 16]>();
            if INVERSE {
                F::dft16::<true>(unsafe { &mut *ptr });
            } else {
                F::dft16::<false>(unsafe { &mut *ptr });
            }
        }
        11 => {
            let ptr = buf.as_mut_ptr().cast::<[Complex<F>; 11]>();
            if INVERSE {
                F::dft11::<true>(unsafe { &mut *ptr })
            } else {
                F::dft11::<false>(unsafe { &mut *ptr })
            }
        }
        13 => {
            let ptr = buf.as_mut_ptr().cast::<[Complex<F>; 13]>();
            if INVERSE {
                F::dft13::<true>(unsafe { &mut *ptr })
            } else {
                F::dft13::<false>(unsafe { &mut *ptr })
            }
        }
        17 => {
            let ptr = buf.as_mut_ptr().cast::<[Complex<F>; 17]>();
            if INVERSE {
                F::dft17::<true>(unsafe { &mut *ptr })
            } else {
                F::dft17::<false>(unsafe { &mut *ptr })
            }
        }
        23 => {
            let ptr = buf.as_mut_ptr().cast::<[Complex<F>; 23]>();
            if INVERSE {
                F::dft23::<true>(unsafe { &mut *ptr })
            } else {
                F::dft23::<false>(unsafe { &mut *ptr })
            }
        }
        _ => unreachable!("unsupported radix {}", R),
    }
}

/// Gather one column from strided src, apply twiddles for column j, write to buf.
///
/// j == 0 → no twiddles (W^0 = 1 for all arms).
/// j > 0 → reads precomputed arm twiddles directly from the flat table:
///   arm k (k=1..R-1) at `stage_twiddles[(k-1)*prev_len + j]` = W^{k·j}.
///
/// ## Layout contract (matches `build_composite_twiddles`)
/// The flat slice `stage_twiddles` for one stage has `(R-1)*prev_len` entries:
///   [0*prev_len .. 1*prev_len): arm1 = W^{1·j}
///   [1*prev_len .. 2*prev_len): arm2 = W^{2·j}
///   …
///   [(R-2)*prev_len .. (R-1)*prev_len): arm(R-1) = W^{(R-1)·j}
///
/// All arm loads are independent table reads — no iterative chain. This enables
/// the out-of-order core to issue all arm loads in parallel and eliminates the
/// (R-2) redundant complex multiplies per column that the previous iterative
/// `tw_k *= base_tw` approach required for radix R ≥ 3.
#[inline]
fn load_and_twiddle<F: ShortWinogradScalar, const R: usize>(
    src: &[Complex<F>],
    stride: usize,
    src_base: usize,
    j: usize,
    prev_len: usize,
    stage_twiddles: &[Complex<F>],
    buf: &mut [Complex<F>; R],
) {
    // Explicit iteration for better LLVM loop unroll hints with const R.
    // The compiler sees the exact iteration count from const R and can unroll fully.
    for k in 0..R {
        buf[k] = *unsafe { src.get_unchecked(k * stride + src_base + j) };
    }
    if j > 0 {
        // Read each arm's precomputed twiddle W^{k·j} from the flat table.
        // Arm k is at stage_twiddles[(k-1)*prev_len + j]; all loads are independent.
        for k in 1..R {
            let tw = *unsafe { stage_twiddles.get_unchecked((k - 1) * prev_len + j) };
            buf[k] = apply_twiddle_impl(buf[k], tw);
        }
    }
}

/// Scatter one butterfly result column to strided dst at column j.
#[inline]
fn store_col<F: Copy, const R: usize>(
    dst: &mut [Complex<F>],
    j: usize,
    prev_len: usize,
    buf: &[Complex<F>; R],
) {
    // Explicit iteration with const R enables full unroll by LLVM.
    for k in 0..R {
        *unsafe { dst.get_unchecked_mut(j + k * prev_len) } = buf[k];
    }
}

pub(super) struct Radix<const R: usize>;

impl<const R: usize> FusedStage for Radix<R> {
    #[inline]
    fn compute_group<F: CompositeCache + ShortWinogradScalar, const INVERSE: bool>(
        src: &[Complex<F>],
        dst: &mut [Complex<F>],
        prev_len: usize,
        b_out: usize,
        groups_out: usize,
        twiddles: &[&[Complex<F>]],
        tw_idx: usize,
        pointwise: Option<&[Complex<F>]>,
    ) {
        let stride = groups_out * prev_len;
        let src_base = b_out * prev_len;
        let stage_twiddles = twiddles[tw_idx];
        let zero = Complex::<F>::zero();
        let mut j = 0;

        // SUBTILE-4: 4 independent column chains per iteration expose ILP.
        // Each (load→twiddle→butterfly→store) chain has zero data dependencies
        // across the 4 lanes; the out-of-order engine issues all loads and
        // twiddle multiplies in parallel.
        while j + 3 < prev_len {
            let mut buf0 = [zero; R];
            let mut buf1 = [zero; R];
            let mut buf2 = [zero; R];
            let mut buf3 = [zero; R];
            load_and_twiddle::<F, R>(
                src,
                stride,
                src_base,
                j,
                prev_len,
                stage_twiddles,
                &mut buf0,
            );
            load_and_twiddle::<F, R>(
                src,
                stride,
                src_base,
                j + 1,
                prev_len,
                stage_twiddles,
                &mut buf1,
            );
            load_and_twiddle::<F, R>(
                src,
                stride,
                src_base,
                j + 2,
                prev_len,
                stage_twiddles,
                &mut buf2,
            );
            load_and_twiddle::<F, R>(
                src,
                stride,
                src_base,
                j + 3,
                prev_len,
                stage_twiddles,
                &mut buf3,
            );
            apply_dft_r::<F, R, INVERSE>(&mut buf0);
            apply_dft_r::<F, R, INVERSE>(&mut buf1);
            apply_dft_r::<F, R, INVERSE>(&mut buf2);
            apply_dft_r::<F, R, INVERSE>(&mut buf3);
            store_col::<F, R>(dst, j, prev_len, &buf0);
            store_col::<F, R>(dst, j + 1, prev_len, &buf1);
            store_col::<F, R>(dst, j + 2, prev_len, &buf2);
            store_col::<F, R>(dst, j + 3, prev_len, &buf3);
            j += 4;
        }

        // SUBTILE-2: 2-wide tail for remaining even columns.
        while j + 1 < prev_len {
            let mut buf0 = [zero; R];
            let mut buf1 = [zero; R];
            load_and_twiddle::<F, R>(
                src,
                stride,
                src_base,
                j,
                prev_len,
                stage_twiddles,
                &mut buf0,
            );
            load_and_twiddle::<F, R>(
                src,
                stride,
                src_base,
                j + 1,
                prev_len,
                stage_twiddles,
                &mut buf1,
            );
            apply_dft_r::<F, R, INVERSE>(&mut buf0);
            apply_dft_r::<F, R, INVERSE>(&mut buf1);
            store_col::<F, R>(dst, j, prev_len, &buf0);
            store_col::<F, R>(dst, j + 1, prev_len, &buf1);
            j += 2;
        }

        // Scalar tail for the last column when prev_len is odd.
        while j < prev_len {
            let mut buf = [zero; R];
            load_and_twiddle::<F, R>(src, stride, src_base, j, prev_len, stage_twiddles, &mut buf);
            apply_dft_r::<F, R, INVERSE>(&mut buf);
            store_col::<F, R>(dst, j, prev_len, &buf);
            j += 1;
        }

        // Pointwise frequency-domain multiply (e.g., convolution spectrum).
        // Applied only on the final stage of a fused pass; inner stages receive None.
        if let Some(pw) = pointwise {
            let len = R * prev_len;
            let dst_ptr = dst.as_mut_ptr();
            let pw_ptr = pw.as_ptr();
            let mut idx = 0;
            while idx < len {
                unsafe {
                    let out = dst_ptr.add(idx);
                    *out = apply_twiddle_impl(*out, *pw_ptr.add(idx));
                }
                idx += 1;
            }
        }
    }
}

/// Dispatch a single-radix butterfly stage via a monomorphized `Radix<R>` call.
///
/// The `r` value is resolved at runtime via a match, emitting 11 independent
/// monomorphizations. The match itself is a single indirect branch amortized
/// over `prev_len` columns; the per-column cost is dominated by butterfly
/// arithmetic, not branch overhead.
#[inline]
pub(super) fn dispatch_single_radix<
    F: CompositeCache + ShortWinogradScalar,
    const INVERSE: bool,
>(
    src: &[Complex<F>],
    dst: &mut [Complex<F>],
    prev_len: usize,
    b_out: usize,
    groups_out: usize,
    r: usize,
    twiddles: &[Complex<F>],
    pointwise: Option<&[Complex<F>]>,
) {
    let tw = &[twiddles];
    match r {
        2 => Radix::<2>::compute_group::<F, INVERSE>(
            src, dst, prev_len, b_out, groups_out, tw, 0, pointwise,
        ),
        3 => Radix::<3>::compute_group::<F, INVERSE>(
            src, dst, prev_len, b_out, groups_out, tw, 0, pointwise,
        ),
        4 => Radix::<4>::compute_group::<F, INVERSE>(
            src, dst, prev_len, b_out, groups_out, tw, 0, pointwise,
        ),
        5 => Radix::<5>::compute_group::<F, INVERSE>(
            src, dst, prev_len, b_out, groups_out, tw, 0, pointwise,
        ),
        7 => Radix::<7>::compute_group::<F, INVERSE>(
            src, dst, prev_len, b_out, groups_out, tw, 0, pointwise,
        ),
        8 => Radix::<8>::compute_group::<F, INVERSE>(
            src, dst, prev_len, b_out, groups_out, tw, 0, pointwise,
        ),
        16 => Radix::<16>::compute_group::<F, INVERSE>(
            src, dst, prev_len, b_out, groups_out, tw, 0, pointwise,
        ),
        11 => Radix::<11>::compute_group::<F, INVERSE>(
            src, dst, prev_len, b_out, groups_out, tw, 0, pointwise,
        ),
        13 => Radix::<13>::compute_group::<F, INVERSE>(
            src, dst, prev_len, b_out, groups_out, tw, 0, pointwise,
        ),
        17 => Radix::<17>::compute_group::<F, INVERSE>(
            src, dst, prev_len, b_out, groups_out, tw, 0, pointwise,
        ),
        23 => Radix::<23>::compute_group::<F, INVERSE>(
            src, dst, prev_len, b_out, groups_out, tw, 0, pointwise,
        ),
        _ => unreachable!("unsupported radix {r}"),
    }
}

#[inline]
fn dispatch_stage_const<
    F: CompositeCache + ShortWinogradScalar,
    const R: usize,
    const INVERSE: bool,
>(
    src: &[Complex<F>],
    dst: &mut [Complex<F>],
    prev_len: usize,
    groups_out: usize,
    twiddles: &[Complex<F>],
    pointwise: Option<&[Complex<F>]>,
) {
    debug_assert!(pointwise.is_none() || groups_out == 1);
    let tw = &[twiddles];
    let stage_chunk = prev_len * R;
    debug_assert!(dst.len() >= groups_out * stage_chunk);
    for (g, dst_block) in dst[..groups_out * stage_chunk]
        .chunks_exact_mut(stage_chunk)
        .enumerate()
    {
        Radix::<R>::compute_group::<F, INVERSE>(
            src, dst_block, prev_len, g, groups_out, tw, 0, pointwise,
        );
    }
}

#[inline]
pub(super) fn dispatch_radix_stage<F: CompositeCache + ShortWinogradScalar, const INVERSE: bool>(
    src: &[Complex<F>],
    dst: &mut [Complex<F>],
    prev_len: usize,
    groups_out: usize,
    r: usize,
    twiddles: &[Complex<F>],
    pointwise: Option<&[Complex<F>]>,
) {
    match r {
        2 => dispatch_stage_const::<F, 2, INVERSE>(
            src, dst, prev_len, groups_out, twiddles, pointwise,
        ),
        3 => dispatch_stage_const::<F, 3, INVERSE>(
            src, dst, prev_len, groups_out, twiddles, pointwise,
        ),
        4 => dispatch_stage_const::<F, 4, INVERSE>(
            src, dst, prev_len, groups_out, twiddles, pointwise,
        ),
        5 => dispatch_stage_const::<F, 5, INVERSE>(
            src, dst, prev_len, groups_out, twiddles, pointwise,
        ),
        7 => dispatch_stage_const::<F, 7, INVERSE>(
            src, dst, prev_len, groups_out, twiddles, pointwise,
        ),
        8 => dispatch_stage_const::<F, 8, INVERSE>(
            src, dst, prev_len, groups_out, twiddles, pointwise,
        ),
        11 => dispatch_stage_const::<F, 11, INVERSE>(
            src, dst, prev_len, groups_out, twiddles, pointwise,
        ),
        13 => dispatch_stage_const::<F, 13, INVERSE>(
            src, dst, prev_len, groups_out, twiddles, pointwise,
        ),
        16 => dispatch_stage_const::<F, 16, INVERSE>(
            src, dst, prev_len, groups_out, twiddles, pointwise,
        ),
        17 => dispatch_stage_const::<F, 17, INVERSE>(
            src, dst, prev_len, groups_out, twiddles, pointwise,
        ),
        23 => dispatch_stage_const::<F, 23, INVERSE>(
            src, dst, prev_len, groups_out, twiddles, pointwise,
        ),
        _ => unreachable!("unsupported radix {r}"),
    }
}
