use crate::application::execution::kernel::components::winograd::ShortWinogradScalar;
use crate::application::execution::kernel::components::winograd::WinogradScalar;
use eunomia::Complex;
use std::cell::RefCell;
use std::sync::Arc;

#[derive(Clone)]
pub struct CompositeTwiddleEntry<C> {
    pub radices: Arc<[usize]>,
    pub twiddles: Arc<[C]>,
    pub offsets: Arc<[usize]>,
}

pub trait CompositeCache: WinogradScalar + ShortWinogradScalar {
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex<Self>]) -> R) -> R;

    /// Runs the batched-layout four-step transform, reporting whether it applied.
    ///
    /// The batched kernel needs bounds (`LaneScalar`, `Pod`) that this trait's
    /// generic callers do not carry, so the concrete scalars route to it here.
    /// Returns `false` when the length is outside the path's domain, leaving the
    /// caller to take its existing route.
    fn try_four_step_batched<const INVERSE: bool>(data: &mut [Complex<Self>]) -> bool;
    fn cached_twiddles<const INVERSE: bool>(
        radices: &[usize],
    ) -> (Arc<[Complex<Self>]>, Arc<[usize]>);

    /// Attempt an AVX2-accelerated flat Stockham pass for radix-4.
    ///
    /// Processes ALL `g_count` groups in one call (not per-group), amortizing
    /// the `#[target_feature]` function-call overhead across the entire stage.
    /// Returns `true` if the pass was handled; `false` if scalar fallback is needed.
    ///
    /// Default: returns `false` (scalar path).
    #[allow(unused_variables)]
    #[inline]
    fn try_flat_pass_r4<const INVERSE: bool>(
        src: &[Complex<Self>],
        dst: &mut [Complex<Self>],
        prev_len: usize,
        g_count: usize,
        stage_chunk: usize,
        tw: &[Complex<Self>],
        pointwise: Option<&[Complex<Self>]>,
    ) -> bool {
        false
    }

    /// Attempt an AVX2-accelerated flat Stockham pass for radix-3.
    ///
    /// Same amortization contract as `try_flat_pass_r4`.
    /// Default: returns `false` (scalar path).
    #[allow(unused_variables)]
    #[inline]
    fn try_flat_pass_r3<const INVERSE: bool>(
        src: &[Complex<Self>],
        dst: &mut [Complex<Self>],
        prev_len: usize,
        g_count: usize,
        stage_chunk: usize,
        tw: &[Complex<Self>],
        pointwise: Option<&[Complex<Self>]>,
    ) -> bool {
        false
    }

    /// Attempt an AVX2-accelerated flat Stockham pass for radix-5.
    ///
    /// Same amortization contract as `try_flat_pass_r4`. Vectorizes the
    /// radix-5 stage (previously scalar) shared by every composite with a
    /// factor of 5 (e.g. N=15, 25, 100, 180, 1000).
    /// Default: returns `false` (scalar path).
    #[allow(unused_variables)]
    #[inline]
    fn try_flat_pass_r5<const INVERSE: bool>(
        src: &[Complex<Self>],
        dst: &mut [Complex<Self>],
        prev_len: usize,
        g_count: usize,
        stage_chunk: usize,
        tw: &[Complex<Self>],
        pointwise: Option<&[Complex<Self>]>,
    ) -> bool {
        false
    }

    /// Attempt an AVX2-accelerated flat Stockham pass for radix-7.
    ///
    /// Same amortization contract; vectorizes the radix-7 stage (previously
    /// scalar) shared by every composite with a factor of 7.
    /// Default: returns `false` (scalar path).
    #[allow(unused_variables)]
    #[inline]
    fn try_flat_pass_r7<const INVERSE: bool>(
        src: &[Complex<Self>],
        dst: &mut [Complex<Self>],
        prev_len: usize,
        g_count: usize,
        stage_chunk: usize,
        tw: &[Complex<Self>],
        pointwise: Option<&[Complex<Self>]>,
    ) -> bool {
        false
    }

    /// Attempt an AVX2-accelerated flat Stockham pass for radix-2.
    ///
    /// Vectorizes the trailing radix-2 stage of odd-power-of-two
    /// decompositions (previously scalar). Default: returns `false`.
    #[allow(unused_variables)]
    #[inline]
    fn try_flat_pass_r2<const INVERSE: bool>(
        src: &[Complex<Self>],
        dst: &mut [Complex<Self>],
        prev_len: usize,
        g_count: usize,
        stage_chunk: usize,
        tw: &[Complex<Self>],
        pointwise: Option<&[Complex<Self>]>,
    ) -> bool {
        false
    }
}

thread_local! {
    static TL_TWIDDLES_FWD_64: RefCell<Vec<CompositeTwiddleEntry<eunomia::Complex64>>> = const { RefCell::new(Vec::new()) };
    static TL_TWIDDLES_INV_64: RefCell<Vec<CompositeTwiddleEntry<eunomia::Complex64>>> = const { RefCell::new(Vec::new()) };

    static TL_TWIDDLES_FWD_32: RefCell<Vec<CompositeTwiddleEntry<eunomia::Complex32>>> = const { RefCell::new(Vec::new()) };
    static TL_TWIDDLES_INV_32: RefCell<Vec<CompositeTwiddleEntry<eunomia::Complex32>>> = const { RefCell::new(Vec::new()) };

    static TL_COMPOSITE_SCRATCH_64: mnemosyne::scratch::ScratchPool<eunomia::Complex64> =
        const { mnemosyne::scratch::ScratchPool::new() };
    static TL_COMPOSITE_SCRATCH_32: mnemosyne::scratch::ScratchPool<eunomia::Complex32> =
        const { mnemosyne::scratch::ScratchPool::new() };
}

/// Whether the batched-layout four-step covers this length.
///
/// Square splits only, which is what the four-step gate already admits and what
/// lets the middle transpose run in place. The upper bound is where the
/// superseded path begins distributing its row transforms across threads: the
/// batched kernel is single-threaded, and its batch dimension is the innermost
/// one, so splitting it across workers hands each a strided view rather than a
/// contiguous chunk. Extending past this bound is a measurement plus a
/// partitioning design, not a constant change.
#[inline]
fn batched_four_step_applies(n: usize) -> bool {
    n.is_power_of_two()
        && n.trailing_zeros() % 2 == 0
        && n >= 4
        && n < crate::application::execution::kernel::components::four_step::PARALLEL_ROW_THRESHOLD
}

fn build_composite_twiddles<F: WinogradScalar, const INVERSE: bool>(
    radices: &[usize],
) -> (Vec<Complex<F>>, Vec<usize>) {
    let sign: f64 = if INVERSE { 1.0 } else { -1.0 };
    // Per-arm layout: (R-1)*prev_len entries per stage.
    // Arm k (k=1..R-1) at stage_offset + (k-1)*prev_len: W^{k*j} for j=0..prev_len-1.
    // Radix-2 stages are unchanged ((2-1)*L = L).
    let total_twiddles: usize = radices
        .iter()
        .scan(1usize, |p, &r| {
            let out = *p * (r - 1);
            *p *= r;
            Some(out)
        })
        .sum();
    let one = Complex::new(F::from_precise(1.0), F::from_precise(0.0));
    let mut all_twiddles = vec![one; total_twiddles];
    let mut stage_offsets = vec![0usize; radices.len()];
    let mut prev_len = 1usize;
    let mut tw_idx = 0;
    let mut offset_idx = 0;
    for &r in radices {
        let stage_len = prev_len * r;
        unsafe { *stage_offsets.get_unchecked_mut(offset_idx) = tw_idx };
        offset_idx += 1;
        // Arms 1..R-1: arm-k[j] = W^{k*j}, evaluated directly.
        //
        // The exponent is reduced modulo `stage_len` before the angle is
        // formed, so every argument stays inside one period and the
        // library's own argument reduction is never the accuracy limit.
        //
        // The superseded form built arm 1 by the recurrence
        // `tw[j] = tw[j-1] * W_base` and each later arm by multiplying the
        // previous arm, so entry `j` of the last stage carried the rounding
        // of `j` complex multiplications. That is `O(N * u)` twiddle error,
        // and it propagates directly into the transform: the `O(log N * u)`
        // FFT forward-error bound (Higham, *Accuracy and Stability of
        // Numerical Algorithms*, 2nd ed., section 24.1) holds only for
        // accurately computed twiddles and degrades to `O(N * u)` without
        // them. Direct evaluation costs one `sin_cos` per entry at plan
        // build, which a plan amortizes; the recurrence saved that at the
        // cost of the bound.
        for k in 1..r {
            for j in 0..prev_len {
                let reduced = (k * j) % stage_len;
                let angle = sign * std::f64::consts::TAU * reduced as f64 / stage_len as f64;
                let (sin, cos) = angle.sin_cos();
                unsafe {
                    *all_twiddles.get_unchecked_mut(tw_idx) =
                        Complex::new(F::from_precise(cos), F::from_precise(sin));
                }
                tw_idx += 1;
            }
        }
        prev_len = stage_len;
    }
    debug_assert_eq!(tw_idx, total_twiddles);
    debug_assert_eq!(offset_idx, radices.len());
    (all_twiddles, stage_offsets)
}

impl CompositeCache for f64 {
    /// AVX2+FMA flat pass for radix-2 f64 (trailing stage of odd powers of two).
    #[inline]
    fn try_flat_pass_r2<const INVERSE: bool>(
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
        prev_len: usize,
        g_count: usize,
        stage_chunk: usize,
        tw: &[Complex<f64>],
        pointwise: Option<&[Complex<f64>]>,
    ) -> bool {
        // Amortization guard: below n=64 the AVX setup (feature check + frame)
        // exceeds the scalar radix-2 cost (measured regression at N=32). Keep
        // tiny radix-2 stages on the scalar path.
        if g_count.saturating_mul(stage_chunk) < 64 {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Feature detection above guarantees AVX2+FMA.
            unsafe {
                super::avx2::flat_pass_r2_f64(
                    src,
                    dst,
                    prev_len,
                    g_count,
                    stage_chunk,
                    tw,
                    pointwise,
                );
            }
            return true;
        }
        false
    }

    /// AVX2+FMA flat pass for radix-7 f64. Checked once per stage (not per group).
    #[inline]
    fn try_flat_pass_r7<const INVERSE: bool>(
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
        prev_len: usize,
        g_count: usize,
        stage_chunk: usize,
        tw: &[Complex<f64>],
        pointwise: Option<&[Complex<f64>]>,
    ) -> bool {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Feature detection above guarantees AVX2+FMA.
            unsafe {
                super::avx2::flat_pass_r7_f64::<INVERSE>(
                    src,
                    dst,
                    prev_len,
                    g_count,
                    stage_chunk,
                    tw,
                    pointwise,
                );
            }
            return true;
        }
        false
    }

    /// AVX2+FMA flat pass for radix-5 f64. Checked once per stage (not per group).
    #[inline]
    fn try_flat_pass_r5<const INVERSE: bool>(
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
        prev_len: usize,
        g_count: usize,
        stage_chunk: usize,
        tw: &[Complex<f64>],
        pointwise: Option<&[Complex<f64>]>,
    ) -> bool {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Feature detection above guarantees AVX2+FMA.
            unsafe {
                super::avx2::flat_pass_r5_f64::<INVERSE>(
                    src,
                    dst,
                    prev_len,
                    g_count,
                    stage_chunk,
                    tw,
                    pointwise,
                );
            }
            return true;
        }
        false
    }

    /// AVX2+FMA flat pass for radix-4 f64. Checked once per stage (not per group).
    #[inline]
    fn try_flat_pass_r4<const INVERSE: bool>(
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
        prev_len: usize,
        g_count: usize,
        stage_chunk: usize,
        tw: &[Complex<f64>],
        pointwise: Option<&[Complex<f64>]>,
    ) -> bool {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Feature detection above guarantees AVX2+FMA.
            unsafe {
                super::avx2::flat_pass_r4_f64::<INVERSE>(
                    src,
                    dst,
                    prev_len,
                    g_count,
                    stage_chunk,
                    tw,
                    pointwise,
                );
            }
            return true;
        }
        false
    }

    /// AVX2+FMA flat pass for radix-3 f64.
    #[inline]
    fn try_flat_pass_r3<const INVERSE: bool>(
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
        prev_len: usize,
        g_count: usize,
        stage_chunk: usize,
        tw: &[Complex<f64>],
        pointwise: Option<&[Complex<f64>]>,
    ) -> bool {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Feature detection above guarantees AVX2+FMA.
            unsafe {
                super::avx2::flat_pass_r3_f64::<INVERSE>(
                    src,
                    dst,
                    prev_len,
                    g_count,
                    stage_chunk,
                    tw,
                    pointwise,
                );
            }
            return true;
        }
        false
    }

    #[inline]
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex<Self>]) -> R) -> R {
        TL_COMPOSITE_SCRATCH_64.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn try_four_step_batched<const INVERSE: bool>(data: &mut [Complex<Self>]) -> bool {
        use crate::application::execution::kernel::components::batched::four_step_batched;
        let n = data.len();
        if !batched_four_step_applies(n) {
            return false;
        }
        // The driver pads each plane row by a cache line to break power-of-two
        // stride aliasing, so its scratch requirement exceeds n; the driver's
        // own helper is the single definition of it.
        Self::with_scratch(
            crate::application::execution::kernel::components::batched::scratch_len(n),
            |scratch| {
                four_step_batched::<Self, INVERSE>(data, scratch);
            },
        );
        true
    }

    #[inline]
    fn cached_twiddles<const INVERSE: bool>(
        radices: &[usize],
    ) -> (Arc<[Complex<Self>]>, Arc<[usize]>) {
        let tl = if INVERSE {
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
        let (tw, offsets) = build_composite_twiddles::<f64, INVERSE>(radices);
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
    /// AVX2+FMA flat pass for radix-2 f32 (trailing stage of odd powers of two).
    #[inline]
    fn try_flat_pass_r2<const INVERSE: bool>(
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
        prev_len: usize,
        g_count: usize,
        stage_chunk: usize,
        tw: &[Complex<f32>],
        pointwise: Option<&[Complex<f32>]>,
    ) -> bool {
        // Amortization guard: below n=64 the AVX setup exceeds the scalar
        // radix-2 cost. Keep tiny radix-2 stages on the scalar path.
        if g_count.saturating_mul(stage_chunk) < 64 {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Feature detection above guarantees AVX2+FMA.
            unsafe {
                super::avx2::flat_pass_r2_f32(
                    src,
                    dst,
                    prev_len,
                    g_count,
                    stage_chunk,
                    tw,
                    pointwise,
                );
            }
            return true;
        }
        false
    }

    /// AVX2+FMA flat pass for radix-7 f32. Processes 4 complex per __m256 register.
    #[inline]
    fn try_flat_pass_r7<const INVERSE: bool>(
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
        prev_len: usize,
        g_count: usize,
        stage_chunk: usize,
        tw: &[Complex<f32>],
        pointwise: Option<&[Complex<f32>]>,
    ) -> bool {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Feature detection above guarantees AVX2+FMA.
            unsafe {
                super::avx2::flat_pass_r7_f32::<INVERSE>(
                    src,
                    dst,
                    prev_len,
                    g_count,
                    stage_chunk,
                    tw,
                    pointwise,
                );
            }
            return true;
        }
        false
    }

    /// AVX2+FMA flat pass for radix-5 f32. Processes 4 complex per __m256 register.
    #[inline]
    fn try_flat_pass_r5<const INVERSE: bool>(
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
        prev_len: usize,
        g_count: usize,
        stage_chunk: usize,
        tw: &[Complex<f32>],
        pointwise: Option<&[Complex<f32>]>,
    ) -> bool {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Feature detection above guarantees AVX2+FMA.
            unsafe {
                super::avx2::flat_pass_r5_f32::<INVERSE>(
                    src,
                    dst,
                    prev_len,
                    g_count,
                    stage_chunk,
                    tw,
                    pointwise,
                );
            }
            return true;
        }
        false
    }

    /// AVX2+FMA flat pass for radix-4 f32. Processes 4 complex per __m256 register.
    #[inline]
    fn try_flat_pass_r4<const INVERSE: bool>(
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
        prev_len: usize,
        g_count: usize,
        stage_chunk: usize,
        tw: &[Complex<f32>],
        pointwise: Option<&[Complex<f32>]>,
    ) -> bool {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Feature detection above guarantees AVX2+FMA.
            unsafe {
                super::avx2::flat_pass_r4_f32::<INVERSE>(
                    src,
                    dst,
                    prev_len,
                    g_count,
                    stage_chunk,
                    tw,
                    pointwise,
                );
            }
            return true;
        }
        false
    }

    /// AVX2+FMA flat pass for radix-3 f32.
    #[inline]
    fn try_flat_pass_r3<const INVERSE: bool>(
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
        prev_len: usize,
        g_count: usize,
        stage_chunk: usize,
        tw: &[Complex<f32>],
        pointwise: Option<&[Complex<f32>]>,
    ) -> bool {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Feature detection above guarantees AVX2+FMA.
            unsafe {
                super::avx2::flat_pass_r3_f32::<INVERSE>(
                    src,
                    dst,
                    prev_len,
                    g_count,
                    stage_chunk,
                    tw,
                    pointwise,
                );
            }
            return true;
        }
        false
    }

    #[inline]
    fn with_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex<Self>]) -> R) -> R {
        TL_COMPOSITE_SCRATCH_32.with(|pool| pool.with_scratch(n, f))
    }

    #[inline]
    fn try_four_step_batched<const INVERSE: bool>(data: &mut [Complex<Self>]) -> bool {
        use crate::application::execution::kernel::components::batched::four_step_batched;
        let n = data.len();
        if !batched_four_step_applies(n) {
            return false;
        }
        // As the f64 impl above: the driver's helper is the single definition
        // of the padded-plane scratch requirement.
        Self::with_scratch(
            crate::application::execution::kernel::components::batched::scratch_len(n),
            |scratch| {
                four_step_batched::<Self, INVERSE>(data, scratch);
            },
        );
        true
    }

    #[inline]
    fn cached_twiddles<const INVERSE: bool>(
        radices: &[usize],
    ) -> (Arc<[Complex<Self>]>, Arc<[usize]>) {
        let tl = if INVERSE {
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
        let (tw, offsets) = build_composite_twiddles::<f32, INVERSE>(radices);
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
