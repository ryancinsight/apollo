use crate::application::execution::kernel::components::winograd::ShortWinogradScalar;
use crate::application::execution::kernel::components::winograd::{
    apply_twiddle_impl, WinogradScalar,
};
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
        let arm1_start = tw_idx;
        unsafe { *stage_offsets.get_unchecked_mut(offset_idx) = tw_idx };
        offset_idx += 1;
        let base_angle = sign * std::f64::consts::TAU / stage_len as f64;
        let w_base = Complex::new(
            F::from_precise(base_angle.cos()),
            F::from_precise(base_angle.sin()),
        );
        // Arm 1: W^j for j=0..prev_len-1.
        let mut tw = one;
        for _ in 0..prev_len {
            unsafe { *all_twiddles.get_unchecked_mut(tw_idx) = tw };
            tw_idx += 1;
            tw = apply_twiddle_impl(tw, w_base);
        }
        // Arms 2..R-1: arm-k[j] = arm-(k-1)[j] * arm-1[j] (element-wise).
        for _ in 2..r {
            let prev_arm = tw_idx - prev_len;
            for j in 0..prev_len {
                let a = unsafe { *all_twiddles.get_unchecked(prev_arm + j) };
                let b = unsafe { *all_twiddles.get_unchecked(arm1_start + j) };
                unsafe { *all_twiddles.get_unchecked_mut(tw_idx) = apply_twiddle_impl(a, b) };
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
