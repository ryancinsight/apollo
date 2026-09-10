//! The flat radix-2 Stockham pass: the trailing stage of odd-power-of-two
//! decompositions (32, 128, 512 lower to `[4, .., 4, 2]`).
//!
//! Butterfly: `b0 = a0 + tw · a1`, `b1 = a0 − tw · a1`; the direction is in
//! the twiddle table.

use super::{apply_pointwise, cmul, load, store};
use crate::application::execution::kernel::components::winograd::WinogradScalar;
use eunomia::Complex;
use hermes_simd::{LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage};

/// One radix-2 pass: `g_count` groups of `prev_len` butterflies, the two
/// source rows `g_count * prev_len` complexes apart, each group's outputs
/// `stage_chunk` complexes apart in `dst`.
pub(in super::super) struct FlatPassR2<'a, T> {
    pub(in super::super) src: &'a [Complex<T>],
    pub(in super::super) dst: &'a mut [Complex<T>],
    pub(in super::super) prev_len: usize,
    pub(in super::super) g_count: usize,
    pub(in super::super) stage_chunk: usize,
    pub(in super::super) tw: &'a [Complex<T>],
    /// The pointwise spectrum multiplied into the output afterwards, on a
    /// convolution's last pass.
    pub(in super::super) pointwise: Option<&'a [Complex<T>]>,
}

impl<T> LaneKernel<T> for FlatPassR2<'_, T>
where
    T: LaneScalar + WinogradScalar,
{
    /// Whether the dispatched width ran the pass; the scalar backend
    /// declines, and the scalar pass runs instead.
    type Output = bool;

    #[expect(
        clippy::inline_always,
        reason = "the body must inline into the dispatcher's target-feature frame"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, _capability: Simd<T, A>) -> bool {
        let per = <A as SimdStorage<T>>::LANE_COUNT / 2;
        if per == 0 {
            return false;
        }
        let Self {
            src,
            dst,
            prev_len,
            g_count,
            stage_chunk,
            tw,
            pointwise,
        } = self;
        let stride = g_count * prev_len;
        assert!(
            src.len() >= 2 * stride
                && dst.len() >= g_count * stage_chunk
                && stage_chunk >= 2 * prev_len
                && tw.len() >= prev_len,
            "invariant: a radix-2 pass reads two rows of g_count * prev_len complexes and prev_len twiddles, and writes g_count blocks of stage_chunk"
        );

        if prev_len == 1 {
            for g in 0..g_count {
                let a0 = src[g];
                let a1 = src[stride + g];
                dst[2 * g] = Complex::new(a0.re + a1.re, a0.im + a1.im);
                dst[2 * g + 1] = Complex::new(a0.re - a1.re, a0.im - a1.im);
            }
        } else {
            for g in 0..g_count {
                let src_base = g * prev_len;
                let dst_base = g * stage_chunk;
                let mut j = 0;
                while j + per <= prev_len {
                    // SAFETY: `src_base + j + per <= stride`, so both source
                    // rows and the twiddle row stay inside their slices, and
                    // `dst_base + j + prev_len + per <= g_count * stage_chunk`,
                    // both by the assertion above.
                    unsafe {
                        let a0 = load::<T, A>(src, src_base + j);
                        let a1 = load::<T, A>(src, stride + src_base + j);
                        let a1 = cmul(a1, load::<T, A>(tw, j));
                        store(a0 + a1, dst, dst_base + j);
                        store(a0 - a1, dst, dst_base + j + prev_len);
                    }
                    j += per;
                }
                while j < prev_len {
                    let v = src[stride + src_base + j];
                    let t = tw[j];
                    let a1 = Complex::new(v.re * t.re - v.im * t.im, v.re * t.im + v.im * t.re);
                    let a0 = src[src_base + j];
                    dst[dst_base + j] = Complex::new(a0.re + a1.re, a0.im + a1.im);
                    dst[dst_base + j + prev_len] = Complex::new(a0.re - a1.re, a0.im - a1.im);
                    j += 1;
                }
            }
        }

        if let Some(factors) = pointwise {
            apply_pointwise::<T, A>(dst, factors);
        }
        true
    }
}
