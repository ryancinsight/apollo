//! The lane-wise `ROWS`-point DIF column pass, shared by both register
//! widths of the instance-major base kernel.
//!
//! Both widths stage the row spectra as `ROWS x GROUPS` chunks — the
//! four-lane layout holds two interleaved complex samples per chunk over
//! eight groups, the eight-lane layout four samples over four groups — and
//! the pass below is identical modulo that `GROUPS` factor: load column
//! chunk `a * GROUPS + g`, apply the mixed-radix twiddle from the
//! dup-split table, run the `ROWS`-point DIF across the row index, and
//! store register `q` to output row `rev(q)` through the selected
//! [`StoreSink`]. Keeping it in one place keeps the sink family available
//! to every width for the same reason it keeps the arithmetic identical.

use super::super::cmul::cmul_chunk;
use super::store::StoreSink;
use super::{MIX_CH, REV2, REV3};
use hermes_simd::{ComplexReg, LaneScalar, Simd, SimdArch, SimdKernel};

/// Runs the column pass over `staging`, storing into `data` (direct) or
/// through `sink` (combining). `col` carries `W_8^1` and `W_8^3` for the
/// eight-row form's distance-4 stage; the four-row form never reads them.
///
/// Inlined into the dispatcher's target-feature frame; the caller's
/// fixed-size arrays keep the view bounds foldable through that inlining.
#[expect(
    clippy::inline_always,
    reason = "the pass must fold into the dispatcher's target-feature frame"
)]
#[inline(always)]
pub(super) fn column_pass<T, A, S, const INVERSE: bool, const ROWS: usize, const GROUPS: usize>(
    simd: Simd<T, A>,
    staging: &[T],
    table: &[T],
    col: &[[T; 2]; 2],
    data: &mut [T],
    mut sink: S,
) where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
    S: StoreSink<T>,
{
    let tab_view = simd.view(table);
    let complex_splat = |sample: [T; 2]| {
        let (interleaved, _) = simd.splat(sample[0]).interleave(simd.splat(sample[1]));
        ComplexReg::from_interleaved(interleaved)
    };
    let w8_1 = complex_splat(col[0]);
    let w8_3 = complex_splat(col[1]);
    let zero_complex = ComplexReg::from_interleaved(simd.zero());
    let stg = simd.view(staging);
    let mut out = simd.view_mut(data);
    for g in 0..GROUPS {
        let mut c = [zero_complex; 8];
        for (a, reg) in c.iter_mut().enumerate().take(ROWS) {
            *reg = ComplexReg::from_interleaved(hermes_simd::Vector::from_view_chunk(
                &stg,
                a * GROUPS + g,
            ));
        }
        for a in 1..ROWS {
            c[a] = cmul_chunk(&tab_view, c[a], MIX_CH + 2 * ((a - 1) * GROUPS + g));
        }
        // Distance 4 (L = 8): v * W8^a on the difference. A four-row
        // column ends before this distance exists.
        if ROWS == 8 {
            for a in 0..4usize {
                let (u, v) = c[a].butterfly(c[a + 4]);
                c[a] = u;
                c[a + 4] = match a {
                    0 => v,
                    1 => v * w8_1,
                    2 => {
                        if INVERSE {
                            v.mul_i()
                        } else {
                            v.mul_neg_i()
                        }
                    }
                    _ => v * w8_3,
                };
            }
        }
        // Distance 2 (L = 4): twiddles [1, -+i].
        for h in 0..ROWS / 4 {
            let base = 4 * h;
            for j in 0..2usize {
                let (u, v) = c[base + j].butterfly(c[base + j + 2]);
                c[base + j] = u;
                c[base + j + 2] = if j == 0 {
                    v
                } else if INVERSE {
                    v.mul_i()
                } else {
                    v.mul_neg_i()
                };
            }
        }
        // Distance 1 (L = 2): twiddle one.
        for h in 0..ROWS / 2 {
            let base = 2 * h;
            let (u, v) = c[base].butterfly(c[base + 1]);
            (c[base], c[base + 1]) = (u, v);
        }
        for (q, reg) in c.iter().enumerate().take(ROWS) {
            let row = if ROWS == 8 { REV3[q] } else { REV2[q] };
            let j = row * GROUPS + g;
            if S::DIRECT {
                reg.into_interleaved().store_to_view_chunk(&mut out, j);
            } else {
                sink.store(&simd, *reg, j);
            }
        }
    }
}
