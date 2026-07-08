#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]
#[cfg(all(test, target_arch = "x86_64"))]
use super::fixed::{fixed_len512_precise_avx_fma, fixed_len512_reduced_avx_fma};
#[cfg(all(test, target_arch = "x86_64"))]
use crate::application::execution::kernel::components::stockham::avx::generic::triple::stage_triple_radix1_avx_fma;
#[cfg(all(test, target_arch = "x86_64"))]
use eunomia::{Complex32, Complex64};

#[cfg(all(test, target_arch = "x86_64"))]
type PackedReducedComplex = [Complex32; 4];

#[cfg(all(test, target_arch = "x86_64"))]
type PackedPreciseComplex = [Complex64; 2];

#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) fn build_butterfly512_twiddles_reduced(
    twiddles: &[Complex32],
) -> [PackedReducedComplex; 120] {
    debug_assert!(twiddles.len() >= 511);
    core::array::from_fn(|index| {
        let row = (index % 15) + 1;
        let col_base = (index / 15) * 4;
        core::array::from_fn(|lane| {
            stockham_mixed_twiddle_reduced::<16, 32>(twiddles, row, col_base + lane)
        })
    })
}

#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) fn build_butterfly512_twiddles_precise(
    twiddles: &[Complex64],
) -> [PackedPreciseComplex; 240] {
    debug_assert!(twiddles.len() >= 511);
    core::array::from_fn(|index| {
        let row = (index % 15) + 1;
        let col_base = (index / 15) * 2;
        core::array::from_fn(|lane| {
            stockham_mixed_twiddle_precise::<16, 32>(twiddles, row, col_base + lane)
        })
    })
}

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn hybrid_radix8x512_precise_avx_fma<const INVERSE: bool>(
    data: &mut [Complex64],
    scratch: &mut [Complex64],
    twiddles: &[Complex64],
) {
    const ROWS: usize = 8;
    const COLS: usize = 512;
    debug_assert_eq!(data.len(), ROWS * COLS);
    debug_assert_eq!(scratch.len(), ROWS * COLS);
    debug_assert!(twiddles.len() >= ROWS * COLS - 1);

    let row_twiddles = &twiddles[..COLS - 1];
    stage_triple_radix1_avx_fma::<f64>(data, scratch, &twiddles[1..3], &twiddles[3..7]);
    let mut r = 1;
    while r < ROWS {
        let row_base = r * COLS;
        let mut c = 1;
        while c < COLS {
            scratch[row_base + c] *= stockham_mixed_twiddle_precise::<ROWS, COLS>(twiddles, r, c);
            c += 1;
        }
        r += 1;
    }

    for r in 0..ROWS {
        let row = &mut scratch[r * COLS..(r + 1) * COLS];
        let row_scratch = &mut data[r * COLS..(r + 1) * COLS];
        fixed_len512_precise_avx_fma(row, row_scratch, row_twiddles);
    }

    for r in 0..ROWS {
        for c in 0..COLS {
            data[c * ROWS + r] = scratch[r * COLS + c];
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
pub(crate) unsafe fn hybrid_radix8x512_reduced_avx_fma<const INVERSE: bool>(
    data: &mut [Complex32],
    scratch: &mut [Complex32],
    twiddles: &[Complex32],
) {
    const ROWS: usize = 8;
    const COLS: usize = 512;
    debug_assert_eq!(data.len(), ROWS * COLS);
    debug_assert_eq!(scratch.len(), ROWS * COLS);
    debug_assert!(twiddles.len() >= ROWS * COLS - 1);

    let row_twiddles = &twiddles[..COLS - 1];
    stage_triple_radix1_avx_fma::<f32>(data, scratch, &twiddles[1..3], &twiddles[3..7]);
    let mut r = 1;
    while r < ROWS {
        let row_base = r * COLS;
        let mut c = 1;
        while c < COLS {
            scratch[row_base + c] *= stockham_mixed_twiddle_reduced::<ROWS, COLS>(twiddles, r, c);
            c += 1;
        }
        r += 1;
    }

    for r in 0..ROWS {
        let row = &mut scratch[r * COLS..(r + 1) * COLS];
        let row_scratch = &mut data[r * COLS..(r + 1) * COLS];
        fixed_len512_reduced_avx_fma(row, row_scratch, row_twiddles);
    }

    for r in 0..ROWS {
        for c in 0..COLS {
            data[c * ROWS + r] = scratch[r * COLS + c];
        }
    }
}

#[inline]
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) fn stockham_mixed_twiddle_reduced<const ROWS: usize, const COLS: usize>(
    twiddles: &[eunomia::Complex32],
    row: usize,
    col: usize,
) -> eunomia::Complex32 {
    let exponent = row * col;
    if exponent == 0 {
        return eunomia::Complex32::new(1.0, 0.0);
    }
    let half_n = (ROWS * COLS) >> 1;
    let stage_base = half_n - 1;
    if exponent < half_n {
        twiddles[stage_base + exponent]
    } else {
        -twiddles[stage_base + exponent - half_n]
    }
}

#[inline]
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) fn stockham_mixed_twiddle_precise<const ROWS: usize, const COLS: usize>(
    twiddles: &[eunomia::Complex64],
    row: usize,
    col: usize,
) -> eunomia::Complex64 {
    let exponent = row * col;
    if exponent == 0 {
        return eunomia::Complex64::new(1.0, 0.0);
    }
    let half_n = (ROWS * COLS) >> 1;
    let stage_base = half_n - 1;
    if exponent < half_n {
        twiddles[stage_base + exponent]
    } else {
        -twiddles[stage_base + exponent - half_n]
    }
}
