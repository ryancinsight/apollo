//! Inline fixed-array DFT codelets for N=96..126.
//!
//! | N   | Decomposition           | Twiddles |
//! |-----|-------------------------|----------|
//! | 96  | Good-Thomas 32×3 (PFA)  | none     |
//! | 98  | Good-Thomas 2×49 (PFA)  | none     |
//! | 99  | Good-Thomas 9×11 (PFA)  | none     |
//! | 105 | Good-Thomas 15×7 (PFA)  | none     |
//! | 112 | Good-Thomas 16×7 (PFA)  | none     |
//! | 120 | Good-Thomas 8×15 (PFA)  | none     |
//! | 126 | Good-Thomas 2×63 (PFA)  | none     |

use super::super::radix::{dft3_impl, dft7_impl, dft8_impl, dft11_impl, dft15_impl};
use super::super::traits::WinogradScalar;
use super::small::dft9_impl;
use super::{dft16_impl, dft32_impl, dft49_impl, dft63_impl};

// ── N=96: Good-Thomas 32×3 ────────────────────────────────────────────────────
//
// n1=32, n2=3. N2⁻¹ mod N1=11; N1⁻¹ mod N2=2. Output: k=(33·k1+64·k2)%96.
#[inline(always)]
pub(crate) fn dft96_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 96],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 3]; 32] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(3 * i1 + 32 * j) % 96]));
    for row in &mut rows {
        dft3_impl(row, inverse);
    }
    for k2 in 0..3usize {
        let mut col: [num_complex::Complex<F>; 32] = core::array::from_fn(|k1| rows[k1][k2]);
        dft32_impl(&mut col, inverse);
        let base = (64 * k2) % 96;
        for k1 in 0..32usize {
            data[(base + 33 * k1) % 96] = col[k1];
        }
    }
}

// ── N=98: Good-Thomas 2×49 ────────────────────────────────────────────────────
//
// n1=2, n2=49. N2⁻¹ mod N1=1; N1⁻¹ mod N2=25. Output: k=(49·k1+50·k2)%98.
// Butterfly: row0=even-indexed data, row1=odd-starting-at-49 data.
#[inline(always)]
pub(crate) fn dft98_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 98],
    inverse: bool,
) {
    let mut sums: [num_complex::Complex<F>; 49] =
        core::array::from_fn(|j| data[2 * j] + data[(2 * j + 49) % 98]);
    let mut diffs: [num_complex::Complex<F>; 49] =
        core::array::from_fn(|j| data[2 * j] - data[(2 * j + 49) % 98]);
    dft49_impl(&mut sums, inverse);
    dft49_impl(&mut diffs, inverse);
    for k2 in 0..49usize {
        data[(50 * k2) % 98] = sums[k2];
        data[(50 * k2 + 49) % 98] = diffs[k2];
    }
}

// ── N=99: Good-Thomas 9×11 ────────────────────────────────────────────────────
//
// n1=9, n2=11. N2⁻¹ mod N1=5; N1⁻¹ mod N2=5. Output: k=(55·k1+45·k2)%99.
#[inline(always)]
pub(crate) fn dft99_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 99],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 11]; 9] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(11 * i1 + 9 * j) % 99]));
    for row in &mut rows {
        dft11_impl(row, inverse);
    }
    for k2 in 0..11usize {
        let mut col: [num_complex::Complex<F>; 9] = core::array::from_fn(|k1| rows[k1][k2]);
        dft9_impl(&mut col, inverse);
        let base = (45 * k2) % 99;
        for k1 in 0..9usize {
            data[(base + 55 * k1) % 99] = col[k1];
        }
    }
}

// ── N=105: Good-Thomas 15×7 ───────────────────────────────────────────────────
//
// n1=15, n2=7. N2⁻¹ mod N1=13; N1⁻¹ mod N2=1. Output: k=(91·k1+15·k2)%105.
#[inline(always)]
pub(crate) fn dft105_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 105],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 7]; 15] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(7 * i1 + 15 * j) % 105]));
    for row in &mut rows {
        dft7_impl(row, inverse);
    }
    for k2 in 0..7usize {
        let mut col: [num_complex::Complex<F>; 15] = core::array::from_fn(|k1| rows[k1][k2]);
        dft15_impl(&mut col, inverse);
        let base = (15 * k2) % 105;
        for k1 in 0..15usize {
            data[(base + 91 * k1) % 105] = col[k1];
        }
    }
}

// ── N=112: Good-Thomas 16×7 ───────────────────────────────────────────────────
//
// n1=16, n2=7. N2⁻¹ mod N1=7; N1⁻¹ mod N2=4. Output: k=(49·k1+64·k2)%112.
#[inline(always)]
pub(crate) fn dft112_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 112],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 7]; 16] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(7 * i1 + 16 * j) % 112]));
    for row in &mut rows {
        dft7_impl(row, inverse);
    }
    for k2 in 0..7usize {
        let mut col: [num_complex::Complex<F>; 16] = core::array::from_fn(|k1| rows[k1][k2]);
        dft16_impl(&mut col, inverse);
        let base = (64 * k2) % 112;
        for k1 in 0..16usize {
            data[(base + 49 * k1) % 112] = col[k1];
        }
    }
}

// ── N=120: Good-Thomas 8×15 ───────────────────────────────────────────────────
//
// n1=8, n2=15. N2⁻¹ mod N1=7; N1⁻¹ mod N2=2. Output: k=(105·k1+16·k2)%120.
#[inline(always)]
pub(crate) fn dft120_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 120],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 15]; 8] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(15 * i1 + 8 * j) % 120]));
    for row in &mut rows {
        dft15_impl(row, inverse);
    }
    for k2 in 0..15usize {
        let mut col: [num_complex::Complex<F>; 8] = core::array::from_fn(|k1| rows[k1][k2]);
        dft8_impl(&mut col, inverse);
        let base = (16 * k2) % 120;
        for k1 in 0..8usize {
            data[(base + 105 * k1) % 120] = col[k1];
        }
    }
}

// ── N=126: Good-Thomas 2×63 ───────────────────────────────────────────────────
//
// n1=2, n2=63. N2⁻¹ mod N1=1; N1⁻¹ mod N2=32. Output: k=(63·k1+64·k2)%126.
// Butterfly: row0=even-indexed data[0,2,..124], row1=data[63,65,..,125,1,..,61].
#[inline(always)]
pub(crate) fn dft126_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 126],
    inverse: bool,
) {
    let mut sums: [num_complex::Complex<F>; 63] =
        core::array::from_fn(|j| data[2 * j] + data[(2 * j + 63) % 126]);
    let mut diffs: [num_complex::Complex<F>; 63] =
        core::array::from_fn(|j| data[2 * j] - data[(2 * j + 63) % 126]);
    dft63_impl(&mut sums, inverse);
    dft63_impl(&mut diffs, inverse);
    for k2 in 0..63usize {
        data[(64 * k2) % 126] = sums[k2];
        data[(64 * k2 + 63) % 126] = diffs[k2];
    }
}
