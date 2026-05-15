//! Inline fixed-array DFT codelets for N=70..90.
//!
//! | N  | Decomposition           | Twiddles |
//! |----|-------------------------|----------|
//! | 70 | Good-Thomas 10×7 (PFA)  | none     |
//! | 72 | Good-Thomas 8×9 (PFA)   | none     |
//! | 75 | Good-Thomas 3×25 (PFA)  | none     |
//! | 77 | Good-Thomas 7×11 (PFA)  | none     |
//! | 80 | Good-Thomas 16×5 (PFA)  | none     |
//! | 84 | Good-Thomas 12×7 (PFA)  | none     |
//! | 88 | Good-Thomas 8×11 (PFA)  | none     |
//! | 90 | Good-Thomas 9×10 (PFA)  | none     |

use super::super::radix::{dft5_array_impl, dft7_impl, dft8_impl, dft11_impl};
use super::super::traits::WinogradScalar;
use super::small::{dft9_impl, dft10_impl, dft12_impl};
use super::{dft16_impl, dft25_impl};

// ── N=70: Good-Thomas 10×7 ────────────────────────────────────────────────────
//
// n1=10, n2=7. N2⁻¹ mod N1=3; N1⁻¹ mod N2=5. Output: k=(21·k1+50·k2)%70.
#[inline(always)]
pub(crate) fn dft70_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 70],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 7]; 10] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(7 * i1 + 10 * j) % 70]));
    for row in &mut rows {
        dft7_impl(row, inverse);
    }
    for k2 in 0..7usize {
        let mut col: [num_complex::Complex<F>; 10] = core::array::from_fn(|k1| rows[k1][k2]);
        dft10_impl(&mut col, inverse);
        let base = (50 * k2) % 70;
        for k1 in 0..10usize {
            data[(base + 21 * k1) % 70] = col[k1];
        }
    }
}

// ── N=72: Good-Thomas 8×9 ─────────────────────────────────────────────────────
//
// n1=8, n2=9. N2⁻¹ mod N1=1; N1⁻¹ mod N2=8. Output: k=(9·k1+64·k2)%72.
#[inline(always)]
pub(crate) fn dft72_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 72],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 9]; 8] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(9 * i1 + 8 * j) % 72]));
    for row in &mut rows {
        dft9_impl(row, inverse);
    }
    for k2 in 0..9usize {
        let mut col: [num_complex::Complex<F>; 8] = core::array::from_fn(|k1| rows[k1][k2]);
        dft8_impl(&mut col, inverse);
        let base = (64 * k2) % 72;
        for k1 in 0..8usize {
            data[(base + 9 * k1) % 72] = col[k1];
        }
    }
}

// ── N=75: Good-Thomas 3×25 ────────────────────────────────────────────────────
//
// n1=3, n2=25. N2⁻¹ mod N1=1; N1⁻¹ mod N2=17. Output: k=(25·k1+51·k2)%75.
#[inline(always)]
pub(crate) fn dft75_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 75],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 25]; 3] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(25 * i1 + 3 * j) % 75]));
    dft25_impl(&mut rows[0], inverse);
    dft25_impl(&mut rows[1], inverse);
    dft25_impl(&mut rows[2], inverse);
    for k2 in 0..25usize {
        let mut col: [num_complex::Complex<F>; 3] = [rows[0][k2], rows[1][k2], rows[2][k2]];
        super::super::radix::dft3_impl(&mut col, inverse);
        let base = (51 * k2) % 75;
        data[base] = col[0];
        data[(base + 25) % 75] = col[1];
        data[(base + 50) % 75] = col[2];
    }
}

// ── N=77: Good-Thomas 7×11 ────────────────────────────────────────────────────
//
// n1=7, n2=11. N2⁻¹ mod N1=2; N1⁻¹ mod N2=8. Output: k=(22·k1+56·k2)%77.
#[inline(always)]
pub(crate) fn dft77_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 77],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 11]; 7] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(11 * i1 + 7 * j) % 77]));
    for row in &mut rows {
        dft11_impl(row, inverse);
    }
    for k2 in 0..11usize {
        let mut col: [num_complex::Complex<F>; 7] = core::array::from_fn(|k1| rows[k1][k2]);
        dft7_impl(&mut col, inverse);
        let base = (56 * k2) % 77;
        for k1 in 0..7usize {
            data[(base + 22 * k1) % 77] = col[k1];
        }
    }
}

// ── N=80: Good-Thomas 16×5 ────────────────────────────────────────────────────
//
// n1=16, n2=5. N2⁻¹ mod N1=13; N1⁻¹ mod N2=1. Output: k=(65·k1+16·k2)%80.
#[inline(always)]
pub(crate) fn dft80_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 80],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 5]; 16] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(5 * i1 + 16 * j) % 80]));
    for row in &mut rows {
        dft5_array_impl(row, inverse);
    }
    for k2 in 0..5usize {
        let mut col: [num_complex::Complex<F>; 16] = core::array::from_fn(|k1| rows[k1][k2]);
        dft16_impl(&mut col, inverse);
        let base = (16 * k2) % 80;
        for k1 in 0..16usize {
            data[(base + 65 * k1) % 80] = col[k1];
        }
    }
}

// ── N=84: Good-Thomas 12×7 ────────────────────────────────────────────────────
//
// n1=12, n2=7. N2⁻¹ mod N1=7; N1⁻¹ mod N2=3. Output: k=(49·k1+36·k2)%84.
#[inline(always)]
pub(crate) fn dft84_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 84],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 7]; 12] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(7 * i1 + 12 * j) % 84]));
    for row in &mut rows {
        dft7_impl(row, inverse);
    }
    for k2 in 0..7usize {
        let mut col: [num_complex::Complex<F>; 12] = core::array::from_fn(|k1| rows[k1][k2]);
        dft12_impl(&mut col, inverse);
        let base = (36 * k2) % 84;
        for k1 in 0..12usize {
            data[(base + 49 * k1) % 84] = col[k1];
        }
    }
}

// ── N=88: Good-Thomas 8×11 ────────────────────────────────────────────────────
//
// n1=8, n2=11. N2⁻¹ mod N1=3; N1⁻¹ mod N2=7. Output: k=(33·k1+56·k2)%88.
#[inline(always)]
pub(crate) fn dft88_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 88],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 11]; 8] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(11 * i1 + 8 * j) % 88]));
    for row in &mut rows {
        dft11_impl(row, inverse);
    }
    for k2 in 0..11usize {
        let mut col: [num_complex::Complex<F>; 8] = core::array::from_fn(|k1| rows[k1][k2]);
        dft8_impl(&mut col, inverse);
        let base = (56 * k2) % 88;
        for k1 in 0..8usize {
            data[(base + 33 * k1) % 88] = col[k1];
        }
    }
}

// ── N=90: Good-Thomas 9×10 ────────────────────────────────────────────────────
//
// n1=9, n2=10. N2⁻¹ mod N1=1; N1⁻¹ mod N2=9. Output: k=(10·k1+81·k2)%90.
#[inline(always)]
pub(crate) fn dft90_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 90],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 10]; 9] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(10 * i1 + 9 * j) % 90]));
    for row in &mut rows {
        dft10_impl(row, inverse);
    }
    for k2 in 0..10usize {
        let mut col: [num_complex::Complex<F>; 9] = core::array::from_fn(|k1| rows[k1][k2]);
        dft9_impl(&mut col, inverse);
        let base = (81 * k2) % 90;
        for k1 in 0..9usize {
            data[(base + 10 * k1) % 90] = col[k1];
        }
    }
}
