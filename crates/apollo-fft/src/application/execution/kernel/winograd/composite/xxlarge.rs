//! Inline fixed-array DFT codelets for N=130..200.
//!
//! | N   | Decomposition            | Twiddles |
//! |-----|--------------------------|----------|
//! | 130 | Good-Thomas 10×13 (PFA)  | none     |
//! | 143 | Good-Thomas 11×13 (PFA)  | none     |
//! | 144 | Good-Thomas 16×9 (PFA)   | none     |
//! | 150 | Good-Thomas 6×25 (PFA)   | none     |
//! | 154 | Good-Thomas 14×11 (PFA)  | none     |
//! | 160 | Good-Thomas 32×5 (PFA)   | none     |
//! | 165 | Good-Thomas 15×11 (PFA)  | none     |
//! | 168 | Good-Thomas 56×3 (PFA)   | none     |
//! | 175 | Good-Thomas 25×7 (PFA)   | none     |
//! | 176 | Good-Thomas 16×11 (PFA)  | none     |
//! | 180 | Good-Thomas 36×5 (PFA)   | none     |
//! | 192 | Good-Thomas 64×3 (PFA)   | none     |
//! | 196 | Good-Thomas 4×49 (PFA)   | none     |
//! | 198 | Good-Thomas 2×99 (PFA)   | none     |
//! | 200 | Good-Thomas 8×25 (PFA)   | none     |

use super::super::radix::{dft3_impl, dft4_array_impl, dft5_array_impl, dft7_impl, dft8_impl, dft11_impl, dft15_impl};
use super::super::radix::dft13::Dft13Scalar;
use super::super::radix::dft13_impl;
use super::super::traits::WinogradScalar;
use super::small::{dft6_impl, dft9_impl, dft10_impl, dft14_impl};
use super::{dft16_impl, dft25_impl, dft32_impl, dft36_impl};
use super::{dft49_impl, dft56_impl, dft64_impl, dft99_impl};

// ── N=130: Good-Thomas 10×13 ──────────────────────────────────────────────────
//
// n1=10, n2=13. N2⁻¹ mod N1=7; N1⁻¹ mod N2=4. Output: k=(40·k2+91·k1)%130.
#[inline(always)]
pub(crate) fn dft130_impl<F: Dft13Scalar>(
    data: &mut [num_complex::Complex<F>; 130],
    inverse: bool,
) {
    if inverse {
        dft130_k::<F, true>(data);
    } else {
        dft130_k::<F, false>(data);
    }
}

#[inline(always)]
fn dft130_k<F: Dft13Scalar, const INVERSE: bool>(data: &mut [num_complex::Complex<F>; 130]) {
    let mut rows: [[num_complex::Complex<F>; 13]; 10] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(13 * i1 + 10 * j) % 130]));
    for row in &mut rows {
        dft13_impl::<F, INVERSE>(row);
    }
    for k2 in 0..13usize {
        let mut col: [num_complex::Complex<F>; 10] = core::array::from_fn(|k1| rows[k1][k2]);
        dft10_impl(&mut col, INVERSE);
        let base = (40 * k2) % 130;
        for k1 in 0..10usize {
            data[(base + 91 * k1) % 130] = col[k1];
        }
    }
}

// ── N=143: Good-Thomas 11×13 ──────────────────────────────────────────────────
//
// n1=11, n2=13. N2⁻¹ mod N1=6; N1⁻¹ mod N2=6. Output: k=(66·k2+78·k1)%143.
#[inline(always)]
pub(crate) fn dft143_impl<F: Dft13Scalar>(
    data: &mut [num_complex::Complex<F>; 143],
    inverse: bool,
) {
    if inverse {
        dft143_k::<F, true>(data);
    } else {
        dft143_k::<F, false>(data);
    }
}

#[inline(always)]
fn dft143_k<F: Dft13Scalar, const INVERSE: bool>(data: &mut [num_complex::Complex<F>; 143]) {
    let mut rows: [[num_complex::Complex<F>; 13]; 11] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(13 * i1 + 11 * j) % 143]));
    for row in &mut rows {
        dft13_impl::<F, INVERSE>(row);
    }
    for k2 in 0..13usize {
        let mut col: [num_complex::Complex<F>; 11] = core::array::from_fn(|k1| rows[k1][k2]);
        dft11_impl(&mut col, INVERSE);
        let base = (66 * k2) % 143;
        for k1 in 0..11usize {
            data[(base + 78 * k1) % 143] = col[k1];
        }
    }
}

// ── N=144: Good-Thomas 16×9 ───────────────────────────────────────────────────
//
// n1=16, n2=9. N2⁻¹ mod N1=9; N1⁻¹ mod N2=4. Output: k=(64·k2+81·k1)%144.
#[inline(always)]
pub(crate) fn dft144_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 144],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 9]; 16] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(9 * i1 + 16 * j) % 144]));
    for row in &mut rows {
        dft9_impl(row, inverse);
    }
    for k2 in 0..9usize {
        let mut col: [num_complex::Complex<F>; 16] = core::array::from_fn(|k1| rows[k1][k2]);
        dft16_impl(&mut col, inverse);
        let base = (64 * k2) % 144;
        for k1 in 0..16usize {
            data[(base + 81 * k1) % 144] = col[k1];
        }
    }
}

// ── N=150: Good-Thomas 6×25 ───────────────────────────────────────────────────
//
// n1=6, n2=25. N2⁻¹ mod N1=1; N1⁻¹ mod N2=21. Output: k=(126·k2+25·k1)%150.
#[inline(always)]
pub(crate) fn dft150_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 150],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 25]; 6] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(25 * i1 + 6 * j) % 150]));
    for row in &mut rows {
        dft25_impl(row, inverse);
    }
    for k2 in 0..25usize {
        let mut col: [num_complex::Complex<F>; 6] = core::array::from_fn(|k1| rows[k1][k2]);
        dft6_impl(&mut col, inverse);
        let base = (126 * k2) % 150;
        for k1 in 0..6usize {
            data[(base + 25 * k1) % 150] = col[k1];
        }
    }
}

// ── N=154: Good-Thomas 14×11 ──────────────────────────────────────────────────
//
// n1=14, n2=11. N2⁻¹ mod N1=9; N1⁻¹ mod N2=4. Output: k=(56·k2+99·k1)%154.
#[inline(always)]
pub(crate) fn dft154_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 154],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 11]; 14] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(11 * i1 + 14 * j) % 154]));
    for row in &mut rows {
        dft11_impl(row, inverse);
    }
    for k2 in 0..11usize {
        let mut col: [num_complex::Complex<F>; 14] = core::array::from_fn(|k1| rows[k1][k2]);
        dft14_impl(&mut col, inverse);
        let base = (56 * k2) % 154;
        for k1 in 0..14usize {
            data[(base + 99 * k1) % 154] = col[k1];
        }
    }
}

// ── N=160: Good-Thomas 32×5 ───────────────────────────────────────────────────
//
// n1=32, n2=5. N2⁻¹ mod N1=13; N1⁻¹ mod N2=3. Output: k=(96·k2+65·k1)%160.
#[inline(always)]
pub(crate) fn dft160_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 160],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 5]; 32] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(5 * i1 + 32 * j) % 160]));
    for row in &mut rows {
        dft5_array_impl(row, inverse);
    }
    for k2 in 0..5usize {
        let mut col: [num_complex::Complex<F>; 32] = core::array::from_fn(|k1| rows[k1][k2]);
        dft32_impl(&mut col, inverse);
        let base = (96 * k2) % 160;
        for k1 in 0..32usize {
            data[(base + 65 * k1) % 160] = col[k1];
        }
    }
}

// ── N=165: Good-Thomas 15×11 ──────────────────────────────────────────────────
//
// n1=15, n2=11. N2⁻¹ mod N1=11; N1⁻¹ mod N2=3. Output: k=(45·k2+121·k1)%165.
#[inline(always)]
pub(crate) fn dft165_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 165],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 11]; 15] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(11 * i1 + 15 * j) % 165]));
    for row in &mut rows {
        dft11_impl(row, inverse);
    }
    for k2 in 0..11usize {
        let mut col: [num_complex::Complex<F>; 15] = core::array::from_fn(|k1| rows[k1][k2]);
        dft15_impl(&mut col, inverse);
        let base = (45 * k2) % 165;
        for k1 in 0..15usize {
            data[(base + 121 * k1) % 165] = col[k1];
        }
    }
}

// ── N=168: Good-Thomas 56×3 ───────────────────────────────────────────────────
//
// n1=56, n2=3. N2⁻¹ mod N1=19; N1⁻¹ mod N2=2. Output: k=(112·k2+57·k1)%168.
#[inline(always)]
pub(crate) fn dft168_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 168],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 3]; 56] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(3 * i1 + 56 * j) % 168]));
    for row in &mut rows {
        dft3_impl(row, inverse);
    }
    for k2 in 0..3usize {
        let mut col: [num_complex::Complex<F>; 56] = core::array::from_fn(|k1| rows[k1][k2]);
        dft56_impl(&mut col, inverse);
        let base = (112 * k2) % 168;
        for k1 in 0..56usize {
            data[(base + 57 * k1) % 168] = col[k1];
        }
    }
}

// ── N=175: Good-Thomas 25×7 ───────────────────────────────────────────────────
//
// n1=25, n2=7. N2⁻¹ mod N1=18; N1⁻¹ mod N2=2. Output: k=(50·k2+126·k1)%175.
#[inline(always)]
pub(crate) fn dft175_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 175],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 7]; 25] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(7 * i1 + 25 * j) % 175]));
    for row in &mut rows {
        dft7_impl(row, inverse);
    }
    for k2 in 0..7usize {
        let mut col: [num_complex::Complex<F>; 25] = core::array::from_fn(|k1| rows[k1][k2]);
        dft25_impl(&mut col, inverse);
        let base = (50 * k2) % 175;
        for k1 in 0..25usize {
            data[(base + 126 * k1) % 175] = col[k1];
        }
    }
}

// ── N=176: Good-Thomas 16×11 ──────────────────────────────────────────────────
//
// n1=16, n2=11. N2⁻¹ mod N1=3; N1⁻¹ mod N2=9. Output: k=(144·k2+33·k1)%176.
#[inline(always)]
pub(crate) fn dft176_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 176],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 11]; 16] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(11 * i1 + 16 * j) % 176]));
    for row in &mut rows {
        dft11_impl(row, inverse);
    }
    for k2 in 0..11usize {
        let mut col: [num_complex::Complex<F>; 16] = core::array::from_fn(|k1| rows[k1][k2]);
        dft16_impl(&mut col, inverse);
        let base = (144 * k2) % 176;
        for k1 in 0..16usize {
            data[(base + 33 * k1) % 176] = col[k1];
        }
    }
}

// ── N=180: Good-Thomas 36×5 ───────────────────────────────────────────────────
//
// n1=36, n2=5. N2⁻¹ mod N1=29; N1⁻¹ mod N2=1. Output: k=(36·k2+145·k1)%180.
#[inline(always)]
pub(crate) fn dft180_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 180],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 5]; 36] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(5 * i1 + 36 * j) % 180]));
    for row in &mut rows {
        dft5_array_impl(row, inverse);
    }
    for k2 in 0..5usize {
        let mut col: [num_complex::Complex<F>; 36] = core::array::from_fn(|k1| rows[k1][k2]);
        dft36_impl(&mut col, inverse);
        let base = (36 * k2) % 180;
        for k1 in 0..36usize {
            data[(base + 145 * k1) % 180] = col[k1];
        }
    }
}

// ── N=192: Good-Thomas 64×3 ───────────────────────────────────────────────────
//
// n1=64, n2=3. N2⁻¹ mod N1=43; N1⁻¹ mod N2=1. Output: k=(64·k2+129·k1)%192.
#[inline(always)]
pub(crate) fn dft192_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 192],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 3]; 64] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(3 * i1 + 64 * j) % 192]));
    for row in &mut rows {
        dft3_impl(row, inverse);
    }
    for k2 in 0..3usize {
        let mut col: [num_complex::Complex<F>; 64] = core::array::from_fn(|k1| rows[k1][k2]);
        dft64_impl(&mut col, inverse);
        let base = (64 * k2) % 192;
        for k1 in 0..64usize {
            data[(base + 129 * k1) % 192] = col[k1];
        }
    }
}

// ── N=196: Good-Thomas 4×49 ───────────────────────────────────────────────────
//
// n1=4, n2=49. N2⁻¹ mod N1=1; N1⁻¹ mod N2=37. Output: k=(148·k2+49·k1)%196.
#[inline(always)]
pub(crate) fn dft196_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 196],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 49]; 4] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(49 * i1 + 4 * j) % 196]));
    for row in &mut rows {
        dft49_impl(row, inverse);
    }
    for k2 in 0..49usize {
        let mut col: [num_complex::Complex<F>; 4] =
            [rows[0][k2], rows[1][k2], rows[2][k2], rows[3][k2]];
        dft4_array_impl(&mut col, inverse);
        let base = (148 * k2) % 196;
        for k1 in 0..4usize {
            data[(base + 49 * k1) % 196] = col[k1];
        }
    }
}

// ── N=198: Good-Thomas 2×99 ───────────────────────────────────────────────────
//
// n1=2, n2=99. N2⁻¹ mod N1=1; N1⁻¹ mod N2=50. Output: k=(100·k2+99·k1)%198.
// Butterfly: sums[j]=data[2j]+data[(2j+99)%198]; diffs[j]=data[2j]-data[(2j+99)%198].
#[inline(always)]
pub(crate) fn dft198_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 198],
    inverse: bool,
) {
    let mut sums: [num_complex::Complex<F>; 99] =
        core::array::from_fn(|j| data[2 * j] + data[(2 * j + 99) % 198]);
    let mut diffs: [num_complex::Complex<F>; 99] =
        core::array::from_fn(|j| data[2 * j] - data[(2 * j + 99) % 198]);
    dft99_impl(&mut sums, inverse);
    dft99_impl(&mut diffs, inverse);
    for k2 in 0..99usize {
        data[(100 * k2) % 198] = sums[k2];
        data[(100 * k2 + 99) % 198] = diffs[k2];
    }
}

// ── N=200: Good-Thomas 8×25 ───────────────────────────────────────────────────
//
// n1=8, n2=25. N2⁻¹ mod N1=1; N1⁻¹ mod N2=22. Output: k=(176·k2+25·k1)%200.
#[inline(always)]
pub(crate) fn dft200_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 200],
    inverse: bool,
) {
    let mut rows: [[num_complex::Complex<F>; 25]; 8] =
        core::array::from_fn(|i1| core::array::from_fn(|j| data[(25 * i1 + 8 * j) % 200]));
    for row in &mut rows {
        dft25_impl(row, inverse);
    }
    for k2 in 0..25usize {
        let mut col: [num_complex::Complex<F>; 8] = core::array::from_fn(|k1| rows[k1][k2]);
        dft8_impl(&mut col, inverse);
        let base = (176 * k2) % 200;
        for k1 in 0..8usize {
            data[(base + 25 * k1) % 200] = col[k1];
        }
    }
}
