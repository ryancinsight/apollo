//! Inline fixed-array DFT codelets for medium composite lengths: 18, 22, 28, 30, 36, 42, 45, 48, 63.
//!
//! ## Algorithms
//!
//! | N  | Decomposition              | Twiddles |
//! |----|----------------------------|----------|
//! | 18 | Good-Thomas 2×9 (PFA)      | none     |
//! | 22 | Good-Thomas 2×11 (PFA)     | none     |
//! | 28 | Good-Thomas 4×7 (PFA)      | none     |
//! | 30 | Good-Thomas 5×6 (PFA)      | none     |
//! | 36 | Good-Thomas 4×9 (PFA)      | none     |
//! | 42 | Good-Thomas 7×6 (PFA)      | none     |
//! | 45 | Good-Thomas 9×5 (PFA)      | none     |
//! | 48 | Good-Thomas 3×16 (PFA)     | none     |
//! | 63 | Good-Thomas 7×9 (PFA)      | none     |
//!
//! All use Good-Thomas PFA (Good 1958, Thomas 1963): for N = n1 × n2 with gcd(n1,n2)=1,
//! the DFT decomposes with no inter-stage twiddle factors.
//!
//! Input permutation:  X[i1, i2] = x[(n2·i1 + n1·i2) mod N]
//! Output permutation: k = n2·(n2⁻¹ mod n1)·k1 + n1·(n1⁻¹ mod n2)·k2  (mod N)

use super::super::radix::{dft11_impl, dft3_impl, dft4_array_impl, dft5_array_impl, dft7_impl};
use super::super::traits::WinogradScalar;
use super::small::{dft6_impl, dft9_impl};

mod tail;
pub(crate) use tail::{dft48_impl, dft63_impl};

// ── N=18: Good-Thomas 2×9 ─────────────────────────────────────────────────────
//
// n1=2, n2=9. Input: X[i1,i2] = data[(9·i1 + 2·i2) % 18].
//   row 0: data[0,2,4,6,8,10,12,14,16]
//   row 1: data[9,11,13,15,17,1,3,5,7]
// N2⁻¹ mod N1 = 9⁻¹ mod 2 = 1; N1⁻¹ mod N2 = 2⁻¹ mod 9 = 5.
// Output: k = (9·k1 + 10·k2) % 18.
//   k1=0: data[0,10,2,12,4,14,6,16,8]; k1=1: data[9,1,11,3,13,5,15,7,17]
#[inline(always)]
pub(crate) fn dft18_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 18],
    inverse: bool,
) {
    let s0 = data[0] + data[9];
    let d0 = data[0] - data[9];
    let s1 = data[2] + data[11];
    let d1 = data[2] - data[11];
    let s2 = data[4] + data[13];
    let d2 = data[4] - data[13];
    let s3 = data[6] + data[15];
    let d3 = data[6] - data[15];
    let s4 = data[8] + data[17];
    let d4 = data[8] - data[17];
    let s5 = data[10] + data[1];
    let d5 = data[10] - data[1];
    let s6 = data[12] + data[3];
    let d6 = data[12] - data[3];
    let s7 = data[14] + data[5];
    let d7 = data[14] - data[5];
    let s8 = data[16] + data[7];
    let d8 = data[16] - data[7];
    let mut sums = [s0, s1, s2, s3, s4, s5, s6, s7, s8];
    let mut diffs = [d0, d1, d2, d3, d4, d5, d6, d7, d8];
    dft9_impl(&mut sums, inverse);
    dft9_impl(&mut diffs, inverse);
    data[0] = sums[0];
    data[10] = sums[1];
    data[2] = sums[2];
    data[12] = sums[3];
    data[4] = sums[4];
    data[14] = sums[5];
    data[6] = sums[6];
    data[16] = sums[7];
    data[8] = sums[8];
    data[9] = diffs[0];
    data[1] = diffs[1];
    data[11] = diffs[2];
    data[3] = diffs[3];
    data[13] = diffs[4];
    data[5] = diffs[5];
    data[15] = diffs[6];
    data[7] = diffs[7];
    data[17] = diffs[8];
}

// ── N=22: Good-Thomas 2×11 ────────────────────────────────────────────────────
//
// n1=2, n2=11. Input: X[i1,i2] = data[(11·i1 + 2·i2) % 22].
//   row 0: data[0,2,4,6,8,10,12,14,16,18,20]
//   row 1: data[11,13,15,17,19,21,1,3,5,7,9]
// N2⁻¹ mod N1=1; N1⁻¹ mod N2=6. Output: k=(11·k1+12·k2)%22.
//   k1=0: data[0,12,2,14,4,16,6,18,8,20,10]; k1=1: data[11,1,13,3,15,5,17,7,19,9,21]
#[inline(always)]
pub(crate) fn dft22_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 22],
    inverse: bool,
) {
    let s0 = data[0] + data[11];
    let d0 = data[0] - data[11];
    let s1 = data[2] + data[13];
    let d1 = data[2] - data[13];
    let s2 = data[4] + data[15];
    let d2 = data[4] - data[15];
    let s3 = data[6] + data[17];
    let d3 = data[6] - data[17];
    let s4 = data[8] + data[19];
    let d4 = data[8] - data[19];
    let s5 = data[10] + data[21];
    let d5 = data[10] - data[21];
    let s6 = data[12] + data[1];
    let d6 = data[12] - data[1];
    let s7 = data[14] + data[3];
    let d7 = data[14] - data[3];
    let s8 = data[16] + data[5];
    let d8 = data[16] - data[5];
    let s9 = data[18] + data[7];
    let d9 = data[18] - data[7];
    let s10 = data[20] + data[9];
    let d10 = data[20] - data[9];
    let mut sums = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10];
    let mut diffs = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10];
    dft11_impl(&mut sums, inverse);
    dft11_impl(&mut diffs, inverse);
    data[0] = sums[0];
    data[12] = sums[1];
    data[2] = sums[2];
    data[14] = sums[3];
    data[4] = sums[4];
    data[16] = sums[5];
    data[6] = sums[6];
    data[18] = sums[7];
    data[8] = sums[8];
    data[20] = sums[9];
    data[10] = sums[10];
    data[11] = diffs[0];
    data[1] = diffs[1];
    data[13] = diffs[2];
    data[3] = diffs[3];
    data[15] = diffs[4];
    data[5] = diffs[5];
    data[17] = diffs[6];
    data[7] = diffs[7];
    data[19] = diffs[8];
    data[9] = diffs[9];
    data[21] = diffs[10];
}

// ── N=28: Good-Thomas 4×7 ─────────────────────────────────────────────────────
//
// n1=4, n2=7. Input: X[i1,i2] = data[(7·i1 + 4·i2) % 28].
// N2⁻¹ mod N1=3; N1⁻¹ mod N2=2. Output: k=(21·k1+8·k2)%28.
//   k1=0: data[0,8,16,24,4,12,20]; k1=1: data[21,1,9,17,25,5,13]
//   k1=2: data[14,22,2,10,18,26,6]; k1=3: data[7,15,23,3,11,19,27]
#[inline(always)]
pub(crate) fn dft28_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 28],
    inverse: bool,
) {
    let mut col0 = [data[0], data[7], data[14], data[21]];
    let mut col1 = [data[4], data[11], data[18], data[25]];
    let mut col2 = [data[8], data[15], data[22], data[1]];
    let mut col3 = [data[12], data[19], data[26], data[5]];
    let mut col4 = [data[16], data[23], data[2], data[9]];
    let mut col5 = [data[20], data[27], data[6], data[13]];
    let mut col6 = [data[24], data[3], data[10], data[17]];
    dft4_array_impl(&mut col0, inverse);
    dft4_array_impl(&mut col1, inverse);
    dft4_array_impl(&mut col2, inverse);
    dft4_array_impl(&mut col3, inverse);
    dft4_array_impl(&mut col4, inverse);
    dft4_array_impl(&mut col5, inverse);
    dft4_array_impl(&mut col6, inverse);
    let mut r0 = [
        col0[0], col1[0], col2[0], col3[0], col4[0], col5[0], col6[0],
    ];
    let mut r1 = [
        col0[1], col1[1], col2[1], col3[1], col4[1], col5[1], col6[1],
    ];
    let mut r2 = [
        col0[2], col1[2], col2[2], col3[2], col4[2], col5[2], col6[2],
    ];
    let mut r3 = [
        col0[3], col1[3], col2[3], col3[3], col4[3], col5[3], col6[3],
    ];
    dft7_impl(&mut r0, inverse);
    dft7_impl(&mut r1, inverse);
    dft7_impl(&mut r2, inverse);
    dft7_impl(&mut r3, inverse);
    data[0] = r0[0];
    data[8] = r0[1];
    data[16] = r0[2];
    data[24] = r0[3];
    data[4] = r0[4];
    data[12] = r0[5];
    data[20] = r0[6];
    data[21] = r1[0];
    data[1] = r1[1];
    data[9] = r1[2];
    data[17] = r1[3];
    data[25] = r1[4];
    data[5] = r1[5];
    data[13] = r1[6];
    data[14] = r2[0];
    data[22] = r2[1];
    data[2] = r2[2];
    data[10] = r2[3];
    data[18] = r2[4];
    data[26] = r2[5];
    data[6] = r2[6];
    data[7] = r3[0];
    data[15] = r3[1];
    data[23] = r3[2];
    data[3] = r3[3];
    data[11] = r3[4];
    data[19] = r3[5];
    data[27] = r3[6];
}

// ── N=30: Good-Thomas 5×6 ─────────────────────────────────────────────────────
//
// n1=5, n2=6. Input: X[i1,i2] = data[(6·i1 + 5·i2) % 30].
// N2⁻¹ mod N1=1; N1⁻¹ mod N2=5. Output: k=(6·k1+25·k2)%30.
//   k1=0: data[0,25,20,15,10,5]; k1=1: data[6,1,26,21,16,11]
//   k1=2: data[12,7,2,27,22,17]; k1=3: data[18,13,8,3,28,23]
//   k1=4: data[24,19,14,9,4,29]
#[inline(always)]
pub(crate) fn dft30_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 30],
    inverse: bool,
) {
    let mut col0 = [data[0], data[6], data[12], data[18], data[24]];
    let mut col1 = [data[5], data[11], data[17], data[23], data[29]];
    let mut col2 = [data[10], data[16], data[22], data[28], data[4]];
    let mut col3 = [data[15], data[21], data[27], data[3], data[9]];
    let mut col4 = [data[20], data[26], data[2], data[8], data[14]];
    let mut col5 = [data[25], data[1], data[7], data[13], data[19]];
    dft5_array_impl(&mut col0, inverse);
    dft5_array_impl(&mut col1, inverse);
    dft5_array_impl(&mut col2, inverse);
    dft5_array_impl(&mut col3, inverse);
    dft5_array_impl(&mut col4, inverse);
    dft5_array_impl(&mut col5, inverse);
    let mut r0 = [col0[0], col1[0], col2[0], col3[0], col4[0], col5[0]];
    let mut r1 = [col0[1], col1[1], col2[1], col3[1], col4[1], col5[1]];
    let mut r2 = [col0[2], col1[2], col2[2], col3[2], col4[2], col5[2]];
    let mut r3 = [col0[3], col1[3], col2[3], col3[3], col4[3], col5[3]];
    let mut r4 = [col0[4], col1[4], col2[4], col3[4], col4[4], col5[4]];
    dft6_impl(&mut r0, inverse);
    dft6_impl(&mut r1, inverse);
    dft6_impl(&mut r2, inverse);
    dft6_impl(&mut r3, inverse);
    dft6_impl(&mut r4, inverse);
    data[0] = r0[0];
    data[25] = r0[1];
    data[20] = r0[2];
    data[15] = r0[3];
    data[10] = r0[4];
    data[5] = r0[5];
    data[6] = r1[0];
    data[1] = r1[1];
    data[26] = r1[2];
    data[21] = r1[3];
    data[16] = r1[4];
    data[11] = r1[5];
    data[12] = r2[0];
    data[7] = r2[1];
    data[2] = r2[2];
    data[27] = r2[3];
    data[22] = r2[4];
    data[17] = r2[5];
    data[18] = r3[0];
    data[13] = r3[1];
    data[8] = r3[2];
    data[3] = r3[3];
    data[28] = r3[4];
    data[23] = r3[5];
    data[24] = r4[0];
    data[19] = r4[1];
    data[14] = r4[2];
    data[9] = r4[3];
    data[4] = r4[4];
    data[29] = r4[5];
}

// ── N=36: Good-Thomas 4×9 ─────────────────────────────────────────────────────
//
// n1=4, n2=9. Input: X[i1,i2] = data[(9·i1 + 4·i2) % 36].
//   row 0: data[0,4,8,12,16,20,24,28,32]   row 1: data[9,13,17,21,25,29,33,1,5]
//   row 2: data[18,22,26,30,34,2,6,10,14]  row 3: data[27,31,35,3,7,11,15,19,23]
// N2⁻¹ mod N1 = 9⁻¹ mod 4 = 1; N1⁻¹ mod N2 = 4⁻¹ mod 9 = 7.
// Output: k = (9·k1 + 28·k2) % 36.
//   k2=0: data[0,9,18,27]   k2=1: data[28,1,10,19]  k2=2: data[20,29,2,11]
//   k2=3: data[12,21,30,3]  k2=4: data[4,13,22,31]  k2=5: data[32,5,14,23]
//   k2=6: data[24,33,6,15]  k2=7: data[16,25,34,7]  k2=8: data[8,17,26,35]
#[inline(always)]
pub(crate) fn dft36_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 36],
    inverse: bool,
) {
    let mut rows = [
        [
            data[0], data[4], data[8], data[12], data[16], data[20], data[24], data[28], data[32],
        ],
        [
            data[9], data[13], data[17], data[21], data[25], data[29], data[33], data[1], data[5],
        ],
        [
            data[18], data[22], data[26], data[30], data[34], data[2], data[6], data[10], data[14],
        ],
        [
            data[27], data[31], data[35], data[3], data[7], data[11], data[15], data[19], data[23],
        ],
    ];
    dft9_impl(&mut rows[0], inverse);
    dft9_impl(&mut rows[1], inverse);
    dft9_impl(&mut rows[2], inverse);
    dft9_impl(&mut rows[3], inverse);
    for k2 in 0..9usize {
        let mut col = [rows[0][k2], rows[1][k2], rows[2][k2], rows[3][k2]];
        dft4_array_impl(&mut col, inverse);
        let base = (28 * k2) % 36;
        data[base] = col[0];
        data[(base + 9) % 36] = col[1];
        data[(base + 18) % 36] = col[2];
        data[(base + 27) % 36] = col[3];
    }
}

// ── N=42: Good-Thomas 7×6 ─────────────────────────────────────────────────────
//
// n1=7, n2=6. Input: X[i1,i2] = data[(6·i1 + 7·i2) % 42].
//   row 0: data[0,7,14,21,28,35]    row 1: data[6,13,20,27,34,41]
//   row 2: data[12,19,26,33,40,5]   row 3: data[18,25,32,39,4,11]
//   row 4: data[24,31,38,3,10,17]   row 5: data[30,37,2,9,16,23]
//   row 6: data[36,1,8,15,22,29]
// N2⁻¹ mod N1 = 6⁻¹ mod 7 = 6; N1⁻¹ mod N2 = 7⁻¹ mod 6 = 1.
// Output: k = (36·k1 + 7·k2) % 42.
//   k2=0: data[0,36,30,24,18,12,6]   k2=1: data[7,1,37,31,25,19,13]
//   k2=2: data[14,8,2,38,32,26,20]   k2=3: data[21,15,9,3,39,33,27]
//   k2=4: data[28,22,16,10,4,40,34]  k2=5: data[35,29,23,17,11,5,41]
#[inline(always)]
pub(crate) fn dft42_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 42],
    inverse: bool,
) {
    let mut rows = [
        [data[0], data[7], data[14], data[21], data[28], data[35]],
        [data[6], data[13], data[20], data[27], data[34], data[41]],
        [data[12], data[19], data[26], data[33], data[40], data[5]],
        [data[18], data[25], data[32], data[39], data[4], data[11]],
        [data[24], data[31], data[38], data[3], data[10], data[17]],
        [data[30], data[37], data[2], data[9], data[16], data[23]],
        [data[36], data[1], data[8], data[15], data[22], data[29]],
    ];
    dft6_impl(&mut rows[0], inverse);
    dft6_impl(&mut rows[1], inverse);
    dft6_impl(&mut rows[2], inverse);
    dft6_impl(&mut rows[3], inverse);
    dft6_impl(&mut rows[4], inverse);
    dft6_impl(&mut rows[5], inverse);
    dft6_impl(&mut rows[6], inverse);
    for k2 in 0..6usize {
        let mut col = [
            rows[0][k2],
            rows[1][k2],
            rows[2][k2],
            rows[3][k2],
            rows[4][k2],
            rows[5][k2],
            rows[6][k2],
        ];
        dft7_impl(&mut col, inverse);
        let base = (7 * k2) % 42;
        data[base] = col[0];
        data[(base + 36) % 42] = col[1];
        data[(base + 30) % 42] = col[2];
        data[(base + 24) % 42] = col[3];
        data[(base + 18) % 42] = col[4];
        data[(base + 12) % 42] = col[5];
        data[(base + 6) % 42] = col[6];
    }
}

// ── N=45: Good-Thomas 9×5 ─────────────────────────────────────────────────────
//
// n1=9, n2=5. Input: X[i1,i2] = data[(5·i1 + 9·i2) % 45].
//   row 0: data[0,9,18,27,36]   row 1: data[5,14,23,32,41]
//   row 2: data[10,19,28,37,1]  row 3: data[15,24,33,42,6]
//   row 4: data[20,29,38,2,11]  row 5: data[25,34,43,7,16]
//   row 6: data[30,39,3,12,21]  row 7: data[35,44,8,17,26]
//   row 8: data[40,4,13,22,31]
// N2⁻¹ mod N1 = 5⁻¹ mod 9 = 2; N1⁻¹ mod N2 = 9⁻¹ mod 5 = 4.
// Output: k = (10·k1 + 36·k2) % 45.
//   k2=0: data[0,10,20,30,40,5,15,25,35]  k2=1: data[36,1,11,21,31,41,6,16,26]
//   k2=2: data[27,37,2,12,22,32,42,7,17]  k2=3: data[18,28,38,3,13,23,33,43,8]
//   k2=4: data[9,19,29,39,4,14,24,34,44]
#[inline(always)]
pub(crate) fn dft45_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 45],
    inverse: bool,
) {
    let mut rows = [
        [data[0], data[9], data[18], data[27], data[36]],
        [data[5], data[14], data[23], data[32], data[41]],
        [data[10], data[19], data[28], data[37], data[1]],
        [data[15], data[24], data[33], data[42], data[6]],
        [data[20], data[29], data[38], data[2], data[11]],
        [data[25], data[34], data[43], data[7], data[16]],
        [data[30], data[39], data[3], data[12], data[21]],
        [data[35], data[44], data[8], data[17], data[26]],
        [data[40], data[4], data[13], data[22], data[31]],
    ];
    dft5_array_impl(&mut rows[0], inverse);
    dft5_array_impl(&mut rows[1], inverse);
    dft5_array_impl(&mut rows[2], inverse);
    dft5_array_impl(&mut rows[3], inverse);
    dft5_array_impl(&mut rows[4], inverse);
    dft5_array_impl(&mut rows[5], inverse);
    dft5_array_impl(&mut rows[6], inverse);
    dft5_array_impl(&mut rows[7], inverse);
    dft5_array_impl(&mut rows[8], inverse);
    for k2 in 0..5usize {
        let mut col = [
            rows[0][k2],
            rows[1][k2],
            rows[2][k2],
            rows[3][k2],
            rows[4][k2],
            rows[5][k2],
            rows[6][k2],
            rows[7][k2],
            rows[8][k2],
        ];
        dft9_impl(&mut col, inverse);
        let base = (36 * k2) % 45;
        data[base] = col[0];
        data[(base + 10) % 45] = col[1];
        data[(base + 20) % 45] = col[2];
        data[(base + 30) % 45] = col[3];
        data[(base + 40) % 45] = col[4];
        data[(base + 5) % 45] = col[5];
        data[(base + 15) % 45] = col[6];
        data[(base + 25) % 45] = col[7];
        data[(base + 35) % 45] = col[8];
    }
}
