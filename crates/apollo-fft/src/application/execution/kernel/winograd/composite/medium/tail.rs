use super::super::super::radix::{dft3_impl, dft7_impl};
use super::super::super::traits::WinogradScalar;
use super::super::dft16_impl;
use super::super::small::dft9_impl;

// ── N=48: Good-Thomas 3×16 ────────────────────────────────────────────────────
//
// n1=3, n2=16. Input: X[i1,i2] = data[(16·i1 + 3·i2) % 48].
//   row 0: data[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45]
//   row 1: data[16,19,22,25,28,31,34,37,40,43,46,1,4,7,10,13]
//   row 2: data[32,35,38,41,44,47,2,5,8,11,14,17,20,23,26,29]
// N2⁻¹ mod N1 = 16⁻¹ mod 3 = 1; N1⁻¹ mod N2 = 3⁻¹ mod 16 = 11.
// Output: k = (16·k1 + 33·k2) % 48.
#[inline(always)]
pub(crate) fn dft48_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 48],
    inverse: bool,
) {
    let mut row0: [num_complex::Complex<F>; 16] = [
        data[0], data[3], data[6], data[9], data[12], data[15], data[18], data[21], data[24],
        data[27], data[30], data[33], data[36], data[39], data[42], data[45],
    ];
    let mut row1: [num_complex::Complex<F>; 16] = [
        data[16], data[19], data[22], data[25], data[28], data[31], data[34], data[37], data[40],
        data[43], data[46], data[1], data[4], data[7], data[10], data[13],
    ];
    let mut row2: [num_complex::Complex<F>; 16] = [
        data[32], data[35], data[38], data[41], data[44], data[47], data[2], data[5], data[8],
        data[11], data[14], data[17], data[20], data[23], data[26], data[29],
    ];
    dft16_impl(&mut row0, inverse);
    dft16_impl(&mut row1, inverse);
    dft16_impl(&mut row2, inverse);
    for k2 in 0..16usize {
        let mut col = [row0[k2], row1[k2], row2[k2]];
        dft3_impl(&mut col, inverse);
        let base = (33 * k2) % 48;
        data[base] = col[0];
        data[(base + 16) % 48] = col[1];
        data[(base + 32) % 48] = col[2];
    }
}

// ── N=63: Good-Thomas 7×9 ─────────────────────────────────────────────────────
//
// n1=7, n2=9. Input: X[i1,i2] = data[(9·i1 + 7·i2) % 63].
//   row 0: data[0,7,14,21,28,35,42,49,56]  row 1: data[9,16,23,30,37,44,51,58,2]
//   row 2: data[18,25,32,39,46,53,60,4,11] row 3: data[27,34,41,48,55,62,6,13,20]
//   row 4: data[36,43,50,57,1,8,15,22,29]  row 5: data[45,52,59,3,10,17,24,31,38]
//   row 6: data[54,61,5,12,19,26,33,40,47]
// N2⁻¹ mod N1 = 9⁻¹ mod 7 = 4; N1⁻¹ mod N2 = 7⁻¹ mod 9 = 4.
// Output: k = (36·k1 + 28·k2) % 63.
//   k2=0: data[0,36,9,45,18,54,27]  k2=1: data[28,1,37,10,46,19,55]
//   k2=2: data[56,29,2,38,11,47,20] k2=3: data[21,57,30,3,39,12,48]
//   k2=4: data[49,22,58,31,4,40,13] k2=5: data[14,50,23,59,32,5,41]
//   k2=6: data[42,15,51,24,60,33,6] k2=7: data[7,43,16,52,25,61,34]
//   k2=8: data[35,8,44,17,53,26,62]
#[inline(always)]
pub(crate) fn dft63_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 63],
    inverse: bool,
) {
    let mut rows = [
        [
            data[0], data[7], data[14], data[21], data[28], data[35], data[42], data[49], data[56],
        ],
        [
            data[9], data[16], data[23], data[30], data[37], data[44], data[51], data[58], data[2],
        ],
        [
            data[18], data[25], data[32], data[39], data[46], data[53], data[60], data[4], data[11],
        ],
        [
            data[27], data[34], data[41], data[48], data[55], data[62], data[6], data[13], data[20],
        ],
        [
            data[36], data[43], data[50], data[57], data[1], data[8], data[15], data[22], data[29],
        ],
        [
            data[45], data[52], data[59], data[3], data[10], data[17], data[24], data[31], data[38],
        ],
        [
            data[54], data[61], data[5], data[12], data[19], data[26], data[33], data[40], data[47],
        ],
    ];
    dft9_impl(&mut rows[0], inverse);
    dft9_impl(&mut rows[1], inverse);
    dft9_impl(&mut rows[2], inverse);
    dft9_impl(&mut rows[3], inverse);
    dft9_impl(&mut rows[4], inverse);
    dft9_impl(&mut rows[5], inverse);
    dft9_impl(&mut rows[6], inverse);
    for k2 in 0..9usize {
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
        let base = (28 * k2) % 63;
        data[base] = col[0];
        data[(base + 36) % 63] = col[1];
        data[(base + 9) % 63] = col[2];
        data[(base + 45) % 63] = col[3];
        data[(base + 18) % 63] = col[4];
        data[(base + 54) % 63] = col[5];
        data[(base + 27) % 63] = col[6];
    }
}
