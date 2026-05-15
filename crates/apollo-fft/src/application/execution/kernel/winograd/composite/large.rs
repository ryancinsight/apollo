//! Inline fixed-array DFT codelets for larger composite lengths: 33, 35, 40, 49, 50, 56.
//!
//! | N  | Decomposition          | Twiddles |
//! |----|------------------------|----------|
//! | 33 | Good-Thomas 3×11 (PFA) | none     |
//! | 35 | Good-Thomas 5×7 (PFA)  | none     |
//! | 40 | Good-Thomas 8×5 (PFA)  | none     |
//! | 49 | Cooley-Tukey 7×7 (DIT) | 36       |
//! | 50 | Good-Thomas 2×25 (PFA) | none     |
//! | 56 | Good-Thomas 8×7 (PFA)  | none     |

use super::super::radix::{dft11_impl, dft3_impl, dft5_array_impl, dft7_impl, dft8_impl};
use super::super::traits::WinogradScalar;

// ── N=33: Good-Thomas 3×11 ────────────────────────────────────────────────────
//
// n1=3, n2=11. Input: X[i1,i2] = data[(11·i1 + 3·i2) % 33].
//   row 0: data[0,3,6,9,12,15,18,21,24,27,30]
//   row 1: data[11,14,17,20,23,26,29,32,2,5,8]
//   row 2: data[22,25,28,31,1,4,7,10,13,16,19]
// N2⁻¹ mod N1=2; N1⁻¹ mod N2=4. Output: k=(22·k1+12·k2)%33.
#[inline(always)]
pub(crate) fn dft33_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 33],
    inverse: bool,
) {
    let mut rows = [
        [
            data[0], data[3], data[6], data[9], data[12], data[15], data[18], data[21], data[24],
            data[27], data[30],
        ],
        [
            data[11], data[14], data[17], data[20], data[23], data[26], data[29], data[32],
            data[2], data[5], data[8],
        ],
        [
            data[22], data[25], data[28], data[31], data[1], data[4], data[7], data[10], data[13],
            data[16], data[19],
        ],
    ];
    dft11_impl(&mut rows[0], inverse);
    dft11_impl(&mut rows[1], inverse);
    dft11_impl(&mut rows[2], inverse);
    for k2 in 0..11usize {
        let mut col = [rows[0][k2], rows[1][k2], rows[2][k2]];
        dft3_impl(&mut col, inverse);
        let base = (12 * k2) % 33;
        data[base] = col[0];
        data[(base + 22) % 33] = col[1];
        data[(base + 44) % 33] = col[2];
    }
}

// ── N=35: Good-Thomas 5×7 ─────────────────────────────────────────────────────
//
// n1=5, n2=7. Input: X[i1,i2] = data[(7·i1 + 5·i2) % 35].
//   row 0: data[0,5,10,15,20,25,30]    row 1: data[7,12,17,22,27,32,2]
//   row 2: data[14,19,24,29,34,4,9]    row 3: data[21,26,31,1,6,11,16]
//   row 4: data[28,33,3,8,13,18,23]
// N2⁻¹ mod N1=3; N1⁻¹ mod N2=3. Output: k=(21·k1+15·k2)%35.
#[inline(always)]
pub(crate) fn dft35_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 35],
    inverse: bool,
) {
    let mut rows = [
        [
            data[0], data[5], data[10], data[15], data[20], data[25], data[30],
        ],
        [
            data[7], data[12], data[17], data[22], data[27], data[32], data[2],
        ],
        [
            data[14], data[19], data[24], data[29], data[34], data[4], data[9],
        ],
        [
            data[21], data[26], data[31], data[1], data[6], data[11], data[16],
        ],
        [
            data[28], data[33], data[3], data[8], data[13], data[18], data[23],
        ],
    ];
    dft7_impl(&mut rows[0], inverse);
    dft7_impl(&mut rows[1], inverse);
    dft7_impl(&mut rows[2], inverse);
    dft7_impl(&mut rows[3], inverse);
    dft7_impl(&mut rows[4], inverse);
    for k2 in 0..7usize {
        let mut col = [
            rows[0][k2],
            rows[1][k2],
            rows[2][k2],
            rows[3][k2],
            rows[4][k2],
        ];
        dft5_array_impl(&mut col, inverse);
        let base = (15 * k2) % 35;
        data[base] = col[0];
        data[(base + 21) % 35] = col[1];
        data[(base + 42) % 35] = col[2];
        data[(base + 63) % 35] = col[3];
        data[(base + 84) % 35] = col[4];
    }
}

// ── N=40: Good-Thomas 8×5 ─────────────────────────────────────────────────────
//
// n1=8, n2=5. Input: X[i1,i2] = data[(5·i1 + 8·i2) % 40].
//   row 0: data[0,8,16,24,32]     row 1: data[5,13,21,29,37]
//   row 2: data[10,18,26,34,2]    row 3: data[15,23,31,39,7]
//   row 4: data[20,28,36,4,12]    row 5: data[25,33,1,9,17]
//   row 6: data[30,38,6,14,22]    row 7: data[35,3,11,19,27]
// N2⁻¹ mod N1=5; N1⁻¹ mod N2=2. Output: k=(25·k1+16·k2)%40.
#[inline(always)]
pub(crate) fn dft40_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 40],
    inverse: bool,
) {
    let mut rows = [
        [data[0], data[8], data[16], data[24], data[32]],
        [data[5], data[13], data[21], data[29], data[37]],
        [data[10], data[18], data[26], data[34], data[2]],
        [data[15], data[23], data[31], data[39], data[7]],
        [data[20], data[28], data[36], data[4], data[12]],
        [data[25], data[33], data[1], data[9], data[17]],
        [data[30], data[38], data[6], data[14], data[22]],
        [data[35], data[3], data[11], data[19], data[27]],
    ];
    dft5_array_impl(&mut rows[0], inverse);
    dft5_array_impl(&mut rows[1], inverse);
    dft5_array_impl(&mut rows[2], inverse);
    dft5_array_impl(&mut rows[3], inverse);
    dft5_array_impl(&mut rows[4], inverse);
    dft5_array_impl(&mut rows[5], inverse);
    dft5_array_impl(&mut rows[6], inverse);
    dft5_array_impl(&mut rows[7], inverse);
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
        ];
        dft8_impl(&mut col, inverse);
        let base = (16 * k2) % 40;
        data[base] = col[0];
        data[(base + 25) % 40] = col[1];
        data[(base + 50) % 40] = col[2];
        data[(base + 75) % 40] = col[3];
        data[(base + 100) % 40] = col[4];
        data[(base + 125) % 40] = col[5];
        data[(base + 150) % 40] = col[6];
        data[(base + 175) % 40] = col[7];
    }
}

// ── N=50: Good-Thomas 2×25 ────────────────────────────────────────────────────
//
// n1=2, n2=25. Input: X[i1,i2] = data[(25·i1 + 2·i2) % 50].
//   row 0: data[0,2,4,...,48]
//   row 1: data[25,27,...,49,1,3,...,23]
// N2⁻¹ mod N1=1; N1⁻¹ mod N2=13. Output: k=(25·k1+26·k2)%50.
#[inline(always)]
pub(crate) fn dft50_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 50],
    inverse: bool,
) {
    let s0 = data[0] + data[25];
    let d0 = data[0] - data[25];
    let s1 = data[2] + data[27];
    let d1 = data[2] - data[27];
    let s2 = data[4] + data[29];
    let d2 = data[4] - data[29];
    let s3 = data[6] + data[31];
    let d3 = data[6] - data[31];
    let s4 = data[8] + data[33];
    let d4 = data[8] - data[33];
    let s5 = data[10] + data[35];
    let d5 = data[10] - data[35];
    let s6 = data[12] + data[37];
    let d6 = data[12] - data[37];
    let s7 = data[14] + data[39];
    let d7 = data[14] - data[39];
    let s8 = data[16] + data[41];
    let d8 = data[16] - data[41];
    let s9 = data[18] + data[43];
    let d9 = data[18] - data[43];
    let s10 = data[20] + data[45];
    let d10 = data[20] - data[45];
    let s11 = data[22] + data[47];
    let d11 = data[22] - data[47];
    let s12 = data[24] + data[49];
    let d12 = data[24] - data[49];
    let s13 = data[26] + data[1];
    let d13 = data[26] - data[1];
    let s14 = data[28] + data[3];
    let d14 = data[28] - data[3];
    let s15 = data[30] + data[5];
    let d15 = data[30] - data[5];
    let s16 = data[32] + data[7];
    let d16 = data[32] - data[7];
    let s17 = data[34] + data[9];
    let d17 = data[34] - data[9];
    let s18 = data[36] + data[11];
    let d18 = data[36] - data[11];
    let s19 = data[38] + data[13];
    let d19 = data[38] - data[13];
    let s20 = data[40] + data[15];
    let d20 = data[40] - data[15];
    let s21 = data[42] + data[17];
    let d21 = data[42] - data[17];
    let s22 = data[44] + data[19];
    let d22 = data[44] - data[19];
    let s23 = data[46] + data[21];
    let d23 = data[46] - data[21];
    let s24 = data[48] + data[23];
    let d24 = data[48] - data[23];
    let mut sums = [
        s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19,
        s20, s21, s22, s23, s24,
    ];
    let mut diffs = [
        d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19,
        d20, d21, d22, d23, d24,
    ];
    super::dft25_impl(&mut sums, inverse);
    super::dft25_impl(&mut diffs, inverse);
    data[0] = sums[0];
    data[26] = sums[1];
    data[2] = sums[2];
    data[28] = sums[3];
    data[4] = sums[4];
    data[30] = sums[5];
    data[6] = sums[6];
    data[32] = sums[7];
    data[8] = sums[8];
    data[34] = sums[9];
    data[10] = sums[10];
    data[36] = sums[11];
    data[12] = sums[12];
    data[38] = sums[13];
    data[14] = sums[14];
    data[40] = sums[15];
    data[16] = sums[16];
    data[42] = sums[17];
    data[18] = sums[18];
    data[44] = sums[19];
    data[20] = sums[20];
    data[46] = sums[21];
    data[22] = sums[22];
    data[48] = sums[23];
    data[24] = sums[24];
    data[25] = diffs[0];
    data[1] = diffs[1];
    data[27] = diffs[2];
    data[3] = diffs[3];
    data[29] = diffs[4];
    data[5] = diffs[5];
    data[31] = diffs[6];
    data[7] = diffs[7];
    data[33] = diffs[8];
    data[9] = diffs[9];
    data[35] = diffs[10];
    data[11] = diffs[11];
    data[37] = diffs[12];
    data[13] = diffs[13];
    data[39] = diffs[14];
    data[15] = diffs[15];
    data[41] = diffs[16];
    data[17] = diffs[17];
    data[43] = diffs[18];
    data[19] = diffs[19];
    data[45] = diffs[20];
    data[21] = diffs[21];
    data[47] = diffs[22];
    data[23] = diffs[23];
    data[49] = diffs[24];
}

// ── N=56: Good-Thomas 8×7 ─────────────────────────────────────────────────────
//
// n1=8, n2=7. Input: X[i1,i2] = data[(7·i1 + 8·i2) % 56].
//   row 0: data[0,8,16,24,32,40,48]   row 1: data[7,15,23,31,39,47,55]
//   row 2: data[14,22,30,38,46,54,6]  row 3: data[21,29,37,45,53,5,13]
//   row 4: data[28,36,44,52,4,12,20]  row 5: data[35,43,51,3,11,19,27]
//   row 6: data[42,50,2,10,18,26,34]  row 7: data[49,1,9,17,25,33,41]
// N2⁻¹ mod N1=7; N1⁻¹ mod N2=1. Output: k=(49·k1+8·k2)%56.
#[inline(always)]
pub(crate) fn dft56_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 56],
    inverse: bool,
) {
    let mut rows = [
        [
            data[0], data[8], data[16], data[24], data[32], data[40], data[48],
        ],
        [
            data[7], data[15], data[23], data[31], data[39], data[47], data[55],
        ],
        [
            data[14], data[22], data[30], data[38], data[46], data[54], data[6],
        ],
        [
            data[21], data[29], data[37], data[45], data[53], data[5], data[13],
        ],
        [
            data[28], data[36], data[44], data[52], data[4], data[12], data[20],
        ],
        [
            data[35], data[43], data[51], data[3], data[11], data[19], data[27],
        ],
        [
            data[42], data[50], data[2], data[10], data[18], data[26], data[34],
        ],
        [
            data[49], data[1], data[9], data[17], data[25], data[33], data[41],
        ],
    ];
    dft7_impl(&mut rows[0], inverse);
    dft7_impl(&mut rows[1], inverse);
    dft7_impl(&mut rows[2], inverse);
    dft7_impl(&mut rows[3], inverse);
    dft7_impl(&mut rows[4], inverse);
    dft7_impl(&mut rows[5], inverse);
    dft7_impl(&mut rows[6], inverse);
    dft7_impl(&mut rows[7], inverse);
    for k2 in 0..7usize {
        let mut col = [
            rows[0][k2],
            rows[1][k2],
            rows[2][k2],
            rows[3][k2],
            rows[4][k2],
            rows[5][k2],
            rows[6][k2],
            rows[7][k2],
        ];
        dft8_impl(&mut col, inverse);
        let base = 8 * k2; // max = 48, no wrap needed
        data[base] = col[0];
        data[(base + 49) % 56] = col[1];
        data[(base + 98) % 56] = col[2];
        data[(base + 147) % 56] = col[3];
        data[(base + 196) % 56] = col[4];
        data[(base + 245) % 56] = col[5];
        data[(base + 294) % 56] = col[6];
        data[(base + 343) % 56] = col[7];
    }
}

// ── N=49: Cooley-Tukey 7×7 DIT ───────────────────────────────────────────────
//
// Step 1: 7 column DFT-7 (col[j][k1] = data[k1*7+j]).
// Step 2: Multiply col[j][k1] by W_49^{k1*j} for k1,j ∈ 1..6.
// Step 3: 7 row DFT-7 → output data[k1*7+k2].
//
// W_49^k = exp(-2πi·k/49) for k=0..36 (forward). Max exponent: 6×6=36.
use num_complex::Complex64;
const TWIDDLE49_FWD: [Complex64; 37] = [
    Complex64::new(1.00000000000000000e0, -0.00000000000000000e0),
    Complex64::new(9.91790013823246164e-1, -1.27877161684505997e-1),
    Complex64::new(9.67294863039029451e-1, -2.53654583909507347e-1),
    Complex64::new(9.26916757346021747e-1, -3.75267004879374089e-1),
    Complex64::new(8.71318704123389409e-1, -4.90717552003937851e-1),
    Complex64::new(8.01413621867956616e-1, -5.98110530491215986e-1),
    Complex64::new(7.18349350097727712e-1, -6.95682550603486383e-1),
    Complex64::new(6.23489801858733594e-1, -7.81831482468029804e-1),
    Complex64::new(5.18392568310525159e-1, -8.55142763005346085e-1),
    Complex64::new(4.04783343122393779e-1, -9.14412623015812498e-1),
    Complex64::new(2.84527586631032448e-1, -9.58667853036660578e-1),
    Complex64::new(1.59599895033379541e-1, -9.87181783414450065e-1),
    Complex64::new(3.20515775716553319e-2, -9.99486216200687894e-1),
    Complex64::new(-9.60230259076817610e-2, -9.95379112949198230e-1),
    Complex64::new(-2.22520933956314337e-1, -9.74927912181823619e-1),
    Complex64::new(-3.45365054421307494e-1, -9.38468422049760442e-1),
    Complex64::new(-4.62538290240835093e-1, -8.86599306373000107e-1),
    Complex64::new(-5.72116660122169773e-1, -8.20172254596955752e-1),
    Complex64::new(-6.72300890261316786e-1, -7.40277997075315497e-1),
    Complex64::new(-7.61445958369134202e-1, -6.48228395307788752e-1),
    Complex64::new(-8.38088104891840602e-1, -5.45534901210548706e-1),
    Complex64::new(-9.00968867902419035e-1, -4.33883739117558231e-1),
    Complex64::new(-9.49055747010668527e-1, -3.15108218023621267e-1),
    Complex64::new(-9.81559156991065329e-1, -1.91158628701372540e-1),
    Complex64::new(-9.97945392750336335e-1, -6.40702199807132305e-2),
    Complex64::new(-9.97945392750336335e-1, 6.40702199807129946e-2),
    Complex64::new(-9.81559156991065329e-1, 1.91158628701372291e-1),
    Complex64::new(-9.49055747010668638e-1, 3.15108218023620601e-1),
    Complex64::new(-9.00968867902419146e-1, 4.33883739117558009e-1),
    Complex64::new(-8.38088104891840713e-1, 5.45534901210548484e-1),
    Complex64::new(-7.61445958369134646e-1, 6.48228395307788197e-1),
    Complex64::new(-6.72300890261317008e-1, 7.40277997075315275e-1),
    Complex64::new(-5.72116660122169995e-1, 8.20172254596955641e-1),
    Complex64::new(-4.62538290240835315e-1, 8.86599306372999996e-1),
    Complex64::new(-3.45365054421307327e-1, 9.38468422049760553e-1),
    Complex64::new(-2.22520933956314587e-1, 9.74927912181823619e-1),
    Complex64::new(-9.60230259076815668e-2, 9.95379112949198230e-1),
];

#[inline(always)]
fn tw49<F: WinogradScalar>(exp: usize, inverse: bool) -> num_complex::Complex<F> {
    let w = TWIDDLE49_FWD[exp];
    let (re, im) = if inverse { (w.re, -w.im) } else { (w.re, w.im) };
    num_complex::Complex::new(F::cast_f64(re), F::cast_f64(im))
}

/// In-place DFT-49 via Cooley-Tukey radix-7×7 DIT.
///
/// **Twiddles**: W_49^{k1·j} for k1,j ∈ 1..6 (36 muls); table TWIDDLE49_FWD.
#[inline(always)]
pub(crate) fn dft49_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 49],
    inverse: bool,
) {
    let mut cols: [[num_complex::Complex<F>; 7]; 7] =
        core::array::from_fn(|j| core::array::from_fn(|k1| data[k1 * 7 + j]));
    for col in &mut cols {
        dft7_impl(col, inverse);
    }
    for k1 in 1..7usize {
        for j in 1..7usize {
            let tw = tw49::<F>(k1 * j, inverse);
            let v = cols[j][k1];
            cols[j][k1] =
                num_complex::Complex::new(v.re * tw.re - v.im * tw.im, v.re * tw.im + v.im * tw.re);
        }
    }
    for k1 in 0..7usize {
        let mut row: [num_complex::Complex<F>; 7] = core::array::from_fn(|j| cols[j][k1]);
        dft7_impl(&mut row, inverse);
        for k2 in 0..7usize {
            data[k2 * 7 + k1] = row[k2];
        }
    }
}
