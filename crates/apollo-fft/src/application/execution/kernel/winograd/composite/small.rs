//! Inline fixed-array DFT codelets for small composite lengths: 6, 9, 10, 12, 14.
//!
//! ## Algorithms
//!
//! | N  | Decomposition              | Twiddles |
//! |----|----------------------------|----------|
//! | 6  | Good-Thomas 2×3 (PFA)      | none     |
//! | 9  | Cooley-Tukey 3×3 (DIT)     | 4        |
//! | 10 | Good-Thomas 2×5 (PFA)      | none     |
//! | 12 | Good-Thomas 4×3 (PFA)      | none     |
//! | 14 | Good-Thomas 2×7 (PFA)      | none     |
//!
//! Good-Thomas PFA (Good 1958, Thomas 1963): for N = n1 × n2 with gcd(n1,n2)=1,
//! the DFT decomposes into sub-transforms with no inter-stage twiddle factors.
//!
//! Input permutation:  X[i1, i2] = x[(n2·i1 + n1·i2) mod N]
//! Output permutation: y[k] where k = n2·(n2⁻¹ mod n1)·k1 + n1·(n1⁻¹ mod n2)·k2  (mod N)

use super::super::radix::{
    dft11_impl, dft3_impl, dft4_array_impl, dft5_array_impl, dft5_impl, dft7_impl,
};
use super::super::traits::WinogradScalar;

/// In-place DFT-6 via Good-Thomas PFA (2×3, gcd=1).
///
/// **Decomposition**: 3 butterfly (DFT-2) + 2 DFT-3. Zero twiddle factors.
///
/// Input map: X[i1,i2] = data[(3·i1 + 2·i2) % 6]
///   row 0: data[0], data[2], data[4]
///   row 1: data[3], data[5], data[1]
///
/// Output map: data[(4·k1 + 3·k2) % 6]
///   k1=0: data[0], data[4], data[2]
///   k1=1: data[3], data[1], data[5]
///
/// **Multiplications**: 8 real (2 DFT-3, each 4 real muls).
/// **Additions**: 18 (6 butterfly + 2×6 DFT-3).
#[inline(always)]
pub(crate) fn dft6_impl<F: WinogradScalar>(data: &mut [num_complex::Complex<F>; 6], inverse: bool) {
    // Step 1: DFT-2 butterfly on each column.
    let s0 = data[0] + data[3];
    let d0 = data[0] - data[3];
    let s1 = data[2] + data[5];
    let d1 = data[2] - data[5];
    let s2 = data[4] + data[1];
    let d2 = data[4] - data[1];
    // Step 2: DFT-3 on sums → data[0], data[4], data[2] (k1=0 output map)
    //         DFT-3 on diffs → data[3], data[1], data[5] (k1=1 output map)
    let sq3o2 = F::cast_f64(0.8660254037844386);
    let half = F::cast_f64(-0.5);
    // Row k1=0:
    let sum_re_0 = s1.re + s2.re;
    let sum_im_0 = s1.im + s2.im;
    let mid0_re = s0.re + sum_re_0 * half;
    let mid0_im = s0.im + sum_im_0 * half;
    let diff_re_0 = s1.re - s2.re;
    let diff_im_0 = s1.im - s2.im;
    let (n0_re, n0_im) = if inverse {
        (-diff_im_0 * sq3o2, diff_re_0 * sq3o2)
    } else {
        (diff_im_0 * sq3o2, -diff_re_0 * sq3o2)
    };
    data[0] = num_complex::Complex::new(s0.re + sum_re_0, s0.im + sum_im_0);
    data[4] = num_complex::Complex::new(mid0_re + n0_re, mid0_im + n0_im);
    data[2] = num_complex::Complex::new(mid0_re - n0_re, mid0_im - n0_im);
    // Row k1=1:
    let sum_re_1 = d1.re + d2.re;
    let sum_im_1 = d1.im + d2.im;
    let mid1_re = d0.re + sum_re_1 * half;
    let mid1_im = d0.im + sum_im_1 * half;
    let diff_re_1 = d1.re - d2.re;
    let diff_im_1 = d1.im - d2.im;
    let (n1_re, n1_im) = if inverse {
        (-diff_im_1 * sq3o2, diff_re_1 * sq3o2)
    } else {
        (diff_im_1 * sq3o2, -diff_re_1 * sq3o2)
    };
    data[3] = num_complex::Complex::new(d0.re + sum_re_1, d0.im + sum_im_1);
    data[1] = num_complex::Complex::new(mid1_re + n1_re, mid1_im + n1_im);
    data[5] = num_complex::Complex::new(mid1_re - n1_re, mid1_im - n1_im);
}

/// In-place DFT-9 via Cooley-Tukey radix-3×3.
///
/// **Decomposition**: 3 column DFT-3 + 4 twiddle muls + 3 row DFT-3.
///
/// Twiddle factors W_9^{k1·j} = exp(−2πi·k1·j/9):
/// - (k1=1, j=1): W_9^1 ≈ 0.76604 − 0.64279i
/// - (k1=1, j=2): W_9^2 ≈ 0.17365 − 0.98481i
/// - (k1=2, j=1): W_9^2
/// - (k1=2, j=2): W_9^4 ≈ −0.93969 − 0.34202i
///
/// **Multiplications**: 6 DFT-3 × 4 real each + 4 complex twiddle muls = 40 real muls.
/// **Additions**: 6 DFT-3 × 6 each + 4 complex adds = 40 real adds.
#[inline(always)]
pub(crate) fn dft9_impl<F: WinogradScalar>(data: &mut [num_complex::Complex<F>; 9], inverse: bool) {
    // Step 1: DFT-3 on 3 stride-3 columns (j=0, 1, 2).
    let mut c0 = [data[0], data[3], data[6]];
    let mut c1 = [data[1], data[4], data[7]];
    let mut c2 = [data[2], data[5], data[8]];
    dft3_impl(&mut c0, inverse);
    dft3_impl(&mut c1, inverse);
    dft3_impl(&mut c2, inverse);
    // Step 2: Twiddle W_9^{k1*j} (forward: exp(-2πi/9)), rows k1=1,2 only.
    let (w1, w2, w4) = if inverse {
        (
            num_complex::Complex::new(
                F::cast_f64(0.766044443118978),
                F::cast_f64(0.642787609686539),
            ),
            num_complex::Complex::new(
                F::cast_f64(0.173648177666930),
                F::cast_f64(0.984807753012208),
            ),
            num_complex::Complex::new(
                F::cast_f64(-0.939692620785908),
                F::cast_f64(0.342020143325669),
            ),
        )
    } else {
        (
            num_complex::Complex::new(
                F::cast_f64(0.766044443118978),
                F::cast_f64(-0.642787609686539),
            ),
            num_complex::Complex::new(
                F::cast_f64(0.173648177666930),
                F::cast_f64(-0.984807753012208),
            ),
            num_complex::Complex::new(
                F::cast_f64(-0.939692620785908),
                F::cast_f64(-0.342020143325669),
            ),
        )
    };
    c1[1] *= w1;
    c2[1] *= w2;
    c1[2] *= w2;
    c2[2] *= w4;
    // Step 3: DFT-3 on 3 rows (k1=0,1,2) → output y[3*k2 + k1].
    let mut r0 = [c0[0], c1[0], c2[0]];
    let mut r1 = [c0[1], c1[1], c2[1]];
    let mut r2 = [c0[2], c1[2], c2[2]];
    dft3_impl(&mut r0, inverse);
    dft3_impl(&mut r1, inverse);
    dft3_impl(&mut r2, inverse);
    data[0] = r0[0];
    data[3] = r0[1];
    data[6] = r0[2];
    data[1] = r1[0];
    data[4] = r1[1];
    data[7] = r1[2];
    data[2] = r2[0];
    data[5] = r2[1];
    data[8] = r2[2];
}

/// In-place DFT-10 via Good-Thomas PFA (2×5, gcd=1).
///
/// **Decomposition**: 5 butterfly (DFT-2) + 2 DFT-5. Zero twiddle factors.
///
/// Input map: X[i1,i2] = data[(5·i1 + 2·i2) % 10]
///   row 0: data[0], data[2], data[4], data[6], data[8]
///   row 1: data[5], data[7], data[9], data[1], data[3]
///
/// Output map: data[(5·k1 + 6·k2) % 10]
///   k1=0: data[0], data[6], data[2], data[8], data[4]
///   k1=1: data[5], data[1], data[7], data[3], data[9]
///
/// **Multiplications**: 16 real (2 DFT-5, each 8 real muls).
/// **Additions**: 30 (10 butterfly + 2×10 DFT-5).
#[inline(always)]
pub(crate) fn dft10_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 10],
    inverse: bool,
) {
    // Step 1: DFT-2 butterfly on each column.
    let s0 = data[0] + data[5];
    let d0 = data[0] - data[5];
    let s1 = data[2] + data[7];
    let d1 = data[2] - data[7];
    let s2 = data[4] + data[9];
    let d2 = data[4] - data[9];
    let s3 = data[6] + data[1];
    let d3 = data[6] - data[1];
    let s4 = data[8] + data[3];
    let d4 = data[8] - data[3];
    // Step 2: DFT-5 on sums → data[0], data[6], data[2], data[8], data[4]
    let mut sums = [s0, s1, s2, s3, s4];
    let mut diffs = [d0, d1, d2, d3, d4];
    dft5_impl(&mut sums, inverse);
    dft5_impl(&mut diffs, inverse);
    data[0] = sums[0];
    data[6] = sums[1];
    data[2] = sums[2];
    data[8] = sums[3];
    data[4] = sums[4];
    data[5] = diffs[0];
    data[1] = diffs[1];
    data[7] = diffs[2];
    data[3] = diffs[3];
    data[9] = diffs[4];
}

/// In-place DFT-12 via Good-Thomas PFA (4×3, gcd=1).
///
/// **Decomposition**: 3 DFT-4 (columns) + 4 DFT-3 (rows). Zero twiddle factors.
///
/// Input map: X[i1,i2] = data[(3·i1 + 4·i2) % 12]
///   col 0: data[0], data[3], data[6], data[9]
///   col 1: data[4], data[7], data[10], data[1]
///   col 2: data[8], data[11], data[2], data[5]
///
/// Output map: data[(9·k1 + 4·k2) % 12]
///   k1=0: data[0], data[4], data[8]
///   k1=1: data[9], data[1], data[5]
///   k1=2: data[6], data[10], data[2]
///   k1=3: data[3], data[7], data[11]
///
/// **Multiplications**: 0 (DFT-4 mul-free) + 4×4 = 16 real (4 DFT-3).
/// **Additions**: 3×16 (DFT-4) + 4×6 (DFT-3) = 72 real.
#[inline(always)]
pub(crate) fn dft12_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 12],
    inverse: bool,
) {
    // Step 1: DFT-4 on each column.
    let mut col0 = [data[0], data[3], data[6], data[9]];
    let mut col1 = [data[4], data[7], data[10], data[1]];
    let mut col2 = [data[8], data[11], data[2], data[5]];
    dft4_array_impl(&mut col0, inverse);
    dft4_array_impl(&mut col1, inverse);
    dft4_array_impl(&mut col2, inverse);
    // Step 2: DFT-3 on each row k1 → output at data[(9*k1 + 4*k2) % 12].
    // Row k1=0: [col0[0], col1[0], col2[0]] → data[0], data[4], data[8]
    // Row k1=1: [col0[1], col1[1], col2[1]] → data[9], data[1], data[5]
    // Row k1=2: [col0[2], col1[2], col2[2]] → data[6], data[10], data[2]
    // Row k1=3: [col0[3], col1[3], col2[3]] → data[3], data[7], data[11]
    let mut r0 = [col0[0], col1[0], col2[0]];
    let mut r1 = [col0[1], col1[1], col2[1]];
    let mut r2 = [col0[2], col1[2], col2[2]];
    let mut r3 = [col0[3], col1[3], col2[3]];
    dft3_impl(&mut r0, inverse);
    dft3_impl(&mut r1, inverse);
    dft3_impl(&mut r2, inverse);
    dft3_impl(&mut r3, inverse);
    data[0] = r0[0];
    data[4] = r0[1];
    data[8] = r0[2];
    data[9] = r1[0];
    data[1] = r1[1];
    data[5] = r1[2];
    data[6] = r2[0];
    data[10] = r2[1];
    data[2] = r2[2];
    data[3] = r3[0];
    data[7] = r3[1];
    data[11] = r3[2];
}

/// In-place DFT-14 via Good-Thomas PFA (2×7, gcd=1).
///
/// **Decomposition**: 7 butterfly (DFT-2) + 2 DFT-7. Zero twiddle factors.
///
/// Input map: X[i1,i2] = data[(7·i1 + 2·i2) % 14]
///   row 0: data[0], data[2], data[4], data[6], data[8], data[10], data[12]
///   row 1: data[7], data[9], data[11], data[13], data[1], data[3], data[5]
///
/// Output map: data[(7·k1 + 8·k2) % 14]
///   k1=0: data[0], data[8], data[2], data[10], data[4], data[12], data[6]
///   k1=1: data[7], data[1], data[9], data[3], data[11], data[5], data[13]
///
/// **Multiplications**: 36 real (2 DFT-7, each 18 real muls).
/// **Additions**: 14 butterfly + 2×26 = 66 real.
#[inline(always)]
pub(crate) fn dft14_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 14],
    inverse: bool,
) {
    // Step 1: DFT-2 butterfly on each column.
    let s0 = data[0] + data[7];
    let d0 = data[0] - data[7];
    let s1 = data[2] + data[9];
    let d1 = data[2] - data[9];
    let s2 = data[4] + data[11];
    let d2 = data[4] - data[11];
    let s3 = data[6] + data[13];
    let d3 = data[6] - data[13];
    let s4 = data[8] + data[1];
    let d4 = data[8] - data[1];
    let s5 = data[10] + data[3];
    let d5 = data[10] - data[3];
    let s6 = data[12] + data[5];
    let d6 = data[12] - data[5];
    // Step 2: DFT-7 on sums → data[0], data[8], data[2], data[10], data[4], data[12], data[6]
    //         DFT-7 on diffs → data[7], data[1], data[9], data[3], data[11], data[5], data[13]
    let mut sums = [s0, s1, s2, s3, s4, s5, s6];
    let mut diffs = [d0, d1, d2, d3, d4, d5, d6];
    dft7_impl(&mut sums, inverse);
    dft7_impl(&mut diffs, inverse);
    data[0] = sums[0];
    data[8] = sums[1];
    data[2] = sums[2];
    data[10] = sums[3];
    data[4] = sums[4];
    data[12] = sums[5];
    data[6] = sums[6];
    data[7] = diffs[0];
    data[1] = diffs[1];
    data[9] = diffs[2];
    data[3] = diffs[3];
    data[11] = diffs[4];
    data[5] = diffs[5];
    data[13] = diffs[6];
}
