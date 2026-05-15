use super::scalar::Dft23Scalar;

/// In-place prime DFT-23.
///
/// Pair symmetry reduces the 23x23 DFT matrix to eleven conjugate input pairs.
/// For each `m=1..11`, define `p_m=x[m]+x[23-m]` and
/// `i_m=i*(x[m]-x[23-m])`. Output rows `k` and `23-k` share the cosine
/// projection and differ only by the sine projection sign.
#[inline(never)]
pub(crate) fn dft23_impl<F: Dft23Scalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>],
) {
    dft23_body::<F, INVERSE>(data);
}

#[inline(always)]
pub(crate) fn dft23_inline_impl<F: Dft23Scalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>],
) {
    dft23_body::<F, INVERSE>(data);
}

#[rustfmt::skip]
#[inline(always)]
fn dft23_body<F: Dft23Scalar, const INVERSE: bool>(data: &mut [num_complex::Complex<F>]) {
    debug_assert!(data.len() >= 23);
    let sign = if INVERSE { F::cast_f64(1.0) } else { F::cast_f64(-1.0) };
    let x0 = data[0];
    let x1 = data[1];
    let x2 = data[2];
    let x3 = data[3];
    let x4 = data[4];
    let x5 = data[5];
    let x6 = data[6];
    let x7 = data[7];
    let x8 = data[8];
    let x9 = data[9];
    let x10 = data[10];
    let x11 = data[11];
    let x12 = data[12];
    let x13 = data[13];
    let x14 = data[14];
    let x15 = data[15];
    let x16 = data[16];
    let x17 = data[17];
    let x18 = data[18];
    let x19 = data[19];
    let x20 = data[20];
    let x21 = data[21];
    let x22 = data[22];
    let p1_re = x1.re + x22.re; let p1_im = x1.im + x22.im; let d1_re = x1.re - x22.re; let d1_im = x1.im - x22.im; let i1_re = -d1_im; let i1_im = d1_re;
    let p2_re = x2.re + x21.re; let p2_im = x2.im + x21.im; let d2_re = x2.re - x21.re; let d2_im = x2.im - x21.im; let i2_re = -d2_im; let i2_im = d2_re;
    let p3_re = x3.re + x20.re; let p3_im = x3.im + x20.im; let d3_re = x3.re - x20.re; let d3_im = x3.im - x20.im; let i3_re = -d3_im; let i3_im = d3_re;
    let p4_re = x4.re + x19.re; let p4_im = x4.im + x19.im; let d4_re = x4.re - x19.re; let d4_im = x4.im - x19.im; let i4_re = -d4_im; let i4_im = d4_re;
    let p5_re = x5.re + x18.re; let p5_im = x5.im + x18.im; let d5_re = x5.re - x18.re; let d5_im = x5.im - x18.im; let i5_re = -d5_im; let i5_im = d5_re;
    let p6_re = x6.re + x17.re; let p6_im = x6.im + x17.im; let d6_re = x6.re - x17.re; let d6_im = x6.im - x17.im; let i6_re = -d6_im; let i6_im = d6_re;
    let p7_re = x7.re + x16.re; let p7_im = x7.im + x16.im; let d7_re = x7.re - x16.re; let d7_im = x7.im - x16.im; let i7_re = -d7_im; let i7_im = d7_re;
    let p8_re = x8.re + x15.re; let p8_im = x8.im + x15.im; let d8_re = x8.re - x15.re; let d8_im = x8.im - x15.im; let i8_re = -d8_im; let i8_im = d8_re;
    let p9_re = x9.re + x14.re; let p9_im = x9.im + x14.im; let d9_re = x9.re - x14.re; let d9_im = x9.im - x14.im; let i9_re = -d9_im; let i9_im = d9_re;
    let p10_re = x10.re + x13.re; let p10_im = x10.im + x13.im; let d10_re = x10.re - x13.re; let d10_im = x10.im - x13.im; let i10_re = -d10_im; let i10_im = d10_re;
    let p11_re = x11.re + x12.re; let p11_im = x11.im + x12.im; let d11_re = x11.re - x12.re; let d11_im = x11.im - x12.im; let i11_re = -d11_im; let i11_im = d11_re;
    data[0] = num_complex::Complex::new(x0.re + p1_re + p2_re + p3_re + p4_re + p5_re + p6_re + p7_re + p8_re + p9_re + p10_re + p11_re, x0.im + p1_im + p2_im + p3_im + p4_im + p5_im + p6_im + p7_im + p8_im + p9_im + p10_im + p11_im);
    let r1_re = x0.re + p1_re * F::C00_00 + p2_re * F::C00_01 + p3_re * F::C00_02 + p4_re * F::C00_03 + p5_re * F::C00_04 + p6_re * F::C00_05 + p7_re * F::C00_06 + p8_re * F::C00_07 + p9_re * F::C00_08 + p10_re * F::C00_09 + p11_re * F::C00_10; let r1_im = x0.im + p1_im * F::C00_00 + p2_im * F::C00_01 + p3_im * F::C00_02 + p4_im * F::C00_03 + p5_im * F::C00_04 + p6_im * F::C00_05 + p7_im * F::C00_06 + p8_im * F::C00_07 + p9_im * F::C00_08 + p10_im * F::C00_09 + p11_im * F::C00_10; let q1_re = sign * (i1_re * F::S00_00 + i2_re * F::S00_01 + i3_re * F::S00_02 + i4_re * F::S00_03 + i5_re * F::S00_04 + i6_re * F::S00_05 + i7_re * F::S00_06 + i8_re * F::S00_07 + i9_re * F::S00_08 + i10_re * F::S00_09 + i11_re * F::S00_10); let q1_im = sign * (i1_im * F::S00_00 + i2_im * F::S00_01 + i3_im * F::S00_02 + i4_im * F::S00_03 + i5_im * F::S00_04 + i6_im * F::S00_05 + i7_im * F::S00_06 + i8_im * F::S00_07 + i9_im * F::S00_08 + i10_im * F::S00_09 + i11_im * F::S00_10);
    data[1] = num_complex::Complex::new(r1_re + q1_re, r1_im + q1_im); data[22] = num_complex::Complex::new(r1_re - q1_re, r1_im - q1_im);
    let r2_re = x0.re + p1_re * F::C01_00 + p2_re * F::C01_01 + p3_re * F::C01_02 + p4_re * F::C01_03 + p5_re * F::C01_04 + p6_re * F::C01_05 + p7_re * F::C01_06 + p8_re * F::C01_07 + p9_re * F::C01_08 + p10_re * F::C01_09 + p11_re * F::C01_10; let r2_im = x0.im + p1_im * F::C01_00 + p2_im * F::C01_01 + p3_im * F::C01_02 + p4_im * F::C01_03 + p5_im * F::C01_04 + p6_im * F::C01_05 + p7_im * F::C01_06 + p8_im * F::C01_07 + p9_im * F::C01_08 + p10_im * F::C01_09 + p11_im * F::C01_10; let q2_re = sign * (i1_re * F::S01_00 + i2_re * F::S01_01 + i3_re * F::S01_02 + i4_re * F::S01_03 + i5_re * F::S01_04 + i6_re * F::S01_05 + i7_re * F::S01_06 + i8_re * F::S01_07 + i9_re * F::S01_08 + i10_re * F::S01_09 + i11_re * F::S01_10); let q2_im = sign * (i1_im * F::S01_00 + i2_im * F::S01_01 + i3_im * F::S01_02 + i4_im * F::S01_03 + i5_im * F::S01_04 + i6_im * F::S01_05 + i7_im * F::S01_06 + i8_im * F::S01_07 + i9_im * F::S01_08 + i10_im * F::S01_09 + i11_im * F::S01_10);
    data[2] = num_complex::Complex::new(r2_re + q2_re, r2_im + q2_im); data[21] = num_complex::Complex::new(r2_re - q2_re, r2_im - q2_im);
    let r3_re = x0.re + p1_re * F::C02_00 + p2_re * F::C02_01 + p3_re * F::C02_02 + p4_re * F::C02_03 + p5_re * F::C02_04 + p6_re * F::C02_05 + p7_re * F::C02_06 + p8_re * F::C02_07 + p9_re * F::C02_08 + p10_re * F::C02_09 + p11_re * F::C02_10; let r3_im = x0.im + p1_im * F::C02_00 + p2_im * F::C02_01 + p3_im * F::C02_02 + p4_im * F::C02_03 + p5_im * F::C02_04 + p6_im * F::C02_05 + p7_im * F::C02_06 + p8_im * F::C02_07 + p9_im * F::C02_08 + p10_im * F::C02_09 + p11_im * F::C02_10; let q3_re = sign * (i1_re * F::S02_00 + i2_re * F::S02_01 + i3_re * F::S02_02 + i4_re * F::S02_03 + i5_re * F::S02_04 + i6_re * F::S02_05 + i7_re * F::S02_06 + i8_re * F::S02_07 + i9_re * F::S02_08 + i10_re * F::S02_09 + i11_re * F::S02_10); let q3_im = sign * (i1_im * F::S02_00 + i2_im * F::S02_01 + i3_im * F::S02_02 + i4_im * F::S02_03 + i5_im * F::S02_04 + i6_im * F::S02_05 + i7_im * F::S02_06 + i8_im * F::S02_07 + i9_im * F::S02_08 + i10_im * F::S02_09 + i11_im * F::S02_10);
    data[3] = num_complex::Complex::new(r3_re + q3_re, r3_im + q3_im); data[20] = num_complex::Complex::new(r3_re - q3_re, r3_im - q3_im);
    let r4_re = x0.re + p1_re * F::C03_00 + p2_re * F::C03_01 + p3_re * F::C03_02 + p4_re * F::C03_03 + p5_re * F::C03_04 + p6_re * F::C03_05 + p7_re * F::C03_06 + p8_re * F::C03_07 + p9_re * F::C03_08 + p10_re * F::C03_09 + p11_re * F::C03_10; let r4_im = x0.im + p1_im * F::C03_00 + p2_im * F::C03_01 + p3_im * F::C03_02 + p4_im * F::C03_03 + p5_im * F::C03_04 + p6_im * F::C03_05 + p7_im * F::C03_06 + p8_im * F::C03_07 + p9_im * F::C03_08 + p10_im * F::C03_09 + p11_im * F::C03_10; let q4_re = sign * (i1_re * F::S03_00 + i2_re * F::S03_01 + i3_re * F::S03_02 + i4_re * F::S03_03 + i5_re * F::S03_04 + i6_re * F::S03_05 + i7_re * F::S03_06 + i8_re * F::S03_07 + i9_re * F::S03_08 + i10_re * F::S03_09 + i11_re * F::S03_10); let q4_im = sign * (i1_im * F::S03_00 + i2_im * F::S03_01 + i3_im * F::S03_02 + i4_im * F::S03_03 + i5_im * F::S03_04 + i6_im * F::S03_05 + i7_im * F::S03_06 + i8_im * F::S03_07 + i9_im * F::S03_08 + i10_im * F::S03_09 + i11_im * F::S03_10);
    data[4] = num_complex::Complex::new(r4_re + q4_re, r4_im + q4_im); data[19] = num_complex::Complex::new(r4_re - q4_re, r4_im - q4_im);
    let r5_re = x0.re + p1_re * F::C04_00 + p2_re * F::C04_01 + p3_re * F::C04_02 + p4_re * F::C04_03 + p5_re * F::C04_04 + p6_re * F::C04_05 + p7_re * F::C04_06 + p8_re * F::C04_07 + p9_re * F::C04_08 + p10_re * F::C04_09 + p11_re * F::C04_10; let r5_im = x0.im + p1_im * F::C04_00 + p2_im * F::C04_01 + p3_im * F::C04_02 + p4_im * F::C04_03 + p5_im * F::C04_04 + p6_im * F::C04_05 + p7_im * F::C04_06 + p8_im * F::C04_07 + p9_im * F::C04_08 + p10_im * F::C04_09 + p11_im * F::C04_10; let q5_re = sign * (i1_re * F::S04_00 + i2_re * F::S04_01 + i3_re * F::S04_02 + i4_re * F::S04_03 + i5_re * F::S04_04 + i6_re * F::S04_05 + i7_re * F::S04_06 + i8_re * F::S04_07 + i9_re * F::S04_08 + i10_re * F::S04_09 + i11_re * F::S04_10); let q5_im = sign * (i1_im * F::S04_00 + i2_im * F::S04_01 + i3_im * F::S04_02 + i4_im * F::S04_03 + i5_im * F::S04_04 + i6_im * F::S04_05 + i7_im * F::S04_06 + i8_im * F::S04_07 + i9_im * F::S04_08 + i10_im * F::S04_09 + i11_im * F::S04_10);
    data[5] = num_complex::Complex::new(r5_re + q5_re, r5_im + q5_im); data[18] = num_complex::Complex::new(r5_re - q5_re, r5_im - q5_im);
    let r6_re = x0.re + p1_re * F::C05_00 + p2_re * F::C05_01 + p3_re * F::C05_02 + p4_re * F::C05_03 + p5_re * F::C05_04 + p6_re * F::C05_05 + p7_re * F::C05_06 + p8_re * F::C05_07 + p9_re * F::C05_08 + p10_re * F::C05_09 + p11_re * F::C05_10; let r6_im = x0.im + p1_im * F::C05_00 + p2_im * F::C05_01 + p3_im * F::C05_02 + p4_im * F::C05_03 + p5_im * F::C05_04 + p6_im * F::C05_05 + p7_im * F::C05_06 + p8_im * F::C05_07 + p9_im * F::C05_08 + p10_im * F::C05_09 + p11_im * F::C05_10; let q6_re = sign * (i1_re * F::S05_00 + i2_re * F::S05_01 + i3_re * F::S05_02 + i4_re * F::S05_03 + i5_re * F::S05_04 + i6_re * F::S05_05 + i7_re * F::S05_06 + i8_re * F::S05_07 + i9_re * F::S05_08 + i10_re * F::S05_09 + i11_re * F::S05_10); let q6_im = sign * (i1_im * F::S05_00 + i2_im * F::S05_01 + i3_im * F::S05_02 + i4_im * F::S05_03 + i5_im * F::S05_04 + i6_im * F::S05_05 + i7_im * F::S05_06 + i8_im * F::S05_07 + i9_im * F::S05_08 + i10_im * F::S05_09 + i11_im * F::S05_10);
    data[6] = num_complex::Complex::new(r6_re + q6_re, r6_im + q6_im); data[17] = num_complex::Complex::new(r6_re - q6_re, r6_im - q6_im);
    let r7_re = x0.re + p1_re * F::C06_00 + p2_re * F::C06_01 + p3_re * F::C06_02 + p4_re * F::C06_03 + p5_re * F::C06_04 + p6_re * F::C06_05 + p7_re * F::C06_06 + p8_re * F::C06_07 + p9_re * F::C06_08 + p10_re * F::C06_09 + p11_re * F::C06_10; let r7_im = x0.im + p1_im * F::C06_00 + p2_im * F::C06_01 + p3_im * F::C06_02 + p4_im * F::C06_03 + p5_im * F::C06_04 + p6_im * F::C06_05 + p7_im * F::C06_06 + p8_im * F::C06_07 + p9_im * F::C06_08 + p10_im * F::C06_09 + p11_im * F::C06_10; let q7_re = sign * (i1_re * F::S06_00 + i2_re * F::S06_01 + i3_re * F::S06_02 + i4_re * F::S06_03 + i5_re * F::S06_04 + i6_re * F::S06_05 + i7_re * F::S06_06 + i8_re * F::S06_07 + i9_re * F::S06_08 + i10_re * F::S06_09 + i11_re * F::S06_10); let q7_im = sign * (i1_im * F::S06_00 + i2_im * F::S06_01 + i3_im * F::S06_02 + i4_im * F::S06_03 + i5_im * F::S06_04 + i6_im * F::S06_05 + i7_im * F::S06_06 + i8_im * F::S06_07 + i9_im * F::S06_08 + i10_im * F::S06_09 + i11_im * F::S06_10);
    data[7] = num_complex::Complex::new(r7_re + q7_re, r7_im + q7_im); data[16] = num_complex::Complex::new(r7_re - q7_re, r7_im - q7_im);
    let r8_re = x0.re + p1_re * F::C07_00 + p2_re * F::C07_01 + p3_re * F::C07_02 + p4_re * F::C07_03 + p5_re * F::C07_04 + p6_re * F::C07_05 + p7_re * F::C07_06 + p8_re * F::C07_07 + p9_re * F::C07_08 + p10_re * F::C07_09 + p11_re * F::C07_10; let r8_im = x0.im + p1_im * F::C07_00 + p2_im * F::C07_01 + p3_im * F::C07_02 + p4_im * F::C07_03 + p5_im * F::C07_04 + p6_im * F::C07_05 + p7_im * F::C07_06 + p8_im * F::C07_07 + p9_im * F::C07_08 + p10_im * F::C07_09 + p11_im * F::C07_10; let q8_re = sign * (i1_re * F::S07_00 + i2_re * F::S07_01 + i3_re * F::S07_02 + i4_re * F::S07_03 + i5_re * F::S07_04 + i6_re * F::S07_05 + i7_re * F::S07_06 + i8_re * F::S07_07 + i9_re * F::S07_08 + i10_re * F::S07_09 + i11_re * F::S07_10); let q8_im = sign * (i1_im * F::S07_00 + i2_im * F::S07_01 + i3_im * F::S07_02 + i4_im * F::S07_03 + i5_im * F::S07_04 + i6_im * F::S07_05 + i7_im * F::S07_06 + i8_im * F::S07_07 + i9_im * F::S07_08 + i10_im * F::S07_09 + i11_im * F::S07_10);
    data[8] = num_complex::Complex::new(r8_re + q8_re, r8_im + q8_im); data[15] = num_complex::Complex::new(r8_re - q8_re, r8_im - q8_im);
    let r9_re = x0.re + p1_re * F::C08_00 + p2_re * F::C08_01 + p3_re * F::C08_02 + p4_re * F::C08_03 + p5_re * F::C08_04 + p6_re * F::C08_05 + p7_re * F::C08_06 + p8_re * F::C08_07 + p9_re * F::C08_08 + p10_re * F::C08_09 + p11_re * F::C08_10; let r9_im = x0.im + p1_im * F::C08_00 + p2_im * F::C08_01 + p3_im * F::C08_02 + p4_im * F::C08_03 + p5_im * F::C08_04 + p6_im * F::C08_05 + p7_im * F::C08_06 + p8_im * F::C08_07 + p9_im * F::C08_08 + p10_im * F::C08_09 + p11_im * F::C08_10; let q9_re = sign * (i1_re * F::S08_00 + i2_re * F::S08_01 + i3_re * F::S08_02 + i4_re * F::S08_03 + i5_re * F::S08_04 + i6_re * F::S08_05 + i7_re * F::S08_06 + i8_re * F::S08_07 + i9_re * F::S08_08 + i10_re * F::S08_09 + i11_re * F::S08_10); let q9_im = sign * (i1_im * F::S08_00 + i2_im * F::S08_01 + i3_im * F::S08_02 + i4_im * F::S08_03 + i5_im * F::S08_04 + i6_im * F::S08_05 + i7_im * F::S08_06 + i8_im * F::S08_07 + i9_im * F::S08_08 + i10_im * F::S08_09 + i11_im * F::S08_10);
    data[9] = num_complex::Complex::new(r9_re + q9_re, r9_im + q9_im); data[14] = num_complex::Complex::new(r9_re - q9_re, r9_im - q9_im);
    let r10_re = x0.re + p1_re * F::C09_00 + p2_re * F::C09_01 + p3_re * F::C09_02 + p4_re * F::C09_03 + p5_re * F::C09_04 + p6_re * F::C09_05 + p7_re * F::C09_06 + p8_re * F::C09_07 + p9_re * F::C09_08 + p10_re * F::C09_09 + p11_re * F::C09_10; let r10_im = x0.im + p1_im * F::C09_00 + p2_im * F::C09_01 + p3_im * F::C09_02 + p4_im * F::C09_03 + p5_im * F::C09_04 + p6_im * F::C09_05 + p7_im * F::C09_06 + p8_im * F::C09_07 + p9_im * F::C09_08 + p10_im * F::C09_09 + p11_im * F::C09_10; let q10_re = sign * (i1_re * F::S09_00 + i2_re * F::S09_01 + i3_re * F::S09_02 + i4_re * F::S09_03 + i5_re * F::S09_04 + i6_re * F::S09_05 + i7_re * F::S09_06 + i8_re * F::S09_07 + i9_re * F::S09_08 + i10_re * F::S09_09 + i11_re * F::S09_10); let q10_im = sign * (i1_im * F::S09_00 + i2_im * F::S09_01 + i3_im * F::S09_02 + i4_im * F::S09_03 + i5_im * F::S09_04 + i6_im * F::S09_05 + i7_im * F::S09_06 + i8_im * F::S09_07 + i9_im * F::S09_08 + i10_im * F::S09_09 + i11_im * F::S09_10);
    data[10] = num_complex::Complex::new(r10_re + q10_re, r10_im + q10_im); data[13] = num_complex::Complex::new(r10_re - q10_re, r10_im - q10_im);
    let r11_re = x0.re + p1_re * F::C10_00 + p2_re * F::C10_01 + p3_re * F::C10_02 + p4_re * F::C10_03 + p5_re * F::C10_04 + p6_re * F::C10_05 + p7_re * F::C10_06 + p8_re * F::C10_07 + p9_re * F::C10_08 + p10_re * F::C10_09 + p11_re * F::C10_10; let r11_im = x0.im + p1_im * F::C10_00 + p2_im * F::C10_01 + p3_im * F::C10_02 + p4_im * F::C10_03 + p5_im * F::C10_04 + p6_im * F::C10_05 + p7_im * F::C10_06 + p8_im * F::C10_07 + p9_im * F::C10_08 + p10_im * F::C10_09 + p11_im * F::C10_10; let q11_re = sign * (i1_re * F::S10_00 + i2_re * F::S10_01 + i3_re * F::S10_02 + i4_re * F::S10_03 + i5_re * F::S10_04 + i6_re * F::S10_05 + i7_re * F::S10_06 + i8_re * F::S10_07 + i9_re * F::S10_08 + i10_re * F::S10_09 + i11_re * F::S10_10); let q11_im = sign * (i1_im * F::S10_00 + i2_im * F::S10_01 + i3_im * F::S10_02 + i4_im * F::S10_03 + i5_im * F::S10_04 + i6_im * F::S10_05 + i7_im * F::S10_06 + i8_im * F::S10_07 + i9_im * F::S10_08 + i10_im * F::S10_09 + i11_im * F::S10_10);
    data[11] = num_complex::Complex::new(r11_re + q11_re, r11_im + q11_im); data[12] = num_complex::Complex::new(r11_re - q11_re, r11_im - q11_im);
}
