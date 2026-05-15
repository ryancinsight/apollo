use super::scalar::Dft17Scalar;

/// In-place prime DFT-17.
///
/// Pair symmetry reduces the 17x17 DFT matrix to eight conjugate input pairs.
/// For each `m=1..8`, define `p_m=x[m]+x[17-m]` and
/// `i_m=i*(x[m]-x[17-m])`. Output rows `k` and `17-k` share the cosine
/// projection and differ only by the sine projection sign.
#[inline(never)]
pub(crate) fn dft17_impl<F: Dft17Scalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>],
) {
    dft17_body::<F, INVERSE>(data);
}

#[inline(always)]
pub(crate) fn dft17_inline_impl<F: Dft17Scalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>],
) {
    dft17_body::<F, INVERSE>(data);
}

#[inline(always)]
fn dft17_body<F: Dft17Scalar, const INVERSE: bool>(data: &mut [num_complex::Complex<F>]) {
    debug_assert!(data.len() >= 17);
    let sign = if INVERSE {
        F::cast_f64(1.0)
    } else {
        F::cast_f64(-1.0)
    };
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
    let p1_re = x1.re + x16.re;
    let p1_im = x1.im + x16.im;
    let d1_re = x1.re - x16.re;
    let d1_im = x1.im - x16.im;
    let i1_re = -d1_im;
    let i1_im = d1_re;
    let p2_re = x2.re + x15.re;
    let p2_im = x2.im + x15.im;
    let d2_re = x2.re - x15.re;
    let d2_im = x2.im - x15.im;
    let i2_re = -d2_im;
    let i2_im = d2_re;
    let p3_re = x3.re + x14.re;
    let p3_im = x3.im + x14.im;
    let d3_re = x3.re - x14.re;
    let d3_im = x3.im - x14.im;
    let i3_re = -d3_im;
    let i3_im = d3_re;
    let p4_re = x4.re + x13.re;
    let p4_im = x4.im + x13.im;
    let d4_re = x4.re - x13.re;
    let d4_im = x4.im - x13.im;
    let i4_re = -d4_im;
    let i4_im = d4_re;
    let p5_re = x5.re + x12.re;
    let p5_im = x5.im + x12.im;
    let d5_re = x5.re - x12.re;
    let d5_im = x5.im - x12.im;
    let i5_re = -d5_im;
    let i5_im = d5_re;
    let p6_re = x6.re + x11.re;
    let p6_im = x6.im + x11.im;
    let d6_re = x6.re - x11.re;
    let d6_im = x6.im - x11.im;
    let i6_re = -d6_im;
    let i6_im = d6_re;
    let p7_re = x7.re + x10.re;
    let p7_im = x7.im + x10.im;
    let d7_re = x7.re - x10.re;
    let d7_im = x7.im - x10.im;
    let i7_re = -d7_im;
    let i7_im = d7_re;
    let p8_re = x8.re + x9.re;
    let p8_im = x8.im + x9.im;
    let d8_re = x8.re - x9.re;
    let d8_im = x8.im - x9.im;
    let i8_re = -d8_im;
    let i8_im = d8_re;
    data[0] = num_complex::Complex::new(
        x0.re + p1_re + p2_re + p3_re + p4_re + p5_re + p6_re + p7_re + p8_re,
        x0.im + p1_im + p2_im + p3_im + p4_im + p5_im + p6_im + p7_im + p8_im,
    );
    let r1_re = x0.re
        + p1_re * F::C00
        + p2_re * F::C01
        + p3_re * F::C02
        + p4_re * F::C03
        + p5_re * F::C04
        + p6_re * F::C05
        + p7_re * F::C06
        + p8_re * F::C07;
    let r1_im = x0.im
        + p1_im * F::C00
        + p2_im * F::C01
        + p3_im * F::C02
        + p4_im * F::C03
        + p5_im * F::C04
        + p6_im * F::C05
        + p7_im * F::C06
        + p8_im * F::C07;
    let q1_re = sign
        * (i1_re * F::S00
            + i2_re * F::S01
            + i3_re * F::S02
            + i4_re * F::S03
            + i5_re * F::S04
            + i6_re * F::S05
            + i7_re * F::S06
            + i8_re * F::S07);
    let q1_im = sign
        * (i1_im * F::S00
            + i2_im * F::S01
            + i3_im * F::S02
            + i4_im * F::S03
            + i5_im * F::S04
            + i6_im * F::S05
            + i7_im * F::S06
            + i8_im * F::S07);
    data[1] = num_complex::Complex::new(r1_re + q1_re, r1_im + q1_im);
    data[16] = num_complex::Complex::new(r1_re - q1_re, r1_im - q1_im);
    let r2_re = x0.re
        + p1_re * F::C10
        + p2_re * F::C11
        + p3_re * F::C12
        + p4_re * F::C13
        + p5_re * F::C14
        + p6_re * F::C15
        + p7_re * F::C16
        + p8_re * F::C17;
    let r2_im = x0.im
        + p1_im * F::C10
        + p2_im * F::C11
        + p3_im * F::C12
        + p4_im * F::C13
        + p5_im * F::C14
        + p6_im * F::C15
        + p7_im * F::C16
        + p8_im * F::C17;
    let q2_re = sign
        * (i1_re * F::S10
            + i2_re * F::S11
            + i3_re * F::S12
            + i4_re * F::S13
            + i5_re * F::S14
            + i6_re * F::S15
            + i7_re * F::S16
            + i8_re * F::S17);
    let q2_im = sign
        * (i1_im * F::S10
            + i2_im * F::S11
            + i3_im * F::S12
            + i4_im * F::S13
            + i5_im * F::S14
            + i6_im * F::S15
            + i7_im * F::S16
            + i8_im * F::S17);
    data[2] = num_complex::Complex::new(r2_re + q2_re, r2_im + q2_im);
    data[15] = num_complex::Complex::new(r2_re - q2_re, r2_im - q2_im);
    let r3_re = x0.re
        + p1_re * F::C20
        + p2_re * F::C21
        + p3_re * F::C22
        + p4_re * F::C23
        + p5_re * F::C24
        + p6_re * F::C25
        + p7_re * F::C26
        + p8_re * F::C27;
    let r3_im = x0.im
        + p1_im * F::C20
        + p2_im * F::C21
        + p3_im * F::C22
        + p4_im * F::C23
        + p5_im * F::C24
        + p6_im * F::C25
        + p7_im * F::C26
        + p8_im * F::C27;
    let q3_re = sign
        * (i1_re * F::S20
            + i2_re * F::S21
            + i3_re * F::S22
            + i4_re * F::S23
            + i5_re * F::S24
            + i6_re * F::S25
            + i7_re * F::S26
            + i8_re * F::S27);
    let q3_im = sign
        * (i1_im * F::S20
            + i2_im * F::S21
            + i3_im * F::S22
            + i4_im * F::S23
            + i5_im * F::S24
            + i6_im * F::S25
            + i7_im * F::S26
            + i8_im * F::S27);
    data[3] = num_complex::Complex::new(r3_re + q3_re, r3_im + q3_im);
    data[14] = num_complex::Complex::new(r3_re - q3_re, r3_im - q3_im);
    let r4_re = x0.re
        + p1_re * F::C30
        + p2_re * F::C31
        + p3_re * F::C32
        + p4_re * F::C33
        + p5_re * F::C34
        + p6_re * F::C35
        + p7_re * F::C36
        + p8_re * F::C37;
    let r4_im = x0.im
        + p1_im * F::C30
        + p2_im * F::C31
        + p3_im * F::C32
        + p4_im * F::C33
        + p5_im * F::C34
        + p6_im * F::C35
        + p7_im * F::C36
        + p8_im * F::C37;
    let q4_re = sign
        * (i1_re * F::S30
            + i2_re * F::S31
            + i3_re * F::S32
            + i4_re * F::S33
            + i5_re * F::S34
            + i6_re * F::S35
            + i7_re * F::S36
            + i8_re * F::S37);
    let q4_im = sign
        * (i1_im * F::S30
            + i2_im * F::S31
            + i3_im * F::S32
            + i4_im * F::S33
            + i5_im * F::S34
            + i6_im * F::S35
            + i7_im * F::S36
            + i8_im * F::S37);
    data[4] = num_complex::Complex::new(r4_re + q4_re, r4_im + q4_im);
    data[13] = num_complex::Complex::new(r4_re - q4_re, r4_im - q4_im);
    let r5_re = x0.re
        + p1_re * F::C40
        + p2_re * F::C41
        + p3_re * F::C42
        + p4_re * F::C43
        + p5_re * F::C44
        + p6_re * F::C45
        + p7_re * F::C46
        + p8_re * F::C47;
    let r5_im = x0.im
        + p1_im * F::C40
        + p2_im * F::C41
        + p3_im * F::C42
        + p4_im * F::C43
        + p5_im * F::C44
        + p6_im * F::C45
        + p7_im * F::C46
        + p8_im * F::C47;
    let q5_re = sign
        * (i1_re * F::S40
            + i2_re * F::S41
            + i3_re * F::S42
            + i4_re * F::S43
            + i5_re * F::S44
            + i6_re * F::S45
            + i7_re * F::S46
            + i8_re * F::S47);
    let q5_im = sign
        * (i1_im * F::S40
            + i2_im * F::S41
            + i3_im * F::S42
            + i4_im * F::S43
            + i5_im * F::S44
            + i6_im * F::S45
            + i7_im * F::S46
            + i8_im * F::S47);
    data[5] = num_complex::Complex::new(r5_re + q5_re, r5_im + q5_im);
    data[12] = num_complex::Complex::new(r5_re - q5_re, r5_im - q5_im);
    let r6_re = x0.re
        + p1_re * F::C50
        + p2_re * F::C51
        + p3_re * F::C52
        + p4_re * F::C53
        + p5_re * F::C54
        + p6_re * F::C55
        + p7_re * F::C56
        + p8_re * F::C57;
    let r6_im = x0.im
        + p1_im * F::C50
        + p2_im * F::C51
        + p3_im * F::C52
        + p4_im * F::C53
        + p5_im * F::C54
        + p6_im * F::C55
        + p7_im * F::C56
        + p8_im * F::C57;
    let q6_re = sign
        * (i1_re * F::S50
            + i2_re * F::S51
            + i3_re * F::S52
            + i4_re * F::S53
            + i5_re * F::S54
            + i6_re * F::S55
            + i7_re * F::S56
            + i8_re * F::S57);
    let q6_im = sign
        * (i1_im * F::S50
            + i2_im * F::S51
            + i3_im * F::S52
            + i4_im * F::S53
            + i5_im * F::S54
            + i6_im * F::S55
            + i7_im * F::S56
            + i8_im * F::S57);
    data[6] = num_complex::Complex::new(r6_re + q6_re, r6_im + q6_im);
    data[11] = num_complex::Complex::new(r6_re - q6_re, r6_im - q6_im);
    let r7_re = x0.re
        + p1_re * F::C60
        + p2_re * F::C61
        + p3_re * F::C62
        + p4_re * F::C63
        + p5_re * F::C64
        + p6_re * F::C65
        + p7_re * F::C66
        + p8_re * F::C67;
    let r7_im = x0.im
        + p1_im * F::C60
        + p2_im * F::C61
        + p3_im * F::C62
        + p4_im * F::C63
        + p5_im * F::C64
        + p6_im * F::C65
        + p7_im * F::C66
        + p8_im * F::C67;
    let q7_re = sign
        * (i1_re * F::S60
            + i2_re * F::S61
            + i3_re * F::S62
            + i4_re * F::S63
            + i5_re * F::S64
            + i6_re * F::S65
            + i7_re * F::S66
            + i8_re * F::S67);
    let q7_im = sign
        * (i1_im * F::S60
            + i2_im * F::S61
            + i3_im * F::S62
            + i4_im * F::S63
            + i5_im * F::S64
            + i6_im * F::S65
            + i7_im * F::S66
            + i8_im * F::S67);
    data[7] = num_complex::Complex::new(r7_re + q7_re, r7_im + q7_im);
    data[10] = num_complex::Complex::new(r7_re - q7_re, r7_im - q7_im);
    let r8_re = x0.re
        + p1_re * F::C70
        + p2_re * F::C71
        + p3_re * F::C72
        + p4_re * F::C73
        + p5_re * F::C74
        + p6_re * F::C75
        + p7_re * F::C76
        + p8_re * F::C77;
    let r8_im = x0.im
        + p1_im * F::C70
        + p2_im * F::C71
        + p3_im * F::C72
        + p4_im * F::C73
        + p5_im * F::C74
        + p6_im * F::C75
        + p7_im * F::C76
        + p8_im * F::C77;
    let q8_re = sign
        * (i1_re * F::S70
            + i2_re * F::S71
            + i3_re * F::S72
            + i4_re * F::S73
            + i5_re * F::S74
            + i6_re * F::S75
            + i7_re * F::S76
            + i8_re * F::S77);
    let q8_im = sign
        * (i1_im * F::S70
            + i2_im * F::S71
            + i3_im * F::S72
            + i4_im * F::S73
            + i5_im * F::S74
            + i6_im * F::S75
            + i7_im * F::S76
            + i8_im * F::S77);
    data[8] = num_complex::Complex::new(r8_re + q8_re, r8_im + q8_im);
    data[9] = num_complex::Complex::new(r8_re - q8_re, r8_im - q8_im);
}
