use super::traits::WinogradScalar;

pub(crate) mod odd_prime_pair;

/// In-place Winograd DFT-4.
///
/// Forward contract:
/// `y[k] = sum_j x[j] * exp(-2*pi*i*j*k/4)`.
/// Inverse uses the conjugate sign. Multiplications are only sign changes and
/// real/imaginary swaps.
#[inline]
#[cfg(test)]
pub(crate) fn dft4_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>],
) {
    debug_assert!(data.len() >= 4);
    let data: &mut [num_complex::Complex<F>; 4] =
        (&mut data[..4]).try_into().expect("length checked");
    dft4_array_impl::<F, INVERSE>(data);
}

#[inline(always)]
pub(crate) fn dft4_array_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>; 4],
) {
    let t0 = data[0] + data[2];
    let t1 = data[0] - data[2];
    let t2 = data[1] + data[3];
    let t3 = data[1] - data[3];
    data[0] = t0 + t2;
    data[2] = t0 - t2;
    let i_t3 = if INVERSE {
        num_complex::Complex::new(-t3.im, t3.re)
    } else {
        num_complex::Complex::new(t3.im, -t3.re)
    };
    data[1] = t1 + i_t3;
    data[3] = t1 - i_t3;
}

#[inline(always)]
pub(crate) fn dft4_values<F: WinogradScalar, const INVERSE: bool>(
    x0: num_complex::Complex<F>,
    x1: num_complex::Complex<F>,
    x2: num_complex::Complex<F>,
    x3: num_complex::Complex<F>,
) -> (
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
) {
    let t0 = x0 + x2;
    let t1 = x0 - x2;
    let t2 = x1 + x3;
    let t3 = x1 - x3;
    let y0 = t0 + t2;
    let y2 = t0 - t2;
    let i_t3 = if INVERSE {
        num_complex::Complex::new(-t3.im, t3.re)
    } else {
        num_complex::Complex::new(t3.im, -t3.re)
    };
    let y1 = t1 + i_t3;
    let y3 = t1 - i_t3;
    (y0, y1, y2, y3)
}

#[inline(always)]
pub(crate) fn dft8_values<F: WinogradScalar, const INVERSE: bool>(
    x0: num_complex::Complex<F>,
    x1: num_complex::Complex<F>,
    x2: num_complex::Complex<F>,
    x3: num_complex::Complex<F>,
    x4: num_complex::Complex<F>,
    x5: num_complex::Complex<F>,
    x6: num_complex::Complex<F>,
    x7: num_complex::Complex<F>,
) -> (
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
) {
    let sq2o2 = F::sq2o2();
    let (e0, e1, e2, e3) = dft4_values::<F, INVERSE>(x0, x2, x4, x6);
    let (o0, o1, o2, o3) = dft4_values::<F, INVERSE>(x1, x3, x5, x7);

    // o2: pure ±i rotation — no multiplication, just sign/swap.
    let o2 = if INVERSE {
        num_complex::Complex::new(-o2.im, o2.re)
    } else {
        num_complex::Complex::new(o2.im, -o2.re)
    };

    // o1, o3: W_8^k twiddles with sign determined by const INVERSE.
    // Original: sign = INVERSE ? +1 : -1.
    // o1_new.re = sq2o2*(o1.re - sign*o1.im), o1_new.im = sq2o2*(sign*o1.re + o1.im)
    // o3_new.re = sq2o2*(-o3.re - sign*o3.im), o3_new.im = sq2o2*(sign*o3.re - o3.im)
    // Expanded per branch (eliminates the runtime sign * x multiply):
    let (o1, o3) = if INVERSE {
        // sign = +1: o1.re = sq2o2*(re - im), o1.im = sq2o2*(re + im)
        //            o3.re = sq2o2*(-re - im), o3.im = sq2o2*(re - im)
        (
            num_complex::Complex::new(sq2o2 * (o1.re - o1.im), sq2o2 * (o1.re + o1.im)),
            num_complex::Complex::new(sq2o2 * (-o3.re - o3.im), sq2o2 * (o3.re - o3.im)),
        )
    } else {
        // sign = -1: o1.re = sq2o2*(re + im), o1.im = sq2o2*(-re + im)
        //            o3.re = sq2o2*(-re + im), o3.im = sq2o2*(-re - im)
        (
            num_complex::Complex::new(sq2o2 * (o1.re + o1.im), sq2o2 * (-o1.re + o1.im)),
            num_complex::Complex::new(sq2o2 * (-o3.re + o3.im), sq2o2 * (-o3.re - o3.im)),
        )
    };

    (
        e0 + o0,
        e1 + o1,
        e2 + o2,
        e3 + o3,
        e0 - o0,
        e1 - o1,
        e2 - o2,
        e3 - o3,
    )
}

#[inline]
#[cfg(test)]
pub(crate) fn dft8_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>],
) {
    debug_assert!(data.len() >= 8);
    let data: &mut [num_complex::Complex<F>; 8] =
        (&mut data[..8]).try_into().expect("length checked");
    dft8_array_impl::<F, INVERSE>(data);
}

#[inline(always)]
pub(crate) fn dft8_array_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>; 8],
) {
    let (y0, y1, y2, y3, y4, y5, y6, y7) = dft8_values::<F, INVERSE>(
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    );
    data[0] = y0;
    data[1] = y1;
    data[2] = y2;
    data[3] = y3;
    data[4] = y4;
    data[5] = y5;
    data[6] = y6;
    data[7] = y7;
}

/// In-place Winograd DFT-7.
///
/// ## Mathematical derivation
///
/// N=7 is prime; exploit Hermitian symmetry of the twiddle matrix.
/// W₇ = exp(−2πi/7). Define xr[n]=x[n]+x[7−n], xi[n]=x[n]−x[7−n] for n=1..3.
/// Then X[k] = x[0] + Σ_{n=1}^{3} [cos(2πkn/7)·xr[n] + sign·sin(2πkn/7)·(i·xi[n])]
/// where sign=+1 for inverse (conjugate twiddles), −1 for forward.
/// X[7−k] = conjugate-symmetric counterpart sharing real parts with X[k].
///
/// Cosine matrix (k=1..3, n=1..3): row-cyclic in [c1,c2,c3]:
///   k=1: [c1,c2,c3],  k=2: [c2,c3,c1],  k=3: [c3,c1,c2].
/// Sine rows: k=1:[s1,s2,s3], k=2:[s2,−s3,−s1], k=3:[s3,−s1,s2].
///
/// **Real multiplications**: 18 scalar×complex (= 36 real muls).
/// Replaces the O(N²) naive DFT that computed trig at every call.
///
/// Constants: cos(2πk/7) and sin(2πk/7) for k=1,2,3.
/// References: Winograd (1978), Blahut (2010) §3.5.
#[inline(always)]
pub(crate) fn dft7_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>; 7],
) {
    *data = dft7_values::<F, INVERSE>(*data);
}

#[inline(always)]
pub(crate) fn dft7_values<F: WinogradScalar, const INVERSE: bool>(
    data: [num_complex::Complex<F>; 7],
) -> [num_complex::Complex<F>; 7] {
    let xr1 = data[1] + data[6];
    let xr2 = data[2] + data[5];
    let xr3 = data[3] + data[4];
    let xi1 = data[1] - data[6];
    let xi2 = data[2] - data[5];
    let xi3 = data[3] - data[4];
    // i·xi[n] = (−xi.im, xi.re)
    let ixi1 = num_complex::Complex::new(-xi1.im, xi1.re);
    let ixi2 = num_complex::Complex::new(-xi2.im, xi2.re);
    let ixi3 = num_complex::Complex::new(-xi3.im, xi3.re);
    let c1 = F::from_precise(0.6234898018587336);
    let c2 = F::from_precise(-0.2225209339563144);
    let c3 = F::from_precise(-0.9009688679024191);
    let s1 = F::from_precise(0.7818314824680298);
    let s2 = F::from_precise(0.9749279121818236);
    let s3 = F::from_precise(0.4338837391175582);
    // sign = +1 for inverse, -1 for forward. Use if/else on const INVERSE:
    // ixi * sign is a complex-scalar multiply (2 real muls each). Replacing
    // with conditional negation costs only a sign-bit flip per element.
    let (sixi1, sixi2, sixi3) = if INVERSE {
        (ixi1, ixi2, ixi3)
    } else {
        (-ixi1, -ixi2, -ixi3)
    };
    let x0 = data[0];
    let re1 = x0 + xr1 * c1 + xr2 * c2 + xr3 * c3;
    let re2 = x0 + xr1 * c2 + xr2 * c3 + xr3 * c1;
    let re3 = x0 + xr1 * c3 + xr2 * c1 + xr3 * c2;
    let d1 = sixi1 * s1 + sixi2 * s2 + sixi3 * s3;
    let d2 = sixi1 * s2 - sixi2 * s3 - sixi3 * s1;
    let d3 = sixi1 * s3 - sixi2 * s1 + sixi3 * s2;
    [
        x0 + xr1 + xr2 + xr3,
        re1 + d1,
        re2 + d2,
        re3 + d3,
        re3 - d3,
        re2 - d2,
        re1 - d1,
    ]
}

pub(crate) mod dft3;
pub(crate) use dft3::{dft3_impl, dft3_values};

/// In-place Good-Thomas DFT-15.
///
/// ## Mathematical derivation
///
/// N=15 = N1×N2 = 3×5, gcd(3,5)=1. Good-Thomas PFA requires no inter-stage
/// twiddle factors because N1 and N2 are coprime (unlike Cooley-Tukey).
///
/// **Input CRT mapping**: grid[i1·5+i2] = data[(5·i1 + 3·i2) mod 15]
/// for i1 ∈ 0..3, i2 ∈ 0..5.
///
/// **Apply DFT-5** on each of the 3 rows (i1=0,1,2).
///
/// **Transpose** 3×5 → 5×3.
///
/// **Apply DFT-3** on each of the 5 columns (now contiguous).
///
/// **Output CRT mapping**:
/// inv(5 mod 3)=2, inv(3 mod 5)=2.
/// k_idx = (10·k1 + 6·k2) mod 15; data[k_idx] = result[k2·3+k1].
///
/// **Real multiplications**: 3×8 + 5×4 = 44 real muls.
/// All storage is on-stack; zero heap allocation.
///
/// References: Good (1958), Thomas (1963), Burrus & Parks (1985) §3.
#[inline]
pub(crate) fn dft15_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>; 15],
) {
    *data = dft15_values::<F, INVERSE>(*data);
}

#[inline(always)]
pub(crate) fn dft15_values<F: WinogradScalar, const INVERSE: bool>(
    data: [num_complex::Complex<F>; 15],
) -> [num_complex::Complex<F>; 15] {
    let r0 = dft5_values::<F, INVERSE>([data[0], data[3], data[6], data[9], data[12]]);
    let r1 = dft5_values::<F, INVERSE>([data[5], data[8], data[11], data[14], data[2]]);
    let r2 = dft5_values::<F, INVERSE>([data[10], data[13], data[1], data[4], data[7]]);

    let (y0_0, y1_0, y2_0) = dft3_values::<F, INVERSE>(r0[0], r1[0], r2[0]);
    let (y0_1, y1_1, y2_1) = dft3_values::<F, INVERSE>(r0[1], r1[1], r2[1]);
    let (y0_2, y1_2, y2_2) = dft3_values::<F, INVERSE>(r0[2], r1[2], r2[2]);
    let (y0_3, y1_3, y2_3) = dft3_values::<F, INVERSE>(r0[3], r1[3], r2[3]);
    let (y0_4, y1_4, y2_4) = dft3_values::<F, INVERSE>(r0[4], r1[4], r2[4]);

    [
        y0_0, // 0
        y1_1, // 1
        y2_2, // 2
        y0_3, // 3
        y1_4, // 4
        y2_0, // 5
        y0_1, // 6
        y1_2, // 7
        y2_3, // 8
        y0_4, // 9
        y1_0, // 10
        y2_1, // 11
        y0_2, // 12
        y1_3, // 13
        y2_4, // 14
    ]
}

/// In-place DFT-5.
///
/// ## Mathematical derivation
///
/// For N=5, W₅ = exp(−2πi/5). The symmetric index pairs (1,4) and (2,3)
/// allow the 5-point DFT to be expressed via sum/difference decomposition:
/// ```text
/// r₁ = X[1]+X[4],  d₁ = X[1]−X[4]
/// r₂ = X[2]+X[3],  d₂ = X[2]−X[3]
///
/// Y[0] = X[0] + r₁ + r₂
/// ar   = X[0] + c₁·r₁ + c₂·r₂       (cosine terms for Y[1],Y[4])
/// br   = X[0] + c₂·r₁ + c₁·r₂       (cosine terms for Y[2],Y[3])
/// id₁  = s₁·d₁ + s₂·d₂               (imaginary term for Y[1],Y[4])
/// id₂  = s₂·d₁ − s₁·d₂               (imaginary term for Y[2],Y[3])
///
/// Y[1] = ar − i·id₁   (fwd)    Y[4] = ar + i·id₁
/// Y[2] = br − i·id₂   (fwd)    Y[3] = br + i·id₂
/// ```
/// Inverse: flip sign of the imaginary rotation (−i ↔ +i).
///
/// Constants:
/// - c₁ = cos(2π/5) = (√5−1)/4 ≈ 0.30902
/// - c₂ = cos(4π/5) = −(√5+1)/4 ≈ −0.80902
/// - s₁ = sin(2π/5) ≈ 0.95106
/// - s₂ = sin(4π/5) ≈ 0.58779
///
/// **Real multiplications**: 8 (c₁,c₂ applied to r₁,r₂; s₁,s₂ applied to
/// d₁,d₂ — each scalar×complex costs 2 real muls). Standard minimal-form
/// derivation: Winograd (1978), Blahut (2010) §3.3.
/// **Complex additions**: 10.
#[inline(always)]
#[cfg(test)]
pub(crate) fn dft5_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>],
) {
    debug_assert!(data.len() >= 5);
    let [y0, y1, y2, y3, y4] =
        dft5_values::<F, INVERSE>([data[0], data[1], data[2], data[3], data[4]]);
    data[0] = y0;
    data[1] = y1;
    data[2] = y2;
    data[3] = y3;
    data[4] = y4;
}

#[inline(always)]
pub(crate) fn dft5_array_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>; 5],
) {
    *data = dft5_values::<F, INVERSE>(*data);
}

#[inline(always)]
pub(crate) fn dft5_values<F: WinogradScalar, const INVERSE: bool>(
    data: [num_complex::Complex<F>; 5],
) -> [num_complex::Complex<F>; 5] {
    let c1 = F::from_precise(0.30901699437494745);
    let c2 = F::from_precise(-0.8090169943749475);
    let s1 = F::from_precise(0.9510565162951535);
    let s2 = F::from_precise(0.5877852522924731);
    // sign = +1 for inverse, -1 for forward. Use if/else on const INVERSE
    // to avoid a runtime scalar multiply on a compile-time-known value.
    let s1 = if INVERSE { s1 } else { -s1 };
    let s2 = if INVERSE { s2 } else { -s2 };
    let x0 = data[0];
    let x1 = data[1];
    let x2 = data[2];
    let x3 = data[3];
    let x4 = data[4];
    let t1_re = x1.re + x4.re;
    let t1_im = x1.im + x4.im;
    let t2_re = x1.re - x4.re;
    let t2_im = x1.im - x4.im;
    let t3_re = x2.re + x3.re;
    let t3_im = x2.im + x3.im;
    let t4_re = x2.re - x3.re;
    let t4_im = x2.im - x3.im;
    let m1_re = t1_re * c1 + t3_re * c2;
    let m1_im = t1_im * c1 + t3_im * c2;
    let m2_re = t1_re * c2 + t3_re * c1;
    let m2_im = t1_im * c2 + t3_im * c1;
    let q3_re = t2_re * s1 + t4_re * s2;
    let q3_im = t2_im * s1 + t4_im * s2;
    let q4_re = t2_re * s2 - t4_re * s1;
    let q4_im = t2_im * s2 - t4_im * s1;
    let a1_re = x0.re + m1_re;
    let a1_im = x0.im + m1_im;
    let a2_re = x0.re + m2_re;
    let a2_im = x0.im + m2_im;
    [
        num_complex::Complex::new(x0.re + t1_re + t3_re, x0.im + t1_im + t3_im),
        num_complex::Complex::new(a1_re - q3_im, a1_im + q3_re),
        num_complex::Complex::new(a2_re - q4_im, a2_im + q4_re),
        num_complex::Complex::new(a2_re + q4_im, a2_im - q4_re),
        num_complex::Complex::new(a1_re + q3_im, a1_im - q3_re),
    ]
}
