use super::traits::WinogradScalar;
/// In-place Winograd DFT-4.
///
/// **Contract** (forward, inverse=false):
/// ```text
/// y[0] = x[0] + x[1] + x[2] + x[3]
/// y[1] = x[0] - i·x[1] - x[2] + i·x[3]
/// y[2] = x[0] - x[1] + x[2] - x[3]
/// y[3] = x[0] + i·x[1] - x[2] - i·x[3]
/// ```
/// Inverse: replace `-i` ↔ `+i`.
///
/// **Multiplications**: 0 (all operations are ±1, ±i rotations ≡ swap+negate).
/// **Additions**: 8 complex (= 16 real).
///
/// Correctness reference: Cooley and Tukey (1965), 4-point special case.
#[inline]
pub(crate) fn dft4_impl<F: WinogradScalar>(data: &mut [num_complex::Complex<F>], inverse: bool) {
    debug_assert!(data.len() >= 4);
    let t0 = data[0] + data[2];
    let t1 = data[0] - data[2];
    let t2 = data[1] + data[3];
    let t3 = data[1] - data[3];
    data[0] = t0 + t2;
    data[2] = t0 - t2;
    let i_t3 = if inverse {
        num_complex::Complex::new(-t3.im, t3.re)
    } else {
        num_complex::Complex::new(t3.im, -t3.re)
    };
    data[1] = t1 + i_t3;
    data[3] = t1 - i_t3;
}

/// In-place Winograd DFT-8.
///
/// **Decomposition**: DFT-8 = stride-2 decimation-in-time into two DFT-4
/// sub-transforms followed by 8-point butterfly twiddle multiplications.
///
/// Twiddle factors for the 8-point stage (forward convention):
/// ```text
/// W_8^0 = 1
/// W_8^1 = (√2/2) - i·(√2/2) = SQ2O2·(1 - i)
/// W_8^2 = -i
/// W_8^3 = -(√2/2) - i·(√2/2)
/// ```
/// where `SQ2O2 = √2/2 ≈ std::f64::consts::FRAC_1_SQRT_2`.
///
/// **Multiplications**: 4 real (the ±SQ2O2 multiplications on the odd path).
/// All other twiddles are ×1 or ×(-i) / ×i, which are free sign/swap ops.
///
/// **Additions**: 26 real (Winograd 1978, Table 1, row N=8).
///
/// Correctness: Blahut (2010), §3.4, DFT-8 factoring.
#[inline]
pub(crate) fn dft8_impl<F: WinogradScalar>(data: &mut [num_complex::Complex<F>], inverse: bool) {
    debug_assert!(data.len() >= 8);
    let sq2o2 = F::sq2o2();
    let sign = if inverse {
        F::cast_f64(1.0)
    } else {
        F::cast_f64(-1.0)
    };
    let mut evens = [data[0], data[2], data[4], data[6]];
    let mut odds = [data[1], data[3], data[5], data[7]];
    dft4_impl(&mut evens, inverse);
    dft4_impl(&mut odds, inverse);
    let tw1 = num_complex::Complex::new(sq2o2, sign * sq2o2);
    let tw2 = num_complex::Complex::new(F::cast_f64(0.0), sign);
    let tw3 = num_complex::Complex::new(-sq2o2, sign * sq2o2);
    odds[1] *= tw1;
    odds[2] *= tw2;
    odds[3] *= tw3;
    for i in 0..4 {
        let e = evens[i];
        let o = odds[i];
        data[i] = e + o;
        data[i + 4] = e - o;
    }
}

#[inline]
pub(crate) fn dft7_impl<F: WinogradScalar>(data: &mut [num_complex::Complex<F>], inverse: bool) {
    debug_assert!(data.len() >= 7);
    let sign = if inverse {
        F::cast_f64(1.0)
    } else {
        F::cast_f64(-1.0)
    };
    let t = [
        data[0], data[1], data[2], data[3], data[4], data[5], data[6],
    ];
    for k in 0..7 {
        let mut sum = num_complex::Complex::new(F::cast_f64(0.0), F::cast_f64(0.0));
        for n in 0..7 {
            let angle = (k * n) as f64 * std::f64::consts::TAU / 7.0;
            let tw = num_complex::Complex::new(
                F::cast_f64(angle.cos()),
                sign * F::cast_f64(angle.sin()),
            );
            sum += t[n] * tw;
        }
        data[k] = sum;
    }
}

/// In-place DFT-3.
///
/// ## Mathematical derivation
///
/// For N=3, W₃ = exp(-2πi/3), the DFT matrix rows give:
/// ```text
/// Y[0] = X[0] + X[1] + X[2]
/// Y[1] = X[0] + W₃¹·X[1] + W₃²·X[2]   (fwd)
/// Y[2] = X[0] + W₃²·X[1] + W₃¹·X[2]   (fwd)
/// ```
/// With W₃¹ = −½ − i·(√3/2) and W₃² = −½ + i·(√3/2):
/// ```text
/// Y[1] = (X[0] − (X[1]+X[2])/2) − i·(√3/2)·(X[1]−X[2])
/// Y[2] = (X[0] − (X[1]+X[2])/2) + i·(√3/2)·(X[1]−X[2])
/// ```
/// Conjugate (flip sign on imaginary twiddle component) for inverse.
///
/// **Real multiplications**: 4 (two by C3=−½ on re/im of s, two by S3=√3/2
/// on re/im of id). Matches Winograd's lower bound for DFT-3.
/// **Complex additions**: 6.
///
/// References: Winograd (1978), Blahut (2010) §3.2.
#[inline]
pub(crate) fn dft3_impl<F: WinogradScalar>(data: &mut [num_complex::Complex<F>], inverse: bool) {
    debug_assert!(data.len() >= 3);
    let s = F::cast_f64(0.8660254037844386);
    let w_r = F::cast_f64(-0.5);
    let w_i = if inverse { s } else { -s };
    let t1 = data[1] + data[2];
    let m0 = data[0] + t1 * w_r;
    let m1 = (data[1] - data[2]) * num_complex::Complex::new(F::cast_f64(0.0), w_i);
    data[0] += t1;
    data[1] = m0 + m1;
    data[2] = m0 - m1;
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
#[inline]
pub(crate) fn dft5_impl<F: WinogradScalar>(data: &mut [num_complex::Complex<F>], inverse: bool) {
    debug_assert!(data.len() >= 5);
    let c1 = F::cast_f64(0.30901699437494745);
    let c2 = F::cast_f64(-0.8090169943749475);
    let s1 = F::cast_f64(0.9510565162951535);
    let s2 = F::cast_f64(0.5877852522924731);
    let sign = if inverse {
        F::cast_f64(1.0)
    } else {
        F::cast_f64(-1.0)
    };
    let s1 = s1 * sign;
    let s2 = s2 * sign;
    let t1 = data[1] + data[4];
    let t2 = data[1] - data[4];
    let t3 = data[2] + data[3];
    let t4 = data[2] - data[3];
    let m0 = data[0] + t1 + t3;
    let m1 = t1 * c1 + t3 * c2;
    let m2 = t1 * c2 + t3 * c1;
    let m3 = num_complex::Complex::new(F::cast_f64(0.0), F::cast_f64(1.0)) * (t2 * s1 + t4 * s2);
    let m4 = num_complex::Complex::new(F::cast_f64(0.0), F::cast_f64(1.0)) * (t2 * s2 - t4 * s1);
    let s1_add = data[0] + m1;
    let s2_add = data[0] + m2;
    data[0] = m0;
    data[1] = s1_add + m3;
    data[4] = s1_add - m3;
    data[2] = s2_add + m4;
    data[3] = s2_add - m4;
}

