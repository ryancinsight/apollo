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
    let data: &mut [num_complex::Complex<F>; 4] =
        (&mut data[..4]).try_into().expect("length checked");
    dft4_array_impl(data, inverse);
}

#[inline(always)]
pub(crate) fn dft4_array_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 4],
    inverse: bool,
) {
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
pub(crate) fn dft7_impl<F: WinogradScalar>(data: &mut [num_complex::Complex<F>; 7], inverse: bool) {
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
    let c1 = F::cast_f64(0.6234898018587336);
    let c2 = F::cast_f64(-0.2225209339563144);
    let c3 = F::cast_f64(-0.9009688679024191);
    let s1 = F::cast_f64(0.7818314824680298);
    let s2 = F::cast_f64(0.9749279121818236);
    let s3 = F::cast_f64(0.4338837391175582);
    // sign = +1 inverse (add sine terms), −1 forward (subtract)
    let sign = if inverse {
        F::cast_f64(1.0)
    } else {
        F::cast_f64(-1.0)
    };
    let sixi1 = ixi1 * sign;
    let sixi2 = ixi2 * sign;
    let sixi3 = ixi3 * sign;
    let x0 = data[0];
    data[0] = x0 + xr1 + xr2 + xr3;
    let re1 = x0 + xr1 * c1 + xr2 * c2 + xr3 * c3;
    let re2 = x0 + xr1 * c2 + xr2 * c3 + xr3 * c1;
    let re3 = x0 + xr1 * c3 + xr2 * c1 + xr3 * c2;
    let d1 = sixi1 * s1 + sixi2 * s2 + sixi3 * s3;
    let d2 = sixi1 * s2 - sixi2 * s3 - sixi3 * s1;
    let d3 = sixi1 * s3 - sixi2 * s1 + sixi3 * s2;
    data[1] = re1 + d1;
    data[6] = re1 - d1;
    data[2] = re2 + d2;
    data[5] = re2 - d2;
    data[3] = re3 + d3;
    data[4] = re3 - d3;
}

const DFT11_COS: [[f64; 5]; 5] = [
    [
        0.8412535328311812,
        0.41541501300188644,
        -0.142314838273285,
        -0.654860733945285,
        -0.9594929736144974,
    ],
    [
        0.41541501300188644,
        -0.654860733945285,
        -0.9594929736144975,
        -0.14231483827328523,
        0.8412535328311812,
    ],
    [
        -0.142314838273285,
        -0.9594929736144975,
        0.41541501300188605,
        0.8412535328311815,
        -0.6548607339452839,
    ],
    [
        -0.654860733945285,
        -0.14231483827328523,
        0.8412535328311815,
        -0.9594929736144974,
        0.4154150130018866,
    ],
    [
        -0.9594929736144974,
        0.8412535328311812,
        -0.6548607339452839,
        0.4154150130018866,
        -0.1423148382732854,
    ],
];

const DFT11_SIN: [[f64; 5]; 5] = [
    [
        0.5406408174555976,
        0.9096319953545183,
        0.9898214418809328,
        0.7557495743542583,
        0.28173255684142967,
    ],
    [
        0.9096319953545183,
        0.7557495743542583,
        -0.2817325568414294,
        -0.9898214418809327,
        -0.5406408174555974,
    ],
    [
        0.9898214418809328,
        -0.2817325568414294,
        -0.9096319953545186,
        0.5406408174555971,
        0.7557495743542594,
    ],
    [
        0.7557495743542583,
        -0.9898214418809327,
        0.5406408174555971,
        0.2817325568414299,
        -0.9096319953545183,
    ],
    [
        0.28173255684142967,
        -0.5406408174555974,
        0.7557495743542594,
        -0.9096319953545183,
        0.9898214418809327,
    ],
];

/// In-place odd-prime DFT using paired twiddle symmetry.
///
/// For `N = 2H + 1`, pairs `(x[m], x[N-m])` share cosine terms and opposite
/// sine terms. This avoids Rader setup for small primes while keeping a single
/// monomorphized implementation parameterized by `N` and `H`.
#[inline(always)]
fn dft_odd_prime_impl<F: WinogradScalar, const N: usize, const H: usize>(
    data: &mut [num_complex::Complex<F>],
    inverse: bool,
    cos: &[[f64; H]; H],
    sin: &[[f64; H]; H],
) {
    debug_assert!(data.len() >= N);
    debug_assert_eq!(N, 2 * H + 1);
    let zero = F::zero();
    let x0 = data[0];
    let mut sums = [num_complex::Complex::new(zero, zero); H];
    let mut idiffs = [num_complex::Complex::new(zero, zero); H];

    let mut m = 0usize;
    while m < H {
        let a = data[m + 1];
        let b = data[N - 1 - m];
        sums[m] = num_complex::Complex::new(a.re + b.re, a.im + b.im);
        let diff_re = a.re - b.re;
        let diff_im = a.im - b.im;
        idiffs[m] = num_complex::Complex::new(-diff_im, diff_re);
        m += 1;
    }

    let mut y0_re = x0.re;
    let mut y0_im = x0.im;
    m = 0;
    while m < H {
        y0_re = y0_re + sums[m].re;
        y0_im = y0_im + sums[m].im;
        m += 1;
    }
    data[0] = num_complex::Complex::new(y0_re, y0_im);

    let sign = if inverse {
        F::cast_f64(1.0)
    } else {
        F::cast_f64(-1.0)
    };
    let mut k = 0usize;
    while k < H {
        let mut even_re = x0.re;
        let mut even_im = x0.im;
        let mut odd_re = zero;
        let mut odd_im = zero;
        m = 0;
        while m < H {
            let c = F::cast_f64(cos[k][m]);
            let s = F::cast_f64(sin[k][m]) * sign;
            even_re = even_re + sums[m].re * c;
            even_im = even_im + sums[m].im * c;
            odd_re = odd_re + idiffs[m].re * s;
            odd_im = odd_im + idiffs[m].im * s;
            m += 1;
        }
        data[k + 1] = num_complex::Complex::new(even_re + odd_re, even_im + odd_im);
        data[N - 1 - k] = num_complex::Complex::new(even_re - odd_re, even_im - odd_im);
        k += 1;
    }
}

#[inline(always)]
pub(crate) fn dft11_impl<F: WinogradScalar>(data: &mut [num_complex::Complex<F>], inverse: bool) {
    dft_odd_prime_impl::<F, 11, 5>(data, inverse, &DFT11_COS, &DFT11_SIN);
}

pub(crate) mod dft13;
pub(crate) use dft13::dft13_impl;

pub(crate) mod dft17;
pub(crate) use dft17::{dft17_impl, dft17_inline_impl};

pub(crate) mod dft23;
pub(crate) use dft23::{dft23_impl, dft23_inline_impl};

pub(crate) mod dft3;
pub(crate) use dft3::dft3_impl;

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
pub(crate) fn dft15_impl<F: WinogradScalar>(data: &mut [num_complex::Complex<F>], inverse: bool) {
    debug_assert!(data.len() >= 15);
    // Input CRT permutation: n_idx = (5·i1 + 3·i2) mod 15.
    let mut grid: [num_complex::Complex<F>; 15] = core::array::from_fn(|idx| {
        let i1 = idx / 5;
        let i2 = idx % 5;
        data[(5 * i1 + 3 * i2) % 15]
    });
    // 3 rows of DFT-5 (no twiddles needed — coprime factors).
    dft5_impl(&mut grid[0..5], inverse);
    dft5_impl(&mut grid[5..10], inverse);
    dft5_impl(&mut grid[10..15], inverse);
    // Transpose 3×5 → 5×3 into a second stack buffer.
    let mut grid2: [num_complex::Complex<F>; 15] = core::array::from_fn(|idx| {
        let i2 = idx / 3;
        let i1 = idx % 3;
        grid[i1 * 5 + i2]
    });
    // 5 columns of DFT-3 (now contiguous rows after transpose).
    dft3_impl(&mut grid2[0..3], inverse);
    dft3_impl(&mut grid2[3..6], inverse);
    dft3_impl(&mut grid2[6..9], inverse);
    dft3_impl(&mut grid2[9..12], inverse);
    dft3_impl(&mut grid2[12..15], inverse);
    // Output CRT permutation: k_idx = (10·k1 + 6·k2) mod 15.
    for k1 in 0..3_usize {
        for k2 in 0..5_usize {
            data[(10 * k1 + 6 * k2) % 15] = grid2[k2 * 3 + k1];
        }
    }
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
pub(crate) fn dft5_impl<F: WinogradScalar>(data: &mut [num_complex::Complex<F>], inverse: bool) {
    debug_assert!(data.len() >= 5);
    let [y0, y1, y2, y3, y4] = dft5_values([data[0], data[1], data[2], data[3], data[4]], inverse);
    data[0] = y0;
    data[1] = y1;
    data[2] = y2;
    data[3] = y3;
    data[4] = y4;
}

#[inline(always)]
pub(crate) fn dft5_array_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 5],
    inverse: bool,
) {
    *data = dft5_values(*data, inverse);
}

#[inline(always)]
fn dft5_values<F: WinogradScalar>(
    data: [num_complex::Complex<F>; 5],
    inverse: bool,
) -> [num_complex::Complex<F>; 5] {
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
