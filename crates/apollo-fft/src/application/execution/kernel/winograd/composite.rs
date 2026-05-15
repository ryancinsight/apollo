mod good_thomas;
mod large;
mod medium;
mod small;
mod xlarge;
mod xlarge2;

pub(crate) use good_thomas::dft100_impl;
pub(crate) use large::{dft33_impl, dft35_impl, dft40_impl, dft49_impl, dft50_impl, dft56_impl};
pub(crate) use medium::{
    dft18_impl, dft22_impl, dft28_impl, dft30_impl, dft36_impl, dft42_impl, dft45_impl, dft48_impl,
    dft63_impl,
};
pub(crate) use small::{dft10_impl, dft12_impl, dft14_impl, dft6_impl, dft9_impl};
pub(crate) use xlarge::{
    dft70_impl, dft72_impl, dft75_impl, dft77_impl, dft80_impl, dft84_impl, dft88_impl,
    dft90_impl,
};
pub(crate) use xlarge2::{
    dft96_impl, dft98_impl, dft99_impl, dft105_impl, dft112_impl, dft120_impl, dft126_impl,
};

use super::radix::{dft5_array_impl, dft8_impl};
use super::traits::{apply_twiddle_impl, WinogradScalar};
use num_complex::Complex64;
/// `W_16^k = W_32^(2k)` for `k = 0..7`: every other entry of `TWIDDLE32_FWD`.
/// Forward convention: `(cos(2πk/16), -sin(2πk/16))`.
const TWIDDLE16_FWD: [Complex64; 8] = [
    Complex64::new(1.0, 0.0),
    Complex64::new(0.9238795325112867, -0.3826834323650898),
    Complex64::new(
        std::f64::consts::FRAC_1_SQRT_2,
        -std::f64::consts::FRAC_1_SQRT_2,
    ),
    Complex64::new(0.38268343236508984, -0.9238795325112867),
    Complex64::new(0.0, -1.0),
    Complex64::new(-0.3826834323650897, -0.9238795325112867),
    Complex64::new(
        -std::f64::consts::FRAC_1_SQRT_2,
        -std::f64::consts::FRAC_1_SQRT_2,
    ),
    Complex64::new(-0.9238795325112867, -0.3826834323650899),
];

#[inline(always)]
fn twiddle16<F: WinogradScalar>(k: usize, inverse: bool) -> num_complex::Complex<F> {
    let w = TWIDDLE16_FWD[k];
    let w = if inverse {
        Complex64::new(w.re, -w.im)
    } else {
        w
    };
    cast_twiddle(w)
}

fn cast_twiddle<F: WinogradScalar>(w: Complex64) -> num_complex::Complex<F> {
    num_complex::Complex::new(F::cast_f64(w.re), F::cast_f64(w.im))
}

/// In-place Winograd DFT-16.
///
/// **Decomposition**: DFT-16 = 2 × DFT-8 (stride-2 DIT) + 16-point twiddle
/// butterfly.
///
/// **Multiplications**: 2 × (DFT-8 ops) + 12 real twiddle mults (k=1..7
/// excluding k=0 trivial, k=4 = ×(-i) free, effectively 10 irrational mults).
///
/// Correctness: Van Loan (1992), §3.3.
#[inline]
pub(crate) fn dft16_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 16],
    inverse: bool,
) {
    // Step 1: two DFT-8 sub-transforms on even and odd sub-arrays.
    let mut even = [
        data[0], data[2], data[4], data[6], data[8], data[10], data[12], data[14],
    ];
    let mut odd = [
        data[1], data[3], data[5], data[7], data[9], data[11], data[13], data[15],
    ];
    dft8_impl(&mut even, inverse);
    dft8_impl(&mut odd, inverse);

    // Step 2: apply W_16^k twiddles to odd outputs and butterfly.
    for k in 0..8 {
        let o = apply_twiddle_impl(odd[k], twiddle16(k, inverse));
        data[k] = even[k] + o;
        data[k + 8] = even[k] - o;
    }
}

// ── DFT-32 butterfly ─────────────────────────────────────────────────────────

const TWIDDLE32_FWD: [Complex64; 16] = [
    Complex64::new(1.0, 0.0),
    Complex64::new(0.9807852804032304, -0.19509032201612825),
    Complex64::new(0.9238795325112867, -0.3826834323650898),
    Complex64::new(0.8314696123025452, -0.5555702330196022),
    Complex64::new(
        std::f64::consts::FRAC_1_SQRT_2,
        -std::f64::consts::FRAC_1_SQRT_2,
    ),
    Complex64::new(0.5555702330196023, -0.8314696123025452),
    Complex64::new(0.38268343236508984, -0.9238795325112867),
    Complex64::new(0.19509032201612833, -0.9807852804032304),
    Complex64::new(0.0, -1.0),
    Complex64::new(-0.1950903220161282, -0.9807852804032304),
    Complex64::new(-0.3826834323650897, -0.9238795325112867),
    Complex64::new(-0.555570233019602, -0.8314696123025455),
    Complex64::new(
        -std::f64::consts::FRAC_1_SQRT_2,
        -std::f64::consts::FRAC_1_SQRT_2,
    ),
    Complex64::new(-0.8314696123025453, -0.5555702330196022),
    Complex64::new(-0.9238795325112867, -0.3826834323650899),
    Complex64::new(-0.9807852804032304, -0.1950903220161286),
];

#[inline]
fn twiddle32<F: WinogradScalar>(k: usize, inverse: bool) -> num_complex::Complex<F> {
    let w = TWIDDLE32_FWD[k];
    let w = if inverse {
        Complex64::new(w.re, -w.im)
    } else {
        w
    };
    cast_twiddle(w)
}

/// In-place Winograd DFT-32.
///
/// **Decomposition**: DFT-32 = 2 × DFT-16 (stride-2 DIT) + 32-point twiddle
/// butterfly.
///
/// **Multiplications**: 2 × (DFT-16 ops) + 28 twiddle mults (k=1..15 minus
/// the trivial/free roots).
///
/// Correctness: Van Loan (1992), §3.3 recursive formulation.
#[inline]
pub(crate) fn dft32_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 32],
    inverse: bool,
) {
    let mut even = [
        data[0], data[2], data[4], data[6], data[8], data[10], data[12], data[14], data[16],
        data[18], data[20], data[22], data[24], data[26], data[28], data[30],
    ];
    let mut odd = [
        data[1], data[3], data[5], data[7], data[9], data[11], data[13], data[15], data[17],
        data[19], data[21], data[23], data[25], data[27], data[29], data[31],
    ];
    dft16_impl(&mut even, inverse);
    dft16_impl(&mut odd, inverse);
    for k in 0..16 {
        let o = apply_twiddle_impl(odd[k], twiddle32(k, inverse));
        data[k] = even[k] + o;
        data[k + 16] = even[k] - o;
    }
}

// ── DFT-64 butterfly ─────────────────────────────────────────────────────────

#[inline]
fn twiddle64<F: WinogradScalar>(k: usize, inverse: bool) -> num_complex::Complex<F> {
    // W_64^(2m) = W_32^m and W_64^(2m+1) = W_32^m * W_64^1.
    // This avoids per-call trig evaluation and avoids lock checks in hot loops.
    let base = twiddle32(k >> 1, inverse);
    if (k & 1) == 0 {
        base
    } else {
        // cos(pi/32) ± i*sin(pi/32)
        let w1 = if inverse {
            Complex64::new(0.9951847266721969, 0.0980171403295606)
        } else {
            Complex64::new(0.9951847266721969, -0.0980171403295606)
        };
        apply_twiddle_impl(base, cast_twiddle(w1))
    }
}

/// In-place Winograd DFT-64.
///
/// **Decomposition**: DFT-64 = 2 × DFT-32 (stride-2 DIT) + 64-point twiddle
/// butterfly.
///
/// **Multiplications**: 2 × (DFT-32 ops) + 60 twiddle mults (k=1..31 minus
/// the trivial/free roots).
///
/// Correctness: Van Loan (1992), §3.3 recursive formulation.
#[inline]
pub(crate) fn dft64_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 64],
    inverse: bool,
) {
    let mut even = core::array::from_fn(|i| data[2 * i]);
    let mut odd = core::array::from_fn(|i| data[2 * i + 1]);
    dft32_impl(&mut even, inverse);
    dft32_impl(&mut odd, inverse);
    for k in 0..32 {
        let o = apply_twiddle_impl(odd[k], twiddle64(k, inverse));
        data[k] = even[k] + o;
        data[k + 32] = even[k] - o;
    }
}

// ── DFT-128 butterfly ────────────────────────────────────────────────────────

#[inline]
fn twiddle128<F: WinogradScalar>(k: usize, inverse: bool) -> num_complex::Complex<F> {
    let base = twiddle64(k >> 1, inverse);
    if (k & 1) == 0 {
        base
    } else {
        let w1 = if inverse {
            Complex64::new(0.9987954562051724, 0.049067674327418015)
        } else {
            Complex64::new(0.9987954562051724, -0.049067674327418015)
        };
        apply_twiddle_impl(base, cast_twiddle(w1))
    }
}

/// In-place Winograd DFT-128.
///
/// **Decomposition**: DFT-128 = 2 × DFT-64 (stride-2 DIT) + 128-point twiddle
/// butterfly.
#[inline]
pub(crate) fn dft128_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 128],
    inverse: bool,
) {
    let mut even = core::array::from_fn(|i| data[2 * i]);
    let mut odd = core::array::from_fn(|i| data[2 * i + 1]);
    dft64_impl(&mut even, inverse);
    dft64_impl(&mut odd, inverse);
    for k in 0..64 {
        let o = apply_twiddle_impl(odd[k], twiddle128(k, inverse));
        data[k] = even[k] + o;
        data[k + 64] = even[k] - o;
    }
}

// ── DFT-25 butterfly ─────────────────────────────────────────────────────────

/// `W_25^k = exp(-2πik/25)` for `k = 0..16` (forward convention).
///
/// Only k=0..16 are needed: the 5×5 Cooley-Tukey twiddle grid has exponents
/// n1·k2 for n1,k2 ∈ 0..4, giving max exponent 16.
/// Values: `(cos(2πk/25), -sin(2πk/25))`.
const TWIDDLE25_FWD: [Complex64; 17] = [
    Complex64::new(1.0, 0.0),
    Complex64::new(0.9685831611286311, -0.2486898871648548),
    Complex64::new(0.8763066800438636, -0.48175367410171534),
    Complex64::new(0.7289686274214116, -0.6845471059286886),
    Complex64::new(0.5358267949789967, -0.8443279255020151),
    Complex64::new(0.30901699437494742, -0.9510565162951535),
    Complex64::new(0.06279051952931337, -0.9980267284282716),
    Complex64::new(-0.18374951781657033, -0.9829730996839018),
    Complex64::new(-0.42577929156507266, -0.9048270524660195),
    Complex64::new(-0.6374239897486896, -0.7705132427757894),
    Complex64::new(-0.8090169943749474, -0.5877852522924731),
    Complex64::new(-0.9297765375436513, -0.36812455268467784),
    Complex64::new(-0.9921147013144779, -0.12533323356430426),
    Complex64::new(-0.9921147013144779, 0.12533323356430426),
    Complex64::new(-0.9297765375436513, 0.36812455268467784),
    Complex64::new(-0.8090169943749474, 0.5877852522924731),
    Complex64::new(-0.6374239897486896, 0.7705132427757894),
];

#[inline(always)]
fn twiddle25_exp<F: WinogradScalar, const INVERSE: bool>(exp: usize) -> num_complex::Complex<F> {
    debug_assert!(exp < TWIDDLE25_FWD.len());
    let w = TWIDDLE25_FWD[exp];
    let w = if INVERSE {
        Complex64::new(w.re, -w.im)
    } else {
        w
    };
    cast_twiddle(w)
}

#[inline(always)]
fn apply_twiddle25_exp<F: WinogradScalar, const INVERSE: bool>(
    v: num_complex::Complex<F>,
    exp: usize,
) -> num_complex::Complex<F> {
    if exp == 0 {
        v
    } else {
        apply_twiddle_impl(v, twiddle25_exp::<F, INVERSE>(exp))
    }
}

#[inline(always)]
fn dft25_column_exp<F: WinogradScalar, const INVERSE: bool>(
    rows: &[[num_complex::Complex<F>; 5]; 5],
    data: &mut [num_complex::Complex<F>; 25],
    k2: usize,
) {
    let mut col = [
        rows[0][k2],
        apply_twiddle25_exp::<F, INVERSE>(rows[1][k2], k2),
        apply_twiddle25_exp::<F, INVERSE>(rows[2][k2], 2 * k2),
        apply_twiddle25_exp::<F, INVERSE>(rows[3][k2], 3 * k2),
        apply_twiddle25_exp::<F, INVERSE>(rows[4][k2], 4 * k2),
    ];
    dft5_array_impl(&mut col, INVERSE);
    data[k2] = col[0];
    data[5 + k2] = col[1];
    data[10 + k2] = col[2];
    data[15 + k2] = col[3];
    data[20 + k2] = col[4];
}

/// In-place DFT-25.
///
/// **Decomposition**: radix-5 DIT (decimation in time).
/// Split x into 5 stride-5 sub-sequences: sub[n1] = {x[n1+5n2]}_{n2=0..4}.
/// Apply DFT-5 to each → G[n1,k2]. Multiply by W_25^{n1·k2} twiddle.
/// Apply DFT-5 along columns (n1 dim, k2 fixed) → Y[k1,k2].
/// Output: data[k1·5+k2] = Y[k1,k2] = y[k] (natural DFT output order).
#[inline]
pub(crate) fn dft25_impl<F: WinogradScalar>(
    data: &mut [num_complex::Complex<F>; 25],
    inverse: bool,
) {
    if inverse {
        dft25_impl_const::<F, true>(data);
    } else {
        dft25_impl_const::<F, false>(data);
    }
}

#[inline(always)]
pub(super) fn dft25_impl_const<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>; 25],
) {
    let mut rows = [
        [data[0], data[5], data[10], data[15], data[20]],
        [data[1], data[6], data[11], data[16], data[21]],
        [data[2], data[7], data[12], data[17], data[22]],
        [data[3], data[8], data[13], data[18], data[23]],
        [data[4], data[9], data[14], data[19], data[24]],
    ];
    dft5_array_impl(&mut rows[0], INVERSE);
    dft5_array_impl(&mut rows[1], INVERSE);
    dft5_array_impl(&mut rows[2], INVERSE);
    dft5_array_impl(&mut rows[3], INVERSE);
    dft5_array_impl(&mut rows[4], INVERSE);

    dft25_column_exp::<F, INVERSE>(&rows, data, 0);
    dft25_column_exp::<F, INVERSE>(&rows, data, 1);
    dft25_column_exp::<F, INVERSE>(&rows, data, 2);
    dft25_column_exp::<F, INVERSE>(&rows, data, 3);
    dft25_column_exp::<F, INVERSE>(&rows, data, 4);
}
