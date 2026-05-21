use super::radix::dft8_impl;
use super::traits::{apply_twiddle_impl, WinogradScalar};
use num_complex::Complex64;
/// Precomputed forward twiddle factors for the 16-point DFT stage.
///
/// `W_16^k = exp(-2πi·k/16)` for `k = 0..7`.
/// Stored as `(cos(2πk/16), -sin(2πk/16))`.
/// Only k=0..7 are needed (k=0 is trivial; k=1..7 are non-trivial).
fn twiddle16_fwd(k: usize) -> Complex64 {
    // Exact values for 16th roots of unity.
    match k {
        0 => Complex64::new(1.0, 0.0),
        1 => {
            // cos(π/8), -sin(π/8)
            let c = (2.0f64 + 2.0f64.sqrt()).sqrt() * 0.5;
            let s = (2.0f64 - 2.0f64.sqrt()).sqrt() * 0.5;
            Complex64::new(c, -s)
        }
        2 => {
            // cos(π/4), -sin(π/4)
            const SQ2O2: f64 = std::f64::consts::FRAC_1_SQRT_2;
            Complex64::new(SQ2O2, -SQ2O2)
        }
        3 => {
            // cos(3π/8), -sin(3π/8)
            let c = (2.0f64 - 2.0f64.sqrt()).sqrt() * 0.5;
            let s = (2.0f64 + 2.0f64.sqrt()).sqrt() * 0.5;
            Complex64::new(c, -s)
        }
        4 => Complex64::new(0.0, -1.0),
        5 => {
            let c = (2.0f64 - 2.0f64.sqrt()).sqrt() * 0.5;
            let s = (2.0f64 + 2.0f64.sqrt()).sqrt() * 0.5;
            Complex64::new(-c, -s)
        }
        6 => {
            const SQ2O2: f64 = std::f64::consts::FRAC_1_SQRT_2;
            Complex64::new(-SQ2O2, -SQ2O2)
        }
        7 => {
            let c = (2.0f64 + 2.0f64.sqrt()).sqrt() * 0.5;
            let s = (2.0f64 - 2.0f64.sqrt()).sqrt() * 0.5;
            Complex64::new(-c, -s)
        }
        _ => unreachable!(),
    }
}

fn twiddle16_inv(k: usize) -> Complex64 {
    let w = twiddle16_fwd(k);
    Complex64::new(w.re, -w.im)
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
        let tw = if inverse {
            cast_twiddle(twiddle16_inv(k))
        } else {
            cast_twiddle(twiddle16_fwd(k))
        };
        let o = apply_twiddle_impl(odd[k], tw);
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
