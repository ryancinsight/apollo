use super::super::radix::dft8_array_impl;
use super::super::traits::{apply_twiddle_impl, WinogradScalar};
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

#[inline]
fn twiddle16<F: WinogradScalar, const INVERSE: bool, const K: usize>() -> num_complex::Complex<F> {
    let w = TWIDDLE16_FWD[K];
    let w = if INVERSE {
        Complex64::new(w.re, -w.im)
    } else {
        w
    };
    cast_twiddle(w)
}

#[inline]
fn cast_twiddle<F: WinogradScalar>(w: Complex64) -> num_complex::Complex<F> {
    num_complex::Complex::new(F::from_precise(w.re), F::from_precise(w.im))
}

#[inline]
fn apply_twiddle16<F: WinogradScalar, const INVERSE: bool, const K: usize>(
    o: num_complex::Complex<F>,
) -> num_complex::Complex<F> {
    let sign = if INVERSE {
        F::from_precise(1.0)
    } else {
        F::from_precise(-1.0)
    };
    match K {
        0 => o,
        4 => {
            if INVERSE {
                num_complex::Complex::new(-o.im, o.re)
            } else {
                num_complex::Complex::new(o.im, -o.re)
            }
        }
        2 => {
            let sq2o2 = F::sq2o2();
            num_complex::Complex::new(sq2o2 * (o.re - sign * o.im), sq2o2 * (sign * o.re + o.im))
        }
        6 => {
            let sq2o2 = F::sq2o2();
            num_complex::Complex::new(sq2o2 * (-o.re - sign * o.im), sq2o2 * (sign * o.re - o.im))
        }
        _ => o * twiddle16::<F, INVERSE, K>(),
    }
}

/// In-place DFT-16 with optional fused 1/N normalization.
///
/// When `NORMALIZE` is true, each output is multiplied by 1/16 during the
/// final write-back, eliminating the separate scale loop in the inverse path.
#[inline]
pub(crate) fn dft16_array_impl<F: WinogradScalar, const INVERSE: bool, const NORMALIZE: bool>(
    data: &mut [num_complex::Complex<F>; 16],
) {
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

    let mut e_arr = [x0, x2, x4, x6, x8, x10, x12, x14];
    dft8_array_impl::<F, INVERSE, false>(&mut e_arr);
    let mut o_arr = [x1, x3, x5, x7, x9, x11, x13, x15];
    dft8_array_impl::<F, INVERSE, false>(&mut o_arr);
    let t0 = apply_twiddle16::<F, INVERSE, 0>(o_arr[0]);
    let t1 = apply_twiddle16::<F, INVERSE, 1>(o_arr[1]);
    let t2 = apply_twiddle16::<F, INVERSE, 2>(o_arr[2]);
    let t3 = apply_twiddle16::<F, INVERSE, 3>(o_arr[3]);
    let t4 = apply_twiddle16::<F, INVERSE, 4>(o_arr[4]);
    let t5 = apply_twiddle16::<F, INVERSE, 5>(o_arr[5]);
    let t6 = apply_twiddle16::<F, INVERSE, 6>(o_arr[6]);
    let t7 = apply_twiddle16::<F, INVERSE, 7>(o_arr[7]);

    if NORMALIZE {
        let q = F::from_precise(1.0 / 16.0);
        data[0] = num_complex::Complex::new((e_arr[0].re + t0.re) * q, (e_arr[0].im + t0.im) * q);
        data[1] = num_complex::Complex::new((e_arr[1].re + t1.re) * q, (e_arr[1].im + t1.im) * q);
        data[2] = num_complex::Complex::new((e_arr[2].re + t2.re) * q, (e_arr[2].im + t2.im) * q);
        data[3] = num_complex::Complex::new((e_arr[3].re + t3.re) * q, (e_arr[3].im + t3.im) * q);
        data[4] = num_complex::Complex::new((e_arr[4].re + t4.re) * q, (e_arr[4].im + t4.im) * q);
        data[5] = num_complex::Complex::new((e_arr[5].re + t5.re) * q, (e_arr[5].im + t5.im) * q);
        data[6] = num_complex::Complex::new((e_arr[6].re + t6.re) * q, (e_arr[6].im + t6.im) * q);
        data[7] = num_complex::Complex::new((e_arr[7].re + t7.re) * q, (e_arr[7].im + t7.im) * q);
        data[8] = num_complex::Complex::new((e_arr[0].re - t0.re) * q, (e_arr[0].im - t0.im) * q);
        data[9] = num_complex::Complex::new((e_arr[1].re - t1.re) * q, (e_arr[1].im - t1.im) * q);
        data[10] = num_complex::Complex::new((e_arr[2].re - t2.re) * q, (e_arr[2].im - t2.im) * q);
        data[11] = num_complex::Complex::new((e_arr[3].re - t3.re) * q, (e_arr[3].im - t3.im) * q);
        data[12] = num_complex::Complex::new((e_arr[4].re - t4.re) * q, (e_arr[4].im - t4.im) * q);
        data[13] = num_complex::Complex::new((e_arr[5].re - t5.re) * q, (e_arr[5].im - t5.im) * q);
        data[14] = num_complex::Complex::new((e_arr[6].re - t6.re) * q, (e_arr[6].im - t6.im) * q);
        data[15] = num_complex::Complex::new((e_arr[7].re - t7.re) * q, (e_arr[7].im - t7.im) * q);
    } else {
        data[0] = e_arr[0] + t0;
        data[1] = e_arr[1] + t1;
        data[2] = e_arr[2] + t2;
        data[3] = e_arr[3] + t3;
        data[4] = e_arr[4] + t4;
        data[5] = e_arr[5] + t5;
        data[6] = e_arr[6] + t6;
        data[7] = e_arr[7] + t7;
        data[8] = e_arr[0] - t0;
        data[9] = e_arr[1] - t1;
        data[10] = e_arr[2] - t2;
        data[11] = e_arr[3] - t3;
        data[12] = e_arr[4] - t4;
        data[13] = e_arr[5] - t5;
        data[14] = e_arr[6] - t6;
        data[15] = e_arr[7] - t7;
    }
}

/// In-place Winograd DFT-16 (public API).
#[inline]
pub(crate) fn dft16_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>; 16],
) {
    dft16_array_impl::<F, INVERSE, false>(data);
}

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
fn twiddle32<F: WinogradScalar, const INVERSE: bool, const K: usize>() -> num_complex::Complex<F> {
    let w = TWIDDLE32_FWD[K];
    let w = if INVERSE {
        Complex64::new(w.re, -w.im)
    } else {
        w
    };
    cast_twiddle(w)
}

#[inline]
fn apply_twiddle32<F: WinogradScalar, const INVERSE: bool, const K: usize>(
    o: num_complex::Complex<F>,
) -> num_complex::Complex<F> {
    let sign = if INVERSE {
        F::from_precise(1.0)
    } else {
        F::from_precise(-1.0)
    };
    match K {
        0 => o,
        8 => {
            if INVERSE {
                num_complex::Complex::new(-o.im, o.re)
            } else {
                num_complex::Complex::new(o.im, -o.re)
            }
        }
        4 => {
            let sq2o2 = F::sq2o2();
            num_complex::Complex::new(sq2o2 * (o.re - sign * o.im), sq2o2 * (sign * o.re + o.im))
        }
        12 => {
            let sq2o2 = F::sq2o2();
            num_complex::Complex::new(sq2o2 * (-o.re - sign * o.im), sq2o2 * (sign * o.re - o.im))
        }
        _ => o * twiddle32::<F, INVERSE, K>(),
    }
}

/// In-place DFT-32 with optional fused 1/N normalization.
///
/// When `NORMALIZE` is true, each output is multiplied by 1/32 during the
/// final write-back, eliminating the separate scale loop in the inverse path.
#[inline]
pub(crate) fn dft32_array_impl<F: WinogradScalar, const INVERSE: bool, const NORMALIZE: bool>(
    data: &mut [num_complex::Complex<F>; 32],
) {
    let mut e_arr = [
        data[0], data[2], data[4], data[6], data[8], data[10], data[12], data[14], data[16],
        data[18], data[20], data[22], data[24], data[26], data[28], data[30],
    ];
    dft16_array_impl::<F, INVERSE, false>(&mut e_arr);
    let mut o_arr = [
        data[1], data[3], data[5], data[7], data[9], data[11], data[13], data[15], data[17],
        data[19], data[21], data[23], data[25], data[27], data[29], data[31],
    ];
    dft16_array_impl::<F, INVERSE, false>(&mut o_arr);
    let t0 = apply_twiddle32::<F, INVERSE, 0>(o_arr[0]);
    let t1 = apply_twiddle32::<F, INVERSE, 1>(o_arr[1]);
    let t2 = apply_twiddle32::<F, INVERSE, 2>(o_arr[2]);
    let t3 = apply_twiddle32::<F, INVERSE, 3>(o_arr[3]);
    let t4 = apply_twiddle32::<F, INVERSE, 4>(o_arr[4]);
    let t5 = apply_twiddle32::<F, INVERSE, 5>(o_arr[5]);
    let t6 = apply_twiddle32::<F, INVERSE, 6>(o_arr[6]);
    let t7 = apply_twiddle32::<F, INVERSE, 7>(o_arr[7]);
    let t8 = apply_twiddle32::<F, INVERSE, 8>(o_arr[8]);
    let t9 = apply_twiddle32::<F, INVERSE, 9>(o_arr[9]);
    let t10 = apply_twiddle32::<F, INVERSE, 10>(o_arr[10]);
    let t11 = apply_twiddle32::<F, INVERSE, 11>(o_arr[11]);
    let t12 = apply_twiddle32::<F, INVERSE, 12>(o_arr[12]);
    let t13 = apply_twiddle32::<F, INVERSE, 13>(o_arr[13]);
    let t14 = apply_twiddle32::<F, INVERSE, 14>(o_arr[14]);
    let t15 = apply_twiddle32::<F, INVERSE, 15>(o_arr[15]);

    if NORMALIZE {
        let q = F::from_precise(1.0 / 32.0);
        data[0] = num_complex::Complex::new((e_arr[0].re + t0.re) * q, (e_arr[0].im + t0.im) * q);
        data[16] = num_complex::Complex::new((e_arr[0].re - t0.re) * q, (e_arr[0].im - t0.im) * q);
        data[1] = num_complex::Complex::new((e_arr[1].re + t1.re) * q, (e_arr[1].im + t1.im) * q);
        data[17] = num_complex::Complex::new((e_arr[1].re - t1.re) * q, (e_arr[1].im - t1.im) * q);
        data[2] = num_complex::Complex::new((e_arr[2].re + t2.re) * q, (e_arr[2].im + t2.im) * q);
        data[18] = num_complex::Complex::new((e_arr[2].re - t2.re) * q, (e_arr[2].im - t2.im) * q);
        data[3] = num_complex::Complex::new((e_arr[3].re + t3.re) * q, (e_arr[3].im + t3.im) * q);
        data[19] = num_complex::Complex::new((e_arr[3].re - t3.re) * q, (e_arr[3].im - t3.im) * q);
        data[4] = num_complex::Complex::new((e_arr[4].re + t4.re) * q, (e_arr[4].im + t4.im) * q);
        data[20] = num_complex::Complex::new((e_arr[4].re - t4.re) * q, (e_arr[4].im - t4.im) * q);
        data[5] = num_complex::Complex::new((e_arr[5].re + t5.re) * q, (e_arr[5].im + t5.im) * q);
        data[21] = num_complex::Complex::new((e_arr[5].re - t5.re) * q, (e_arr[5].im - t5.im) * q);
        data[6] = num_complex::Complex::new((e_arr[6].re + t6.re) * q, (e_arr[6].im + t6.im) * q);
        data[22] = num_complex::Complex::new((e_arr[6].re - t6.re) * q, (e_arr[6].im - t6.im) * q);
        data[7] = num_complex::Complex::new((e_arr[7].re + t7.re) * q, (e_arr[7].im + t7.im) * q);
        data[23] = num_complex::Complex::new((e_arr[7].re - t7.re) * q, (e_arr[7].im - t7.im) * q);
        data[8] = num_complex::Complex::new((e_arr[8].re + t8.re) * q, (e_arr[8].im + t8.im) * q);
        data[24] = num_complex::Complex::new((e_arr[8].re - t8.re) * q, (e_arr[8].im - t8.im) * q);
        data[9] = num_complex::Complex::new((e_arr[9].re + t9.re) * q, (e_arr[9].im + t9.im) * q);
        data[25] = num_complex::Complex::new((e_arr[9].re - t9.re) * q, (e_arr[9].im - t9.im) * q);
        data[10] =
            num_complex::Complex::new((e_arr[10].re + t10.re) * q, (e_arr[10].im + t10.im) * q);
        data[26] =
            num_complex::Complex::new((e_arr[10].re - t10.re) * q, (e_arr[10].im - t10.im) * q);
        data[11] =
            num_complex::Complex::new((e_arr[11].re + t11.re) * q, (e_arr[11].im + t11.im) * q);
        data[27] =
            num_complex::Complex::new((e_arr[11].re - t11.re) * q, (e_arr[11].im - t11.im) * q);
        data[12] =
            num_complex::Complex::new((e_arr[12].re + t12.re) * q, (e_arr[12].im + t12.im) * q);
        data[28] =
            num_complex::Complex::new((e_arr[12].re - t12.re) * q, (e_arr[12].im - t12.im) * q);
        data[13] =
            num_complex::Complex::new((e_arr[13].re + t13.re) * q, (e_arr[13].im + t13.im) * q);
        data[29] =
            num_complex::Complex::new((e_arr[13].re - t13.re) * q, (e_arr[13].im - t13.im) * q);
        data[14] =
            num_complex::Complex::new((e_arr[14].re + t14.re) * q, (e_arr[14].im + t14.im) * q);
        data[30] =
            num_complex::Complex::new((e_arr[14].re - t14.re) * q, (e_arr[14].im - t14.im) * q);
        data[15] =
            num_complex::Complex::new((e_arr[15].re + t15.re) * q, (e_arr[15].im + t15.im) * q);
        data[31] =
            num_complex::Complex::new((e_arr[15].re - t15.re) * q, (e_arr[15].im - t15.im) * q);
    } else {
        data[0] = e_arr[0] + t0;
        data[16] = e_arr[0] - t0;
        data[1] = e_arr[1] + t1;
        data[17] = e_arr[1] - t1;
        data[2] = e_arr[2] + t2;
        data[18] = e_arr[2] - t2;
        data[3] = e_arr[3] + t3;
        data[19] = e_arr[3] - t3;
        data[4] = e_arr[4] + t4;
        data[20] = e_arr[4] - t4;
        data[5] = e_arr[5] + t5;
        data[21] = e_arr[5] - t5;
        data[6] = e_arr[6] + t6;
        data[22] = e_arr[6] - t6;
        data[7] = e_arr[7] + t7;
        data[23] = e_arr[7] - t7;
        data[8] = e_arr[8] + t8;
        data[24] = e_arr[8] - t8;
        data[9] = e_arr[9] + t9;
        data[25] = e_arr[9] - t9;
        data[10] = e_arr[10] + t10;
        data[26] = e_arr[10] - t10;
        data[11] = e_arr[11] + t11;
        data[27] = e_arr[11] - t11;
        data[12] = e_arr[12] + t12;
        data[28] = e_arr[12] - t12;
        data[13] = e_arr[13] + t13;
        data[29] = e_arr[13] - t13;
        data[14] = e_arr[14] + t14;
        data[30] = e_arr[14] - t14;
        data[15] = e_arr[15] + t15;
        data[31] = e_arr[15] - t15;
    }
}

/// In-place Winograd DFT-32 (public API).
#[inline]
pub(crate) fn dft32_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>; 32],
) {
    dft32_array_impl::<F, INVERSE, false>(data);
}

#[inline]
fn twiddle32_runtime<F: WinogradScalar, const INVERSE: bool>(k: usize) -> num_complex::Complex<F> {
    let w = TWIDDLE32_FWD[k];
    let w = if INVERSE {
        Complex64::new(w.re, -w.im)
    } else {
        w
    };
    cast_twiddle(w)
}

/// Twiddle factor W_64^k using half-size table lookup.
/// For even k (k=2*m), returns W_32^m directly.
/// For odd k (k=2*m+1), returns W_64^(2m+1) = W_32^m * W_64^1.
/// LLVM optimizes `k >> 1` and `if (k & 1) == 0` at -O3 for runtime k.
#[inline]
fn twiddle64<F: WinogradScalar, const INVERSE: bool>(k: usize) -> num_complex::Complex<F> {
    let base = twiddle32_runtime::<F, INVERSE>(k >> 1);
    if (k & 1) == 0 {
        base
    } else {
        let w1 = if INVERSE {
            Complex64::new(0.9951847266721969, 0.0980171403295606)
        } else {
            Complex64::new(0.9951847266721969, -0.0980171403295606)
        };
        apply_twiddle_impl(base, cast_twiddle(w1))
    }
}

/// In-place DFT-64 with optional fused 1/N normalization.
///
/// When `NORMALIZE` is true, each output is multiplied by 1/64 during the
/// final write-back, eliminating the separate scale loop in the inverse path.
#[inline]
pub(crate) fn dft64_array_impl<F: WinogradScalar, const INVERSE: bool, const NORMALIZE: bool>(
    data: &mut [num_complex::Complex<F>; 64],
) {
    F::with_winograd_scratch(64, |scratch| {
        let ptr = scratch.as_mut_ptr();
        for i in 0..32 {
            unsafe {
                std::ptr::write(ptr.add(i), data[2 * i]);
                std::ptr::write(ptr.add(i + 32), data[2 * i + 1]);
            }
        }
        let (even, odd) = scratch.split_at_mut(32);
        dft32_array_impl::<F, INVERSE, false>(even.try_into().unwrap());
        dft32_array_impl::<F, INVERSE, false>(odd.try_into().unwrap());
        if NORMALIZE {
            let q = F::from_precise(1.0 / 64.0);
            for k in 0..32 {
                let o = apply_twiddle_impl(odd[k], twiddle64::<F, INVERSE>(k));
                data[k] =
                    num_complex::Complex::new((even[k].re + o.re) * q, (even[k].im + o.im) * q);
                data[k + 32] =
                    num_complex::Complex::new((even[k].re - o.re) * q, (even[k].im - o.im) * q);
            }
        } else {
            for k in 0..32 {
                let o = apply_twiddle_impl(odd[k], twiddle64::<F, INVERSE>(k));
                data[k] = even[k] + o;
                data[k + 32] = even[k] - o;
            }
        }
    });
}

/// In-place Winograd DFT-64 (public API).
#[inline]
pub(crate) fn dft64_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>; 64],
) {
    dft64_array_impl::<F, INVERSE, false>(data);
}

/// Twiddle factor W_128^k using quarter-size table lookup.
/// For even k (k=2*m), returns W_64^m directly.
/// For odd k (k=2*m+1), returns W_128^(2m+1) = W_64^m * W_128^1.
/// LLVM optimizes `k >> 1` and `if (k & 1) == 0` at -O3 for runtime k.
#[inline]
fn twiddle128<F: WinogradScalar, const INVERSE: bool>(k: usize) -> num_complex::Complex<F> {
    let base = twiddle64::<F, INVERSE>(k >> 1);
    if (k & 1) == 0 {
        base
    } else {
        let w1 = if INVERSE {
            Complex64::new(0.9987954562051724, 0.049067674327418015)
        } else {
            Complex64::new(0.9987954562051724, -0.049067674327418015)
        };
        apply_twiddle_impl(base, cast_twiddle(w1))
    }
}

/// In-place DFT-128 with optional fused 1/N normalization.
///
/// When `NORMALIZE` is true, each output is multiplied by 1/128 during the
/// final write-back, eliminating the separate scale loop in the inverse path.
#[inline]
pub(crate) fn dft128_array_impl<F: WinogradScalar, const INVERSE: bool, const NORMALIZE: bool>(
    data: &mut [num_complex::Complex<F>; 128],
) {
    F::with_winograd_scratch(128, |scratch| {
        let ptr = scratch.as_mut_ptr();
        for i in 0..64 {
            unsafe {
                std::ptr::write(ptr.add(i), data[2 * i]);
                std::ptr::write(ptr.add(i + 64), data[2 * i + 1]);
            }
        }
        let (even, odd) = scratch.split_at_mut(64);
        dft64_array_impl::<F, INVERSE, false>(even.try_into().unwrap());
        dft64_array_impl::<F, INVERSE, false>(odd.try_into().unwrap());
        if NORMALIZE {
            let q = F::from_precise(1.0 / 128.0);
            for k in 0..64 {
                let o = apply_twiddle_impl(odd[k], twiddle128::<F, INVERSE>(k));
                data[k] =
                    num_complex::Complex::new((even[k].re + o.re) * q, (even[k].im + o.im) * q);
                data[k + 64] =
                    num_complex::Complex::new((even[k].re - o.re) * q, (even[k].im - o.im) * q);
            }
        } else {
            for k in 0..64 {
                let o = apply_twiddle_impl(odd[k], twiddle128::<F, INVERSE>(k));
                data[k] = even[k] + o;
                data[k + 64] = even[k] - o;
            }
        }
    });
}

/// In-place Winograd DFT-128 (public API).
#[inline]
pub(crate) fn dft128_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>; 128],
) {
    dft128_array_impl::<F, INVERSE, false>(data);
}
