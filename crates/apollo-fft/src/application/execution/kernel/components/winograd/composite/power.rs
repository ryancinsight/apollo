use super::super::radix::dft8_values;
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

#[inline(always)]
fn twiddle16<F: WinogradScalar, const INVERSE: bool, const K: usize>() -> num_complex::Complex<F> {
    let w = TWIDDLE16_FWD[K];
    let w = if INVERSE {
        Complex64::new(w.re, -w.im)
    } else {
        w
    };
    cast_twiddle(w)
}

#[inline(always)]
fn cast_twiddle<F: WinogradScalar>(w: Complex64) -> num_complex::Complex<F> {
    num_complex::Complex::new(F::from_precise(w.re), F::from_precise(w.im))
}

#[inline(always)]
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

#[inline(always)]
pub(crate) fn dft16_values<F: WinogradScalar, const INVERSE: bool>(
    x0: num_complex::Complex<F>,
    x1: num_complex::Complex<F>,
    x2: num_complex::Complex<F>,
    x3: num_complex::Complex<F>,
    x4: num_complex::Complex<F>,
    x5: num_complex::Complex<F>,
    x6: num_complex::Complex<F>,
    x7: num_complex::Complex<F>,
    x8: num_complex::Complex<F>,
    x9: num_complex::Complex<F>,
    x10: num_complex::Complex<F>,
    x11: num_complex::Complex<F>,
    x12: num_complex::Complex<F>,
    x13: num_complex::Complex<F>,
    x14: num_complex::Complex<F>,
    x15: num_complex::Complex<F>,
) -> (
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
    num_complex::Complex<F>,
) {
    let (e0, e1, e2, e3, e4, e5, e6, e7) =
        dft8_values::<F, INVERSE>(x0, x2, x4, x6, x8, x10, x12, x14);
    let (o0, o1, o2, o3, o4, o5, o6, o7) =
        dft8_values::<F, INVERSE>(x1, x3, x5, x7, x9, x11, x13, x15);
    let t0 = apply_twiddle16::<F, INVERSE, 0>(o0);
    let t1 = apply_twiddle16::<F, INVERSE, 1>(o1);
    let t2 = apply_twiddle16::<F, INVERSE, 2>(o2);
    let t3 = apply_twiddle16::<F, INVERSE, 3>(o3);
    let t4 = apply_twiddle16::<F, INVERSE, 4>(o4);
    let t5 = apply_twiddle16::<F, INVERSE, 5>(o5);
    let t6 = apply_twiddle16::<F, INVERSE, 6>(o6);
    let t7 = apply_twiddle16::<F, INVERSE, 7>(o7);
    (
        e0 + t0,
        e1 + t1,
        e2 + t2,
        e3 + t3,
        e4 + t4,
        e5 + t5,
        e6 + t6,
        e7 + t7,
        e0 - t0,
        e1 - t1,
        e2 - t2,
        e3 - t3,
        e4 - t4,
        e5 - t5,
        e6 - t6,
        e7 - t7,
    )
}

/// In-place Winograd DFT-16.
#[inline(always)]
pub(crate) fn dft16_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>; 16],
) {
    let (y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15) =
        dft16_values::<F, INVERSE>(
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
            data[9], data[10], data[11], data[12], data[13], data[14], data[15],
        );
    data[0] = y0;
    data[1] = y1;
    data[2] = y2;
    data[3] = y3;
    data[4] = y4;
    data[5] = y5;
    data[6] = y6;
    data[7] = y7;
    data[8] = y8;
    data[9] = y9;
    data[10] = y10;
    data[11] = y11;
    data[12] = y12;
    data[13] = y13;
    data[14] = y14;
    data[15] = y15;
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

#[inline(always)]
fn twiddle32<F: WinogradScalar, const INVERSE: bool, const K: usize>() -> num_complex::Complex<F> {
    let w = TWIDDLE32_FWD[K];
    let w = if INVERSE {
        Complex64::new(w.re, -w.im)
    } else {
        w
    };
    cast_twiddle(w)
}

#[inline(always)]
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

/// In-place Winograd DFT-32.
#[inline(always)]
pub(crate) fn dft32_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>; 32],
) {
    let (e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15) =
        dft16_values::<F, INVERSE>(
            data[0], data[2], data[4], data[6], data[8], data[10], data[12], data[14], data[16],
            data[18], data[20], data[22], data[24], data[26], data[28], data[30],
        );
    let (o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15) =
        dft16_values::<F, INVERSE>(
            data[1], data[3], data[5], data[7], data[9], data[11], data[13], data[15], data[17],
            data[19], data[21], data[23], data[25], data[27], data[29], data[31],
        );
    let t0 = apply_twiddle32::<F, INVERSE, 0>(o0);
    let t1 = apply_twiddle32::<F, INVERSE, 1>(o1);
    let t2 = apply_twiddle32::<F, INVERSE, 2>(o2);
    let t3 = apply_twiddle32::<F, INVERSE, 3>(o3);
    let t4 = apply_twiddle32::<F, INVERSE, 4>(o4);
    let t5 = apply_twiddle32::<F, INVERSE, 5>(o5);
    let t6 = apply_twiddle32::<F, INVERSE, 6>(o6);
    let t7 = apply_twiddle32::<F, INVERSE, 7>(o7);
    let t8 = apply_twiddle32::<F, INVERSE, 8>(o8);
    let t9 = apply_twiddle32::<F, INVERSE, 9>(o9);
    let t10 = apply_twiddle32::<F, INVERSE, 10>(o10);
    let t11 = apply_twiddle32::<F, INVERSE, 11>(o11);
    let t12 = apply_twiddle32::<F, INVERSE, 12>(o12);
    let t13 = apply_twiddle32::<F, INVERSE, 13>(o13);
    let t14 = apply_twiddle32::<F, INVERSE, 14>(o14);
    let t15 = apply_twiddle32::<F, INVERSE, 15>(o15);

    data[0] = e0 + t0;
    data[16] = e0 - t0;
    data[1] = e1 + t1;
    data[17] = e1 - t1;
    data[2] = e2 + t2;
    data[18] = e2 - t2;
    data[3] = e3 + t3;
    data[19] = e3 - t3;
    data[4] = e4 + t4;
    data[20] = e4 - t4;
    data[5] = e5 + t5;
    data[21] = e5 - t5;
    data[6] = e6 + t6;
    data[22] = e6 - t6;
    data[7] = e7 + t7;
    data[23] = e7 - t7;
    data[8] = e8 + t8;
    data[24] = e8 - t8;
    data[9] = e9 + t9;
    data[25] = e9 - t9;
    data[10] = e10 + t10;
    data[26] = e10 - t10;
    data[11] = e11 + t11;
    data[27] = e11 - t11;
    data[12] = e12 + t12;
    data[28] = e12 - t12;
    data[13] = e13 + t13;
    data[29] = e13 - t13;
    data[14] = e14 + t14;
    data[30] = e14 - t14;
    data[15] = e15 + t15;
    data[31] = e15 - t15;
}

#[inline(always)]
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
#[inline(always)]
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

/// In-place Winograd DFT-64.
#[inline(always)]
pub(crate) fn dft64_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>; 64],
) {
    let mut scratch = unsafe {
        let mut arr: std::mem::MaybeUninit<[num_complex::Complex<F>; 64]> = std::mem::MaybeUninit::uninit();
        let ptr = arr.as_mut_ptr() as *mut num_complex::Complex<F>;
        for i in 0..32 {
            std::ptr::write(ptr.add(i), data[2 * i]);
            std::ptr::write(ptr.add(i + 32), data[2 * i + 1]);
        }
        arr.assume_init()
    };
    let (even, odd) = scratch.split_at_mut(32);
    dft32_impl::<F, INVERSE>(even.try_into().unwrap());
    dft32_impl::<F, INVERSE>(odd.try_into().unwrap());
    for k in 0..32 {
        let o = apply_twiddle_impl(odd[k], twiddle64::<F, INVERSE>(k));
        data[k] = even[k] + o;
        data[k + 32] = even[k] - o;
    }
}

/// Twiddle factor W_128^k using quarter-size table lookup.
/// For even k (k=2*m), returns W_64^m directly.
/// For odd k (k=2*m+1), returns W_128^(2m+1) = W_64^m * W_128^1.
/// LLVM optimizes `k >> 1` and `if (k & 1) == 0` at -O3 for runtime k.
#[inline(always)]
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

/// In-place Winograd DFT-128.
#[inline(always)]
pub(crate) fn dft128_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [num_complex::Complex<F>; 128],
) {
    let mut scratch = unsafe {
        let mut arr: std::mem::MaybeUninit<[num_complex::Complex<F>; 128]> = std::mem::MaybeUninit::uninit();
        let ptr = arr.as_mut_ptr() as *mut num_complex::Complex<F>;
        for i in 0..64 {
            std::ptr::write(ptr.add(i), data[2 * i]);
            std::ptr::write(ptr.add(i + 64), data[2 * i + 1]);
        }
        arr.assume_init()
    };
    let (even, odd) = scratch.split_at_mut(64);
    dft64_impl::<F, INVERSE>(even.try_into().unwrap());
    dft64_impl::<F, INVERSE>(odd.try_into().unwrap());
    for k in 0..64 {
        let o = apply_twiddle_impl(odd[k], twiddle128::<F, INVERSE>(k));
        data[k] = even[k] + o;
        data[k + 64] = even[k] - o;
    }
}
