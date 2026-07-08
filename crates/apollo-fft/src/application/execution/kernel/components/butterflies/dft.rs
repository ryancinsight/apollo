//! Shared small-DFT codelets (canonical zero-cost implementations).
//!
//! These are the common in-place Winograd-style DFT-N kernels for small N
//! (2,3,4,5,7,8 and array variants with optional fused normalize).
//! Moved here from winograd/radix/ and winograd/traits to provide a single
//! authoritative source used by Winograd, Good-Thomas (Cook-Toom), Rader
//! (pair), radix-composite, and future paths.
//!
//! All are generic over MixedRadixScalar (which implies the required
//! WinogradScalar ops via supertrait), monomorphized with zero overhead.
//! Call sites remain inlined where marked.
//!
//! Normalization (when const NORMALIZE=true) fuses the 1/N scale into the
//! final stores for inverse paths, avoiding a separate pass.

use eunomia::Complex;

use crate::application::execution::kernel::components::winograd::traits::WinogradScalar;

/// In-place DFT-2 (trivial add/sub).
#[inline]
pub(crate) fn dft2_impl<F: WinogradScalar>(data: &mut [Complex<F>; 2]) {
    let a = data[0];
    let b = data[1];
    data[0] = a + b;
    data[1] = a - b;
}

/// In-place DFT-3 with optional fused 1/N normalization.
#[inline]
pub(crate) fn dft3_impl<F: WinogradScalar, const INVERSE: bool, const NORMALIZE: bool>(
    data: &mut [Complex<F>],
) {
    debug_assert!(data.len() >= 3);
    let s = F::from_precise(0.8660254037844386);
    let w_r = F::from_precise(-0.5);
    let x0 = data[0];
    let x1 = data[1];
    let x2 = data[2];
    let sum_re = x1.re + x2.re;
    let sum_im = x1.im + x2.im;
    let diff_re = x1.re - x2.re;
    let diff_im = x1.im - x2.im;
    let m0_re = x0.re + sum_re * w_r;
    let m0_im = x0.im + sum_im * w_r;
    let (m1_re, m1_im) = if INVERSE {
        (-diff_im * s, diff_re * s)
    } else {
        (diff_im * s, -diff_re * s)
    };
    if NORMALIZE {
        let q = F::from_precise(1.0 / 3.0);
        data[0] = Complex::new((x0.re + sum_re) * q, (x0.im + sum_im) * q);
        data[1] = Complex::new((m0_re + m1_re) * q, (m0_im + m1_im) * q);
        data[2] = Complex::new((m0_re - m1_re) * q, (m0_im - m1_im) * q);
    } else {
        data[0] = Complex::new(x0.re + sum_re, x0.im + sum_im);
        data[1] = Complex::new(m0_re + m1_re, m0_im + m1_im);
        data[2] = Complex::new(m0_re - m1_re, m0_im - m1_im);
    }
}

/// In-place DFT-4 with optional fused 1/N normalization.
#[inline]
pub(crate) fn dft4_array_impl<F: WinogradScalar, const INVERSE: bool, const NORMALIZE: bool>(
    data: &mut [Complex<F>; 4],
) {
    let t0 = data[0] + data[2];
    let t1 = data[0] - data[2];
    let t2 = data[1] + data[3];
    let t3 = data[1] - data[3];
    let i_t3 = if INVERSE {
        Complex::new(-t3.im, t3.re)
    } else {
        Complex::new(t3.im, -t3.re)
    };
    if NORMALIZE {
        let q = F::from_precise(0.25);
        data[0] = Complex::new((t0.re + t2.re) * q, (t0.im + t2.im) * q);
        data[1] = Complex::new((t1.re + i_t3.re) * q, (t1.im + i_t3.im) * q);
        data[2] = Complex::new((t0.re - t2.re) * q, (t0.im - t2.im) * q);
        data[3] = Complex::new((t1.re - i_t3.re) * q, (t1.im - i_t3.im) * q);
    } else {
        data[0] = t0 + t2;
        data[2] = t0 - t2;
        data[1] = t1 + i_t3;
        data[3] = t1 - i_t3;
    }
}

/// In-place DFT-8 with optional fused 1/N normalization.
#[inline]
pub(crate) fn dft8_array_impl<F: WinogradScalar, const INVERSE: bool, const NORMALIZE: bool>(
    data: &mut [Complex<F>; 8],
) {
    // Read all inputs into locals before any writes.
    let x0 = data[0];
    let x1 = data[1];
    let x2 = data[2];
    let x3 = data[3];
    let x4 = data[4];
    let x5 = data[5];
    let x6 = data[6];
    let x7 = data[7];

    let sq2o2 = F::sq2o2();
    let mut e_arr = [x0, x2, x4, x6];
    dft4_array_impl::<F, INVERSE, false>(&mut e_arr);
    let mut o_arr = [x1, x3, x5, x7];
    dft4_array_impl::<F, INVERSE, false>(&mut o_arr);

    // o2: pure ±i rotation — no multiplication, just sign/swap.
    let o2 = if INVERSE {
        Complex::new(-o_arr[2].im, o_arr[2].re)
    } else {
        Complex::new(o_arr[2].im, -o_arr[2].re)
    };

    // o1, o3: W_8^k twiddles with sign determined by const INVERSE.
    let (o1, o3) = if INVERSE {
        (
            Complex::new(
                sq2o2 * (o_arr[1].re - o_arr[1].im),
                sq2o2 * (o_arr[1].re + o_arr[1].im),
            ),
            Complex::new(
                sq2o2 * (-o_arr[3].re - o_arr[3].im),
                sq2o2 * (o_arr[3].re - o_arr[3].im),
            ),
        )
    } else {
        (
            Complex::new(
                sq2o2 * (o_arr[1].re + o_arr[1].im),
                sq2o2 * (-o_arr[1].re + o_arr[1].im),
            ),
            Complex::new(
                sq2o2 * (-o_arr[3].re + o_arr[3].im),
                sq2o2 * (-o_arr[3].re - o_arr[3].im),
            ),
        )
    };

    if NORMALIZE {
        let q = F::from_precise(0.125);
        data[0] = Complex::new(
            (e_arr[0].re + o_arr[0].re) * q,
            (e_arr[0].im + o_arr[0].im) * q,
        );
        data[1] = Complex::new((e_arr[1].re + o1.re) * q, (e_arr[1].im + o1.im) * q);
        data[2] = Complex::new((e_arr[2].re + o2.re) * q, (e_arr[2].im + o2.im) * q);
        data[3] = Complex::new((e_arr[3].re + o3.re) * q, (e_arr[3].im + o3.im) * q);
        data[4] = Complex::new(
            (e_arr[0].re - o_arr[0].re) * q,
            (e_arr[0].im - o_arr[0].im) * q,
        );
        data[5] = Complex::new((e_arr[1].re - o1.re) * q, (e_arr[1].im - o1.im) * q);
        data[6] = Complex::new((e_arr[2].re - o2.re) * q, (e_arr[2].im - o2.im) * q);
        data[7] = Complex::new((e_arr[3].re - o3.re) * q, (e_arr[3].im - o3.im) * q);
    } else {
        data[0] = e_arr[0] + o_arr[0];
        data[1] = e_arr[1] + o1;
        data[2] = e_arr[2] + o2;
        data[3] = e_arr[3] + o3;
        data[4] = e_arr[0] - o_arr[0];
        data[5] = e_arr[1] - o1;
        data[6] = e_arr[2] - o2;
        data[7] = e_arr[3] - o3;
    }
}

/// In-place DFT-5 with optional fused 1/N normalization.
#[inline]
pub(crate) fn dft5_array_impl<F: WinogradScalar, const INVERSE: bool, const NORMALIZE: bool>(
    data: &mut [Complex<F>; 5],
) {
    let c1 = F::from_precise(0.30901699437494745);
    let c2 = F::from_precise(-0.8090169943749475);
    let s1 = F::from_precise(0.9510565162951535);
    let s2 = F::from_precise(0.5877852522924731);
    let s1 = if INVERSE { s1 } else { -s1 };
    let s2 = if INVERSE { s2 } else { -s2 };
    // Read all inputs into locals before any writes.
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
    if NORMALIZE {
        let q = F::from_precise(0.2);
        data[0] = Complex::new((x0.re + t1_re + t3_re) * q, (x0.im + t1_im + t3_im) * q);
        data[1] = Complex::new((a1_re - q3_im) * q, (a1_im + q3_re) * q);
        data[2] = Complex::new((a2_re - q4_im) * q, (a2_im + q4_re) * q);
        data[3] = Complex::new((a2_re + q4_im) * q, (a2_im - q4_re) * q);
        data[4] = Complex::new((a1_re + q3_im) * q, (a1_im - q3_re) * q);
    } else {
        data[0] = Complex::new(x0.re + t1_re + t3_re, x0.im + t1_im + t3_im);
        data[1] = Complex::new(a1_re - q3_im, a1_im + q3_re);
        data[2] = Complex::new(a2_re - q4_im, a2_im + q4_re);
        data[3] = Complex::new(a2_re + q4_im, a2_im - q4_re);
        data[4] = Complex::new(a1_re + q3_im, a1_im - q3_re);
    }
}

/// In-place Winograd DFT-7 (array form).
#[inline]
pub(crate) fn dft7_impl<F: WinogradScalar, const INVERSE: bool, const NORMALIZE: bool>(
    data: &mut [Complex<F>; 7],
) {
    // Read all inputs into locals before any writes.
    let x0 = data[0];
    let x1 = data[1];
    let x2 = data[2];
    let x3 = data[3];
    let x4 = data[4];
    let x5 = data[5];
    let x6 = data[6];
    let xr1 = x1 + x6;
    let xr2 = x2 + x5;
    let xr3 = x3 + x4;
    let xi1 = x1 - x6;
    let xi2 = x2 - x5;
    let xi3 = x3 - x4;
    // i·xi[n] = (−xi.im, xi.re)
    let ixi1 = Complex::new(-xi1.im, xi1.re);
    let ixi2 = Complex::new(-xi2.im, xi2.re);
    let ixi3 = Complex::new(-xi3.im, xi3.re);
    let c1 = F::from_precise(0.6234898018587336);
    let c2 = F::from_precise(-0.2225209339563144);
    let c3 = F::from_precise(-0.9009688679024191);
    let s1 = F::from_precise(0.7818314824680298);
    let s2 = F::from_precise(0.9749279121818236);
    let s3 = F::from_precise(0.4338837391175582);
    let (sixi1, sixi2, sixi3) = if INVERSE {
        (ixi1, ixi2, ixi3)
    } else {
        (-ixi1, -ixi2, -ixi3)
    };
    let re1 = x0 + xr1 * c1 + xr2 * c2 + xr3 * c3;
    let re2 = x0 + xr1 * c2 + xr2 * c3 + xr3 * c1;
    let re3 = x0 + xr1 * c3 + xr2 * c1 + xr3 * c2;
    let d1 = sixi1 * s1 + sixi2 * s2 + sixi3 * s3;
    let d2 = sixi1 * s2 - sixi2 * s3 - sixi3 * s1;
    let d3 = sixi1 * s3 - sixi2 * s1 + sixi3 * s2;
    if NORMALIZE {
        let q = F::from_precise(1.0 / 7.0);
        data[0] = Complex::new(
            (x0.re + xr1.re + xr2.re + xr3.re) * q,
            (x0.im + xr1.im + xr2.im + xr3.im) * q,
        );
        data[1] = Complex::new((re1.re + d1.re) * q, (re1.im + d1.im) * q);
        data[2] = Complex::new((re2.re + d2.re) * q, (re2.im + d2.im) * q);
        data[3] = Complex::new((re3.re + d3.re) * q, (re3.im + d3.im) * q);
        data[4] = Complex::new((re3.re - d3.re) * q, (re3.im - d3.im) * q);
        data[5] = Complex::new((re2.re - d2.re) * q, (re2.im - d2.im) * q);
        data[6] = Complex::new((re1.re - d1.re) * q, (re1.im - d1.im) * q);
    } else {
        data[0] = x0 + xr1 + xr2 + xr3;
        data[1] = re1 + d1;
        data[2] = re2 + d2;
        data[3] = re3 + d3;
        data[4] = re3 - d3;
        data[5] = re2 - d2;
        data[6] = re1 - d1;
    }
}

/// In-place Good-Thomas / PFA DFT-15 (3x5).
///
/// Uses row DFT-5 + col DFT-3 after CRT mapping (no twiddles because coprime).
/// Moved to shared for dupe reduction (used by winograd composite and GT paths).
#[inline]
pub(crate) fn dft15_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [eunomia::Complex<F>; 15],
) {
    // Read CRT-mapped inputs into locals before any writes.
    // Row mapping: grid[i1*5+i2] = data[(5*i1 + 3*i2) mod 15]
    let mut row0 = [data[0], data[3], data[6], data[9], data[12]];
    crate::application::execution::kernel::components::butterflies::dft5_array_impl::<
        F,
        INVERSE,
        false,
    >(&mut row0);
    let mut row1 = [data[5], data[8], data[11], data[14], data[2]];
    crate::application::execution::kernel::components::butterflies::dft5_array_impl::<
        F,
        INVERSE,
        false,
    >(&mut row1);
    let mut row2 = [data[10], data[13], data[1], data[4], data[7]];
    crate::application::execution::kernel::components::butterflies::dft5_array_impl::<
        F,
        INVERSE,
        false,
    >(&mut row2);

    let mut col0 = [row0[0], row1[0], row2[0]];
    crate::application::execution::kernel::components::butterflies::dft3_impl::<F, INVERSE, false>(
        &mut col0,
    );
    let mut col1 = [row0[1], row1[1], row2[1]];
    crate::application::execution::kernel::components::butterflies::dft3_impl::<F, INVERSE, false>(
        &mut col1,
    );
    let mut col2 = [row0[2], row1[2], row2[2]];
    crate::application::execution::kernel::components::butterflies::dft3_impl::<F, INVERSE, false>(
        &mut col2,
    );
    let mut col3 = [row0[3], row1[3], row2[3]];
    crate::application::execution::kernel::components::butterflies::dft3_impl::<F, INVERSE, false>(
        &mut col3,
    );
    let mut col4 = [row0[4], row1[4], row2[4]];
    crate::application::execution::kernel::components::butterflies::dft3_impl::<F, INVERSE, false>(
        &mut col4,
    );

    // Output CRT mapping: k_idx = (10*k1 + 6*k2) mod 15 → data[k_idx] = col_{k1}[k2]
    data[0] = col0[0];
    data[1] = col1[1];
    data[2] = col2[2];
    data[3] = col3[0];
    data[4] = col4[1];
    data[5] = col0[2];
    data[6] = col1[0];
    data[7] = col2[1];
    data[8] = col3[2];
    data[9] = col4[0];
    data[10] = col0[1];
    data[11] = col1[2];
    data[12] = col2[0];
    data[13] = col3[1];
    data[14] = col4[2];
}

// ── Composite DFT re-exports (canonical definitions in winograd/composite/) ──
//
// These composite-sized DFT implementations are macro-generated in
// winograd/composite/ (via apollo_fft_macros::generate_winograd_composites!).
// They are re-exported here so that butterflies::dft serves as the SSOT for
// ALL small-DFT lookups, eliminating the impl_short_dft! macro's special
// dependency on winograd::.
//
// Power-of-two composites (dft16 goes through ShortWinogradScalar::dft16 trait method):
pub(crate) use super::super::winograd::composite::{dft128_impl, dft32_impl, dft64_impl};
// Small composites (N ≤ 63):
pub(crate) use super::super::winograd::composite::{
    dft10_impl, dft12_impl, dft14_impl, dft18_impl, dft20_impl, dft21_impl, dft22_impl, dft24_impl,
    dft25_impl, dft26_impl, dft27_impl, dft28_impl, dft30_impl, dft33_impl, dft34_impl, dft35_impl,
    dft36_impl, dft38_impl, dft39_impl, dft40_impl, dft42_impl, dft44_impl, dft45_impl, dft46_impl,
    dft48_impl, dft49_impl, dft50_impl, dft51_impl, dft52_impl, dft54_impl, dft55_impl, dft56_impl,
    dft58_impl, dft60_impl, dft62_impl, dft63_impl, dft6_impl, dft81_impl, dft9_impl,
};
// Medium composites (N ≥ 72):
pub(crate) use super::super::winograd::composite::{
    dft108_impl, dft112_impl, dft120_impl, dft121_impl, dft126_impl, dft144_impl, dft154_impl,
    dft168_impl, dft180_impl, dft189_impl, dft222_impl, dft242_impl, dft246_impl, dft259_impl,
    dft275_impl, dft280_impl, dft296_impl, dft363_impl, dft400_impl, dft484_impl, dft72_impl,
    dft96_impl, dft99_impl,
};
