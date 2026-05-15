use super::dft25_impl_const;
use crate::application::execution::kernel::winograd::radix::dft4_array_impl;
use crate::application::execution::kernel::winograd::traits::WinogradScalar;
use num_complex::Complex;

/// In-place Good-Thomas PFA DFT-100.
///
/// N = 100 = 4 x 25 with gcd(4, 25) = 1. The Chinese-remainder mapping removes
/// inter-stage twiddles and remains the verified DFT-100 path. Pease was tested
/// for this size and rejected because it failed the f64 direct-comparison bound.
#[inline]
pub(crate) fn dft100_impl<F: WinogradScalar>(data: &mut [Complex<F>; 100], inverse: bool) {
    if inverse {
        dft100_good_thomas::<F, true>(data);
    } else {
        dft100_good_thomas::<F, false>(data);
    }
}

#[inline(always)]
fn dft100_good_thomas<F: WinogradScalar, const INVERSE: bool>(data: &mut [Complex<F>; 100]) {
    let mut rows = [
        input_row::<F, 0>(data),
        input_row::<F, 1>(data),
        input_row::<F, 2>(data),
        input_row::<F, 3>(data),
    ];
    dft25_impl_const::<F, INVERSE>(&mut rows[0]);
    dft25_impl_const::<F, INVERSE>(&mut rows[1]);
    dft25_impl_const::<F, INVERSE>(&mut rows[2]);
    dft25_impl_const::<F, INVERSE>(&mut rows[3]);

    for k2 in 0..25 {
        output_column::<F, INVERSE>(&rows, data, k2);
    }
}

const OUTPUT_COLUMNS: [[usize; 4]; 25] = output_column_indices();

const fn output_column_indices() -> [[usize; 4]; 25] {
    let mut cols = [[0; 4]; 25];
    let mut k2 = 0;
    while k2 < 25 {
        let mut k1 = 0;
        while k1 < 4 {
            cols[k2][k1] = (25 * k1 + 76 * k2) % 100;
            k1 += 1;
        }
        k2 += 1;
    }
    cols
}

#[inline(always)]
fn input_row<F: WinogradScalar, const I1: usize>(data: &[Complex<F>; 100]) -> [Complex<F>; 25] {
    core::array::from_fn(|i2| data[(25 * I1 + 4 * i2) % 100])
}

#[inline(always)]
fn output_column<F: WinogradScalar, const INVERSE: bool>(
    rows: &[[Complex<F>; 25]; 4],
    data: &mut [Complex<F>; 100],
    k2: usize,
) {
    let mut col = [rows[0][k2], rows[1][k2], rows[2][k2], rows[3][k2]];
    dft4_array_impl(&mut col, INVERSE);
    let out = OUTPUT_COLUMNS[k2];
    data[out[0]] = col[0];
    data[out[1]] = col[1];
    data[out[2]] = col[2];
    data[out[3]] = col[3];
}
