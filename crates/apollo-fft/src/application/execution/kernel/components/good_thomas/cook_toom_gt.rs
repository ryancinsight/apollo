//! Cook-Toom-GT fused kernels for composite sizes.
//!
//! This module implements the Good-Thomas PFA with Cook-Toom (Lagrange
//! interpolation) optimization for the row transforms. The key insight from
//! the papers (arxiv 1903.01521, etc.) is that combining GT permutation with
//! Cook-Toom interpolation can reduce multiplications compared to separate
//! GT + Winograd stages.
//!
//! ## Supported sizes
//!
//! - **N=84** (4×21): Row DFT-21 = 3×7 Good-Thomas, column DFT-4
//! - **N=60** (4×15): Row DFT-15 = 3×5 Good-Thomas, column DFT-4
//! - **N=90** (9×10): Row DFT-10 = 2×5 Good-Thomas, column DFT-9 (3×3 prime-power)
//! - **N=150** (6×25): Row DFT-25 = 5×5 prime-power, column DFT-6 = 2×3 Good-Thomas
//!
//! References:
//! - Good (1958), Thomas (1963) - PFA permutation
//! - Cook-Toom (1965) - Lagrange interpolation for polynomial evaluation
//! - Burrus & Parks (1985) - DFT via polynomial interpolation

use crate::application::execution::kernel::components::winograd::composite::dft25_impl;
use crate::application::execution::kernel::components::winograd::composite::dft9_impl;
// Use shared canonical small DFTs (moved to butterflies/ for dupe reduction across
// winograd, GT, rader, etc.). dft9/25 remain winograd-specific for now.
use crate::application::execution::kernel::components::butterflies::{
    dft3_impl, dft4_array_impl, dft5_array_impl, dft7_impl,
};
use crate::application::execution::kernel::components::winograd::traits::WinogradScalar;
use crate::application::execution::kernel::mixed_radix::traits::ShortWinogradScalar;

/// In-place Cook-Toom-GT DFT-84.
///
/// Refactored to a flat, zero-copy 3-way Good-Thomas PFA transform with fused permutations.
#[inline]
pub(crate) fn dft84_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [eunomia::Complex<F>],
) {
    debug_assert!(data.len() >= 84);

    let mut scratch = [eunomia::Complex::<F>::ZERO; 84];

    // Stage 1: Load with input permutation and perform DFT-7 row transforms
    for i1 in 0..4 {
        let src_idx1 = i1 * 21;
        for i2 in 0..3 {
            let mut src_idx2 = src_idx1 + i2 * 28;
            if src_idx2 >= 84 {
                src_idx2 -= 84;
            }

            let mut src_idx = src_idx2;
            let mut row_arr = [eunomia::Complex::<F>::ZERO; 7];
            for i3 in 0..7 {
                row_arr[i3] = unsafe { *data.get_unchecked(src_idx) };
                src_idx += 12;
                if src_idx >= 84 {
                    src_idx -= 84;
                }
            }
            dft7_impl::<F, INVERSE, false>(&mut row_arr);
            let row_start = (i1 * 3 + i2) * 7;
            unsafe {
                *scratch.get_unchecked_mut(row_start) = row_arr[0];
                *scratch.get_unchecked_mut(row_start + 1) = row_arr[1];
                *scratch.get_unchecked_mut(row_start + 2) = row_arr[2];
                *scratch.get_unchecked_mut(row_start + 3) = row_arr[3];
                *scratch.get_unchecked_mut(row_start + 4) = row_arr[4];
                *scratch.get_unchecked_mut(row_start + 5) = row_arr[5];
                *scratch.get_unchecked_mut(row_start + 6) = row_arr[6];
            }
        }
    }

    // Stage 2: DFT-3 column transforms in-place on scratch
    for i1 in 0..4 {
        let offset = i1 * 21;
        for i3 in 0..7 {
            let idx0 = offset + i3;
            let idx1 = idx0 + 7;
            let idx2 = idx0 + 14;

            let mut col = unsafe {
                [
                    *scratch.get_unchecked(idx0),
                    *scratch.get_unchecked(idx1),
                    *scratch.get_unchecked(idx2),
                ]
            };
            dft3_impl::<F, INVERSE, false>(&mut col);
            unsafe {
                *scratch.get_unchecked_mut(idx0) = col[0];
                *scratch.get_unchecked_mut(idx1) = col[1];
                *scratch.get_unchecked_mut(idx2) = col[2];
            }
        }
    }

    // Stage 3: DFT-4 column transforms and write directly to data with output permutation
    for i2 in 0..3 {
        let offset = i2 * 7;
        let dest_idx2_base = i2 * 28;
        for i3 in 0..7 {
            let idx0 = offset + i3;
            let idx1 = idx0 + 21;
            let idx2 = idx0 + 42;
            let idx3 = idx0 + 63;

            let mut col = unsafe {
                [
                    *scratch.get_unchecked(idx0),
                    *scratch.get_unchecked(idx1),
                    *scratch.get_unchecked(idx2),
                    *scratch.get_unchecked(idx3),
                ]
            };
            dft4_array_impl::<F, INVERSE, false>(&mut col);

            let dest_idx_base = dest_idx2_base + i3 * 36;
            let dest_idx0 = dest_idx_base % 84;

            let mut dest_idx1 = dest_idx0 + 21;
            if dest_idx1 >= 84 {
                dest_idx1 -= 84;
            }

            let mut dest_idx2 = dest_idx0 + 42;
            if dest_idx2 >= 84 {
                dest_idx2 -= 84;
            }

            let mut dest_idx3 = dest_idx0 + 63;
            if dest_idx3 >= 84 {
                dest_idx3 -= 84;
            }

            unsafe {
                *data.get_unchecked_mut(dest_idx0) = col[0];
                *data.get_unchecked_mut(dest_idx1) = col[1];
                *data.get_unchecked_mut(dest_idx2) = col[2];
                *data.get_unchecked_mut(dest_idx3) = col[3];
            }
        }
    }
}

/// In-place Cook-Toom-GT DFT-60.
///
/// Refactored to a flat, zero-copy 3-way Good-Thomas PFA transform with fused permutations.
#[inline]
pub(crate) fn dft60_impl<F: WinogradScalar, const INVERSE: bool>(
    data: &mut [eunomia::Complex<F>],
) {
    debug_assert!(data.len() >= 60);

    let mut scratch = [eunomia::Complex::<F>::ZERO; 60];

    // Stage 1: Load with input permutation and perform DFT-5 row transforms
    for i1 in 0..4 {
        let src_idx1 = i1 * 15;
        for i2 in 0..3 {
            let mut src_idx2 = src_idx1 + i2 * 20;
            if src_idx2 >= 60 {
                src_idx2 -= 60;
            }

            let mut src_idx = src_idx2;
            let mut row_arr = [eunomia::Complex::<F>::ZERO; 5];
            for i3 in 0..5 {
                row_arr[i3] = unsafe { *data.get_unchecked(src_idx) };
                src_idx += 12;
                if src_idx >= 60 {
                    src_idx -= 60;
                }
            }
            dft5_array_impl::<F, INVERSE, false>(&mut row_arr);
            let row_start = (i1 * 3 + i2) * 5;
            unsafe {
                *scratch.get_unchecked_mut(row_start) = row_arr[0];
                *scratch.get_unchecked_mut(row_start + 1) = row_arr[1];
                *scratch.get_unchecked_mut(row_start + 2) = row_arr[2];
                *scratch.get_unchecked_mut(row_start + 3) = row_arr[3];
                *scratch.get_unchecked_mut(row_start + 4) = row_arr[4];
            }
        }
    }

    // Stage 2: DFT-3 column transforms in-place on scratch
    for i1 in 0..4 {
        let offset = i1 * 15;
        for i3 in 0..5 {
            let idx0 = offset + i3;
            let idx1 = idx0 + 5;
            let idx2 = idx0 + 10;

            let mut col = unsafe {
                [
                    *scratch.get_unchecked(idx0),
                    *scratch.get_unchecked(idx1),
                    *scratch.get_unchecked(idx2),
                ]
            };
            dft3_impl::<F, INVERSE, false>(&mut col);
            unsafe {
                *scratch.get_unchecked_mut(idx0) = col[0];
                *scratch.get_unchecked_mut(idx1) = col[1];
                *scratch.get_unchecked_mut(idx2) = col[2];
            }
        }
    }

    // Stage 3: DFT-4 column transforms and write directly to data with output permutation
    for i2 in 0..3 {
        let offset = i2 * 5;
        let dest_idx2_base = i2 * 40;
        for i3 in 0..5 {
            let idx0 = offset + i3;
            let idx1 = idx0 + 15;
            let idx2 = idx0 + 30;
            let idx3 = idx0 + 45;

            let mut col = unsafe {
                [
                    *scratch.get_unchecked(idx0),
                    *scratch.get_unchecked(idx1),
                    *scratch.get_unchecked(idx2),
                    *scratch.get_unchecked(idx3),
                ]
            };
            dft4_array_impl::<F, INVERSE, false>(&mut col);

            let dest_idx_base = dest_idx2_base + i3 * 36;
            let dest_idx0 = dest_idx_base % 60;

            let mut dest_idx1 = dest_idx0 + 45;
            if dest_idx1 >= 60 {
                dest_idx1 -= 60;
            }

            let mut dest_idx2 = dest_idx0 + 30;
            if dest_idx2 >= 60 {
                dest_idx2 -= 60;
            }

            let mut dest_idx3 = dest_idx0 + 15;
            if dest_idx3 >= 60 {
                dest_idx3 -= 60;
            }

            unsafe {
                *data.get_unchecked_mut(dest_idx0) = col[0];
                *data.get_unchecked_mut(dest_idx1) = col[1];
                *data.get_unchecked_mut(dest_idx2) = col[2];
                *data.get_unchecked_mut(dest_idx3) = col[3];
            }
        }
    }
}

/// In-place Cook-Toom-GT DFT-90.
///
/// Refactored to a flat, zero-copy 3-way Good-Thomas PFA transform with fused permutations.
#[inline]
pub(crate) fn dft90_impl<F: ShortWinogradScalar, const INVERSE: bool>(
    data: &mut [eunomia::Complex<F>],
) {
    debug_assert!(data.len() >= 90);

    let mut scratch = [eunomia::Complex::<F>::ZERO; 90];

    // Stage 1: Load with input permutation and perform DFT-5 row transforms
    for i1 in 0..2 {
        let src_idx1 = i1 * 45;
        for i2 in 0..9 {
            let mut src_idx2 = src_idx1 + i2 * 10;
            if src_idx2 >= 90 {
                src_idx2 -= 90;
            }

            let mut src_idx = src_idx2;
            let mut row_arr = [eunomia::Complex::<F>::ZERO; 5];
            for i3 in 0..5 {
                row_arr[i3] = unsafe { *data.get_unchecked(src_idx) };
                src_idx += 18;
                if src_idx >= 90 {
                    src_idx -= 90;
                }
            }
            dft5_array_impl::<F, INVERSE, false>(&mut row_arr);
            let row_start = (i1 * 9 + i2) * 5;
            unsafe {
                *scratch.get_unchecked_mut(row_start) = row_arr[0];
                *scratch.get_unchecked_mut(row_start + 1) = row_arr[1];
                *scratch.get_unchecked_mut(row_start + 2) = row_arr[2];
                *scratch.get_unchecked_mut(row_start + 3) = row_arr[3];
                *scratch.get_unchecked_mut(row_start + 4) = row_arr[4];
            }
        }
    }

    // Stage 2: DFT-9 column transforms (along i2, stride 5)
    for i1 in 0..2 {
        let offset1 = i1 * 45;
        for i3 in 0..5 {
            let offset = offset1 + i3;
            let mut col = unsafe {
                [
                    *scratch.get_unchecked(offset),
                    *scratch.get_unchecked(offset + 5),
                    *scratch.get_unchecked(offset + 10),
                    *scratch.get_unchecked(offset + 15),
                    *scratch.get_unchecked(offset + 20),
                    *scratch.get_unchecked(offset + 25),
                    *scratch.get_unchecked(offset + 30),
                    *scratch.get_unchecked(offset + 35),
                    *scratch.get_unchecked(offset + 40),
                ]
            };

            dft9_impl::<F, INVERSE>(&mut col);
            unsafe {
                *scratch.get_unchecked_mut(offset) = col[0];
                *scratch.get_unchecked_mut(offset + 5) = col[1];
                *scratch.get_unchecked_mut(offset + 10) = col[2];
                *scratch.get_unchecked_mut(offset + 15) = col[3];
                *scratch.get_unchecked_mut(offset + 20) = col[4];
                *scratch.get_unchecked_mut(offset + 25) = col[5];
                *scratch.get_unchecked_mut(offset + 30) = col[6];
                *scratch.get_unchecked_mut(offset + 35) = col[7];
                *scratch.get_unchecked_mut(offset + 40) = col[8];
            }
        }
    }

    // Stage 3: DFT-2 column transforms and write directly to data with output permutation
    for i2 in 0..9 {
        let offset = i2 * 5;
        let dest_idx2_base = i2 * 10;
        for i3 in 0..5 {
            let idx0 = offset + i3;
            let idx1 = idx0 + 45;

            let mut col = unsafe { [*scratch.get_unchecked(idx0), *scratch.get_unchecked(idx1)] };
            F::dft2(&mut col);

            let dest_idx_base = dest_idx2_base + i3 * 36;
            let dest_idx0 = dest_idx_base % 90;

            let mut dest_idx1 = dest_idx0 + 45;
            if dest_idx1 >= 90 {
                dest_idx1 -= 90;
            }

            unsafe {
                *data.get_unchecked_mut(dest_idx0) = col[0];
                *data.get_unchecked_mut(dest_idx1) = col[1];
            }
        }
    }
}

/// In-place Cook-Toom-GT DFT-150.
///
/// Refactored to a flat, zero-copy 3-way Good-Thomas PFA transform with fused permutations.
#[inline]
pub(crate) fn dft150_impl<F: ShortWinogradScalar, const INVERSE: bool>(
    data: &mut [eunomia::Complex<F>],
) {
    debug_assert!(data.len() >= 150);

    let mut scratch = [eunomia::Complex::<F>::ZERO; 150];

    // Stage 1: Load with input permutation and perform DFT-25 row transforms
    for i1 in 0..2 {
        let src_idx1 = i1 * 75;
        for i2 in 0..3 {
            let mut src_idx2 = src_idx1 + i2 * 50;
            if src_idx2 >= 150 {
                src_idx2 -= 150;
            }

            let mut src_idx = src_idx2;
            let mut row_arr = [eunomia::Complex::<F>::ZERO; 25];
            for i3 in 0..25 {
                row_arr[i3] = unsafe { *data.get_unchecked(src_idx) };
                src_idx += 6;
                if src_idx >= 150 {
                    src_idx -= 150;
                }
            }
            unsafe {
                dft25_impl::<F, INVERSE>(&mut row_arr);
            }
            let row_start = (i1 * 3 + i2) * 25;
            for i3 in 0..25 {
                unsafe {
                    *scratch.get_unchecked_mut(row_start + i3) = row_arr[i3];
                }
            }
        }
    }

    // Stage 2: DFT-3 column transforms (along i2, stride 25)
    for i1 in 0..2 {
        let offset1 = i1 * 75;
        for i3 in 0..25 {
            let idx0 = offset1 + i3;
            let idx1 = idx0 + 25;
            let idx2 = idx0 + 50;

            let mut col = unsafe {
                [
                    *scratch.get_unchecked(idx0),
                    *scratch.get_unchecked(idx1),
                    *scratch.get_unchecked(idx2),
                ]
            };
            dft3_impl::<F, INVERSE, false>(&mut col);
            unsafe {
                *scratch.get_unchecked_mut(idx0) = col[0];
                *scratch.get_unchecked_mut(idx1) = col[1];
                *scratch.get_unchecked_mut(idx2) = col[2];
            }
        }
    }

    // Stage 3: DFT-2 column transforms and write directly to data with output permutation
    for i2 in 0..3 {
        let offset = i2 * 25;
        let dest_idx2_base = i2 * 100;
        for i3 in 0..25 {
            let idx0 = offset + i3;
            let idx1 = idx0 + 75;

            let mut col = unsafe { [*scratch.get_unchecked(idx0), *scratch.get_unchecked(idx1)] };
            F::dft2(&mut col);

            let dest_idx_base = dest_idx2_base + i3 * 126;
            let dest_idx0 = dest_idx_base % 150;

            let mut dest_idx1 = dest_idx0 + 75;
            if dest_idx1 >= 150 {
                dest_idx1 -= 150;
            }

            unsafe {
                *data.get_unchecked_mut(dest_idx0) = col[0];
                *data.get_unchecked_mut(dest_idx1) = col[1];
            }
        }
    }
}

/// Try to execute using the Cook-Toom-GT fused kernel.
/// Returns true if successful, false if N is not a supported size.
///
/// Checked against total size (n1 * n2) and specific valid coprime pairs to ensure matching
/// works properly regardless of the coprime factor ordering from cache lookup while rejecting
/// non-supported decompositions.
#[inline]
pub(crate) fn try_fft<F: ShortWinogradScalar, const INVERSE: bool>(
    data: &mut [eunomia::Complex<F>],
    n1: usize,
    n2: usize,
) -> bool {
    let match_84 = (n1 == 7 && n2 == 12)
        || (n1 == 12 && n2 == 7)
        || (n1 == 4 && n2 == 21)
        || (n1 == 21 && n2 == 4);
    let match_60 = (n1 == 5 && n2 == 12)
        || (n1 == 12 && n2 == 5)
        || (n1 == 4 && n2 == 15)
        || (n1 == 15 && n2 == 4);
    let match_90 = (n1 == 5 && n2 == 18)
        || (n1 == 18 && n2 == 5)
        || (n1 == 9 && n2 == 10)
        || (n1 == 10 && n2 == 9);
    let match_150 = (n1 == 6 && n2 == 25)
        || (n1 == 25 && n2 == 6)
        || (n1 == 2 && n2 == 75)
        || (n1 == 75 && n2 == 2)
        || (n1 == 3 && n2 == 50)
        || (n1 == 50 && n2 == 3);

    if match_84 {
        dft84_impl::<F, INVERSE>(data);
        true
    } else if match_60 {
        dft60_impl::<F, INVERSE>(data);
        true
    } else if match_90 {
        dft90_impl::<F, INVERSE>(data);
        true
    } else if match_150 {
        dft150_impl::<F, INVERSE>(data);
        true
    } else {
        false
    }
}
