//! Walsh-Hadamard transform kernel.
//!
//! ## Mathematical contract
//!
//! The N-point WHT is defined by the Hadamard matrix H_N where
//! `H_N[k,j] = (-1)^{popcount(k & j)}`.
//!
//! H_N is self-inverse up to scaling: H_N^2 = N * I.
//! Normalization: forward is unnormalized; inverse divides by N.
//!
//! ## Involution theorem
//!
//! Theorem: WHT(WHT(x)) = N * x.
//! Proof: H_N^2 = N * I (Hadamard 1893, Walsh 1923 -- H^T * H = N * I, H = H^T).
//! Therefore WHT(WHT(x)) = H_N * (H_N * x) = N * x.

use hermes_simd::{PreferredArch, SimdArch, SimdKernel, SimdScalar, Vector};
use num_complex::Complex64;
use std::any::TypeId;
use std::ops::{Add, Sub};

const PAR_THRESHOLD: usize = 1024;

/// Verify that a length is a non-zero power of two.
#[must_use]
pub fn is_valid_length(n: usize) -> bool {
    n > 0 && n.is_power_of_two()
}

/// In-place fast Walsh-Hadamard transform over a slice of type `T`.
///
/// ## Mathematical contract
/// For N = 2^k, the WHT butterfly (a, b) -> (a+b, a-b) is applied at each
/// dyadic scale, yielding O(N log N) operations (Hadamard 1893, Walsh 1923).
///
/// ## Preconditions
/// `data.len()` must be a power of two (enforced by `debug_assert` in debug builds).
/// For release builds, the caller must ensure this invariant; the `FwhtPlan` validates it.
pub fn wht_inplace<T>(data: &mut [T])
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Send + Sync + 'static,
{
    let tid = TypeId::of::<T>();
    if tid == TypeId::of::<f64>() {
        let ptr = data.as_mut_ptr().cast::<f64>();
        // SAFETY: The TypeId check proves T is exactly f64, so the pointer cast
        // preserves element layout and length.
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, data.len()) };
        fwht_inplace(slice);
    } else if tid == TypeId::of::<Complex64>() {
        let ptr = data.as_mut_ptr().cast::<Complex64>();
        // SAFETY: The TypeId check proves T is exactly Complex64, so the pointer
        // cast preserves element layout and length.
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, data.len()) };
        fwht_complex_inplace(slice);
    } else if tid == TypeId::of::<f32>() {
        let ptr = data.as_mut_ptr().cast::<f32>();
        // SAFETY: The TypeId check proves T is exactly f32, so the pointer cast
        // preserves element layout and length.
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, data.len()) };
        wht_inplace_f32(slice, 1);
    } else {
        wht_inplace_fallback(data);
    }
}

fn wht_inplace_simd<T, Arch>(data: &mut [T], start_step: usize)
where
    T: SimdScalar + Add<Output = T> + Sub<Output = T> + Send + Sync + 'static,
    Arch: SimdArch + SimdKernel<T>,
{
    let n = data.len();
    if n <= start_step {
        return;
    }
    debug_assert!(n.is_power_of_two(), "WHT requires power-of-2 length");

    let lane_count = Arch::LANE_COUNT;
    let mut step = start_step;
    while step < n {
        let block = step * 2;
        if block >= PAR_THRESHOLD {
            moirai::for_each_chunk_mut_with::<moirai::Adaptive, _, _>(data, block, |chunk| {
                let (left, right) = chunk.split_at_mut(step);
                if step >= lane_count {
                    let mut i = 0;
                    while i < step {
                        unsafe {
                            // SAFETY: step is a power of two and is at least the
                            // SIMD lane count, so each lane-width block stays
                            // inside the split left/right halves.
                            let ptr_a = left.as_mut_ptr().add(i);
                            let ptr_b = right.as_mut_ptr().add(i);
                            let va = Vector::<T, Arch>::load_unaligned(ptr_a);
                            let vb = Vector::<T, Arch>::load_unaligned(ptr_b);
                            let v_add = va + vb;
                            let v_sub = va - vb;
                            v_add.store_unaligned(ptr_a);
                            v_sub.store_unaligned(ptr_b);
                        }
                        i += lane_count;
                    }
                } else {
                    for i in 0..step {
                        let a = left[i];
                        let b = right[i];
                        left[i] = a + b;
                        right[i] = a - b;
                    }
                }
            });
        } else {
            for chunk in data.chunks_mut(block) {
                let (left, right) = chunk.split_at_mut(step);
                if step >= lane_count {
                    let mut i = 0;
                    while i < step {
                        unsafe {
                            // SAFETY: step is a power of two and is at least the
                            // SIMD lane count, so each lane-width block stays
                            // inside the split left/right halves.
                            let ptr_a = left.as_mut_ptr().add(i);
                            let ptr_b = right.as_mut_ptr().add(i);
                            let va = Vector::<T, Arch>::load_unaligned(ptr_a);
                            let vb = Vector::<T, Arch>::load_unaligned(ptr_b);
                            let v_add = va + vb;
                            let v_sub = va - vb;
                            v_add.store_unaligned(ptr_a);
                            v_sub.store_unaligned(ptr_b);
                        }
                        i += lane_count;
                    }
                } else {
                    for i in 0..step {
                        let a = left[i];
                        let b = right[i];
                        left[i] = a + b;
                        right[i] = a - b;
                    }
                }
            }
        }
        step <<= 1;
    }
}

/// Specialized in-place Fast Walsh-Hadamard Transform for f64 slices.
///
/// `start_step` specifies the initial step size (1 for real, 2 for complex-as-real).
fn wht_inplace_f64(data: &mut [f64], start_step: usize) {
    wht_inplace_simd::<f64, PreferredArch>(data, start_step);
}

/// Specialized in-place Fast Walsh-Hadamard Transform for f32 slices.
fn wht_inplace_f32(data: &mut [f32], start_step: usize) {
    wht_inplace_simd::<f32, PreferredArch>(data, start_step);
}

/// Fallback generic in-place transform.
fn wht_inplace_fallback<T>(data: &mut [T])
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Send + Sync,
{
    let n = data.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "WHT requires power-of-2 length");
    let mut step = 1usize;
    while step < n {
        let block = step * 2;
        if block >= PAR_THRESHOLD {
            moirai::for_each_chunk_mut_with::<moirai::Adaptive, _, _>(data, block, |chunk| {
                let (left, right) = chunk.split_at_mut(step);
                for i in 0..step {
                    let a = left[i];
                    let b = right[i];
                    left[i] = a + b;
                    right[i] = a - b;
                }
            });
        } else {
            for chunk in data.chunks_mut(block) {
                let (left, right) = chunk.split_at_mut(step);
                for i in 0..step {
                    let a = left[i];
                    let b = right[i];
                    left[i] = a + b;
                    right[i] = a - b;
                }
            }
        }
        step <<= 1;
    }
}

/// In-place Walsh-Hadamard transform over a real slice.
pub fn fwht_inplace(data: &mut [f64]) {
    wht_inplace_f64(data, 1);
}

/// In-place Walsh-Hadamard transform over a complex slice.
pub fn fwht_complex_inplace(data: &mut [Complex64]) {
    let slice_f64 = bytemuck::cast_slice_mut(data);
    wht_inplace_f64(slice_f64, 2);
}
