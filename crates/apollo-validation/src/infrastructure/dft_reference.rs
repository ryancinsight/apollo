//! Pure O(N²) DFT reference oracle — no external dependencies.
//!
//! Implements the textbook Discrete Fourier Transform:
//! X\[k\] = Σ_{n=0}^{N-1} x\[n\] · exp(-2πi·k·n/N)
//!
//! O(N²) complexity by construction. This is the gold-standard reference:
//! trivially correct, matching the mathematical definition exactly, with no
//! optimized FFT algorithm, no approximations, and no external dependencies.
//! Used as the authoritative oracle for validating Apollo's O(N log N) FFT.
//!
//! # Theorem: Discrete Fourier Transform
//! X\[k\] = Σ_{n=0}^{N-1} x\[n\] · (cos(2π·k·n/N) — i·sin(2π·k·n/N))

use eunomia::Complex64;
use leto::{Array1, Array3};
use std::f64::consts::TAU;

/// Compute a single 1D DFT bin: X[k] = Σ x[n] · exp(-2πi·k·n/N)
fn dft_bin(x: &[f64], k: usize) -> Complex64 {
    let n = x.len();
    let phase = -TAU * k as f64 / n as f64;
    let mut re = 0.0;
    let mut im = 0.0;
    for (n_idx, &xn) in x.iter().enumerate() {
        let angle = phase * n_idx as f64;
        re += xn * angle.cos();
        im += xn * angle.sin();
    }
    Complex64::new(re, im)
}

/// 1D DFT via O(N²) direct summation.
///
/// Input is a contiguous real-valued signal. Returns N complex Fourier
/// coefficients matching the standard DFT definition (no normalization).
pub fn dft_1d_real(input: &[f64]) -> Vec<Complex64> {
    let n = input.len();
    (0..n).map(|k| dft_bin(input, k)).collect()
}

/// Compute the full 1D DFT from a `leto::Array1<f64>` view.
pub fn dft_1d_array(input: &Array1<f64>) -> Vec<Complex64> {
    dft_1d_real(input.as_slice().expect("contiguous input"))
}

/// Execute a separable 3D DFT using O(N²) 1D transforms along each axis.
///
/// Applies the 1D DFT separably along Z, Y, X axes — same decomposition as
/// the prior `RustFftPlan3D` but using the direct O(N²) oracle with zero
/// external dependencies.
pub fn dft_3d_real(input: &Array3<f64>) -> Array3<Complex64> {
    let shape = input.shape();
    let [ni, nj, nk] = shape;
    let contiguous = input.to_contiguous();
    let mut data = Array3::from_shape_fn((ni, nj, nk), |[i, j, k]| {
        Complex64::new(contiguous[[i, j, k]], 0.0)
    });

    let mut lane: Vec<f64> = Vec::with_capacity(ni.max(nj).max(nk));

    // Z-axis transforms (each [i, j, :] lane)
    for i in 0..ni {
        for j in 0..nj {
            lane.clear();
            for k in 0..nk {
                lane.push(data[[i, j, k]].re);
            }
            let transformed = dft_1d_real(&lane);
            for (k, val) in transformed.into_iter().enumerate() {
                data[[i, j, k]] = val;
            }
        }
    }

    // Y-axis transforms (each [i, :, k] lane)
    for i in 0..ni {
        for k in 0..nk {
            lane.clear();
            for j in 0..nj {
                lane.push(data[[i, j, k]].re);
            }
            let transformed = dft_1d_real(&lane);
            for (j, val) in transformed.into_iter().enumerate() {
                data[[i, j, k]] = val;
            }
        }
    }

    // X-axis transforms (each [:, j, k] lane)
    for j in 0..nj {
        for k in 0..nk {
            lane.clear();
            for i in 0..ni {
                lane.push(data[[i, j, k]].re);
            }
            let transformed = dft_1d_real(&lane);
            for (i, val) in transformed.into_iter().enumerate() {
                data[[i, j, k]] = val;
            }
        }
    }

    data
}
