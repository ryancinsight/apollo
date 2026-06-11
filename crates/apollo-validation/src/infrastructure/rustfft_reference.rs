#![cfg(feature = "external-references")]
//! `rustfft`-based reference transforms used by validation.
//!
//! This module is compiled only when the `external-references` feature is enabled.
//! It provides reusable reference FFT plans for validation comparisons without
//! imposing `rustfft` on non-validation builds.

use ndarray::{Array1, Array3};
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Reusable `rustfft` plan for reference 1D FFTs.
pub struct RustFftPlan1D {
    fft: Arc<dyn Fft<f64>>,
    scratch_len: usize,
}

impl RustFftPlan1D {
    /// Create a reusable forward 1D reference plan.
    #[must_use]
    pub fn new(n: usize) -> Self {
        let fft = FftPlanner::<f64>::new().plan_fft_forward(n);
        Self {
            scratch_len: fft.get_inplace_scratch_len(),
            fft,
        }
    }

    /// Return the required scratch length for in-place FFT execution.
    #[must_use]
    pub fn scratch_len(&self) -> usize {
        self.scratch_len
    }

    /// Execute a forward real-input FFT into a caller-owned complex buffer.
    pub fn forward_real_into(
        &self,
        input: &Array1<f64>,
        output: &mut [Complex64],
        scratch: &mut [Complex64],
    ) {
        assert_eq!(input.len(), output.len(), "rustfft 1D length mismatch");
        assert!(
            scratch.len() >= self.scratch_len,
            "rustfft 1D scratch buffer too small"
        );
        output
            .iter_mut()
            .zip(input.iter().copied())
            .for_each(|(dst, src)| *dst = Complex64::new(src, 0.0));
        self.fft
            .process_with_scratch(output, &mut scratch[..self.scratch_len]);
    }
}

/// Reusable separable `rustfft` plan for reference 3D FFTs.
pub struct RustFftPlan3D {
    shape: (usize, usize, usize),
    fft_x: Arc<dyn Fft<f64>>,
    fft_y: Arc<dyn Fft<f64>>,
    fft_z: Arc<dyn Fft<f64>>,
    lane_len: usize,
    scratch_len: usize,
}

impl RustFftPlan3D {
    /// Create a reusable forward 3D reference plan.
    #[must_use]
    pub fn new(shape: (usize, usize, usize)) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        let fft_x = planner.plan_fft_forward(shape.0);
        let fft_y = planner.plan_fft_forward(shape.1);
        let fft_z = planner.plan_fft_forward(shape.2);
        let scratch_len = fft_x
            .get_inplace_scratch_len()
            .max(fft_y.get_inplace_scratch_len())
            .max(fft_z.get_inplace_scratch_len());
        Self {
            shape,
            fft_x,
            fft_y,
            fft_z,
            lane_len: shape.0.max(shape.1).max(shape.2),
            scratch_len,
        }
    }

    /// Return the maximum scratch length required by any axis pass.
    #[must_use]
    pub fn scratch_len(&self) -> usize {
        self.scratch_len
    }

    /// Execute a separable real-input 3D FFT into caller-owned buffers.
    pub fn forward_real_into(
        &self,
        input: &Array3<f64>,
        output: &mut Array3<Complex64>,
        lane_buffer: &mut [Complex64],
        fft_scratch: &mut [Complex64],
    ) {
        let shape = input.dim();
        assert_eq!(shape, self.shape, "rustfft 3D input shape mismatch");
        assert_eq!(output.dim(), self.shape, "rustfft 3D output shape mismatch");
        assert!(
            lane_buffer.len() >= self.lane_len,
            "rustfft 3D lane buffer too small"
        );
        assert!(
            fft_scratch.len() >= self.scratch_len,
            "rustfft 3D scratch buffer too small"
        );

        output
            .iter_mut()
            .zip(input.iter().copied())
            .for_each(|(dst, src)| *dst = Complex64::new(src, 0.0));

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                let limit = shape.2;
                for k in 0..limit {
                    lane_buffer[k] = output[(i, j, k)];
                }
                self.fft_z.process_with_scratch(
                    &mut lane_buffer[..limit],
                    &mut fft_scratch[..self.fft_z.get_inplace_scratch_len()],
                );
                for k in 0..limit {
                    output[(i, j, k)] = lane_buffer[k];
                }
            }
        }

        for i in 0..shape.0 {
            for k in 0..shape.2 {
                let limit = shape.1;
                for j in 0..limit {
                    lane_buffer[j] = output[(i, j, k)];
                }
                self.fft_y.process_with_scratch(
                    &mut lane_buffer[..limit],
                    &mut fft_scratch[..self.fft_y.get_inplace_scratch_len()],
                );
                for j in 0..limit {
                    output[(i, j, k)] = lane_buffer[j];
                }
            }
        }

        for j in 0..shape.1 {
            for k in 0..shape.2 {
                let limit = shape.0;
                for i in 0..limit {
                    lane_buffer[i] = output[(i, j, k)];
                }
                self.fft_x.process_with_scratch(
                    &mut lane_buffer[..limit],
                    &mut fft_scratch[..self.fft_x.get_inplace_scratch_len()],
                );
                for i in 0..limit {
                    output[(i, j, k)] = lane_buffer[i];
                }
            }
        }
    }
}

/// Execute a reference 1D FFT using `rustfft`.
///
/// # Theorem: 1D Discrete Fourier Transform
/// Analytically executes the exact continuous basis projection bounding signals onto Fourier coefficients:
/// $$ X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-i 2\pi k n / N} $$
pub fn fft1_real(input: &leto::ArrayView1<'_, f64>) -> Vec<Complex64> {
    let len = input.shape()[0];
    let plan = RustFftPlan1D::new(len);
    let mut buffer = vec![Complex64::new(0.0, 0.0); len];
    let mut scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len];
    let cow = apollo_fft::application::utilities::leto_interop::view1_cow(input);
    buffer
        .iter_mut()
        .zip(cow.iter().copied())
        .for_each(|(dst, src)| *dst = Complex64::new(src, 0.0));
    plan.fft
        .process_with_scratch(&mut buffer, &mut scratch[..plan.scratch_len]);
    buffer
}

/// Execute a separable 3D FFT using `rustfft`.
///
/// # Theorem: Separable 3D Discrete Fourier Transform
/// The 3D DFT decomposes into three orthogonal 1D DFT evaluations
/// bypassing multidimensional recursive spatial bounds factoring limits globally:
/// $$ X_{k_1, k_2, k_3} = \sum_{n_1} \sum_{n_2} \sum_{n_3} x_{n_1, n_2, n_3} \cdot e^{-i 2\pi (\frac{k_1 n_1}{N_1} + \frac{k_2 n_2}{N_2} + \frac{k_3 n_3}{N_3})} $стә
///
/// **Zero Inner-Loop Allocation Guarantee:** This limits created buffers replacing
/// dimensional iterations into strict pre-allocated boundary matrices globally without nested heap fragmentations.
pub fn fft3_real(input: &leto::ArrayView3<'_, f64>) -> Array3<Complex64> {
    let shape = input.shape();
    let nd_array = apollo_fft::application::utilities::leto_interop::array3_from_view(input);
    let plan = RustFftPlan3D::new((shape[0], shape[1], shape[2]));
    let mut data = Array3::from_elem((shape[0], shape[1], shape[2]), Complex64::new(0.0, 0.0));
    let mut lane_buffer = vec![Complex64::new(0.0, 0.0); shape[0].max(shape[1]).max(shape[2])];
    let mut fft_scratch = vec![Complex64::new(0.0, 0.0); plan.scratch_len];
    plan.forward_real_into(&nd_array, &mut data, &mut lane_buffer, &mut fft_scratch);
    data
}
