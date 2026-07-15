//! Reusable typed accelerator buffers for the fast NUFFT paths.

use eunomia::Complex32;
use hephaestus_core::ComputeDevice;
use hephaestus_wgpu::{WgpuBuffer, WgpuDevice};

use super::descriptors::Position3Pod;
use crate::infrastructure::transport::gpu::domain::error::{NufftWgpuError, NufftWgpuResult};

/// Snapshot of a complex grid for diagnostics and value-semantic tests.
#[cfg(any(test, feature = "diagnostics"))]
#[derive(Clone, Debug)]
pub struct NufftGridSnapshot {
    /// Real components in storage order.
    pub re: Vec<f32>,
    /// Imaginary components in storage order.
    pub im: Vec<f32>,
}

/// Intermediate Type-2 grids captured at the declared diagnostic boundary.
#[cfg(any(test, feature = "diagnostics"))]
#[derive(Clone, Debug)]
pub struct NufftType2GridDiagnostics {
    /// Grid after loading the spectral coefficients.
    pub after_load: NufftGridSnapshot,
    /// Grid after the inverse FFT.
    pub after_ifft: NufftGridSnapshot,
}

/// Pre-allocated provider buffers for repeated one-dimensional fast NUFFT execution.
#[derive(Debug)]
pub struct NufftGpuBuffers1D {
    pub(crate) position_buffer: WgpuBuffer<Complex32>,
    pub(crate) value_buffer: WgpuBuffer<Complex32>,
    pub(crate) deconv_buffer: WgpuBuffer<Complex32>,
    pub(crate) real_grid: WgpuBuffer<f32>,
    pub(crate) imaginary_grid: WgpuBuffer<f32>,
    pub(crate) output_buffer: WgpuBuffer<Complex32>,
    pub(crate) padding_buffer: WgpuBuffer<Complex32>,
    /// Output Fourier-mode count.
    pub(crate) n: usize,
    /// Oversampled grid length.
    pub(crate) m: usize,
    /// Maximum non-uniform sample count per dispatch.
    pub(crate) max_samples: usize,
}

impl NufftGpuBuffers1D {
    /// Allocate all provider buffers for one fast one-dimensional configuration.
    pub fn new(
        device: &WgpuDevice,
        n: usize,
        m: usize,
        max_samples: usize,
    ) -> NufftWgpuResult<Self> {
        let sample_capacity = max_samples.max(1);
        let output_capacity = n.max(max_samples).max(1);
        Ok(Self {
            position_buffer: device.alloc_zeroed(sample_capacity)?,
            value_buffer: device.alloc_zeroed(sample_capacity)?,
            deconv_buffer: device.alloc_zeroed(n.max(1))?,
            real_grid: device.alloc_zeroed(m.max(1))?,
            imaginary_grid: device.alloc_zeroed(m.max(1))?,
            output_buffer: device.alloc_zeroed(output_capacity)?,
            padding_buffer: device.upload(&[Complex32::new(0.0, 0.0)])?,
            n,
            m,
            max_samples,
        })
    }
}

/// Pre-allocated provider buffers for repeated three-dimensional fast NUFFT execution.
#[derive(Debug)]
pub struct NufftGpuBuffers3D {
    pub(crate) position_buffer: WgpuBuffer<Position3Pod>,
    pub(crate) value_buffer: WgpuBuffer<Complex32>,
    pub(crate) deconv_buffer: WgpuBuffer<f32>,
    pub(crate) real_grid: WgpuBuffer<f32>,
    pub(crate) imaginary_grid: WgpuBuffer<f32>,
    pub(crate) output_buffer: WgpuBuffer<Complex32>,
    pub(crate) padding_buffer: WgpuBuffer<Complex32>,
    /// Output shape `(nx, ny, nz)`.
    pub(crate) shape: (usize, usize, usize),
    /// Oversampled grid dimensions `(mx, my, mz)`.
    pub(crate) oversampled: (usize, usize, usize),
    /// Maximum non-uniform sample count per dispatch.
    pub(crate) max_samples: usize,
}

impl NufftGpuBuffers3D {
    /// Allocate all provider buffers for one fast three-dimensional configuration.
    pub fn new(
        device: &WgpuDevice,
        shape: (usize, usize, usize),
        oversampled: (usize, usize, usize),
        max_samples: usize,
    ) -> NufftWgpuResult<Self> {
        let grid_len = oversampled
            .0
            .checked_mul(oversampled.1)
            .and_then(|value| value.checked_mul(oversampled.2))
            .ok_or(NufftWgpuError::InvalidPlan {
                message: "oversampled 3D grid length overflows usize",
            })?;
        let mode_len = shape
            .0
            .checked_mul(shape.1)
            .and_then(|value| value.checked_mul(shape.2))
            .ok_or(NufftWgpuError::InvalidPlan {
                message: "3D mode-grid length overflows usize",
            })?;
        let deconv_len = shape
            .0
            .checked_add(shape.1)
            .and_then(|value| value.checked_add(shape.2))
            .ok_or(NufftWgpuError::InvalidPlan {
                message: "3D deconvolution length overflows usize",
            })?;
        let sample_capacity = max_samples.max(1);
        let output_capacity = mode_len.max(max_samples).max(1);
        Ok(Self {
            position_buffer: device.alloc_zeroed(sample_capacity)?,
            value_buffer: device.alloc_zeroed(sample_capacity)?,
            deconv_buffer: device.alloc_zeroed(deconv_len.max(1))?,
            real_grid: device.alloc_zeroed(grid_len.max(1))?,
            imaginary_grid: device.alloc_zeroed(grid_len.max(1))?,
            output_buffer: device.alloc_zeroed(output_capacity)?,
            padding_buffer: device.upload(&[Complex32::new(0.0, 0.0)])?,
            shape,
            oversampled,
            max_samples,
        })
    }
}

pub(crate) fn ensure_sample_capacity(max_samples: usize, actual: usize) -> NufftWgpuResult<()> {
    if actual > max_samples {
        return Err(NufftWgpuError::InputLengthMismatch {
            expected: max_samples,
            actual,
        });
    }
    Ok(())
}
