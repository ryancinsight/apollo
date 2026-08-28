//! Reusable typed accelerator buffers for the fast NUFFT paths.

use eunomia::Complex32;
use hephaestus_core::{
    ComputeDevice, FftDirection, FftOperands, FftOps, KernelDevice, StridedView,
};
use hephaestus_wgpu::{
    WgpuBuffer, WgpuCommandStream, WgpuDevice, WgpuFftOps, WgpuPrepared, WgpuPreparedFft,
};
use leto::Layout;

use super::descriptors::{
    ExtractOne, ExtractThree, FastOneKernel, FastThreeKernel, InterpolateOne, InterpolateThree,
    LoadOne, LoadThree, Position3Pod, SpreadOne, SpreadThree,
};
use crate::infrastructure::transport::gpu::domain::error::{NufftWgpuError, NufftWgpuResult};

struct PreparedFftPair<const R: usize> {
    forward: WgpuPreparedFft<R>,
    inverse: WgpuPreparedFft<R>,
}

impl<const R: usize> core::fmt::Debug for PreparedFftPair<R> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let Self {
            forward: _,
            inverse: _,
        } = self;
        formatter
            .debug_struct("PreparedFftPair")
            .field("forward", &"prepared")
            .field("inverse", &"prepared")
            .finish()
    }
}

impl<const R: usize> PreparedFftPair<R> {
    fn new(
        device: &WgpuDevice,
        real: &WgpuBuffer<f32>,
        imaginary: &WgpuBuffer<f32>,
        shape: [usize; R],
    ) -> NufftWgpuResult<Self> {
        let layout = Layout::c_contiguous(shape).map_err(|_| NufftWgpuError::InvalidPlan {
            message: "oversampled FFT shape cannot form a dense layout",
        })?;
        let operands = || FftOperands {
            real: StridedView::new(real, &layout),
            imaginary: StridedView::new(imaginary, &layout),
        };
        let ops = WgpuFftOps;
        Ok(Self {
            forward: ops.prepare_fft(device, operands(), FftDirection::Forward)?,
            inverse: ops.prepare_fft(device, operands(), FftDirection::Inverse)?,
        })
    }

    fn encode_forward(
        &self,
        device: &WgpuDevice,
        stream: &mut WgpuCommandStream<'_>,
    ) -> NufftWgpuResult<()> {
        Ok(WgpuFftOps.encode_fft(device, &self.forward, stream)?)
    }

    fn encode_inverse(
        &self,
        device: &WgpuDevice,
        stream: &mut WgpuCommandStream<'_>,
    ) -> NufftWgpuResult<()> {
        Ok(WgpuFftOps.encode_fft(device, &self.inverse, stream)?)
    }
}

#[derive(Debug)]
pub(super) struct PreparedNufftKernels1D {
    pub(super) spread: WgpuPrepared<FastOneKernel<SpreadOne>>,
    pub(super) extract: WgpuPrepared<FastOneKernel<ExtractOne>>,
    pub(super) load: WgpuPrepared<FastOneKernel<LoadOne>>,
    pub(super) interpolate: WgpuPrepared<FastOneKernel<InterpolateOne>>,
}

impl PreparedNufftKernels1D {
    fn new(device: &WgpuDevice) -> NufftWgpuResult<Self> {
        Ok(Self {
            spread: device.prepare(&FastOneKernel::<SpreadOne>::new())?,
            extract: device.prepare(&FastOneKernel::<ExtractOne>::new())?,
            load: device.prepare(&FastOneKernel::<LoadOne>::new())?,
            interpolate: device.prepare(&FastOneKernel::<InterpolateOne>::new())?,
        })
    }
}

#[derive(Debug)]
pub(super) struct PreparedNufftKernels3D {
    pub(super) spread: WgpuPrepared<FastThreeKernel<SpreadThree>>,
    pub(super) extract: WgpuPrepared<FastThreeKernel<ExtractThree>>,
    pub(super) load: WgpuPrepared<FastThreeKernel<LoadThree>>,
    pub(super) interpolate: WgpuPrepared<FastThreeKernel<InterpolateThree>>,
}

impl PreparedNufftKernels3D {
    fn new(device: &WgpuDevice) -> NufftWgpuResult<Self> {
        Ok(Self {
            spread: device.prepare(&FastThreeKernel::<SpreadThree>::new())?,
            extract: device.prepare(&FastThreeKernel::<ExtractThree>::new())?,
            load: device.prepare(&FastThreeKernel::<LoadThree>::new())?,
            interpolate: device.prepare(&FastThreeKernel::<InterpolateThree>::new())?,
        })
    }
}

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
    pub(crate) coefficient_buffer: WgpuBuffer<Complex32>,
    pub(crate) padding_buffer: WgpuBuffer<Complex32>,
    pub(super) host_positions: Vec<Complex32>,
    pub(super) host_deconvolution: Vec<Complex32>,
    pub(super) kernels: PreparedNufftKernels1D,
    fft: PreparedFftPair<1>,
    /// Output Fourier-mode count.
    pub(crate) n: usize,
    /// Oversampled grid length.
    pub(crate) m: usize,
    /// Maximum non-uniform sample count per dispatch.
    pub(crate) max_samples: usize,
}

impl NufftGpuBuffers1D {
    /// Allocate buffers and prepare every pipeline used by one fast one-dimensional configuration.
    ///
    /// Reusing the returned allocation avoids provider-buffer allocation and pipeline preparation
    /// on warm executions.
    ///
    /// # Errors
    ///
    /// Returns an error when the requested layout is invalid or the device cannot allocate a
    /// buffer or prepare a pipeline.
    pub fn new(
        device: &WgpuDevice,
        n: usize,
        m: usize,
        max_samples: usize,
    ) -> NufftWgpuResult<Self> {
        let sample_capacity = max_samples.max(1);
        let output_capacity = n.max(max_samples).max(1);
        let real_grid = device.alloc_zeroed(m.max(1))?;
        let imaginary_grid = device.alloc_zeroed(m.max(1))?;
        let fft = PreparedFftPair::new(device, &real_grid, &imaginary_grid, [m])?;
        let kernels = PreparedNufftKernels1D::new(device)?;
        Ok(Self {
            position_buffer: device.alloc_zeroed(sample_capacity)?,
            value_buffer: device.alloc_zeroed(sample_capacity)?,
            deconv_buffer: device.alloc_zeroed(n.max(1))?,
            real_grid,
            imaginary_grid,
            output_buffer: device.alloc_zeroed(output_capacity)?,
            coefficient_buffer: device.alloc_zeroed(n.max(1))?,
            padding_buffer: device.upload(&[Complex32::new(0.0, 0.0)])?,
            host_positions: Vec::with_capacity(sample_capacity),
            host_deconvolution: Vec::with_capacity(n.max(1)),
            kernels,
            fft,
            n,
            m,
            max_samples,
        })
    }

    pub(crate) fn encode_forward(
        &self,
        device: &WgpuDevice,
        stream: &mut WgpuCommandStream<'_>,
    ) -> NufftWgpuResult<()> {
        self.fft.encode_forward(device, stream)
    }

    pub(crate) fn encode_inverse(
        &self,
        device: &WgpuDevice,
        stream: &mut WgpuCommandStream<'_>,
    ) -> NufftWgpuResult<()> {
        self.fft.encode_inverse(device, stream)
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
    pub(crate) coefficient_buffer: WgpuBuffer<Complex32>,
    pub(crate) padding_buffer: WgpuBuffer<Complex32>,
    pub(super) host_positions: Vec<Position3Pod>,
    pub(super) kernels: PreparedNufftKernels3D,
    fft: PreparedFftPair<3>,
    /// Output shape `(nx, ny, nz)`.
    pub(crate) shape: (usize, usize, usize),
    /// Oversampled grid dimensions `(mx, my, mz)`.
    pub(crate) oversampled: (usize, usize, usize),
    /// Maximum non-uniform sample count per dispatch.
    pub(crate) max_samples: usize,
}

impl NufftGpuBuffers3D {
    /// Allocate buffers and prepare every pipeline used by one fast three-dimensional configuration.
    ///
    /// Reusing the returned allocation avoids provider-buffer allocation and pipeline preparation
    /// on warm executions.
    ///
    /// # Errors
    ///
    /// Returns an error when a shape product overflows, the requested layout is invalid, or the
    /// device cannot allocate a buffer or prepare a pipeline.
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
        let real_grid = device.alloc_zeroed(grid_len.max(1))?;
        let imaginary_grid = device.alloc_zeroed(grid_len.max(1))?;
        let fft = PreparedFftPair::new(
            device,
            &real_grid,
            &imaginary_grid,
            [oversampled.0, oversampled.1, oversampled.2],
        )?;
        let kernels = PreparedNufftKernels3D::new(device)?;
        Ok(Self {
            position_buffer: device.alloc_zeroed(sample_capacity)?,
            value_buffer: device.alloc_zeroed(sample_capacity)?,
            deconv_buffer: device.alloc_zeroed(deconv_len.max(1))?,
            real_grid,
            imaginary_grid,
            output_buffer: device.alloc_zeroed(output_capacity)?,
            coefficient_buffer: device.alloc_zeroed(mode_len.max(1))?,
            padding_buffer: device.upload(&[Complex32::new(0.0, 0.0)])?,
            host_positions: Vec::with_capacity(sample_capacity),
            kernels,
            fft,
            shape,
            oversampled,
            max_samples,
        })
    }

    pub(crate) fn encode_forward(
        &self,
        device: &WgpuDevice,
        stream: &mut WgpuCommandStream<'_>,
    ) -> NufftWgpuResult<()> {
        self.fft.encode_forward(device, stream)
    }

    pub(crate) fn encode_inverse(
        &self,
        device: &WgpuDevice,
        stream: &mut WgpuCommandStream<'_>,
    ) -> NufftWgpuResult<()> {
        self.fft.encode_inverse(device, stream)
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
