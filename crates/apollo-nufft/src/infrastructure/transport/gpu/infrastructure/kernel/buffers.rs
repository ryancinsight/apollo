use super::Position3Pod;
use crate::infrastructure::transport::gpu::domain::error::{NufftWgpuError, NufftWgpuResult};
use num_complex::Complex32;

/// Pre-allocated GPU buffers for repeated 1D NUFFT fast-path execution.
///
/// Buffers are sized for a specific transform configuration (`n`, `m`, `max_samples`).
/// Reusing these buffers across calls eliminates per-dispatch GPU buffer creation overhead.
pub struct NufftGpuBuffers1D {
    pub(crate) position_buffer: wgpu::Buffer,
    pub(crate) value_buffer: wgpu::Buffer,
    pub(crate) deconv_buffer: wgpu::Buffer,
    pub(crate) re_buffer: wgpu::Buffer,
    pub(crate) im_buffer: wgpu::Buffer,
    pub(crate) output_buffer: wgpu::Buffer,
    pub(crate) staging_buffer: wgpu::Buffer,
    /// Output grid length (number of Fourier modes).
    pub(crate) n: usize,
    /// Oversampled grid length.
    pub(crate) m: usize,
    /// Maximum non-uniform sample count per dispatch.
    pub(crate) max_samples: usize,
}

impl NufftGpuBuffers1D {
    /// Pre-allocate all GPU buffers for 1D fast-path transforms of the given configuration.
    ///
    /// `n` is the output grid length, `m` is the oversampled grid length, and
    /// `max_samples` is the maximum number of non-uniform samples per dispatch.
    /// Each call to `execute_fast_type1_1d_with_buffers` or
    /// `execute_fast_type2_1d_with_buffers` may use fewer samples; the excess
    /// capacity is unused but not reallocated.
    pub fn new(device: &wgpu::Device, n: usize, m: usize, max_samples: usize) -> Self {
        let upload_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let grid_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let position_size = (max_samples.max(1) * std::mem::size_of::<Complex32>()) as u64;
        let position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers1d positions"),
            size: position_size,
            usage: upload_usage,
            mapped_at_creation: false,
        });

        let value_size = (max_samples.max(1) * std::mem::size_of::<Complex32>()) as u64;
        let value_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers1d values"),
            size: value_size,
            usage: upload_usage,
            mapped_at_creation: false,
        });

        let deconv_size = (m.max(1) * std::mem::size_of::<Complex32>()) as u64;
        let deconv_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers1d deconv"),
            size: deconv_size,
            usage: upload_usage,
            mapped_at_creation: false,
        });

        let grid_elem_size = (m.max(1) * std::mem::size_of::<f32>()) as u64;
        let re_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers1d grid re"),
            size: grid_elem_size,
            usage: grid_usage,
            mapped_at_creation: false,
        });
        let im_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers1d grid im"),
            size: grid_elem_size,
            usage: grid_usage,
            mapped_at_creation: false,
        });

        let output_size = (n.max(1) * std::mem::size_of::<Complex32>()) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers1d output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_size = output_size;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers1d staging"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            position_buffer,
            value_buffer,
            deconv_buffer,
            re_buffer,
            im_buffer,
            output_buffer,
            staging_buffer,
            n,
            m,
            max_samples,
        }
    }
}

/// Pre-allocated GPU buffers for repeated 3D NUFFT fast-path execution.
///
/// Buffers are sized for a specific transform configuration (`shape`, `oversampled`,
/// `max_samples`). Reusing these buffers across calls eliminates per-dispatch GPU
/// buffer creation overhead.
pub struct NufftGpuBuffers3D {
    pub(crate) position_buffer: wgpu::Buffer,
    pub(crate) value_buffer: wgpu::Buffer,
    pub(crate) deconv_buffer: wgpu::Buffer,
    pub(crate) re_buffer: wgpu::Buffer,
    pub(crate) im_buffer: wgpu::Buffer,
    pub(crate) output_buffer: wgpu::Buffer,
    pub(crate) staging_buffer: wgpu::Buffer,
    /// Output shape `(nx, ny, nz)`.
    pub(crate) shape: (usize, usize, usize),
    /// Oversampled grid dimensions `(mx, my, mz)`.
    pub(crate) oversampled: (usize, usize, usize),
    /// Maximum non-uniform sample count per dispatch.
    pub(crate) max_samples: usize,
}

/// Diagnostic snapshot of split real/imaginary NUFFT grid state.
///
/// This type is compiled only for tests or the explicit `diagnostics` feature.
/// It records computed GPU grid values after named fast-path checkpoints without
/// changing production dispatch behavior.
#[cfg(any(test, feature = "diagnostics"))]
#[derive(Clone, Debug)]
pub struct NufftGridSnapshot {
    /// Real grid component in row-major storage order.
    pub re: Vec<f32>,
    /// Imaginary grid component in row-major storage order.
    pub im: Vec<f32>,
}

/// Diagnostic checkpoints for fast type-2 NUFFT execution.
#[cfg(any(test, feature = "diagnostics"))]
#[derive(Clone, Debug)]
pub struct NufftType2GridDiagnostics {
    /// Grid state immediately after coefficient load/deconvolution.
    pub after_load: NufftGridSnapshot,
    /// Grid state immediately after inverse FFT and before interpolation.
    pub after_ifft: NufftGridSnapshot,
}

impl NufftGpuBuffers3D {
    /// Pre-allocate all GPU buffers for 3D fast-path transforms of the given configuration.
    ///
    /// `shape` is the output grid dimensions `(nx, ny, nz)`, `oversampled` is the
    /// oversampled grid dimensions `(mx, my, mz)`, and `max_samples` is the maximum
    /// number of non-uniform samples per dispatch.
    pub fn new(
        device: &wgpu::Device,
        shape: (usize, usize, usize),
        oversampled: (usize, usize, usize),
        max_samples: usize,
    ) -> Self {
        let upload_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let grid_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let (nx, ny, nz) = shape;
        let (mx, my, mz) = oversampled;
        let grid_len = mx * my * mz;
        let output_len = nx * ny * nz;

        let position_size = (max_samples.max(1) * std::mem::size_of::<Position3Pod>()) as u64;
        let position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers3d positions"),
            size: position_size,
            usage: upload_usage,
            mapped_at_creation: false,
        });

        let value_size = (max_samples.max(1) * std::mem::size_of::<Complex32>()) as u64;
        let value_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers3d values"),
            size: value_size,
            usage: upload_usage,
            mapped_at_creation: false,
        });

        let deconv_size = (grid_len.max(1) * std::mem::size_of::<f32>()) as u64;
        let deconv_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers3d deconv"),
            size: deconv_size,
            usage: upload_usage,
            mapped_at_creation: false,
        });

        let grid_elem_size = (grid_len.max(1) * std::mem::size_of::<f32>()) as u64;
        let re_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers3d grid re"),
            size: grid_elem_size,
            usage: grid_usage,
            mapped_at_creation: false,
        });
        let im_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers3d grid im"),
            size: grid_elem_size,
            usage: grid_usage,
            mapped_at_creation: false,
        });

        let output_size = (output_len.max(1) * std::mem::size_of::<Complex32>()) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers3d output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_size = output_size;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu buffers3d staging"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            position_buffer,
            value_buffer,
            deconv_buffer,
            re_buffer,
            im_buffer,
            output_buffer,
            staging_buffer,
            shape,
            oversampled,
            max_samples,
        }
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
