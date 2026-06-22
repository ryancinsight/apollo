//! Domain strategies for GPU FFT evaluations.

/// Execution strategy chosen for a single FFT axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisStrategy {
    /// Native radix-2 decomposition for power-of-two lengths.
    Radix2,
    /// Bluestein / Chirp-Z reduction for arbitrary lengths.
    ///
    /// `n` is the original transform length and `m` is the padded radix-2
    /// convolution length used by the reduction.
    ChirpZ {
        /// Original transform length.
        n: usize,
        /// Padded radix-2 convolution length.
        m: usize,
    },
}

/// Cartesian axis of a 3D FFT sweep.
#[derive(Debug, Clone, Copy)]
pub enum Axis {
    /// X dimension.
    X,
    /// Y dimension.
    Y,
    /// Z dimension.
    Z,
}

impl Axis {
    /// Return the transform length for this axis.
    pub fn len(self, nx: usize, ny: usize, nz: usize) -> usize {
        match self {
            Self::X => nx,
            Self::Y => ny,
            Self::Z => nz,
        }
    }

    /// Return the number of batched 1D transforms induced by this axis.
    pub fn batch_count(self, nx: usize, ny: usize, nz: usize) -> usize {
        match self {
            Self::X => ny * nz,
            Self::Y => nx * nz,
            Self::Z => nx * ny,
        }
    }
}

/// Precomputed bind groups for the radix-2 FFT stages of one axis pass.
pub struct RadixStages {
    /// Uniform parameter buffers, retained to keep the bind groups alive.
    pub _param_bufs: Vec<wgpu::Buffer>,
    /// Bind groups for the radix stages and optional inverse scaling stage.
    pub bgs: Vec<wgpu::BindGroup>,
    /// FFT length used by this staged pipeline.
    pub fft_m: u32,
    /// Number of batched transforms executed by this stage set.
    pub batch_count: u32,
    /// Whether this stage set dispatches true radix-4 kernels.
    pub radix4: bool,
}

impl RadixStages {
    /// Create an empty sentinel used for non-radix strategies.
    pub fn empty() -> Self {
        Self {
            _param_bufs: Vec::new(),
            bgs: Vec::new(),
            fft_m: 0,
            batch_count: 0,
            radix4: false,
        }
    }

    /// Precompute all parameter buffers and bind groups for a radix-2 pass.
    pub fn precompute(
        device: &wgpu::Device,
        params_layout: &wgpu::BindGroupLayout,
        fft_m: u32,
        batch_count: u32,
        inverse: bool,
    ) -> Self {
        use wgpu::util::DeviceExt;
        let inv_flag = u32::from(inverse);
        let log2_m = fft_m.trailing_zeros() as usize;
        let stage_count = 1 + log2_m + usize::from(inverse);
        let mut param_bufs = Vec::with_capacity(stage_count);
        let mut bgs = Vec::with_capacity(stage_count);

        let mut push_stage = |data: [u32; 4]| {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-fft-wgpu radix2 params"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-fft-wgpu radix2 params bg"),
                layout: params_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            param_bufs.push(buf);
            bg
        };

        bgs.push(push_stage([fft_m, 0, inv_flag, batch_count]));
        for stage in 0..log2_m as u32 {
            bgs.push(push_stage([fft_m, stage, inv_flag, batch_count]));
        }
        if inverse {
            bgs.push(push_stage([fft_m, 0, 1, batch_count]));
        }

        Self {
            _param_bufs: param_bufs,
            bgs,
            fft_m,
            batch_count,
            radix4: false,
        }
    }

    /// Precompute all parameter buffers and bind groups for a radix-4 pass.
    pub fn precompute_radix4(
        device: &wgpu::Device,
        params_layout: &wgpu::BindGroupLayout,
        fft_m: u32,
        batch_count: u32,
        inverse: bool,
    ) -> Self {
        use wgpu::util::DeviceExt;
        debug_assert!(fft_m.is_power_of_two() && (fft_m.trailing_zeros() % 2 == 0));
        let inv_flag = u32::from(inverse);
        let log4_m = (fft_m.trailing_zeros() / 2) as usize;
        let stage_count = 1 + log4_m + usize::from(inverse);
        let mut param_bufs = Vec::with_capacity(stage_count);
        let mut bgs = Vec::with_capacity(stage_count);

        let mut push_stage = |data: [u32; 4]| {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-fft-wgpu radix4 params"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-fft-wgpu radix4 params bg"),
                layout: params_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            param_bufs.push(buf);
            bg
        };

        bgs.push(push_stage([fft_m, 0, inv_flag, batch_count]));
        for stage in 0..log4_m as u32 {
            bgs.push(push_stage([fft_m, stage, inv_flag, batch_count]));
        }
        if inverse {
            bgs.push(push_stage([fft_m, 0, 1, batch_count]));
        }

        Self {
            _param_bufs: param_bufs,
            bgs,
            fft_m,
            batch_count,
            radix4: true,
        }
    }
}

/// Precomputed resources for a Bluestein / Chirp-Z axis pass.
pub struct ChirpData {
    /// Real component of the padded chirp FFT buffer.
    pub _h_fft_re: wgpu::Buffer,
    /// Imaginary component of the padded chirp FFT buffer.
    pub _h_fft_im: wgpu::Buffer,
    /// Kernel that premultiplies by the chirp sequence.
    pub premul_pipeline: wgpu::ComputePipeline,
    /// Kernel that performs pointwise multiplication in the padded domain.
    pub pointmul_pipeline: wgpu::ComputePipeline,
    /// Kernel that applies the inverse scaling factor.
    pub scale_pipeline: wgpu::ComputePipeline,
    /// Kernel that postmultiplies by the chirp sequence.
    pub postmul_pipeline: wgpu::ComputePipeline,
    /// Kernel that conjugates the imaginary component for inverse passes.
    pub negate_im_pipeline: wgpu::ComputePipeline,
    /// Original transform length.
    pub n: u32,
    /// Padded radix-2 length used by the Bluestein reduction.
    pub m: u32,
    /// Number of batched transforms executed by this chirp plan.
    pub batch_count: u32,
    /// Bind group holding the data and chirp buffers.
    pub data_chirp_bg: wgpu::BindGroup,
    /// Uniform parameter buffer retained for bind-group lifetime.
    pub _params_buf: wgpu::Buffer,
    /// Bind group for the chirp parameters.
    pub params_bg: wgpu::BindGroup,
    /// Forward radix-2 stages over the padded length.
    pub radix2_fwd: RadixStages,
    /// Inverse radix-2 stages over the padded length.
    pub radix2_inv: RadixStages,
}
