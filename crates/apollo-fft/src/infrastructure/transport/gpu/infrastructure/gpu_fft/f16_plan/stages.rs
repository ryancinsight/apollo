//! Native-f16 WGPU staging retained until the reduced-precision plan migrates.
//!
//! These concrete WGPU objects belong only to `GpuFft3dF16Native`.  The f32
//! provider migration must not depend on them: its plan records typed
//! Hephaestus descriptors instead.

/// Precomputed raw WGPU bind groups for one native-f16 axis pack/unpack pass.
pub(super) struct AxisPackStage {
    pub(super) _fft_params_buf: wgpu::Buffer,
    pub(super) fft_params_bg: wgpu::BindGroup,
    pub(super) _params_buf: wgpu::Buffer,
    pub(super) bg: wgpu::BindGroup,
}

/// Raw WGPU staging for one native-f16 radix execution plan.
pub(super) struct NativeRadixStages {
    pub(super) _param_bufs: Vec<wgpu::Buffer>,
    pub(super) bgs: Vec<wgpu::BindGroup>,
    pub(super) fft_m: u32,
    pub(super) batch_count: u32,
}

impl NativeRadixStages {
    pub(super) fn empty() -> Self {
        Self {
            _param_bufs: Vec::new(),
            bgs: Vec::new(),
            fft_m: 0,
            batch_count: 0,
        }
    }

    pub(super) fn precompute(
        device: &wgpu::Device,
        params_layout: &wgpu::BindGroupLayout,
        fft_m: u32,
        batch_count: u32,
        inverse: bool,
    ) -> Self {
        use wgpu::util::DeviceExt;

        let inv_flag = u32::from(inverse);
        let stage_count = 1 + fft_m.trailing_zeros() as usize + usize::from(inverse);
        let mut parameter_buffers = Vec::with_capacity(stage_count);
        let mut bind_groups = Vec::with_capacity(stage_count);
        let mut push_stage = |data: [u32; 4]| {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-fft-native-f16 radix parameters"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-fft-native-f16 radix parameters"),
                layout: params_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
            parameter_buffers.push(buffer);
            bind_group
        };

        bind_groups.push(push_stage([fft_m, 0, inv_flag, batch_count]));
        for stage in 0..fft_m.trailing_zeros() {
            bind_groups.push(push_stage([fft_m, stage, inv_flag, batch_count]));
        }
        if inverse {
            bind_groups.push(push_stage([fft_m, 0, 1, batch_count]));
        }

        Self {
            _param_bufs: parameter_buffers,
            bgs: bind_groups,
            fft_m,
            batch_count,
        }
    }
}

/// Raw WGPU staging for one native-f16 Bluestein axis plan.
pub(super) struct NativeChirpData {
    pub(super) _h_fft_re: wgpu::Buffer,
    pub(super) _h_fft_im: wgpu::Buffer,
    pub(super) premul_pipeline: wgpu::ComputePipeline,
    pub(super) pointmul_pipeline: wgpu::ComputePipeline,
    pub(super) scale_pipeline: wgpu::ComputePipeline,
    pub(super) postmul_pipeline: wgpu::ComputePipeline,
    pub(super) negate_im_pipeline: wgpu::ComputePipeline,
    pub(super) n: u32,
    pub(super) m: u32,
    pub(super) batch_count: u32,
    pub(super) data_chirp_bg: wgpu::BindGroup,
    pub(super) _params_buf: wgpu::Buffer,
    pub(super) params_bg: wgpu::BindGroup,
    pub(super) radix2_fwd: NativeRadixStages,
    pub(super) radix2_inv: NativeRadixStages,
}
