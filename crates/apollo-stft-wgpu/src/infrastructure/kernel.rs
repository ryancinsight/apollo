//! GPU compute kernels for the forward and inverse Short-Time Fourier Transform.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Forward execution kernels for power-of-two frame sizes.
pub mod forward;
/// Forward execution kernels for non-power-of-two frame sizes via Chirp-Z.
pub mod forward_chirp;
/// Inverse execution kernels for power-of-two frame sizes.
pub mod inverse;
/// Inverse execution kernels for non-power-of-two frame sizes via Bluestein/Chirp-Z.
pub mod inverse_chirp;

/// Workgroup size for the OLA reconstruction pass (matches `@workgroup_size(64)` in
/// `stft_inverse.wgsl`).
pub(crate) const WORKGROUP_SIZE: u32 = 64;

/// Workgroup size for the four FFT inverse passes (matches `@workgroup_size(256)` in
/// `stft_inverse_fft.wgsl`).
pub(crate) const FFT_WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct ComplexPod {
    pub(crate) re: f32,
    pub(crate) im: f32,
}

/// Uniform parameter block for forward pass and OLA pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct StftParams {
    pub(crate) signal_len: u32,
    pub(crate) frame_len: u32,
    pub(crate) hop_len: u32,
    pub(crate) frame_count: u32,
}

/// Uniform parameter block for the FFT inverse passes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct FftStageParams {
    pub(crate) frame_count: u32,
    pub(crate) frame_len: u32,
    pub(crate) stage: u32,
    pub(crate) _pad: u32,
}

/// Uniform parameter block for the FFT forward passes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct FwdFftStageParams {
    pub(crate) frame_count: u32,
    pub(crate) frame_len: u32,
    pub(crate) hop_len: u32,
    pub(crate) stage: u32,
}

/// GPU compute kernel encapsulating the forward and inverse FFT-accelerated STFT pipelines.
#[derive(Debug)]
pub struct StftGpuKernel {
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) params_buffer: wgpu::Buffer,
    pub(crate) inverse_ola_pipeline: wgpu::ComputePipeline,
    pub(crate) fft_data_bgl: wgpu::BindGroupLayout,
    pub(crate) fft_params_bgl: wgpu::BindGroupLayout,
    pub(crate) deinterleave_pipeline: wgpu::ComputePipeline,
    pub(crate) bitrev_pipeline: wgpu::ComputePipeline,
    pub(crate) butterfly_pipeline: wgpu::ComputePipeline,
    pub(crate) scale_window_pipeline: wgpu::ComputePipeline,
    pub(crate) fwd_pack_window_pipeline: wgpu::ComputePipeline,
    pub(crate) fwd_bitrev_pipeline: wgpu::ComputePipeline,
    pub(crate) fwd_butterfly_pipeline: wgpu::ComputePipeline,
    pub(crate) fwd_interleave_pipeline: wgpu::ComputePipeline,
}

impl StftGpuKernel {
    /// Create a new kernel by compiling all WGSL shaders and building all pipelines.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-stft-wgpu BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(std::mem::size_of::<StftParams>() as u64)
                                .expect("nonzero size"),
                        ),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-stft-wgpu pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let inverse_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-stft-wgpu inverse shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/stft_inverse.wgsl").into()),
        });
        let inverse_ola_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-stft-wgpu inverse ola pipeline"),
                layout: Some(&pipeline_layout),
                module: &inverse_shader,
                entry_point: Some("stft_inverse_ola"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-stft-wgpu params"),
            contents: bytemuck::bytes_of(&StftParams {
                signal_len: 0,
                frame_len: 0,
                hop_len: 0,
                frame_count: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let fft_data_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-stft-wgpu FFT data BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let fft_params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-stft-wgpu FFT params BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(std::mem::size_of::<FftStageParams>() as u64)
                            .expect("nonzero size"),
                    ),
                },
                count: None,
            }],
        });
        let fft_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-stft-wgpu FFT pipeline layout"),
            bind_group_layouts: &[&fft_data_bgl, &fft_params_bgl],
            push_constant_ranges: &[],
        });

        let fft_inv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-stft-wgpu FFT inverse shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/stft_inverse_fft.wgsl").into()),
        });
        let deinterleave_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-stft-wgpu deinterleave pipeline"),
                layout: Some(&fft_pipeline_layout),
                module: &fft_inv_shader,
                entry_point: Some("stft_deinterleave"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let bitrev_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-stft-wgpu bitrev pipeline"),
            layout: Some(&fft_pipeline_layout),
            module: &fft_inv_shader,
            entry_point: Some("stft_bitrev"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let butterfly_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-stft-wgpu butterfly pipeline"),
            layout: Some(&fft_pipeline_layout),
            module: &fft_inv_shader,
            entry_point: Some("stft_butterfly"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let scale_window_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-stft-wgpu scale-window pipeline"),
                layout: Some(&fft_pipeline_layout),
                module: &fft_inv_shader,
                entry_point: Some("stft_scale_and_window"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let fft_fwd_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-stft-wgpu FFT forward shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/stft_forward_fft.wgsl").into()),
        });
        let fwd_pack_window_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-stft-wgpu fwd pack-window pipeline"),
                layout: Some(&fft_pipeline_layout),
                module: &fft_fwd_shader,
                entry_point: Some("stft_fwd_pack_window"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fwd_bitrev_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-stft-wgpu fwd bitrev pipeline"),
                layout: Some(&fft_pipeline_layout),
                module: &fft_fwd_shader,
                entry_point: Some("stft_fwd_bitrev"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fwd_butterfly_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-stft-wgpu fwd butterfly pipeline"),
                layout: Some(&fft_pipeline_layout),
                module: &fft_fwd_shader,
                entry_point: Some("stft_fwd_butterfly"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fwd_interleave_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-stft-wgpu fwd interleave pipeline"),
                layout: Some(&fft_pipeline_layout),
                module: &fft_fwd_shader,
                entry_point: Some("stft_fwd_interleave"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            bind_group_layout,
            params_buffer,
            inverse_ola_pipeline,
            fft_data_bgl,
            fft_params_bgl,
            deinterleave_pipeline,
            bitrev_pipeline,
            butterfly_pipeline,
            scale_window_pipeline,
            fwd_pack_window_pipeline,
            fwd_bitrev_pipeline,
            fwd_butterfly_pipeline,
            fwd_interleave_pipeline,
        }
    }
}

pub(crate) fn dispatch_count(total: u32) -> u32 {
    total.div_ceil(WORKGROUP_SIZE)
}

pub(crate) fn fft_dispatch_count(total: u32) -> u32 {
    total.div_ceil(FFT_WORKGROUP_SIZE)
}
