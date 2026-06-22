//! GPU compute kernels for the forward and inverse Short-Time Fourier Transform.

use bytemuck::{Pod, Zeroable};

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

    // Chirp-Z / Bluestein precompiled layouts and pipelines
    pub(crate) chirp_data_bgl: wgpu::BindGroupLayout,
    pub(crate) chirp_params_bgl: wgpu::BindGroupLayout,
    pub(crate) chirp_io_bgl: wgpu::BindGroupLayout,
    pub(crate) chirp_radix2_params_bgl: wgpu::BindGroupLayout,
    pub(crate) premul_fwd_pipeline: wgpu::ComputePipeline,
    pub(crate) premul_inv_pipeline: wgpu::ComputePipeline,
    pub(crate) pointmul_pipeline: wgpu::ComputePipeline,
    pub(crate) postmul_fwd_pipeline: wgpu::ComputePipeline,
    pub(crate) postmul_inv_pipeline: wgpu::ComputePipeline,
    pub(crate) chirp_bitrev_pipeline: wgpu::ComputePipeline,
    pub(crate) chirp_fwd_butterfly_pipeline: wgpu::ComputePipeline,
    pub(crate) chirp_inv_butterfly_pipeline: wgpu::ComputePipeline,
    pub(crate) chirp_scale_pipeline: wgpu::ComputePipeline,
    pub(crate) pointmul_fwd_pipeline: wgpu::ComputePipeline,
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

        let bgl_storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let chirp_data_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-stft-wgpu chirp data BGL"),
            entries: &[
                bgl_storage_entry(0, false),
                bgl_storage_entry(1, false),
                bgl_storage_entry(2, true),
                bgl_storage_entry(3, true),
            ],
        });

        let chirp_params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-stft-wgpu chirp params BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let chirp_io_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-stft-wgpu chirp IO BGL"),
            entries: &[bgl_storage_entry(0, true), bgl_storage_entry(1, false)],
        });

        let chirp_radix2_params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-stft-wgpu chirp radix2 params BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let chirp_io_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-stft-wgpu chirp IO pipeline layout"),
            bind_group_layouts: &[&chirp_data_bgl, &chirp_params_bgl, &chirp_io_bgl],
            push_constant_ranges: &[],
        });

        let chirp_radix2_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-stft-wgpu chirp radix2 pipeline layout"),
            bind_group_layouts: &[&chirp_data_bgl, &chirp_radix2_params_bgl],
            push_constant_ranges: &[],
        });

        let chirp_pm_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-stft-wgpu chirp pointmul pipeline layout"),
            bind_group_layouts: &[&chirp_data_bgl, &chirp_params_bgl],
            push_constant_ranges: &[],
        });

        let chirp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-stft-wgpu chirp shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/stft_chirp.wgsl").into()),
        });

        let chirp_fft_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-stft-wgpu chirp sub-FFT shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/stft_chirp_fft.wgsl").into()),
        });

        let build_chirp_pipeline = |layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        let premul_fwd_pipeline = build_chirp_pipeline(&chirp_io_pipeline_layout, &chirp_shader, "stft_chirp_premul_fwd");
        let premul_inv_pipeline = build_chirp_pipeline(&chirp_io_pipeline_layout, &chirp_shader, "stft_chirp_premul_inv");
        let pointmul_pipeline = build_chirp_pipeline(&chirp_pm_pipeline_layout, &chirp_shader, "stft_chirp_pointmul");
        let postmul_fwd_pipeline = build_chirp_pipeline(&chirp_io_pipeline_layout, &chirp_shader, "stft_chirp_postmul_fwd");
        let postmul_inv_pipeline = build_chirp_pipeline(&chirp_io_pipeline_layout, &chirp_shader, "stft_chirp_postmul_inv");
        let chirp_bitrev_pipeline = build_chirp_pipeline(&chirp_radix2_pipeline_layout, &chirp_fft_shader, "chirp_fft_bitrev");
        let chirp_fwd_butterfly_pipeline = build_chirp_pipeline(&chirp_radix2_pipeline_layout, &chirp_fft_shader, "chirp_fft_butterfly_fwd");
        let chirp_inv_butterfly_pipeline = build_chirp_pipeline(&chirp_radix2_pipeline_layout, &chirp_fft_shader, "chirp_fft_butterfly_inv");
        let chirp_scale_pipeline = build_chirp_pipeline(&chirp_radix2_pipeline_layout, &chirp_fft_shader, "chirp_fft_scale");
        let pointmul_fwd_pipeline = build_chirp_pipeline(&chirp_pm_pipeline_layout, &chirp_shader, "stft_chirp_pointmul_fwd");

        Self {
            bind_group_layout,
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
            chirp_data_bgl,
            chirp_params_bgl,
            chirp_io_bgl,
            chirp_radix2_params_bgl,
            premul_fwd_pipeline,
            premul_inv_pipeline,
            pointmul_pipeline,
            postmul_fwd_pipeline,
            postmul_inv_pipeline,
            chirp_bitrev_pipeline,
            chirp_fwd_butterfly_pipeline,
            chirp_inv_butterfly_pipeline,
            chirp_scale_pipeline,
            pointmul_fwd_pipeline,
        }

    }
}

pub(crate) fn dispatch_count(total: u32) -> u32 {
    total.div_ceil(WORKGROUP_SIZE)
}

pub(crate) fn fft_dispatch_count(total: u32) -> u32 {
    total.div_ceil(FFT_WORKGROUP_SIZE)
}
