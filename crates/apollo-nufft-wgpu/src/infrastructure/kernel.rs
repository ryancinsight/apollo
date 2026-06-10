//! Direct and fast-gridded NUFFT WGPU kernels.

/// GPU buffer allocations and diagnostics.
pub mod buffers;
/// Fast Type-1 execution pipelines.
pub mod fast_type1;
/// Fast Type-2 1D execution pipelines.
pub mod fast_type2_1d;
/// Fast Type-2 3D execution pipelines.
pub mod fast_type2_3d;
/// Direct Type-1 execution pipelines.
pub mod type1;
/// Direct Type-2 execution pipelines.
pub mod type2;
/// Internal helper functions.
pub(crate) mod helpers;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// Re-export GPU buffers so they remain under the same public path.
pub use buffers::{NufftGpuBuffers1D, NufftGpuBuffers3D};

#[cfg(any(test, feature = "diagnostics"))]
pub use buffers::{NufftGridSnapshot, NufftType2GridDiagnostics};

// Re-export helpers to keep imports in submodules simple.
pub(crate) use helpers::{
    binding, complex_to_pods, dispatch_count, positions_to_complex_pods_1d,
    read_complex_buffer, read_complex_buffer_with_staging, real_to_complex_pods,
    real_to_complex_pods_scaled, split_grid_buffers, storage_buffer,
};

#[cfg(any(test, feature = "diagnostics"))]
pub(crate) use helpers::read_split_grid_snapshot;

pub(crate) const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct ComplexPod {
    pub(crate) re: f32,
    pub(crate) im: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct Position3Pod {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) z: f32,
    pub(crate) _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct NufftParams {
    pub(crate) n0: u32,
    pub(crate) n1: u32,
    pub(crate) n2: u32,
    pub(crate) sample_count: u32,
    pub(crate) l0: f32,
    pub(crate) l1: f32,
    pub(crate) l2: f32,
    pub(crate) _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct FastNufftParams {
    pub(crate) n: u32,
    pub(crate) m: u32,
    pub(crate) sample_count: u32,
    pub(crate) kernel_width: u32,
    pub(crate) length: f32,
    pub(crate) beta: f32,
    pub(crate) i0_beta: f32,
    pub(crate) _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct FastNufftParams3D {
    pub(crate) nx: u32,
    pub(crate) ny: u32,
    pub(crate) nz: u32,
    pub(crate) mx: u32,
    pub(crate) my: u32,
    pub(crate) mz: u32,
    pub(crate) sample_count: u32,
    pub(crate) kernel_width: u32,
    pub(crate) lx: f32,
    pub(crate) ly: f32,
    pub(crate) lz: f32,
    pub(crate) beta: f32,
    pub(crate) i0_beta: f32,
    pub(crate) _pad0: f32,
    pub(crate) _pad1: f32,
    pub(crate) _pad2: f32,
}

/// Cached WGPU state for direct and fast-gridded NUFFT dispatches.
#[derive(Debug)]
pub struct NufftGpuKernel {
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) fast_spread_layout: wgpu::BindGroupLayout,
    pub(crate) fast_extract_layout: wgpu::BindGroupLayout,
    pub(crate) fast_3d_spread_layout: wgpu::BindGroupLayout,
    pub(crate) fast_3d_extract_layout: wgpu::BindGroupLayout,
    pub(crate) params_buffer: wgpu::Buffer,
    pub(crate) fast_params_buffer: wgpu::Buffer,
    pub(crate) layout_padding_buffer: wgpu::Buffer,
    pub(crate) type1_1d_pipeline: wgpu::ComputePipeline,
    pub(crate) type2_1d_pipeline: wgpu::ComputePipeline,
    pub(crate) type1_3d_pipeline: wgpu::ComputePipeline,
    pub(crate) type2_3d_pipeline: wgpu::ComputePipeline,
    pub(crate) fast_type1_spread_1d_pipeline: wgpu::ComputePipeline,
    pub(crate) fast_type1_extract_1d_pipeline: wgpu::ComputePipeline,
    pub(crate) fast_type2_load_1d_pipeline: wgpu::ComputePipeline,
    pub(crate) fast_type2_interpolate_1d_pipeline: wgpu::ComputePipeline,
    pub(crate) fast_3d_params_buffer: wgpu::Buffer,
    pub(crate) fast_type1_spread_3d_pipeline: wgpu::ComputePipeline,
    pub(crate) fast_type1_extract_3d_pipeline: wgpu::ComputePipeline,
    pub(crate) fast_type2_load_3d_pipeline: wgpu::ComputePipeline,
    pub(crate) fast_type2_interpolate_3d_pipeline: wgpu::ComputePipeline,
}

impl NufftGpuKernel {
    /// Compile shader state and allocate the uniform parameter buffer.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-nufft-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/nufft.wgsl").into()),
        });
        let fast_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-nufft-wgpu fast gridding shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/nufft_fast_1d.wgsl").into()),
        });
        let fast_3d_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-nufft-wgpu fast 3d gridding shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/nufft_fast_3d.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-nufft-wgpu bind group layout"),
            entries: &[
                helpers::storage_layout_entry(0, true),
                helpers::storage_layout_entry(1, true),
                helpers::storage_layout_entry(2, false),
                helpers::uniform_layout_entry(3),
            ],
        });
        let fast_spread_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("apollo-nufft-wgpu fast 1d layout"),
                entries: &[
                    helpers::storage_layout_entry(0, true),
                    helpers::storage_layout_entry(1, true),
                    helpers::storage_layout_entry(2, false),
                    helpers::storage_layout_entry(3, false),
                    helpers::storage_layout_entry(4, true),
                    helpers::storage_layout_entry(5, false),
                    helpers::storage_layout_entry(6, true),
                    helpers::uniform_layout_entry(7),
                ],
            });
        let fast_extract_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("apollo-nufft-wgpu fast 1d layout duplicate"),
                entries: &[
                    helpers::storage_layout_entry(0, true),
                    helpers::storage_layout_entry(1, true),
                    helpers::storage_layout_entry(2, false),
                    helpers::storage_layout_entry(3, false),
                    helpers::storage_layout_entry(4, true),
                    helpers::storage_layout_entry(5, false),
                    helpers::storage_layout_entry(6, true),
                    helpers::uniform_layout_entry(7),
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-nufft-wgpu pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let fast_spread_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("apollo-nufft-wgpu fast spread pipeline layout"),
                bind_group_layouts: &[&fast_spread_layout],
                push_constant_ranges: &[],
            });
        let fast_extract_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("apollo-nufft-wgpu fast extract pipeline layout"),
                bind_group_layouts: &[&fast_extract_layout],
                push_constant_ranges: &[],
            });
        let fast_3d_spread_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("apollo-nufft-wgpu fast 3d spread layout"),
                entries: &[
                    helpers::storage_layout_entry(0, true),
                    helpers::storage_layout_entry(1, true),
                    helpers::storage_layout_entry(2, false),
                    helpers::storage_layout_entry(3, false),
                    helpers::storage_layout_entry(4, true),
                    helpers::storage_layout_entry(5, false),
                    helpers::storage_layout_entry(6, true),
                    helpers::uniform_layout_entry_3d(7),
                ],
            });
        let fast_3d_extract_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("apollo-nufft-wgpu fast 3d extract layout"),
                entries: &[
                    helpers::storage_layout_entry(0, true),
                    helpers::storage_layout_entry(1, true),
                    helpers::storage_layout_entry(2, false),
                    helpers::storage_layout_entry(3, false),
                    helpers::storage_layout_entry(4, true),
                    helpers::storage_layout_entry(5, false),
                    helpers::storage_layout_entry(6, true),
                    helpers::uniform_layout_entry_3d(7),
                ],
            });
        let fast_3d_spread_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("apollo-nufft-wgpu fast 3d spread pipeline layout"),
                bind_group_layouts: &[&fast_3d_spread_layout],
                push_constant_ranges: &[],
            });
        let fast_3d_extract_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("apollo-nufft-wgpu fast 3d extract pipeline layout"),
                bind_group_layouts: &[&fast_3d_extract_layout],
                push_constant_ranges: &[],
            });
        let type1_1d_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-nufft-wgpu type1 1d pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("nufft_type1_1d"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let type1_3d_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-nufft-wgpu type1 3d pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("nufft_type1_3d"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let type2_1d_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-nufft-wgpu type2 1d pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("nufft_type2_1d"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let type2_3d_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-nufft-wgpu type2 3d pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("nufft_type2_3d"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let fast_type1_spread_1d_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-nufft-wgpu fast type1 1d spread pipeline"),
                layout: Some(&fast_spread_pipeline_layout),
                module: &fast_shader,
                entry_point: Some("fast_type1_spread_1d"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fast_type1_extract_1d_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-nufft-wgpu fast type1 1d extract pipeline"),
                layout: Some(&fast_extract_pipeline_layout),
                module: &fast_shader,
                entry_point: Some("fast_type1_extract_1d"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fast_type2_load_1d_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-nufft-wgpu fast type2 1d load pipeline"),
                layout: Some(&fast_extract_pipeline_layout),
                module: &fast_shader,
                entry_point: Some("fast_type2_load_1d"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fast_type2_interpolate_1d_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-nufft-wgpu fast type2 1d interpolate pipeline"),
                layout: Some(&fast_spread_pipeline_layout),
                module: &fast_shader,
                entry_point: Some("fast_type2_interpolate_1d"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fast_3d_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-nufft-wgpu fast 3d params"),
            contents: bytemuck::bytes_of(&FastNufftParams3D {
                nx: 1,
                ny: 1,
                nz: 1,
                mx: 2,
                my: 2,
                mz: 2,
                sample_count: 0,
                kernel_width: 6,
                lx: 1.0,
                ly: 1.0,
                lz: 1.0,
                beta: 1.0,
                i0_beta: 1.0,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let fast_type1_spread_3d_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-nufft-wgpu fast type1 3d spread pipeline"),
                layout: Some(&fast_3d_spread_pipeline_layout),
                module: &fast_3d_shader,
                entry_point: Some("fast_type1_spread_3d"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fast_type1_extract_3d_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-nufft-wgpu fast type1 3d extract pipeline"),
                layout: Some(&fast_3d_extract_pipeline_layout),
                module: &fast_3d_shader,
                entry_point: Some("fast_type1_extract_3d"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fast_type2_load_3d_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-nufft-wgpu fast type2 3d load pipeline"),
                layout: Some(&fast_3d_extract_pipeline_layout),
                module: &fast_3d_shader,
                entry_point: Some("fast_type2_load_3d"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let fast_type2_interpolate_3d_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-nufft-wgpu fast type2 3d interpolate pipeline"),
                layout: Some(&fast_3d_spread_pipeline_layout),
                module: &fast_3d_shader,
                entry_point: Some("fast_type2_interpolate_3d"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-nufft-wgpu params"),
            contents: bytemuck::bytes_of(&NufftParams {
                n0: 0,
                n1: 1,
                n2: 1,
                sample_count: 0,
                l0: 1.0,
                l1: 1.0,
                l2: 1.0,
                _pad: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let fast_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-nufft-wgpu fast params"),
            contents: bytemuck::bytes_of(&FastNufftParams {
                n: 0,
                m: 0,
                sample_count: 0,
                kernel_width: 0,
                length: 1.0,
                beta: 1.0,
                i0_beta: 1.0,
                _pad: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let layout_padding_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-nufft-wgpu layout padding complex"),
            contents: bytemuck::bytes_of(&ComplexPod { re: 0.0, im: 0.0 }),
            usage: wgpu::BufferUsages::STORAGE,
        });
        Self {
            bind_group_layout,
            fast_spread_layout,
            fast_extract_layout,
            fast_3d_spread_layout,
            fast_3d_extract_layout,
            params_buffer,
            fast_params_buffer,
            layout_padding_buffer,
            type1_1d_pipeline,
            type2_1d_pipeline,
            type1_3d_pipeline,
            type2_3d_pipeline,
            fast_type1_spread_1d_pipeline,
            fast_type1_extract_1d_pipeline,
            fast_type2_load_1d_pipeline,
            fast_type2_interpolate_1d_pipeline,
            fast_3d_params_buffer,
            fast_type1_spread_3d_pipeline,
            fast_type1_extract_3d_pipeline,
            fast_type2_load_3d_pipeline,
            fast_type2_interpolate_3d_pipeline,
        }
    }
}

#[cfg(feature = "debug-readbacks")]
impl NufftGpuKernel {
    /// Read back 1D grid.
    pub fn read_grid_1d(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        re_buffer: &wgpu::Buffer,
        im_buffer: &wgpu::Buffer,
        len: usize,
    ) -> NufftWgpuResult<Vec<Complex32>> {
        helpers::read_split_grid(device, queue, re_buffer, im_buffer, len)
    }

    /// Read back 3D grid.
    pub fn read_grid_3d(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        re_buffer: &wgpu::Buffer,
        im_buffer: &wgpu::Buffer,
        len: usize,
    ) -> NufftWgpuResult<Vec<Complex32>> {
        helpers::read_split_grid(device, queue, re_buffer, im_buffer, len)
    }
}
