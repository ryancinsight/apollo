//! GPU execution kernel for the graph Fourier transform.
//!
//! Forward (mode 0): `X[k] = sum_i U[i,k] * x[i]`  (U^T x)
//! Inverse (mode 1): `x[i] = sum_k U[i,k] * X[k]`  (U X)
//!
//! The basis matrix U is column-major: `basis[i + k*N] = U[i,k]`.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use apollo_wgpu_helpers::hephaestus_wgpu::ComputeDevice;
use apollo_wgpu_helpers::WgpuDevice;

const WORKGROUP_SIZE: u32 = 64;

/// Uniform parameter block (16 bytes). Fields match WGSL GftParams exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GftParams {
    len: u32,
    mode: u32,
    _padding: [u32; 2],
}

/// Cached WGPU pipeline and layout state for repeated GFT dispatches.
#[derive(Debug)]
pub struct GftGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl GftGpuKernel {
    /// Compile the GFT shader and allocate the uniform parameter buffer.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-gft-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gft.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-gft-wgpu bgl"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(std::mem::size_of::<GftParams>() as u64)
                                .expect("nonzero uniform size"),
                        ),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-gft-wgpu pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-gft-wgpu pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("gft_transform"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            bind_group_layout,
            pipeline,
        }
    }

    /// Execute one GFT dispatch on the GPU.
    ///
    /// mode 0 = forward (U^T x), mode 1 = inverse (U X).
    pub fn execute(
        &self,
        device: &WgpuDevice,
        input: &[f32],
        basis: &[f32],
        len: usize,
        mode: u32,
    ) -> WgpuResult<Vec<f32>> {
        let hep_device = device.hephaestus();
        let input_buf = hep_device
            .upload(input)
            .map_err(|e| WgpuError::BufferMapFailed {
                message: e.to_string(),
            })?;
        let output_buf =
            hep_device
                .alloc_zeroed::<f32>(len)
                .map_err(|e| WgpuError::BufferMapFailed {
                    message: e.to_string(),
                })?;
        let basis_buf = hep_device
            .upload(basis)
            .map_err(|e| WgpuError::BufferMapFailed {
                message: e.to_string(),
            })?;
        let params_buffer = device
            .inner()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-gft-wgpu params"),
                contents: bytemuck::bytes_of(&GftParams {
                    len: len as u32,
                    mode,
                    _padding: [0; 2],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-gft-wgpu bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: basis_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-gft-wgpu encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-gft-wgpu pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(len as u32), 1, 1);
        }
        device.queue().submit(std::iter::once(encoder.finish()));

        let mut output = vec![0.0; len];
        hep_device
            .download(&output_buf, &mut output)
            .map_err(|e| WgpuError::BufferMapFailed {
                message: e.to_string(),
            })?;
        Ok(output)
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}
