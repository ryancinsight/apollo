//! GPU execution for the 1D Discrete Hartley Transform.
//!
//! The Hartley transform is
//! `H[k] = sum_{n=0}^{N-1} x[n] cas(2*pi*k*n/N)` with `cas(t) = cos(t) + sin(t)`.
//! The Hartley matrix is symmetric and satisfies `H_N^2 = N I`, so the inverse
//! reuses the same kernel followed by multiplication by `1 / N`.

use apollo_wgpu_helpers::hephaestus_wgpu::ComputeDevice;
use apollo_wgpu_helpers::WgpuDevice;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct DhtParams {
    len: u32,
    _padding: [u32; 3],
}

/// Cached WGPU kernel state for repeated DHT dispatches.
#[derive(Debug)]
pub struct DhtGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    transform_pipeline: wgpu::ComputePipeline,
    scale_pipeline: wgpu::ComputePipeline,
}

impl DhtGpuKernel {
    /// Compile shader state and allocate the uniform parameter buffer.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-dht-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/dht.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-dht-wgpu bind group layout"),
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
                            std::num::NonZeroU64::new(std::mem::size_of::<DhtParams>() as u64)
                                .expect("nonzero uniform size"),
                        ),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-dht-wgpu pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let transform_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-dht-wgpu transform pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("dht_transform"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let scale_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-dht-wgpu scale pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("dht_scale_inverse"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            bind_group_layout,
            transform_pipeline,
            scale_pipeline,
        }
    }

    /// Execute the forward or inverse 1D DHT on a real-valued `f32` slice.
    pub fn execute(
        &self,
        device: &WgpuDevice,
        input: &[f32],
        inverse: bool,
    ) -> WgpuResult<Vec<f32>> {
        let hep_device = device.hephaestus();
        let len = input.len();
        let input_buffer = hep_device
            .upload(input)
            .map_err(|e| WgpuError::BufferMapFailed {
                message: e.to_string(),
            })?;
        let output_buffer =
            hep_device
                .alloc_zeroed::<f32>(len)
                .map_err(|e| WgpuError::BufferMapFailed {
                    message: e.to_string(),
                })?;
        let params_buffer = device
            .inner()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-dht-wgpu params"),
                contents: bytemuck::bytes_of(&DhtParams {
                    len: len as u32,
                    _padding: [0; 3],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-dht-wgpu bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-dht-wgpu encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-dht-wgpu transform pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.transform_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(len as u32), 1, 1);
        }
        if inverse {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-dht-wgpu scale pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scale_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(len as u32), 1, 1);
        }
        device.queue().submit(std::iter::once(encoder.finish()));

        let mut output = vec![0.0_f32; len];
        hep_device
            .download(&output_buffer, &mut output)
            .map_err(|e| WgpuError::BufferMapFailed {
                message: e.to_string(),
            })?;
        Ok(output)
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}
