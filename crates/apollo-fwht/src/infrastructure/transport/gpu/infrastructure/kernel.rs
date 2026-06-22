//! GPU execution for the 1D Walsh-Hadamard butterfly network.
//!
//! Let `H_n` denote the Hadamard matrix for `n = 2^m`, with entries
//! `H_n[k, j] = (-1)^{popcount(k & j)}`. The radix-2 butterfly factorization
//! of `H_n` is exactly the iterative stage sequence
//! `(a, b) -> (a + b, a - b)` at strides `1, 2, 4, ..., n / 2`.
//! Because `H_n^2 = n I`, the inverse applies the same butterfly network
//! followed by multiplication by `1 / n`.

use std::num::NonZeroU64;
use apollo_wgpu_helpers::hephaestus_wgpu::ComputeDevice;
use apollo_wgpu_helpers::WgpuDevice;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};

const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FwhtParams {
    len: u32,
    stride: u32,
    _padding: [u32; 2],
}

/// Cached WGPU kernel state for repeated FWHT dispatches.
#[derive(Debug)]
pub struct FwhtGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    butterfly_pipeline: wgpu::ComputePipeline,
    scale_pipeline: wgpu::ComputePipeline,
}

impl FwhtGpuKernel {
    /// Compile shader state and allocate the uniform parameter buffer.
    pub fn new(device: &wgpu::Device) -> WgpuResult<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-fwht-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fwht.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-fwht-wgpu bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<f32>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<FwhtParams>() as u64),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-fwht-wgpu pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let butterfly_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-fwht-wgpu butterfly pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("fwht_butterfly"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let scale_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-fwht-wgpu scale pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("fwht_scale_inverse"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Ok(Self {
            bind_group_layout,
            butterfly_pipeline,
            scale_pipeline,
        })
    }

    /// Execute the forward or inverse 1D FWHT on a real-valued `f32` slice.
    pub fn execute(
        &self,
        device: &WgpuDevice,
        input: &[f32],
        inverse: bool,
    ) -> WgpuResult<Vec<f32>> {
        let hep_device = device.hephaestus();
        let len = input.len();
        let storage = hep_device.upload(input).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        let mut stride = 1usize;
        while stride < len {
            let params_buffer = device.inner().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-fwht-wgpu params stage"),
                contents: bytemuck::bytes_of(&FwhtParams {
                    len: len as u32,
                    stride: stride as u32,
                    _padding: [0; 2],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.inner().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-fwht-wgpu bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: storage.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = device.inner().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-fwht-wgpu butterfly encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("apollo-fwht-wgpu butterfly pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.butterfly_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(dispatch_count((len / 2) as u32), 1, 1);
            }
            device.queue().submit(std::iter::once(encoder.finish()));
            stride <<= 1;
        }

        if inverse {
            let params_buffer = device.inner().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-fwht-wgpu params scale"),
                contents: bytemuck::bytes_of(&FwhtParams {
                    len: len as u32,
                    stride: 0,
                    _padding: [0; 2],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.inner().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-fwht-wgpu scale bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: storage.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = device.inner().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-fwht-wgpu scale encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("apollo-fwht-wgpu scale pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scale_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(dispatch_count(len as u32), 1, 1);
            }
            device.queue().submit(std::iter::once(encoder.finish()));
        }

        let mut output = vec![0.0_f32; len];
        hep_device.download(&storage, &mut output).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;
        Ok(output)
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}
