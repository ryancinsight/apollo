//! Direct dense DFT GPU kernel used by sparse Fourier execution.
//!
//! The sparse transform's mathematical owner remains `apollo-sft`. This module
//! owns only the WGPU execution policy: compute the dense DFT or inverse DFT in
//! `f32` and return values to the backend for sparse-domain projection.

use bytemuck::{Pod, Zeroable};
use eunomia::Complex32;
use wgpu::util::DeviceExt;

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use apollo_wgpu_helpers::hephaestus_wgpu::ComputeDevice;
use apollo_wgpu_helpers::WgpuDevice;

const WORKGROUP_SIZE: u32 = 64;

/// Execution mode for the direct dense transform.
#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum SftMode {
    /// Forward DFT with negative phase.
    Forward = 0,
    /// Normalized inverse DFT with positive phase.
    Inverse = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ComplexPod {
    re: f32,
    im: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SftParams {
    len: u32,
    mode: u32,
    _padding: [u32; 2],
}

/// Cached WGPU state for direct dense SFT dispatches.
#[derive(Debug)]
pub struct SftGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl SftGpuKernel {
    /// Compile shader state and allocate the uniform parameter buffer.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-sft-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sft.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-sft-wgpu bind group layout"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, false),
                uniform_layout_entry(2),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-sft-wgpu pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-sft-wgpu direct DFT pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sft_direct_dft"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            bind_group_layout,
            pipeline,
        }
    }

    /// Execute the dense direct transform for the requested mode.
    pub fn execute(
        &self,
        device: &WgpuDevice,
        input: &[Complex32],
        len: usize,
        mode: SftMode,
    ) -> WgpuResult<Vec<Complex32>> {
        let hep_device = device.hephaestus();
        let input_data: Vec<ComplexPod> = input
            .iter()
            .map(|value| ComplexPod {
                re: value.re,
                im: value.im,
            })
            .collect();
        let input_buffer =
            hep_device
                .upload(&input_data)
                .map_err(|e| WgpuError::BufferMapFailed {
                    message: e.to_string(),
                })?;
        let output_buffer =
            hep_device
                .alloc_zeroed::<ComplexPod>(len)
                .map_err(|e| WgpuError::BufferMapFailed {
                    message: e.to_string(),
                })?;
        let params_buffer = device
            .inner()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-sft-wgpu params"),
                contents: bytemuck::bytes_of(&SftParams {
                    len: len as u32,
                    mode: mode as u32,
                    _padding: [0; 2],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-sft-wgpu bind group"),
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
                label: Some("apollo-sft-wgpu encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-sft-wgpu direct DFT pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(len as u32), 1, 1);
        }
        device.queue().submit(std::iter::once(encoder.finish()));

        let mut pods = vec![ComplexPod::zeroed(); len];
        hep_device
            .download(&output_buffer, &mut pods)
            .map_err(|e| WgpuError::BufferMapFailed {
                message: e.to_string(),
            })?;

        Ok(pods
            .iter()
            .map(|value| Complex32::new(value.re, value.im))
            .collect())
    }
}

fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(
                std::num::NonZeroU64::new(std::mem::size_of::<SftParams>() as u64)
                    .expect("nonzero uniform size"),
            ),
        },
        count: None,
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}
