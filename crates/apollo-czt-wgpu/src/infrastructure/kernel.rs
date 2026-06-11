//! GPU execution for the direct complex chirp z-transform.
//!
//! For input `x[n]`, starting point `A`, and spiral ratio `W`, the CZT is
//! `X[k] = sum_{n=0}^{N-1} x[n] A^{-n} W^{nk}`. This kernel evaluates that
//! definition directly in `O(NM)` time on the GPU for the implemented `f32`
//! complex execution surface.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use num_complex::Complex32;
use wgpu::util::DeviceExt;

use crate::application::plan::CztWgpuPlan;
use crate::domain::error::{WgpuError, WgpuResult};

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CztParams {
    input_len: u32,
    output_len: u32,
    a_re: f32,
    a_im: f32,
    w_re: f32,
    w_im: f32,
    _padding: [u32; 2],
}

/// Cached WGPU kernel state for repeated CZT dispatches.
#[derive(Debug)]
pub struct CztGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    forward_pipeline: wgpu::ComputePipeline,
    inverse_pipeline: wgpu::ComputePipeline,
}

impl CztGpuKernel {
    /// Compile shader state and allocate the uniform parameter buffer.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-czt-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/czt.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-czt-wgpu bind group layout"),
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
                            std::num::NonZeroU64::new(std::mem::size_of::<CztParams>() as u64)
                                .expect("nonzero uniform size"),
                        ),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-czt-wgpu pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let forward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-czt-wgpu forward pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("czt_forward"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let inverse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-czt-wgpu inverse pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("czt_inverse"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-czt-wgpu params"),
            contents: bytemuck::bytes_of(&CztParams {
                input_len: 0,
                output_len: 0,
                a_re: 0.0,
                a_im: 0.0,
                w_re: 0.0,
                w_im: 0.0,
                _padding: [0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            bind_group_layout,
            params_buffer,
            forward_pipeline,
            inverse_pipeline,
        }
    }

    /// Execute the direct forward CZT.
    pub fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        plan: &CztWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        let output_len = plan.output_len();
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-czt-wgpu input"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-czt-wgpu output"),
            size: (output_len * std::mem::size_of::<Complex32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-czt-wgpu staging"),
            size: (output_len * std::mem::size_of::<Complex32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let a = plan.a();
        let w = plan.w();
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&CztParams {
                input_len: plan.input_len() as u32,
                output_len: output_len as u32,
                a_re: a.re,
                a_im: a.im,
                w_re: w.re,
                w_im: w.im,
                _padding: [0; 2],
            }),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-czt-wgpu bind group"),
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
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-czt-wgpu encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-czt-wgpu forward pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(output_len as u32), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging,
            0,
            (output_len * std::mem::size_of::<Complex32>()) as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait);
        match receiver.recv() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                return Err(WgpuError::BufferMapFailed {
                    message: error.to_string(),
                });
            }
            Err(error) => {
                return Err(WgpuError::BufferMapFailed {
                    message: error.to_string(),
                });
            }
        }

        let output = {
            let mapped = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, Complex32>(&mapped).to_vec()
        };
        staging.unmap();
        Ok(output)
    }

    /// Execute the adjoint inverse CZT.
    ///
    /// Computes `x[n] = (A^n / M) · Σ_k X[k] · W^{-nk}` where `M = N = plan.input_len()`.
    /// Requires `plan.input_len() == plan.output_len()` (square CZT only).
    pub fn execute_inverse(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        plan: &CztWgpuPlan,
        spectrum: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        let n = plan.input_len();
        let m = plan.output_len();

        let spectrum_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-czt-wgpu inverse input (spectrum)"),
            contents: bytemuck::cast_slice(spectrum),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let signal_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-czt-wgpu inverse output (signal)"),
            size: (n * std::mem::size_of::<Complex32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-czt-wgpu inverse staging"),
            size: (n * std::mem::size_of::<Complex32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let a = plan.a();
        let w = plan.w();
        // For the inverse kernel: input_len = N (output size), output_len = M (spectrum size).
        // Since M = N for the square inverse, both params fields hold N.
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&CztParams {
                input_len: n as u32,
                output_len: m as u32,
                a_re: a.re,
                a_im: a.im,
                w_re: w.re,
                w_im: w.im,
                _padding: [0; 2],
            }),
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-czt-wgpu inverse bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spectrum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: signal_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-czt-wgpu inverse encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-czt-wgpu inverse pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inverse_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(n as u32), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &signal_buffer,
            0,
            &staging,
            0,
            (n * std::mem::size_of::<Complex32>()) as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait);
        match receiver.recv() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                return Err(WgpuError::BufferMapFailed {
                    message: error.to_string(),
                });
            }
            Err(error) => {
                return Err(WgpuError::BufferMapFailed {
                    message: error.to_string(),
                });
            }
        }

        let output = {
            let mapped = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, Complex32>(&mapped).to_vec()
        };
        staging.unmap();
        Ok(output)
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}
