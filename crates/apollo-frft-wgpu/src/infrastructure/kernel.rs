//! GPU execution kernel for the discrete fractional Fourier transform.
//!
//! Evaluates the direct O(N^2) FrFT on centred coordinates.
//! Five dispatch modes cover integer-order degenerate cases and general chirp.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use num_complex::Complex32;
use wgpu::util::DeviceExt;

use crate::domain::error::{WgpuError, WgpuResult};

const WORKGROUP_SIZE: u32 = 64;

/// Uniform parameter block (32 bytes). Fields match WGSL FrftParams exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FrftParams {
    len: u32,
    mode: u32,
    cot: f32,
    csc: f32,
    scale_re: f32,
    scale_im: f32,
    _padding: [u32; 2],
}

/// Cached WGPU pipeline and layout state for repeated FrFT dispatches.
#[derive(Debug)]
pub struct FrftGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
}

impl FrftGpuKernel {
    /// Compile the FrFT shader and allocate the uniform parameter buffer.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-frft-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/frft.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-frft-wgpu bgl"),
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
                            std::num::NonZeroU64::new(std::mem::size_of::<FrftParams>() as u64)
                                .expect("nonzero uniform size"),
                        ),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-frft-wgpu pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-frft-wgpu pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("frft_transform"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-frft-wgpu params"),
            contents: bytemuck::bytes_of(&FrftParams {
                len: 0,
                mode: 0,
                cot: 0.0,
                csc: 0.0,
                scale_re: 1.0,
                scale_im: 0.0,
                _padding: [0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            bind_group_layout,
            params_buffer,
            pipeline,
        }
    }

    /// Execute one FrFT dispatch on the GPU.
    ///
    /// mode: 0=identity, 1=centred DFT, 2=reversal, 3=centred IDFT, 4=general.
    /// cot/csc/scale_re/scale_im are used only for mode 4.
    pub fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[Complex32],
        len: usize,
        mode: u32,
        cot: f32,
        csc: f32,
        scale_re: f32,
        scale_im: f32,
    ) -> WgpuResult<Vec<Complex32>> {
        let byte_len = (len * std::mem::size_of::<Complex32>()) as u64;
        let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-frft-wgpu input"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-frft-wgpu output"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-frft-wgpu staging"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&FrftParams {
                len: len as u32,
                mode,
                cot,
                csc,
                scale_re,
                scale_im,
                _padding: [0; 2],
            }),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-frft-wgpu bind group"),
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
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-frft-wgpu encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-frft-wgpu pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(len as u32), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, byte_len);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return Err(WgpuError::BufferMapFailed {
                    message: e.to_string(),
                })
            }
            Err(e) => {
                return Err(WgpuError::BufferMapFailed {
                    message: e.to_string(),
                })
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
