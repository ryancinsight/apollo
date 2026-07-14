//! GPU compute kernel for the multi-level Haar DWT.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use apollo_wgpu_helpers::hephaestus_wgpu::ComputeDevice;
use apollo_wgpu_helpers::WgpuDevice;

const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct WaveletParams {
    len: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

/// GPU compute kernel encapsulating analysis and synthesis pipelines for the Haar DWT.
#[derive(Debug)]
pub struct WaveletGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    analysis_pipeline: wgpu::ComputePipeline,
    synthesis_pipeline: wgpu::ComputePipeline,
}

impl WaveletGpuKernel {
    /// Create a new kernel by compiling the WGSL shader and building both compute pipelines.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-wavelet-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/wavelet.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-wavelet-wgpu BGL"),
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
                            std::num::NonZeroU64::new(std::mem::size_of::<WaveletParams>() as u64)
                                .expect("nonzero size"),
                        ),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-wavelet-wgpu pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let analysis_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-wavelet-wgpu analysis pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("haar_analysis"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let synthesis_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-wavelet-wgpu synthesis pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("haar_synthesis"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            bind_group_layout,
            analysis_pipeline,
            synthesis_pipeline,
        }
    }

    /// Execute the forward multi-level Haar analysis. Returns Mallat-ordered coefficients.
    pub fn execute_forward(
        &self,
        device: &WgpuDevice,
        signal: &[f32],
        len: usize,
        levels: usize,
    ) -> WgpuResult<Vec<f32>> {
        self.run_passes(device, signal, len, levels, false)
    }

    /// Execute the inverse multi-level Haar synthesis. Expects Mallat-ordered coefficients.
    pub fn execute_inverse(
        &self,
        device: &WgpuDevice,
        coefficients: &[f32],
        len: usize,
        levels: usize,
    ) -> WgpuResult<Vec<f32>> {
        self.run_passes(device, coefficients, len, levels, true)
    }

    fn run_passes(
        &self,
        device: &WgpuDevice,
        input: &[f32],
        len: usize,
        levels: usize,
        inverse: bool,
    ) -> WgpuResult<Vec<f32>> {
        let hep_device = device.hephaestus();

        let main_buf = hep_device
            .upload(input)
            .map_err(|e| WgpuError::BufferMapFailed {
                message: e.to_string(),
            })?;
        let temp_buf =
            hep_device
                .alloc_zeroed::<f32>(len)
                .map_err(|e| WgpuError::BufferMapFailed {
                    message: e.to_string(),
                })?;

        let params_buffer = device
            .inner()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-wavelet-wgpu params"),
                contents: bytemuck::bytes_of(&WaveletParams {
                    len: 0,
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-wavelet-wgpu bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: main_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: temp_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
        let level_lens: Vec<usize> = if inverse {
            (0..levels).rev().map(|l| len >> l).collect()
        } else {
            (0..levels).map(|l| len >> l).collect()
        };
        for &current_len in &level_lens {
            let half = (current_len / 2) as u32;
            let pass_bytes = (current_len * std::mem::size_of::<f32>()) as u64;
            device.queue().write_buffer(
                &params_buffer,
                0,
                bytemuck::bytes_of(&WaveletParams {
                    len: current_len as u32,
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                }),
            );
            let pipeline = if inverse {
                &self.synthesis_pipeline
            } else {
                &self.analysis_pipeline
            };
            let mut encoder =
                device
                    .inner()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("apollo-wavelet-wgpu pass encoder"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("apollo-wavelet-wgpu pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(dispatch_count(half), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&temp_buf, 0, &main_buf, 0, pass_bytes);
            device.queue().submit(std::iter::once(encoder.finish()));
        }

        let mut output = vec![0.0f32; len];
        hep_device
            .download(&main_buf, &mut output)
            .map_err(|e| WgpuError::BufferMapFailed {
                message: e.to_string(),
            })?;
        Ok(output)
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}
