//! GPU execution for direct SDFT bin computation (forward and inverse).
//!
//! Forward: `X[b] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*b*n/N)` for b = 0..K.
//! Matches `SdftPlan::direct_bins` on the CPU.
//!
//! Inverse: `x[n] = (1/K) * sum_{b=0}^{K-1} X[b] * exp(+2*pi*i*b*n/K)` for n = 0..N.
//! Reconstructs the real signal from K complex DFT bins.

use bytemuck::{Pod, Zeroable};
use eunomia::Complex32;
use wgpu::util::DeviceExt;

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use apollo_wgpu_helpers::hephaestus_wgpu::ComputeDevice;
use apollo_wgpu_helpers::WgpuDevice;

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ComplexPod {
    re: f32,
    im: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SdftParams {
    window_len: u32,
    bin_count: u32,
    _padding: [u32; 2],
}

/// Cached WGPU kernel state for repeated SDFT direct-bins dispatches.
#[derive(Debug)]
pub struct SdftGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    forward_pipeline: wgpu::ComputePipeline,
    inverse_pipeline: wgpu::ComputePipeline,
}

impl SdftGpuKernel {
    /// Compile shader state and allocate the uniform parameter buffer.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-sdft-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sdft.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-sdft-wgpu bind group layout"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, false),
                uniform_layout_entry(2),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-sdft-wgpu pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let forward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-sdft-wgpu forward pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sdft_direct_bins"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let inverse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-sdft-wgpu inverse pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sdft_inverse_bins"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            bind_group_layout,
            forward_pipeline,
            inverse_pipeline,
        }
    }

    /// Execute the direct SDFT bins computation.
    pub fn execute(
        &self,
        device: &WgpuDevice,
        window: &[f32],
        window_len: usize,
        bin_count: usize,
    ) -> WgpuResult<Vec<Complex32>> {
        let hep_device = device.hephaestus();
        let input_buffer = hep_device
            .upload(window)
            .map_err(|e| WgpuError::BufferMapFailed {
                message: e.to_string(),
            })?;
        let output_buffer = hep_device
            .alloc_zeroed::<ComplexPod>(bin_count)
            .map_err(|e| WgpuError::BufferMapFailed {
                message: e.to_string(),
            })?;
        let params_buffer = device
            .inner()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-sdft-wgpu params"),
                contents: bytemuck::bytes_of(&SdftParams {
                    window_len: window_len as u32,
                    bin_count: bin_count as u32,
                    _padding: [0; 2],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-sdft-wgpu bind group"),
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
                label: Some("apollo-sdft-wgpu encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-sdft-wgpu direct bins pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(bin_count as u32), 1, 1);
        }
        device.queue().submit(std::iter::once(encoder.finish()));

        let mut pods = vec![ComplexPod::zeroed(); bin_count];
        hep_device
            .download(&output_buffer, &mut pods)
            .map_err(|e| WgpuError::BufferMapFailed {
                message: e.to_string(),
            })?;
        Ok(pods.iter().map(|v| Complex32::new(v.re, v.im)).collect())
    }

    /// Execute the inverse SDFT: reconstruct real signal from complex bins.
    ///
    /// Input `bins` has `bin_count` complex values (interleaved as flat f32 pairs).
    /// Output has `window_len` real values extracted from the real part of each ComplexPod.
    pub fn execute_inverse(
        &self,
        device: &WgpuDevice,
        bins: &[Complex32],
        bin_count: usize,
        window_len: usize,
    ) -> WgpuResult<Vec<f32>> {
        let hep_device = device.hephaestus();
        // Pack complex bins as interleaved f32 pairs for the read-only binding 0.
        let flat_bins: Vec<f32> = bins.iter().flat_map(|c| [c.re, c.im]).collect();

        let input_buffer =
            hep_device
                .upload(&flat_bins)
                .map_err(|e| WgpuError::BufferMapFailed {
                    message: e.to_string(),
                })?;

        let output_buffer = hep_device
            .alloc_zeroed::<ComplexPod>(window_len)
            .map_err(|e| WgpuError::BufferMapFailed {
                message: e.to_string(),
            })?;

        let params_buffer = device
            .inner()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-sdft-wgpu params"),
                contents: bytemuck::bytes_of(&SdftParams {
                    window_len: window_len as u32,
                    bin_count: bin_count as u32,
                    _padding: [0; 2],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-sdft-wgpu inverse bind group"),
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
                label: Some("apollo-sdft-wgpu inverse encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-sdft-wgpu inverse bins pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inverse_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(window_len as u32), 1, 1);
        }

        device.queue().submit(std::iter::once(encoder.finish()));

        let mut pods = vec![ComplexPod::zeroed(); window_len];
        hep_device
            .download(&output_buffer, &mut pods)
            .map_err(|e| WgpuError::BufferMapFailed {
                message: e.to_string(),
            })?;

        Ok(pods.iter().map(|v| v.re).collect())
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
                std::num::NonZeroU64::new(std::mem::size_of::<SdftParams>() as u64)
                    .expect("nonzero uniform size"),
            ),
        },
        count: None,
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}
