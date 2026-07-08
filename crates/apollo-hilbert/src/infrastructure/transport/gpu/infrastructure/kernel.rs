//! GPU execution for the discrete Hilbert transform and analytic signal.
//!
//! Forward path (3 passes in one encoder):
//!   1. hilbert_forward_dft: DFT of real input -> spectrum in inout_b
//!   2. hilbert_apply_mask: double positive frequencies, zero negative -> inout_b
//!   3. hilbert_inverse_dft: IDFT of masked spectrum -> analytic signal in output
//!
//! Inverse path (3 passes in one encoder):
//!   1. hilbert_forward_dft: DFT of quadrature input -> spectrum
//!   2. hilbert_undo_mask: divide positive by 2, reconstruct negative via hermitian symmetry -> recovered
//!   3. hilbert_inverse_dft: IDFT of recovered spectrum -> original real signal in output

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
struct HilbertParams {
    len: u32,
    _padding: [u32; 3],
}

/// Cached WGPU kernel state for repeated Hilbert dispatches.
#[derive(Debug)]
pub struct HilbertGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    forward_pipeline: wgpu::ComputePipeline,
    mask_pipeline: wgpu::ComputePipeline,
    inverse_pipeline: wgpu::ComputePipeline,
    inverse_mask_pipeline: wgpu::ComputePipeline,
}

impl HilbertGpuKernel {
    /// Compile shader state and allocate the uniform parameter buffer.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-hilbert-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/hilbert.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-hilbert-wgpu bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                            std::num::NonZeroU64::new(std::mem::size_of::<HilbertParams>() as u64)
                                .expect("nonzero uniform size"),
                        ),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-hilbert-wgpu pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let forward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-hilbert-wgpu forward pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("hilbert_forward_dft"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let mask_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-hilbert-wgpu mask pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("hilbert_apply_mask"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let inverse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-hilbert-wgpu inverse pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("hilbert_inverse_dft"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let inverse_mask_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-hilbert-wgpu inverse mask pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("hilbert_inverse_mask"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Self {
            bind_group_layout,
            forward_pipeline,
            mask_pipeline,
            inverse_pipeline,
            inverse_mask_pipeline,
        }
    }

    /// Execute the analytic signal path: `x[n] + i H{x}[n]`.
    pub fn execute(&self, device: &WgpuDevice, input: &[f32]) -> WgpuResult<Vec<Complex32>> {
        let len = input.len();
        let hep_device = device.hephaestus();
        let input_data: Vec<ComplexPod> = input
            .iter()
            .map(|value| ComplexPod {
                re: *value,
                im: 0.0,
            })
            .collect();
        let real_buffer =
            hep_device
                .upload(&input_data)
                .map_err(|e| WgpuError::BufferMapFailed {
                    message: e.to_string(),
                })?;
        let spectrum_buffer =
            hep_device
                .alloc_zeroed::<ComplexPod>(len)
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
                label: Some("apollo-hilbert-wgpu params"),
                contents: bytemuck::bytes_of(&HilbertParams {
                    len: len as u32,
                    _padding: [0; 3],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let forward_bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-hilbert-wgpu forward bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: real_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: spectrum_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
        let spectrum_bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-hilbert-wgpu spectrum bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: spectrum_buffer.as_entire_binding(),
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
                label: Some("apollo-hilbert-wgpu encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-hilbert-wgpu forward pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward_pipeline);
            pass.set_bind_group(0, &forward_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(len as u32), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-hilbert-wgpu mask pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mask_pipeline);
            pass.set_bind_group(0, &forward_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(len as u32), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-hilbert-wgpu inverse pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inverse_pipeline);
            pass.set_bind_group(0, &spectrum_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(len as u32), 1, 1);
        }
        device.queue().submit(std::iter::once(encoder.finish()));

        let mut pods = vec![ComplexPod::zeroed(); len];
        hep_device
            .download(&output_buffer, &mut pods)
            .map_err(|e| WgpuError::BufferMapFailed {
                message: e.to_string(),
            })?;
        let mut output: Vec<Complex32> = pods
            .iter()
            .map(|value| Complex32::new(value.re, value.im))
            .collect();

        // Overwrite the real part with the original input so that the output
        // represents the analytic signal x + i H{x} with numerically exact real part.
        for (sample, original) in output.iter_mut().zip(input.iter()) {
            sample.re = *original;
        }
        Ok(output)
    }

    /// Execute the inverse Hilbert transform: recover `x[n]` from `H{x}[n]`.
    ///
    /// Algorithm (3 GPU passes in one command encoder):
    ///   1. DFT of quadrature input: `Q[k] = M[k] * X[k]` where M is the analytic mask
    ///   2. Undo the analytic mask: divide positive by 2, reconstruct negative via
    ///      hermitian symmetry. Uses a separate recovered_buffer to avoid in-place
    ///      data races between threads reading positive bins and writing negative bins.
    ///   3. IDFT of recovered spectrum: gives the original real signal `x[n]`
    ///
    /// Mathematical justification: for a real signal x, the DFT coefficients satisfy
    /// `X[N-k] = conj(X[k])`. The forward mask M doubles positive frequencies and zeros
    /// negative frequencies. Since `Q[k] = M[k]*X[k]`, we recover `X[k] = Q[k]/M[k]`
    /// for positive frequencies and `X[k] = conj(X[N-k])` for negative frequencies.
    pub fn execute_inverse(&self, device: &WgpuDevice, quadrature: &[f32]) -> WgpuResult<Vec<f32>> {
        let len = quadrature.len();
        let hep_device = device.hephaestus();
        let input_data: Vec<ComplexPod> = quadrature
            .iter()
            .map(|value| ComplexPod {
                re: *value,
                im: 0.0,
            })
            .collect();
        let input_buffer =
            hep_device
                .upload(&input_data)
                .map_err(|e| WgpuError::BufferMapFailed {
                    message: e.to_string(),
                })?;
        let spectrum_buffer =
            hep_device
                .alloc_zeroed::<ComplexPod>(len)
                .map_err(|e| WgpuError::BufferMapFailed {
                    message: e.to_string(),
                })?;
        // Separate buffer for mask-undo output to avoid in-place data races:
        // the undo-mask reads positive-frequency bins (from spectrum) and writes
        // reconstructed negative-frequency bins (to recovered). Using separate
        // buffers ensures no cross-thread data race.
        let recovered_buffer =
            hep_device
                .alloc_zeroed::<ComplexPod>(len)
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
                label: Some("apollo-hilbert-wgpu params"),
                contents: bytemuck::bytes_of(&HilbertParams {
                    len: len as u32,
                    _padding: [0; 3],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        // Pass 1: DFT of quadrature. inout_a = input, inout_b = spectrum.
        let dft_bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-hilbert-wgpu inverse dft bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: spectrum_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
        // Pass 2: Undo mask. Read spectrum (binding 0), write recovered (binding 1).
        let undo_bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-hilbert-wgpu undo mask bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: spectrum_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: recovered_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
        // Pass 3: IDFT. Read recovered (binding 0), write output (binding 1).
        let idft_bind_group = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-hilbert-wgpu inverse idft bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: recovered_buffer.as_entire_binding(),
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
                label: Some("apollo-hilbert-wgpu inverse encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-hilbert-wgpu inverse dft pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward_pipeline);
            pass.set_bind_group(0, &dft_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(len as u32), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-hilbert-wgpu undo mask pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inverse_mask_pipeline);
            pass.set_bind_group(0, &undo_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(len as u32), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-hilbert-wgpu inverse idft pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inverse_pipeline);
            pass.set_bind_group(0, &idft_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(len as u32), 1, 1);
        }
        device.queue().submit(std::iter::once(encoder.finish()));

        let mut pods = vec![ComplexPod::zeroed(); len];
        hep_device
            .download(&output_buffer, &mut pods)
            .map_err(|e| WgpuError::BufferMapFailed {
                message: e.to_string(),
            })?;
        let output: Vec<f32> = pods.iter().map(|v| v.re).collect();
        Ok(output)
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}
