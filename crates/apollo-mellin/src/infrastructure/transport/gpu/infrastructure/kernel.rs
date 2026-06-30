//! GPU execution for the forward Mellin log-frequency spectrum.
//!
//! This kernel mirrors the owning CPU crate:
//! 1. log-resample the positive-domain signal onto the plan scale grid,
//! 2. compute the direct log-frequency Mellin spectrum over that grid.

use bytemuck::{Pod, Zeroable};
use eunomia::Complex32;
use wgpu::util::DeviceExt;

use apollo_wgpu_helpers::hephaestus_wgpu::ComputeDevice;
use apollo_wgpu_helpers::WgpuDevice;
use crate::infrastructure::transport::gpu::application::plan::MellinWgpuPlan;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ComplexPod {
    re: f32,
    im: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MellinParams {
    signal_len: u32,
    samples: u32,
    signal_min: f32,
    signal_max: f32,
    log_min: f32,
    log_max: f32,
    _padding: [u32; 2],
}

/// Uniform params shared by both inverse passes.
/// Layout matches `InverseMellinParams` and `ExpResampleParams` in the WGSL.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct InverseMellinParamsPod {
    samples: u32,
    out_len: u32,
    log_min: f32,
    log_max: f32,
    out_min: f32,
    out_max: f32,
    _pad0: u32,
    _pad1: u32,
}

/// Cached WGPU kernel state for repeated Mellin forward dispatches.
#[derive(Debug)]
pub struct MellinGpuKernel {
    resample_layout: wgpu::BindGroupLayout,
    spectrum_layout: wgpu::BindGroupLayout,
    resample_pipeline: wgpu::ComputePipeline,
    spectrum_pipeline: wgpu::ComputePipeline,
    inverse_spectrum_pipeline: wgpu::ComputePipeline,
    exp_resample_pipeline: wgpu::ComputePipeline,
}

impl MellinGpuKernel {
    /// Compile shader state and allocate the uniform parameter buffer.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-mellin-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mellin.wgsl").into()),
        });
        let resample_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-mellin-wgpu resample layout"),
            entries: &[
                buffer_layout_entry(0, true),
                buffer_layout_entry(1, false),
                uniform_layout_entry(2),
            ],
        });
        let spectrum_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-mellin-wgpu spectrum layout"),
            entries: &[
                buffer_layout_entry(0, true),
                complex_buffer_layout_entry(1),
                uniform_layout_entry(2),
            ],
        });
        let resample_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("apollo-mellin-wgpu resample pipeline layout"),
                bind_group_layouts: &[&resample_layout],
                push_constant_ranges: &[],
            });
        let spectrum_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("apollo-mellin-wgpu spectrum pipeline layout"),
                bind_group_layouts: &[&spectrum_layout],
                push_constant_ranges: &[],
            });
        let resample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-mellin-wgpu resample pipeline"),
            layout: Some(&resample_pipeline_layout),
            module: &shader,
            entry_point: Some("mellin_resample"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let spectrum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-mellin-wgpu spectrum pipeline"),
            layout: Some(&spectrum_pipeline_layout),
            module: &shader,
            entry_point: Some("mellin_spectrum"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let inverse_spectrum_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-mellin-wgpu inverse-spectrum pipeline"),
                layout: Some(&resample_pipeline_layout),
                module: &shader,
                entry_point: Some("mellin_inverse_spectrum"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let exp_resample_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("apollo-mellin-wgpu exp-resample pipeline"),
                layout: Some(&resample_pipeline_layout),
                module: &shader,
                entry_point: Some("mellin_exp_resample"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Self {
            resample_layout,
            spectrum_layout,
            resample_pipeline,
            spectrum_pipeline,
            inverse_spectrum_pipeline,
            exp_resample_pipeline,
        }
    }

    /// Execute the forward Mellin spectrum path.
    pub fn execute(
        &self,
        device: &WgpuDevice,
        plan: &MellinWgpuPlan,
        signal: &[f32],
        signal_min: f64,
        signal_max: f64,
     ) -> WgpuResult<Vec<Complex32>> {
        let hep_device = device.hephaestus();
        let signal_buffer = hep_device.upload(signal).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;
        let resample_buffer = hep_device.alloc_zeroed::<f32>(plan.samples()).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;
        let spectrum_buffer = hep_device.alloc_zeroed::<ComplexPod>(plan.samples()).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        let params_buffer = device.inner().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-mellin-wgpu params"),
            contents: bytemuck::bytes_of(&MellinParams {
                signal_len: signal.len() as u32,
                samples: plan.samples() as u32,
                signal_min: signal_min as f32,
                signal_max: signal_max as f32,
                log_min: plan.min_scale().ln() as f32,
                log_max: plan.max_scale().ln() as f32,
                _padding: [0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let resample_bind_group = device.inner().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-mellin-wgpu resample bind group"),
            layout: &self.resample_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: signal_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resample_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let spectrum_bind_group = device.inner().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-mellin-wgpu spectrum bind group"),
            layout: &self.spectrum_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resample_buffer.as_entire_binding(),
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

        let mut encoder = device.inner().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-mellin-wgpu encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-mellin-wgpu resample pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resample_pipeline);
            pass.set_bind_group(0, &resample_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(plan.samples() as u32), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-mellin-wgpu spectrum pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.spectrum_pipeline);
            pass.set_bind_group(0, &spectrum_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(plan.samples() as u32), 1, 1);
        }
        device.queue().submit(std::iter::once(encoder.finish()));

        let mut pods = vec![ComplexPod::zeroed(); plan.samples()];
        hep_device.download(&spectrum_buffer, &mut pods).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        let output = pods.iter()
            .map(|value| Complex32::new(value.re, value.im))
            .collect();
        Ok(output)
    }

    /// Execute the inverse Mellin spectrum path.
    ///
    /// Two GPU passes: (1) IDFT to recover log-domain samples, (2) exp-resample
    /// back to the linear output domain `[out_min, out_max]` at `out_len` points.
    pub fn execute_inverse(
        &self,
        device: &WgpuDevice,
        plan: &MellinWgpuPlan,
        spectrum: &[Complex32],
        out_min: f64,
        out_max: f64,
        out_len: usize,
    ) -> WgpuResult<Vec<f32>> {
        let n = plan.samples();
        let hep_device = device.hephaestus();

        let spectrum_pods: Vec<ComplexPod> = spectrum
            .iter()
            .map(|v| ComplexPod { re: v.re, im: v.im })
            .collect();

        let spectrum_buffer = hep_device.upload(&spectrum_pods).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;
        let log_samples_buffer = hep_device.alloc_zeroed::<f32>(n).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;
        let output_buffer = hep_device.alloc_zeroed::<f32>(out_len).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        let log_min = plan.min_scale().ln() as f32;
        let log_max = plan.max_scale().ln() as f32;
        let inv_params_buffer = device.inner().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-mellin-wgpu inverse params"),
            contents: bytemuck::bytes_of(&InverseMellinParamsPod {
                samples: n as u32,
                out_len: out_len as u32,
                log_min,
                log_max,
                out_min: out_min as f32,
                out_max: out_max as f32,
                _pad0: 0,
                _pad1: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Pass 1: mellin_inverse_spectrum — spectrum -> log_samples
        let inv_spectrum_bg = device.inner().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-mellin-wgpu inv-spectrum bind group"),
            layout: &self.resample_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spectrum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: log_samples_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: inv_params_buffer.as_entire_binding(),
                },
            ],
        });
        // Pass 2: mellin_exp_resample — log_samples -> output
        let exp_resample_bg = device.inner().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-mellin-wgpu exp-resample bind group"),
            layout: &self.resample_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: log_samples_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: inv_params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.inner().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-mellin-wgpu inverse encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-mellin-wgpu inverse-spectrum pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inverse_spectrum_pipeline);
            pass.set_bind_group(0, &inv_spectrum_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(n as u32), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-mellin-wgpu exp-resample pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.exp_resample_pipeline);
            pass.set_bind_group(0, &exp_resample_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(out_len as u32), 1, 1);
        }
        device.queue().submit(std::iter::once(encoder.finish()));

        let mut output = vec![0.0; out_len];
        hep_device.download(&output_buffer, &mut output).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;
        Ok(output)
    }
}

fn buffer_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

fn complex_buffer_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    buffer_layout_entry(binding, false)
}

fn uniform_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(
                std::num::NonZeroU64::new(std::mem::size_of::<MellinParams>() as u64)
                    .expect("nonzero uniform size"),
            ),
        },
        count: None,
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}
