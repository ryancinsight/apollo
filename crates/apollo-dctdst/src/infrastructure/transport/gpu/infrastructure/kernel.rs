//! GPU execution for DCT-II, DCT-III, DST-II, DST-III, DCT-I, DCT-IV, DST-I, and DST-IV.
//!
//! For Apollo's unnormalized convention,
//! `DCT2_k(x) = sum_n x[n] cos(pi/N * (n + 1/2) * k)` and
//! `DCT3_k(x) = 1/2 x[0] + sum_{n=1}^{N-1} x[n] cos(pi/N * n * (k + 1/2))`.
//! The inverse pair satisfies `DCT3(DCT2(x)) = (N / 2) x`, so the inverse path
//! reuses the opposite cosine kernel followed by multiplication by `2 / N`.
//! Likewise, `DST3(DST2(x)) = (N / 2) x`, so the sine-transform inverse path
//! reuses the paired sine kernel with the same normalization.

use std::cell::RefCell;
use std::collections::HashMap;

use apollo_wgpu_helpers::hephaestus_wgpu::{ComputeDevice, WgpuBuffer};
use apollo_wgpu_helpers::WgpuDevice;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};

const WORKGROUP_SIZE: u32 = 64;

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
/// Implemented real-to-real transform modes: DCT-II, DCT-III, DST-II, DST-III, DCT-I, DCT-IV, DST-I, and DST-IV.
pub enum DctMode {
    /// Type-II discrete cosine transform.
    Dct2 = 0,
    /// Type-III discrete cosine transform.
    Dct3 = 1,
    /// Type-II discrete sine transform.
    Dst2 = 2,
    /// Type-III discrete sine transform.
    Dst3 = 3,
    /// Type-I discrete cosine transform.
    Dct1 = 4,
    /// Type-IV discrete cosine transform.
    Dct4 = 5,
    /// Type-I discrete sine transform.
    Dst1 = 6,
    /// Type-IV discrete sine transform.
    Dst4 = 7,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct DctParams {
    len: u32,
    mode: u32,
    scale_bits: u32,
    _padding: u32,
}

/// GPU buffer set reused across same-length dispatches.
///
/// Separable 2D/3D paths issue `O(n)`–`O(n²)` fixed-length 1D dispatches;
/// caching by byte length removes three buffer allocations and one bind-group
/// creation from every dispatch after the first.
#[derive(Debug)]
struct DispatchBuffers {
    byte_len: u64,
    input: WgpuBuffer<f32>,
    output: WgpuBuffer<f32>,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
}

thread_local! {
    static DCT_BUFFERS_CACHE: RefCell<HashMap<usize, Option<DispatchBuffers>>> =
        RefCell::new(HashMap::new());
}

/// Cached WGPU kernel state for repeated DCT/DST dispatches.
#[derive(Debug)]
pub struct DctGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    transform_pipeline: wgpu::ComputePipeline,
    scale_pipeline: wgpu::ComputePipeline,
}

impl Drop for DctGpuKernel {
    fn drop(&mut self) {
        let self_id = self as *const Self as usize;
        let _ = DCT_BUFFERS_CACHE.with(|cache| cache.borrow_mut().remove(&self_id));
    }
}

impl DctGpuKernel {
    /// Compile shader state and allocate the uniform parameter buffer.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-dctdst-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/dct.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-dctdst-wgpu bind group layout"),
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
                            std::num::NonZeroU64::new(std::mem::size_of::<DctParams>() as u64)
                                .expect("nonzero uniform size"),
                        ),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-dctdst-wgpu pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let transform_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-dctdst-wgpu transform pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("dct_transform"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let scale_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-dctdst-wgpu scale pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("dct_scale"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            bind_group_layout,
            transform_pipeline,
            scale_pipeline,
        }
    }

    /// Execute the selected real-to-real mode and apply the requested output scale.
    pub fn execute(
        &self,
        device: &WgpuDevice,
        input: &[f32],
        mode: DctMode,
        scale: f32,
    ) -> WgpuResult<Vec<f32>> {
        let hep_device = device.hephaestus();
        let len = input.len();
        let byte_len = (len * std::mem::size_of::<f32>()) as u64;
        let self_id = self as *const Self as usize;

        let mut map_err = None;
        let output = DCT_BUFFERS_CACHE.with(|cache| {
            let mut cache_borrow = cache.borrow_mut();
            let cache_entry = cache_borrow.entry(self_id).or_insert(None);

            let reusable = matches!(cache_entry.as_ref(), Some(set) if set.byte_len == byte_len);
            if reusable {
                let set = cache_entry.as_ref().unwrap();
                hep_device.write_buffer(&set.input, input).expect("Failed to write to device buffer");
            } else {
                let input_buffer = hep_device.upload(input).expect("Failed to allocate input buffer");
                let output_buffer = hep_device.alloc_zeroed::<f32>(len).expect("Failed to allocate output buffer");
                let params_buffer = device.inner().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("apollo-dctdst-wgpu params"),
                    contents: bytemuck::bytes_of(&DctParams {
                        len: 0,
                        mode: 0,
                        scale_bits: 1.0_f32.to_bits(),
                        _padding: 0,
                    }),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                let bind_group = device.inner().create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("apollo-dctdst-wgpu bind group"),
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
                *cache_entry = Some(DispatchBuffers {
                    byte_len,
                    input: input_buffer,
                    output: output_buffer,
                    bind_group,
                    params_buffer,
                });
            }

            let set = cache_entry.as_ref().unwrap();
            device.queue().write_buffer(
                &set.params_buffer,
                0,
                bytemuck::bytes_of(&DctParams {
                    len: len as u32,
                    mode: mode as u32,
                    scale_bits: scale.to_bits(),
                    _padding: 0,
                }),
            );

            let mut encoder = device.inner().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-dctdst-wgpu encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("apollo-dctdst-wgpu transform pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.transform_pipeline);
                pass.set_bind_group(0, &set.bind_group, &[]);
                pass.dispatch_workgroups(dispatch_count(len as u32), 1, 1);
            }
            if scale != 1.0 {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("apollo-dctdst-wgpu scale pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scale_pipeline);
                pass.set_bind_group(0, &set.bind_group, &[]);
                pass.dispatch_workgroups(dispatch_count(len as u32), 1, 1);
            }
            device.queue().submit(std::iter::once(encoder.finish()));

            let mut output = vec![0.0_f32; len];
            if let Err(e) = hep_device.download(&set.output, &mut output) {
                map_err = Some(e.to_string());
                return Vec::new();
            }
            output
        });

        if let Some(message) = map_err {
            DCT_BUFFERS_CACHE.with(|cache| {
                cache.borrow_mut().insert(self_id, None);
            });
            return Err(WgpuError::BufferMapFailed { message });
        }

        Ok(output)
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}
