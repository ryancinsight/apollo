//! WGPU radix-5 batch dispatch for matrix-factorized column kernels.

use crate::f16;
use eunomia::Complex32;
use std::sync::Arc;
use wgpu::util::DeviceExt;

const RADIX5_SHADER: &str = include_str!("../../shaders/radix5_batch.wgsl");

/// WGPU compute pipeline for batched radix-5 columns using native shader-f16 arithmetic.
///
/// # Theorem
///
/// The shader indexes packed pairs by `point * pair_count + pair`, which is the
/// row-major bijection used by `crate::domain::storage::FftPlanarMut`
/// after grouping adjacent lanes in `vec2<f16>` storage. One compute invocation
/// owns one lane pair, so distinct invocations write disjoint output pairs for
/// all five radix bins. The shader arithmetic matches the CPU radix-5
/// Winograd/FMA projection in native f16. Devices without
/// `wgpu::Features::SHADER_F16` are rejected instead of widened to f32.
pub struct GpuRadix5Batch {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    lanes: usize,
    pair_count: usize,
    pipeline: wgpu::ComputePipeline,
    data_layout: wgpu::BindGroupLayout,
    params_layout: wgpu::BindGroupLayout,
}

impl GpuRadix5Batch {
    /// Request a default WGPU device and queue for validation or local dispatch.
    pub fn try_default_device() -> Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>), String> {
        let dev = hephaestus_wgpu::WgpuDevice::try_default_with_features_and_limits(
            "apollo-fft-wgpu radix5 batch",
            wgpu::Features::SHADER_F16,
            wgpu::Limits::downlevel_defaults(),
        )
        .map_err(|e| e.to_string())?;
        Ok((dev.device().clone(), dev.queue().clone()))
    }

    /// Create a radix-5 batch dispatcher for `lanes` independent columns.
    ///
    /// # Panics
    ///
    /// Panics if `lanes == 0` or `lanes > 8`.
    #[must_use]
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, lanes: usize) -> Self {
        assert!(lanes > 0, "radix-5 GPU batch requires at least one lane");
        assert!(lanes <= 8, "radix-5 GPU batch accepts at most 8 lanes");
        let pair_count = lanes.div_ceil(2);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-fft-wgpu radix5 batch shader"),
            source: wgpu::ShaderSource::Wgsl(RADIX5_SHADER.into()),
        });
        let data_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-fft-wgpu radix5 batch data layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
            ],
        });
        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-fft-wgpu radix5 batch params layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-fft-wgpu radix5 batch pipeline layout"),
            bind_group_layouts: &[&data_layout, &params_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-fft-wgpu radix5 batch pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("radix5_batch_native_f16"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            device,
            queue,
            lanes,
            pair_count,
            pipeline,
            data_layout,
            params_layout,
        }
    }

    /// Dispatch an unnormalized radix-5 batch and return row-major bins.
    ///
    /// When `inverse` is `false`, computes the forward-sign DFT.
    /// When `inverse` is `true`, computes the inverse-sign DFT without `1/N`
    /// normalization.
    #[must_use]
    pub fn transform_f16(
        &self,
        re: &[[f16; 8]; 5],
        im: &[[f16; 8]; 5],
        inverse: bool,
    ) -> Vec<Complex32> {
        let packed_len = 5 * self.pair_count;
        let mut host_re = vec![0_u32; packed_len];
        let mut host_im = vec![0_u32; packed_len];
        for pair in 0..self.pair_count {
            let lane0 = 2 * pair;
            let lane1 = lane0 + 1;
            for point in 0..5 {
                let idx = point * self.pair_count + pair;
                host_re[idx] = pack_f16_pair(
                    re[point][lane0],
                    if lane1 < self.lanes {
                        re[point][lane1]
                    } else {
                        f16::ZERO
                    },
                );
                host_im[idx] = pack_f16_pair(
                    im[point][lane0],
                    if lane1 < self.lanes {
                        im[point][lane1]
                    } else {
                        f16::ZERO
                    },
                );
            }
        }

        let re_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-fft-wgpu radix5 input re"),
                contents: bytemuck::cast_slice(&host_re),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let im_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-fft-wgpu radix5 input im"),
                contents: bytemuck::cast_slice(&host_im),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_size = (packed_len * std::mem::size_of::<u32>()) as u64;
        let output_re = storage_output(
            &self.device,
            "apollo-fft-wgpu radix5 output re",
            output_size,
        );
        let output_im = storage_output(
            &self.device,
            "apollo-fft-wgpu radix5 output im",
            output_size,
        );
        let staging_re = staging_buffer(
            &self.device,
            "apollo-fft-wgpu radix5 staging re",
            output_size,
        );
        let staging_im = staging_buffer(
            &self.device,
            "apollo-fft-wgpu radix5 staging im",
            output_size,
        );
        let params = [
            self.lanes as u32,
            self.pair_count as u32,
            if inverse { 1 } else { 0 },
            0,
        ];
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-fft-wgpu radix5 params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let data_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-fft-wgpu radix5 data bind group"),
            layout: &self.data_layout,
            entries: &[
                buffer_entry(0, &re_buf),
                buffer_entry(1, &im_buf),
                buffer_entry(2, &output_re),
                buffer_entry(3, &output_im),
            ],
        });
        let params_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-fft-wgpu radix5 params bind group"),
            layout: &self.params_layout,
            entries: &[buffer_entry(0, &params_buf)],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-fft-wgpu radix5 encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-fft-wgpu radix5 pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &data_bg, &[]);
            pass.set_bind_group(1, &params_bg, &[]);
            pass.dispatch_workgroups((self.pair_count as u32).div_ceil(128), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_re, 0, &staging_re, 0, output_size);
        encoder.copy_buffer_to_buffer(&output_im, 0, &staging_im, 0, output_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let re_slice = staging_re.slice(..output_size);
        let im_slice = staging_im.slice(..output_size);
        re_slice.map_async(wgpu::MapMode::Read, |_| {});
        im_slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait);
        let mapped_re = re_slice.get_mapped_range();
        let mapped_im = im_slice.get_mapped_range();
        let packed_re: &[u32] = bytemuck::cast_slice(&mapped_re);
        let packed_im: &[u32] = bytemuck::cast_slice(&mapped_im);
        let mut out = vec![Complex32::new(0.0, 0.0); 5 * self.lanes];
        for pair in 0..self.pair_count {
            let lane0 = 2 * pair;
            let lane1 = lane0 + 1;
            for point in 0..5 {
                let idx = point * self.pair_count + pair;
                let (re0, re1) = unpack_f16_pair(packed_re[idx]);
                let (im0, im1) = unpack_f16_pair(packed_im[idx]);
                out[point * self.lanes + lane0] = Complex32::new(re0.to_f32(), im0.to_f32());
                if lane1 < self.lanes {
                    out[point * self.lanes + lane1] = Complex32::new(re1.to_f32(), im1.to_f32());
                }
            }
        }
        drop(mapped_re);
        drop(mapped_im);
        staging_re.unmap();
        staging_im.unmap();
        out
    }

    /// Dispatch an unnormalized forward radix-5 batch and return row-major bins.
    #[must_use]
    pub fn forward_f16(&self, re: &[[f16; 8]; 5], im: &[[f16; 8]; 5]) -> Vec<Complex32> {
        self.transform_f16(re, im, false)
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn storage_output(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn staging_buffer(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn pack_f16_pair(lo: f16, hi: f16) -> u32 {
    u32::from(lo.to_bits()) | (u32::from(hi.to_bits()) << 16)
}

fn unpack_f16_pair(bits: u32) -> (f16, f16) {
    (
        f16::from_bits((bits & 0xffff) as u16),
        f16::from_bits((bits >> 16) as u16),
    )
}
