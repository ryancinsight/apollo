//! Core WGPU 3D FFT planning structures and dispatch operations.

use crate::infrastructure::transport::gpu::infrastructure::gpu_fft::strategy::{
    Axis, AxisStrategy, ChirpData, RadixStages,
};
use crate::{fft_1d_complex_inplace, Complex64};
use leto::Array1;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub(crate) struct AxisPackStage {
    pub(crate) _fft_params_buf: wgpu::Buffer,
    pub(crate) fft_params_bg: wgpu::BindGroup,
    pub(crate) _params_buf: wgpu::Buffer,
    pub(crate) bg: wgpu::BindGroup,
}

/// Returns true when this crate is linked and the WGPU backend is available.
#[must_use]
pub fn gpu_fft_available() -> bool {
    true
}

/// GPU-backed 3D FFT plan.
pub struct GpuFft3d {
    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) nz: usize,
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) strategy_x: AxisStrategy,
    pub(crate) strategy_y: AxisStrategy,
    pub(crate) strategy_z: AxisStrategy,
    pub(crate) bitrev_pipeline: wgpu::ComputePipeline,
    pub(crate) bitrev_radix4_pipeline: wgpu::ComputePipeline,
    pub(crate) forward_pipeline: wgpu::ComputePipeline,
    pub(crate) forward_radix4_pipeline: wgpu::ComputePipeline,
    pub(crate) scale_pipeline: wgpu::ComputePipeline,
    pub(crate) pack_pipeline: wgpu::ComputePipeline,
    pub(crate) unpack_pipeline: wgpu::ComputePipeline,
    pub(crate) _bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) _params_layout: wgpu::BindGroupLayout,
    pub(crate) _volume_layout: wgpu::BindGroupLayout,
    pub(crate) _re_buf: wgpu::Buffer,
    pub(crate) _im_buf: wgpu::Buffer,
    pub(crate) full_re_buf: wgpu::Buffer,
    pub(crate) full_im_buf: wgpu::Buffer,
    pub(crate) full_re_staging: wgpu::Buffer,
    pub(crate) full_im_staging: wgpu::Buffer,
    pub(crate) chirp_x: Option<ChirpData>,
    pub(crate) chirp_y: Option<ChirpData>,
    pub(crate) chirp_z: Option<ChirpData>,
    pub(crate) data_bg: wgpu::BindGroup,
    pub(crate) pack_bg_x: AxisPackStage,
    pub(crate) pack_bg_y: AxisPackStage,
    pub(crate) pack_bg_z: AxisPackStage,
    pub(crate) axis_fwd_x: RadixStages,
    pub(crate) axis_inv_x: RadixStages,
    pub(crate) axis_fwd_y: RadixStages,
    pub(crate) axis_inv_y: RadixStages,
    pub(crate) axis_fwd_z: RadixStages,
    pub(crate) axis_inv_z: RadixStages,
}

fn next_pow2(n: usize) -> usize {
    let mut m = 1usize;
    while m < n {
        m <<= 1;
    }
    m
}

#[inline]
fn is_pow2(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

fn axis_strategy_for(n: usize) -> AxisStrategy {
    if is_pow2(n) {
        AxisStrategy::Radix2
    } else {
        AxisStrategy::ChirpZ {
            n,
            m: next_pow2(2 * n - 1),
        }
    }
}

fn axis_workspace_elems(nx: usize, ny: usize, nz: usize, axis: Axis) -> usize {
    let len = axis.len(nx, ny, nz);
    let batch_count = axis.batch_count(nx, ny, nz);
    let fft_len = match axis_strategy_for(len) {
        AxisStrategy::Radix2 => len,
        AxisStrategy::ChirpZ { m, .. } => m,
    };
    fft_len * batch_count
}

fn validate_dimensions(
    max_buffer_size: u64,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Result<(), String> {
    for (name, n) in [("nx", nx), ("ny", ny), ("nz", nz)] {
        if n == 0 {
            return Err(format!("GpuFft3d: {name} must be > 0"));
        }
    }
    let required = [
        axis_workspace_elems(nx, ny, nz, Axis::X),
        axis_workspace_elems(nx, ny, nz, Axis::Y),
        axis_workspace_elems(nx, ny, nz, Axis::Z),
    ]
    .into_iter()
    .max()
    .unwrap_or(0) as u64
        * std::mem::size_of::<f32>() as u64;
    if required > max_buffer_size {
        return Err(format!(
            "GpuFft3d: workspace requires {required} bytes, exceeds device max_buffer_size={max_buffer_size}"
        ));
    }
    Ok(())
}

impl GpuFft3d {
    /// Create a new WGPU-backed 3D FFT plan.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<Self, String> {
        validate_dimensions(device.limits().max_buffer_size, nx, ny, nz)?;

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-fft-wgpu fft shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fft.wgsl").into()),
        });

        let data_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-fft-wgpu fft data layout"),
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
            ],
        });

        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-fft-wgpu fft params layout"),
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

        let volume_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-fft-wgpu volume layout"),
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
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-fft-wgpu fft pipeline layout"),
            bind_group_layouts: &[&data_layout, &params_layout],
            push_constant_ranges: &[],
        });

        let build_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let bitrev_pipeline = build_pipeline("fft_bitrev");
        let bitrev_radix4_pipeline = build_pipeline("fft_bitrev_radix4");
        let forward_pipeline = build_pipeline("fft_forward");
        let forward_radix4_pipeline = build_pipeline("fft_forward_radix4");
        let scale_pipeline = build_pipeline("fft_scale");
        let pack_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-fft-wgpu pack shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/pack.wgsl").into()),
        });
        let pack_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-fft-wgpu pack pipeline layout"),
            bind_group_layouts: &[&data_layout, &params_layout, &volume_layout],
            push_constant_ranges: &[],
        });
        let build_pack_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pack_layout),
                module: &pack_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pack_pipeline = build_pack_pipeline("fft_pack_axis");
        let unpack_pipeline = build_pack_pipeline("fft_unpack_axis");

        let strategy_x = axis_strategy_for(nx);
        let strategy_y = axis_strategy_for(ny);
        let strategy_z = axis_strategy_for(nz);
        let batch_x = (ny * nz) as u32;
        let batch_y = (nx * nz) as u32;
        let batch_z = (nx * ny) as u32;
        let workspace_capacity = [
            axis_workspace_elems(nx, ny, nz, Axis::X),
            axis_workspace_elems(nx, ny, nz, Axis::Y),
            axis_workspace_elems(nx, ny, nz, Axis::Z),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        let buf_size = (workspace_capacity * std::mem::size_of::<f32>()) as u64;
        let working_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let staging_usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;

        let re_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-fft-wgpu re"),
            size: buf_size,
            usage: working_usage,
            mapped_at_creation: false,
        });
        let im_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-fft-wgpu im"),
            size: buf_size,
            usage: working_usage,
            mapped_at_creation: false,
        });
        let full_buf_size = (nx * ny * nz * std::mem::size_of::<f32>()) as u64;
        let full_re_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-fft-wgpu full re"),
            size: full_buf_size,
            usage: working_usage,
            mapped_at_creation: false,
        });
        let full_im_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-fft-wgpu full im"),
            size: full_buf_size,
            usage: working_usage,
            mapped_at_creation: false,
        });
        let full_re_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-fft-wgpu full re staging"),
            size: full_buf_size,
            usage: staging_usage,
            mapped_at_creation: false,
        });
        let full_im_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-fft-wgpu full im staging"),
            size: full_buf_size,
            usage: staging_usage,
            mapped_at_creation: false,
        });

        let data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-fft-wgpu data bg"),
            layout: &data_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: re_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: im_buf.as_entire_binding(),
                },
            ],
        });

        let build_axis_pack = |axis: Axis, axis_len: u32, fft_len: u32, batch_count: u32| {
            let axis_code = match axis {
                Axis::X => 0u32,
                Axis::Y => 1u32,
                Axis::Z => 2u32,
            };
            let fft_params_data = [axis_len, 0, 0, batch_count];
            let fft_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-fft-wgpu pack fft params"),
                contents: bytemuck::cast_slice(&fft_params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let fft_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-fft-wgpu pack fft params bg"),
                layout: &params_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fft_params_buf.as_entire_binding(),
                }],
            });
            let params_data = [nx as u32, ny as u32, nz as u32, axis_code, fft_len, 0, 0, 0];
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-fft-wgpu pack params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-fft-wgpu volume bg"),
                layout: &volume_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: full_re_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: full_im_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            AxisPackStage {
                _fft_params_buf: fft_params_buf,
                fft_params_bg,
                _params_buf: params_buf,
                bg,
            }
        };
        let pack_bg_x = build_axis_pack(
            Axis::X,
            nx as u32,
            match strategy_x {
                AxisStrategy::Radix2 => nx as u32,
                AxisStrategy::ChirpZ { m, .. } => m as u32,
            },
            batch_x,
        );
        let pack_bg_y = build_axis_pack(
            Axis::Y,
            ny as u32,
            match strategy_y {
                AxisStrategy::Radix2 => ny as u32,
                AxisStrategy::ChirpZ { m, .. } => m as u32,
            },
            batch_y,
        );
        let pack_bg_z = build_axis_pack(
            Axis::Z,
            nz as u32,
            match strategy_z {
                AxisStrategy::Radix2 => nz as u32,
                AxisStrategy::ChirpZ { m, .. } => m as u32,
            },
            batch_z,
        );

        let chirp_x = match strategy_x {
            AxisStrategy::Radix2 => None,
            AxisStrategy::ChirpZ { n, m } => Some(Self::build_chirp_data(
                &device,
                &data_layout,
                &params_layout,
                &re_buf,
                &im_buf,
                n,
                m,
                batch_x,
            )),
        };
        let chirp_y = match strategy_y {
            AxisStrategy::Radix2 => None,
            AxisStrategy::ChirpZ { n, m } => Some(Self::build_chirp_data(
                &device,
                &data_layout,
                &params_layout,
                &re_buf,
                &im_buf,
                n,
                m,
                batch_y,
            )),
        };
        let chirp_z = match strategy_z {
            AxisStrategy::Radix2 => None,
            AxisStrategy::ChirpZ { n, m } => Some(Self::build_chirp_data(
                &device,
                &data_layout,
                &params_layout,
                &re_buf,
                &im_buf,
                n,
                m,
                batch_z,
            )),
        };

        let axis_fwd_x = match strategy_x {
            AxisStrategy::Radix2 => {
                if nx.is_power_of_two() && (nx.trailing_zeros() % 2 == 0) {
                    RadixStages::precompute_radix4(
                        &device,
                        &params_layout,
                        nx as u32,
                        batch_x,
                        false,
                    )
                } else {
                    RadixStages::precompute(&device, &params_layout, nx as u32, batch_x, false)
                }
            }
            AxisStrategy::ChirpZ { .. } => RadixStages::empty(),
        };
        let axis_inv_x = match strategy_x {
            AxisStrategy::Radix2 => {
                if nx.is_power_of_two() && (nx.trailing_zeros() % 2 == 0) {
                    RadixStages::precompute_radix4(
                        &device,
                        &params_layout,
                        nx as u32,
                        batch_x,
                        true,
                    )
                } else {
                    RadixStages::precompute(&device, &params_layout, nx as u32, batch_x, true)
                }
            }
            AxisStrategy::ChirpZ { .. } => RadixStages::empty(),
        };
        let axis_fwd_y = match strategy_y {
            AxisStrategy::Radix2 => {
                if ny.is_power_of_two() && (ny.trailing_zeros() % 2 == 0) {
                    RadixStages::precompute_radix4(
                        &device,
                        &params_layout,
                        ny as u32,
                        batch_y,
                        false,
                    )
                } else {
                    RadixStages::precompute(&device, &params_layout, ny as u32, batch_y, false)
                }
            }
            AxisStrategy::ChirpZ { .. } => RadixStages::empty(),
        };
        let axis_inv_y = match strategy_y {
            AxisStrategy::Radix2 => {
                if ny.is_power_of_two() && (ny.trailing_zeros() % 2 == 0) {
                    RadixStages::precompute_radix4(
                        &device,
                        &params_layout,
                        ny as u32,
                        batch_y,
                        true,
                    )
                } else {
                    RadixStages::precompute(&device, &params_layout, ny as u32, batch_y, true)
                }
            }
            AxisStrategy::ChirpZ { .. } => RadixStages::empty(),
        };
        let axis_fwd_z = match strategy_z {
            AxisStrategy::Radix2 => {
                if nz.is_power_of_two() && (nz.trailing_zeros() % 2 == 0) {
                    RadixStages::precompute_radix4(
                        &device,
                        &params_layout,
                        nz as u32,
                        batch_z,
                        false,
                    )
                } else {
                    RadixStages::precompute(&device, &params_layout, nz as u32, batch_z, false)
                }
            }
            AxisStrategy::ChirpZ { .. } => RadixStages::empty(),
        };
        let axis_inv_z = match strategy_z {
            AxisStrategy::Radix2 => {
                if nz.is_power_of_two() && (nz.trailing_zeros() % 2 == 0) {
                    RadixStages::precompute_radix4(
                        &device,
                        &params_layout,
                        nz as u32,
                        batch_z,
                        true,
                    )
                } else {
                    RadixStages::precompute(&device, &params_layout, nz as u32, batch_z, true)
                }
            }
            AxisStrategy::ChirpZ { .. } => RadixStages::empty(),
        };

        Ok(Self {
            nx,
            ny,
            nz,
            device,
            queue,
            strategy_x,
            strategy_y,
            strategy_z,
            bitrev_pipeline,
            bitrev_radix4_pipeline,
            forward_pipeline,
            forward_radix4_pipeline,
            scale_pipeline,
            pack_pipeline,
            unpack_pipeline,
            _bind_group_layout: data_layout,
            _params_layout: params_layout,
            _volume_layout: volume_layout,
            _re_buf: re_buf,
            _im_buf: im_buf,
            full_re_buf,
            full_im_buf,
            full_re_staging,
            full_im_staging,
            chirp_x,
            chirp_y,
            chirp_z,
            data_bg,
            pack_bg_x,
            pack_bg_y,
            pack_bg_z,
            axis_fwd_x,
            axis_inv_x,
            axis_fwd_y,
            axis_inv_y,
            axis_fwd_z,
            axis_inv_z,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_chirp_data(
        device: &wgpu::Device,
        _data_layout: &wgpu::BindGroupLayout,
        params_layout: &wgpu::BindGroupLayout,
        re_buf: &wgpu::Buffer,
        im_buf: &wgpu::Buffer,
        n: usize,
        m: usize,
        batch_count: u32,
    ) -> ChirpData {
        let mut h = Array1::<Complex64>::zeros([m]);
        for idx in 0..n {
            let arg = std::f32::consts::PI * (idx * idx) as f32 / n as f32;
            let value = Complex64::new(arg.cos() as f64, arg.sin() as f64);
            h[[idx]] = value;
            if idx > 0 {
                h[[m - idx]] = value;
            }
        }
        fft_1d_complex_inplace(&mut h);
        let h_re: Vec<f32> = h.iter().map(|value| value.re as f32).collect();
        let h_im: Vec<f32> = h.iter().map(|value| value.im as f32).collect();

        let h_fft_re = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-fft-wgpu chirp re"),
            contents: bytemuck::cast_slice(&h_re),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let h_fft_im = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-fft-wgpu chirp im"),
            contents: bytemuck::cast_slice(&h_im),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let data_chirp_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-fft-wgpu chirp data layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let chirp_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-fft-wgpu chirp shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/chirp.wgsl").into()),
        });
        let chirp_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-fft-wgpu chirp pipeline layout"),
            bind_group_layouts: &[&data_chirp_layout, params_layout],
            push_constant_ranges: &[],
        });
        let build_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&chirp_layout),
                module: &chirp_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let data_chirp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-fft-wgpu chirp data bg"),
            layout: &data_chirp_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: re_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: im_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: h_fft_re.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: h_fft_im.as_entire_binding(),
                },
            ],
        });

        let params_data = [n as u32, m as u32, batch_count, 0];
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-fft-wgpu chirp params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-fft-wgpu chirp params bg"),
            layout: params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            }],
        });

        ChirpData {
            _h_fft_re: h_fft_re,
            _h_fft_im: h_fft_im,
            premul_pipeline: build_pipeline("chirp_premul"),
            pointmul_pipeline: build_pipeline("chirp_pointmul"),
            scale_pipeline: build_pipeline("chirp_scale"),
            postmul_pipeline: build_pipeline("chirp_postmul"),
            negate_im_pipeline: build_pipeline("chirp_negate_im"),
            n: n as u32,
            m: m as u32,
            batch_count,
            data_chirp_bg,
            _params_buf: params_buf,
            params_bg,
            radix2_fwd: RadixStages::precompute(
                device,
                params_layout,
                m as u32,
                batch_count,
                false,
            ),
            radix2_inv: RadixStages::precompute(device, params_layout, m as u32, batch_count, true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        axis_strategy_for, axis_workspace_elems, next_pow2, validate_dimensions, Axis, AxisStrategy,
    };

    #[test]
    fn next_pow2_rounds_up_non_power_of_two_lengths() {
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(12), 16);
        assert_eq!(next_pow2(127), 128);
    }

    #[test]
    fn axis_strategy_uses_radix2_only_for_power_of_two_lengths() {
        assert_eq!(axis_strategy_for(64), AxisStrategy::Radix2);
        assert_eq!(
            axis_strategy_for(96),
            AxisStrategy::ChirpZ { n: 96, m: 256 }
        );
    }

    #[test]
    fn axis_workspace_matches_axis_batch_geometry() {
        assert_eq!(axis_workspace_elems(8, 4, 2, Axis::X), 8 * 4 * 2);
        assert_eq!(axis_workspace_elems(8, 4, 2, Axis::Y), 4 * 8 * 2);
        assert_eq!(axis_workspace_elems(8, 4, 2, Axis::Z), 2 * 8 * 4);
        assert_eq!(axis_workspace_elems(12, 5, 3, Axis::X), 32 * 5 * 3);
    }

    #[test]
    fn validate_dimensions_rejects_zero_sizes() {
        let error = validate_dimensions(1 << 20, 0, 4, 4).unwrap_err();
        assert!(error.contains("nx must be > 0"));
    }

    #[test]
    fn validate_dimensions_rejects_workspace_that_exceeds_device_limits() {
        let error = validate_dimensions(64, 16, 16, 16).unwrap_err();
        assert!(error.contains("exceeds device max_buffer_size"));
    }

    #[test]
    fn validate_dimensions_accepts_small_valid_shapes() {
        validate_dimensions(1 << 20, 8, 8, 8).unwrap();
    }
}
