//! Native f16 WGPU FFT plan using WGSL `enable f16;`.
//!
//! # Architecture
//!
//! `GpuFft3dF16Native` requires a device created with `wgpu::Features::SHADER_F16`.
//! All shader arithmetic executes in `f16`. Host conversion (f32→f16 upload,
//! f16→f32 readback) occurs at the buffer boundary using little-endian IEEE 754
//! half-precision bit patterns.
//!
//! Power-of-two dimensions use the radix-2 Cooley-Tukey kernel
//! (`fft_native_f16.wgsl`). Non-power-of-two dimensions use the Bluestein
//! chirp-Z reduction (`chirp_native_f16.wgsl`), which reduces the N-point DFT
//! to an M-point radix-2 FFT where M = next_pow2(2N − 1).
//!
//! # Mathematical contract
//!
//! The 3D DFT contract is identical to `GpuFft3d`:
//! `F[kx,ky,kz] = Σ_{x,y,z} f[x,y,z] · W^{kx·x + ky·y + kz·z}`
//! where `W = exp(-2πi/N)` for each axis of length N.
//!
//! Twiddle factors are computed in f32 and narrowed to f16, bounding twiddle
//! error at f32 precision before quantization. Butterfly accumulation error is
//! bounded by O(log₂N) · ε_f16 · ‖input‖_∞ where ε_f16 ≈ 9.77×10⁻⁴.
//! For Bluestein axes, N is replaced by M = next_pow2(2N − 1), giving
//! O(log₂M) · ε_f16 ≈ O(log₂N) · ε_f16 asymptotically.
//!
//! # Feature gate
//!
//! This module is compiled only when the `native-f16` feature is enabled.

use crate::infrastructure::transport::gpu::infrastructure::gpu_fft::pipeline::AxisPackStage;
use crate::infrastructure::transport::gpu::infrastructure::gpu_fft::strategy::{Axis, AxisStrategy, ChirpData, RadixStages};
use crate::{f16 as HalfF16, fft_1d_complex_inplace, Complex64};
use leto::{Array1, Array3};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Validates that all three dimensions are at least 2.
///
/// Power-of-two axes use the radix-2 kernel; non-power-of-two axes use the
/// Bluestein chirp-Z reduction. Both require N ≥ 2 (f16 buffer alignment).
fn validate_dimensions_f16(nx: usize, ny: usize, nz: usize) -> Result<(), String> {
    for (name, n) in [("nx", nx), ("ny", ny), ("nz", nz)] {
        if n == 0 {
            return Err(format!("{name}=0 is invalid for GpuFft3dF16Native"));
        }
        if n < 2 {
            return Err(format!(
                "{name}={n} < 2; f16 buffer alignment requires at least 2 elements per axis"
            ));
        }
    }
    Ok(())
}

/// Smallest power of two >= `n`.
#[inline]
fn f16_next_pow2(n: usize) -> usize {
    let mut m = 1usize;
    while m < n {
        m <<= 1;
    }
    m
}

/// Select radix-2 or Bluestein strategy for a single axis of length `n`.
#[inline]
fn f16_axis_strategy(n: usize) -> AxisStrategy {
    if n > 0 && (n & (n - 1)) == 0 {
        AxisStrategy::Radix2
    } else {
        let m = f16_next_pow2(2 * n - 1);
        AxisStrategy::ChirpZ { n, m }
    }
}

/// Workspace element count (f16 values) required for a single axis pass.
fn f16_axis_workspace_elems(nx: usize, ny: usize, nz: usize, axis: Axis) -> usize {
    let len = axis.len(nx, ny, nz);
    let batch = axis.batch_count(nx, ny, nz);
    let fft_len = match f16_axis_strategy(len) {
        AxisStrategy::Radix2 => len,
        AxisStrategy::ChirpZ { m, .. } => m,
    };
    fft_len * batch
}

/// GPU-backed 3D FFT plan executing all arithmetic in native f16.
///
/// Requires a device created with [`wgpu::Features::SHADER_F16`]. All WGSL
/// shaders use `enable f16;` and operate on `array<f16>` storage buffers.
/// Host-side data is converted f32↔f16 at the upload/readback boundary.
///
/// Power-of-two axes use the radix-2 kernel; non-power-of-two axes use the
/// Bluestein chirp-Z reduction. All axes must be ≥ 2.
pub struct GpuFft3dF16Native {
    /// X dimension.
    nx: usize,
    /// Y dimension.
    ny: usize,
    /// Z dimension.
    nz: usize,
    /// WGPU device (must have `SHADER_F16` enabled).
    device: Arc<wgpu::Device>,
    /// WGPU queue.
    queue: Arc<wgpu::Queue>,
    /// Bit-reversal permutation pipeline (f16 workspace).
    bitrev_pipeline: wgpu::ComputePipeline,
    /// Butterfly pipeline (f16 workspace).
    forward_pipeline: wgpu::ComputePipeline,
    /// Normalization scale pipeline (f16 workspace).
    scale_pipeline: wgpu::ComputePipeline,
    /// Axis pack pipeline (volume f16 → workspace f16).
    pack_pipeline: wgpu::ComputePipeline,
    /// Axis unpack pipeline (workspace f16 → volume f16).
    unpack_pipeline: wgpu::ComputePipeline,
    /// Retained data bind group layout.
    _data_layout: wgpu::BindGroupLayout,
    /// Retained params bind group layout.
    _params_layout: wgpu::BindGroupLayout,
    /// Retained volume bind group layout.
    _volume_layout: wgpu::BindGroupLayout,
    /// Workspace real buffer (f16, size = workspace_capacity × 2 bytes).
    _re_buf: wgpu::Buffer,
    /// Workspace imaginary buffer (f16).
    _im_buf: wgpu::Buffer,
    /// Full-volume real buffer (f16, size = nx × ny × nz × 2 bytes).
    full_re_buf: wgpu::Buffer,
    /// Full-volume imaginary buffer (f16).
    full_im_buf: wgpu::Buffer,
    /// Staging buffer for real readback.
    full_re_staging: wgpu::Buffer,
    /// Staging buffer for imaginary readback.
    full_im_staging: wgpu::Buffer,
    /// Bind group for workspace (data_re, data_im).
    data_bg: wgpu::BindGroup,
    /// Pack bind group for X axis.
    pack_bg_x: AxisPackStage,
    /// Pack bind group for Y axis.
    pack_bg_y: AxisPackStage,
    /// Pack bind group for Z axis.
    pack_bg_z: AxisPackStage,
    /// Forward radix-2 stages for X axis.
    axis_fwd_x: RadixStages,
    /// Inverse radix-2 stages for X axis.
    axis_inv_x: RadixStages,
    /// Forward radix-2 stages for Y axis.
    axis_fwd_y: RadixStages,
    /// Inverse radix-2 stages for Y axis.
    axis_inv_y: RadixStages,
    /// Forward radix-2 stages for Z axis (empty sentinel for ChirpZ axes).
    axis_fwd_z: RadixStages,
    /// Inverse radix-2 stages for Z axis (empty sentinel for ChirpZ axes).
    axis_inv_z: RadixStages,
    /// FFT strategy for the X axis.
    strategy_x: AxisStrategy,
    /// FFT strategy for the Y axis.
    strategy_y: AxisStrategy,
    /// FFT strategy for the Z axis.
    strategy_z: AxisStrategy,
    /// Bluestein chirp data for the X axis (None when radix-2).
    chirp_x: Option<ChirpData>,
    /// Bluestein chirp data for the Y axis (None when radix-2).
    chirp_y: Option<ChirpData>,
    /// Bluestein chirp data for the Z axis (None when radix-2).
    chirp_z: Option<ChirpData>,
}

impl GpuFft3dF16Native {
    /// Return true when the adapter advertises `SHADER_F16` support.
    #[must_use]
    pub fn device_supports_f16(adapter: &wgpu::Adapter) -> bool {
        adapter.features().contains(wgpu::Features::SHADER_F16)
    }

    /// Create a plan by requesting a new WGPU device with `SHADER_F16` enabled.
    ///
    /// Returns `Err` if no adapter is available, if the adapter does not
    /// support `SHADER_F16`, or if any dimension is < 2.
    pub fn try_new(nx: usize, ny: usize, nz: usize) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| format!("no WGPU adapter: {e}"))?;
        if !Self::device_supports_f16(&adapter) {
            return Err("adapter does not support SHADER_F16".to_string());
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("apollo-fft-wgpu native-f16"),
            required_features: wgpu::Features::SHADER_F16,
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("device request failed: {e}"))?;
        Self::try_from_device(Arc::new(device), Arc::new(queue), nx, ny, nz)
    }

    /// Create a plan from a caller-supplied device and queue.
    ///
    /// Returns `Err` if the device was not created with `SHADER_F16`, or if
    /// any dimension violates the power-of-two constraint.
    pub fn try_from_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<Self, String> {
        validate_dimensions_f16(nx, ny, nz)?;
        if !device.features().contains(wgpu::Features::SHADER_F16) {
            return Err("device does not have SHADER_F16 enabled; \
                 create the device with wgpu::Features::SHADER_F16"
                .to_string());
        }

        // --- Shader modules --------------------------------------------------
        let fft_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-fft-wgpu fft-native-f16 shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/fft_native_f16.wgsl").into()),
        });
        let pack_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-fft-wgpu pack-native-f16 shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/pack_native_f16.wgsl").into(),
            ),
        });

        // --- Bind group layouts ----------------------------------------------
        let make_storage_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let make_uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let data_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-fft-wgpu f16 data layout"),
            entries: &[make_storage_entry(0), make_storage_entry(1)],
        });
        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-fft-wgpu f16 params layout"),
            entries: &[make_uniform_entry(0)],
        });
        let volume_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-fft-wgpu f16 volume layout"),
            entries: &[
                make_storage_entry(0),
                make_storage_entry(1),
                make_uniform_entry(2),
            ],
        });

        // --- Pipeline layouts ------------------------------------------------
        let fft_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-fft-wgpu f16 fft pipeline layout"),
            bind_group_layouts: &[&data_layout, &params_layout],
            push_constant_ranges: &[],
        });
        let pack_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-fft-wgpu f16 pack pipeline layout"),
            bind_group_layouts: &[&data_layout, &params_layout, &volume_layout],
            push_constant_ranges: &[],
        });

        // --- Compute pipelines -----------------------------------------------
        let build_fft_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&fft_pipeline_layout),
                module: &fft_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let bitrev_pipeline = build_fft_pipeline("fft_bitrev");
        let forward_pipeline = build_fft_pipeline("fft_forward");
        let scale_pipeline = build_fft_pipeline("fft_scale");

        let build_pack_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pack_pipeline_layout),
                module: &pack_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pack_pipeline = build_pack_pipeline("fft_pack_axis");
        let unpack_pipeline = build_pack_pipeline("fft_unpack_axis");

        // --- Buffers ---------------------------------------------------------
        // f16 element = 2 bytes; for power-of-two nx,ny,nz >= 2:
        //   n = nx*ny*nz >= 8, so n*2 >= 16, always a multiple of COPY_BUFFER_ALIGNMENT (4).
        let n = nx * ny * nz;
        let f16_buf_size = (n * 2) as u64;

        // Compute per-axis strategies before sizing the workspace buffers.
        let strategy_x = f16_axis_strategy(nx);
        let strategy_y = f16_axis_strategy(ny);
        let strategy_z = f16_axis_strategy(nz);

        // Workspace capacity = max over all axes of (fft_len * batch_count).
        // For radix-2 axes this equals nx*ny*nz; for ChirpZ axes the padded
        // length m > n grows the required capacity.
        let workspace_capacity = [
            f16_axis_workspace_elems(nx, ny, nz, Axis::X),
            f16_axis_workspace_elems(nx, ny, nz, Axis::Y),
            f16_axis_workspace_elems(nx, ny, nz, Axis::Z),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        // Workspace buffer size: workspace_capacity f16 elements × 2 bytes each.
        // Guaranteed ≥ 4 bytes (COPY_BUFFER_ALIGNMENT) because workspace_capacity
        // ≥ n ≥ 8 (nx,ny,nz ≥ 2 each) and each f16 is 2 bytes.
        let workspace_size = (workspace_capacity * 2) as u64;

        let working_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let staging_usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;

        let re_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-fft-wgpu f16 workspace re"),
            size: workspace_size,
            usage: working_usage,
            mapped_at_creation: false,
        });
        let im_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-fft-wgpu f16 workspace im"),
            size: workspace_size,
            usage: working_usage,
            mapped_at_creation: false,
        });
        let full_re_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-fft-wgpu f16 full re"),
            size: f16_buf_size,
            usage: working_usage,
            mapped_at_creation: false,
        });
        let full_im_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-fft-wgpu f16 full im"),
            size: f16_buf_size,
            usage: working_usage,
            mapped_at_creation: false,
        });
        let full_re_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-fft-wgpu f16 full re staging"),
            size: f16_buf_size,
            usage: staging_usage,
            mapped_at_creation: false,
        });
        let full_im_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-fft-wgpu f16 full im staging"),
            size: f16_buf_size,
            usage: staging_usage,
            mapped_at_creation: false,
        });

        // --- Bind groups -----------------------------------------------------
        let data_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-fft-wgpu f16 data bg"),
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

        let build_axis_pack =
            |axis: Axis, axis_len: u32, fft_len: u32, batch_count: u32| -> AxisPackStage {
                let axis_code: u32 = match axis {
                    Axis::X => 0,
                    Axis::Y => 1,
                    Axis::Z => 2,
                };
                let fft_params_data = [axis_len, 0u32, 0u32, batch_count];
                let fft_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("apollo-fft-wgpu f16 pack fft params"),
                    contents: bytemuck::cast_slice(&fft_params_data),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let fft_params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("apollo-fft-wgpu f16 pack fft params bg"),
                    layout: &params_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: fft_params_buf.as_entire_binding(),
                    }],
                });
                let params_data = [
                    nx as u32, ny as u32, nz as u32, axis_code, fft_len, 0u32, 0u32, 0u32,
                ];
                let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("apollo-fft-wgpu f16 pack params"),
                    contents: bytemuck::cast_slice(&params_data),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("apollo-fft-wgpu f16 volume bg"),
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

        let batch_x = (ny * nz) as u32;
        let batch_y = (nx * nz) as u32;
        let batch_z = (nx * ny) as u32;

        // For ChirpZ axes the pack stride (fft_len) is the padded length m,
        // not the original axis length n; the pack shader uses fft_len as the
        // workspace row stride so that the padded region is addressable.
        let fft_len_x = match strategy_x {
            AxisStrategy::Radix2 => nx as u32,
            AxisStrategy::ChirpZ { m, .. } => m as u32,
        };
        let fft_len_y = match strategy_y {
            AxisStrategy::Radix2 => ny as u32,
            AxisStrategy::ChirpZ { m, .. } => m as u32,
        };
        let fft_len_z = match strategy_z {
            AxisStrategy::Radix2 => nz as u32,
            AxisStrategy::ChirpZ { m, .. } => m as u32,
        };

        let pack_bg_x = build_axis_pack(Axis::X, nx as u32, fft_len_x, batch_x);
        let pack_bg_y = build_axis_pack(Axis::Y, ny as u32, fft_len_y, batch_y);
        let pack_bg_z = build_axis_pack(Axis::Z, nz as u32, fft_len_z, batch_z);

        // --- RadixStages ----------------------------------------------------
        // Radix-2 stages are only precomputed for Radix2 axes.  ChirpZ axes
        // use an empty sentinel here; the actual radix-2 sub-steps are
        // embedded in the ChirpData (radix2_fwd / radix2_inv fields) and
        // dispatched by dispatch_chirp_f16.
        let axis_fwd_x = match strategy_x {
            AxisStrategy::Radix2 => {
                RadixStages::precompute(&device, &params_layout, nx as u32, batch_x, false)
            }
            AxisStrategy::ChirpZ { .. } => RadixStages::empty(),
        };
        let axis_inv_x = match strategy_x {
            AxisStrategy::Radix2 => {
                RadixStages::precompute(&device, &params_layout, nx as u32, batch_x, true)
            }
            AxisStrategy::ChirpZ { .. } => RadixStages::empty(),
        };
        let axis_fwd_y = match strategy_y {
            AxisStrategy::Radix2 => {
                RadixStages::precompute(&device, &params_layout, ny as u32, batch_y, false)
            }
            AxisStrategy::ChirpZ { .. } => RadixStages::empty(),
        };
        let axis_inv_y = match strategy_y {
            AxisStrategy::Radix2 => {
                RadixStages::precompute(&device, &params_layout, ny as u32, batch_y, true)
            }
            AxisStrategy::ChirpZ { .. } => RadixStages::empty(),
        };
        let axis_fwd_z = match strategy_z {
            AxisStrategy::Radix2 => {
                RadixStages::precompute(&device, &params_layout, nz as u32, batch_z, false)
            }
            AxisStrategy::ChirpZ { .. } => RadixStages::empty(),
        };
        let axis_inv_z = match strategy_z {
            AxisStrategy::Radix2 => {
                RadixStages::precompute(&device, &params_layout, nz as u32, batch_z, true)
            }
            AxisStrategy::ChirpZ { .. } => RadixStages::empty(),
        };

        // --- Bluestein chirp data (built after workspace buffers exist) ------
        let chirp_x = match strategy_x {
            AxisStrategy::Radix2 => None,
            AxisStrategy::ChirpZ { n, m } => Some(Self::build_chirp_data_f16(
                &device,
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
            AxisStrategy::ChirpZ { n, m } => Some(Self::build_chirp_data_f16(
                &device,
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
            AxisStrategy::ChirpZ { n, m } => Some(Self::build_chirp_data_f16(
                &device,
                &params_layout,
                &re_buf,
                &im_buf,
                n,
                m,
                batch_z,
            )),
        };

        Ok(Self {
            nx,
            ny,
            nz,
            device,
            queue,
            bitrev_pipeline,
            forward_pipeline,
            scale_pipeline,
            pack_pipeline,
            unpack_pipeline,
            _data_layout: data_layout,
            _params_layout: params_layout,
            _volume_layout: volume_layout,
            _re_buf: re_buf,
            _im_buf: im_buf,
            full_re_buf,
            full_im_buf,
            full_re_staging,
            full_im_staging,
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
            strategy_x,
            strategy_y,
            strategy_z,
            chirp_x,
            chirp_y,
            chirp_z,
        })
    }

    // -------------------------------------------------------------------------
    // Public transform methods
    // -------------------------------------------------------------------------

    /// Forward 3D FFT of a real f32 field.
    ///
    /// Returns an interleaved complex `Vec<f32>` of length `2 · nx · ny · nz`
    /// ordered as `[re₀, im₀, re₁, im₁, …]`. All GPU arithmetic executes in f16.
    ///
    /// # Mathematical contract
    ///
    /// `F[kx,ky,kz] = Σ_{x,y,z} f[x,y,z] · exp(-2πi(kx·x/Nx + ky·y/Ny + kz·z/Nz))`
    ///
    /// Absolute error per element is bounded by O(log₂N) · ε_f16 · ‖f‖_∞
    /// where ε_f16 ≈ 9.77×10⁻⁴.
    #[must_use]
    pub fn forward_native_f16(&self, field: &Array3<f32>) -> Vec<f32> {
        assert_eq!(
            field.shape(),
            (self.nx, self.ny, self.nz),
            "field dimensions must match plan"
        );
        let n = self.nx * self.ny * self.nz;

        // Convert f32 input → f16 bytes (little-endian IEEE 754 half).
        let re_bytes: Vec<u8> = field
            .iter()
            .flat_map(|&v| HalfF16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let im_bytes = vec![0u8; n * 2];

        self.queue.write_buffer(&self.full_re_buf, 0, &re_bytes);
        self.queue.write_buffer(&self.full_im_buf, 0, &im_bytes);

        self.run_f16_axis_fft(Axis::Z, false);
        self.run_f16_axis_fft(Axis::Y, false);
        self.run_f16_axis_fft(Axis::X, false);

        let re_out = self.read_back_f16_as_f32(&self.full_re_buf, &self.full_re_staging, n);
        let im_out = self.read_back_f16_as_f32(&self.full_im_buf, &self.full_im_staging, n);

        let mut result = Vec::with_capacity(2 * n);
        for (re, im) in re_out.into_iter().zip(im_out.into_iter()) {
            result.push(re);
            result.push(im);
        }
        result
    }

    /// Inverse 3D FFT of an interleaved complex f32 spectrum.
    ///
    /// `field_hat` must have length `2 · nx · ny · nz` (interleaved `[re, im]`
    /// pairs). Returns the real part of the inverse transform as `Vec<f32>` of
    /// length `nx · ny · nz`. All GPU arithmetic executes in f16.
    ///
    /// # Normalization
    ///
    /// Each axis is normalized by `1/N_axis`, matching the FFTW convention.
    #[must_use]
    pub fn inverse_native_f16(&self, field_hat: &[f32]) -> Vec<f32> {
        let n = self.nx * self.ny * self.nz;
        assert_eq!(
            field_hat.len(),
            2 * n,
            "field_hat length must be 2·nx·ny·nz"
        );

        // De-interleave and convert to f16 bytes.
        let re_bytes: Vec<u8> = field_hat
            .chunks_exact(2)
            .flat_map(|pair| HalfF16::from_f32(pair[0]).to_bits().to_le_bytes())
            .collect();
        let im_bytes: Vec<u8> = field_hat
            .chunks_exact(2)
            .flat_map(|pair| HalfF16::from_f32(pair[1]).to_bits().to_le_bytes())
            .collect();

        self.queue.write_buffer(&self.full_re_buf, 0, &re_bytes);
        self.queue.write_buffer(&self.full_im_buf, 0, &im_bytes);

        self.run_f16_axis_fft(Axis::X, true);
        self.run_f16_axis_fft(Axis::Y, true);
        self.run_f16_axis_fft(Axis::Z, true);

        self.read_back_f16_as_f32(&self.full_re_buf, &self.full_re_staging, n)
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /// Size in bytes of a single f16 component buffer (re or im) for the full volume.
    #[inline]
    fn volume_buf_size(&self) -> u64 {
        (self.nx * self.ny * self.nz * 2) as u64
    }

    /// Run one complete axis FFT pass (pack → [radix2 | chirp] → unpack) and submit.
    ///
    /// Dispatches the radix-2 kernel for power-of-two axes and the Bluestein
    /// chirp-Z kernel for non-power-of-two axes.
    fn run_f16_axis_fft(&self, axis: Axis, inverse: bool) {
        let axis_len = axis.len(self.nx, self.ny, self.nz) as u32;
        let batch_count = axis.batch_count(self.nx, self.ny, self.nz) as u32;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-fft-wgpu f16 axis encoder"),
            });

        self.dispatch_pack(&mut encoder, axis, axis_len, batch_count);

        match axis {
            Axis::Z => match self.strategy_z {
                AxisStrategy::Radix2 => {
                    let stages = if inverse {
                        &self.axis_inv_z
                    } else {
                        &self.axis_fwd_z
                    };
                    self.dispatch_radix2(&mut encoder, stages);
                }
                AxisStrategy::ChirpZ { .. } => {
                    self.dispatch_chirp_f16(
                        &mut encoder,
                        self.chirp_z
                            .as_ref()
                            .expect("chirp_z present for ChirpZ strategy"),
                        inverse,
                    );
                }
            },
            Axis::Y => match self.strategy_y {
                AxisStrategy::Radix2 => {
                    let stages = if inverse {
                        &self.axis_inv_y
                    } else {
                        &self.axis_fwd_y
                    };
                    self.dispatch_radix2(&mut encoder, stages);
                }
                AxisStrategy::ChirpZ { .. } => {
                    self.dispatch_chirp_f16(
                        &mut encoder,
                        self.chirp_y
                            .as_ref()
                            .expect("chirp_y present for ChirpZ strategy"),
                        inverse,
                    );
                }
            },
            Axis::X => match self.strategy_x {
                AxisStrategy::Radix2 => {
                    let stages = if inverse {
                        &self.axis_inv_x
                    } else {
                        &self.axis_fwd_x
                    };
                    self.dispatch_radix2(&mut encoder, stages);
                }
                AxisStrategy::ChirpZ { .. } => {
                    self.dispatch_chirp_f16(
                        &mut encoder,
                        self.chirp_x
                            .as_ref()
                            .expect("chirp_x present for ChirpZ strategy"),
                        inverse,
                    );
                }
            },
        }

        self.dispatch_unpack(&mut encoder, axis, axis_len, batch_count);
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    // -------------------------------------------------------------------------
    // Bluestein / chirp-Z helpers
    // -------------------------------------------------------------------------

    /// Build all GPU resources for one Bluestein chirp-Z axis pass in f16.
    ///
    /// # Mathematical contract
    ///
    /// **Theorem (Bluestein 1970):** For any N ≥ 1, the N-point DFT
    /// `X[k] = Σ_{n=0}^{N-1} x[n] · W^{nk}` reduces via the identity
    /// `nk = (n² + k² − (k−n)²)/2` to the length-M circular convolution
    /// `X[k] = W^{-k²/2} · (a ∗ h)[k]`
    /// where `a[n] = x[n]·W^{n²/2}`, `h[j] = W^{-j²/2}`, and
    /// `M = next_pow2(2N−1)` ensures linear convolution via zero-padding.
    ///
    /// Twiddle factors for `h` are computed in f32 and narrowed to f16 once
    /// at plan-creation time, bounding twiddle error at f32 precision.
    fn build_chirp_data_f16(
        device: &wgpu::Device,
        params_layout: &wgpu::BindGroupLayout,
        re_buf: &wgpu::Buffer,
        im_buf: &wgpu::Buffer,
        n: usize,
        m: usize,
        batch_count: u32,
    ) -> ChirpData {
        // Compute h[j] = exp(-πi j²/N) in f32, then narrow to f16.
        // The symmetric extension h[m-j] = h[j] for j=1..N-1 ensures the
        // circular convolution of length M produces the correct linear sum.
        let mut h = Array1::<Complex64>::zeros(m);
        for idx in 0..n {
            let arg = std::f32::consts::PI * (idx * idx) as f32 / n as f32;
            let value = Complex64::new(arg.cos() as f64, arg.sin() as f64);
            h[idx] = value;
            if idx > 0 {
                h[m - idx] = value;
            }
        }
        fft_1d_complex_inplace(&mut h);

        // Convert to f16 bit patterns (u16, little-endian).
        let h_re_bits: Vec<u16> = h
            .iter()
            .map(|value| HalfF16::from_f32(value.re as f32).to_bits())
            .collect();
        let h_im_bits: Vec<u16> = h
            .iter()
            .map(|value| HalfF16::from_f32(value.im as f32).to_bits())
            .collect();

        let h_fft_re = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-fft-wgpu f16 chirp re"),
            contents: bytemuck::cast_slice(&h_re_bits),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let h_fft_im = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-fft-wgpu f16 chirp im"),
            contents: bytemuck::cast_slice(&h_im_bits),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // 4-binding layout: (data_re, data_im, chirp_re, chirp_im).
        // All four use Storage read_write at the API level; the shader treats
        // bindings 2/3 as read-only at the WGSL level.
        let make_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let data_chirp_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-fft-wgpu f16 chirp data layout"),
            entries: &[make_rw(0), make_rw(1), make_rw(2), make_rw(3)],
        });

        let chirp_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-fft-wgpu chirp-native-f16 shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/chirp_native_f16.wgsl").into(),
            ),
        });
        let chirp_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-fft-wgpu f16 chirp pipeline layout"),
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
            label: Some("apollo-fft-wgpu f16 chirp data bg"),
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

        let params_data = [n as u32, m as u32, batch_count, 0u32];
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-fft-wgpu f16 chirp params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-fft-wgpu f16 chirp params bg"),
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

    /// Dispatch a complete Bluestein chirp-Z pass using the f16 pipelines.
    ///
    /// Uses flat 1D dispatch `(total + 255) / 256, 1, 1` throughout to avoid
    /// data races that would arise from a 2D dispatch with a shader that only
    /// reads `gid.x` as the flat element index.
    ///
    /// # Dispatch sequence
    ///
    /// Forward (inverse=false):
    ///   1. chirp_premul  — a[n] = x[n]·exp(+πi n²/N), zero-pad n≥N
    ///   2. radix2_fwd    — M-point forward FFT of padded sequence
    ///   3. chirp_pointmul — pointwise multiply with H = precomputed FFT(h)
    ///   4. radix2_inv    — M-point inverse FFT (includes convolution 1/M)
    ///   5. chirp_postmul — X[k] *= exp(+πi k²/N)
    ///
    /// Inverse (inverse=true): additionally negate_im (between premul and
    /// radix2_fwd) and scale by `1/N` after postmul.
    fn dispatch_chirp_f16(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        chirp: &ChirpData,
        inverse: bool,
    ) {
        if chirp.n == 0 || chirp.m == 0 {
            return;
        }

        let premul_total = chirp.m * chirp.batch_count;
        let postmul_total = chirp.n * chirp.batch_count;

        if inverse {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-fft-wgpu f16 chirp pre-conjugate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&chirp.negate_im_pipeline);
            pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
            pass.set_bind_group(1, &chirp.params_bg, &[]);
            pass.dispatch_workgroups(postmul_total.div_ceil(256), 1, 1);
        }

        // Step 1: premultiply (a[n] = x[n]·exp(-πi n²/N), zero padding n≥N).
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-fft-wgpu f16 chirp premul"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&chirp.premul_pipeline);
            pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
            pass.set_bind_group(1, &chirp.params_bg, &[]);
            pass.dispatch_workgroups(premul_total.div_ceil(256), 1, 1);
        }

        // Step 2: forward radix-2 FFT of padded sequence (uses self.data_bg).
        self.dispatch_radix2(encoder, &chirp.radix2_fwd);

        // Step 3: pointwise multiply with H.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-fft-wgpu f16 chirp pointmul"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&chirp.pointmul_pipeline);
            pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
            pass.set_bind_group(1, &chirp.params_bg, &[]);
            pass.dispatch_workgroups(premul_total.div_ceil(256), 1, 1);
        }

        // Step 4: inverse radix-2 FFT applies the convolution 1/M normalization.
        self.dispatch_radix2(encoder, &chirp.radix2_inv);

        // Step 5: postmultiply X[k] *= exp(+πi k²/N).
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-fft-wgpu f16 chirp postmul"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&chirp.postmul_pipeline);
            pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
            pass.set_bind_group(1, &chirp.params_bg, &[]);
            pass.dispatch_workgroups(postmul_total.div_ceil(256), 1, 1);
        }

        if inverse {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-fft-wgpu f16 chirp post-conjugate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&chirp.negate_im_pipeline);
            pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
            pass.set_bind_group(1, &chirp.params_bg, &[]);
            pass.dispatch_workgroups(postmul_total.div_ceil(256), 1, 1);
        }

        // Scale by 1/N for inverse DFT normalization on the original axis.
        if inverse {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-fft-wgpu f16 chirp scale"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&chirp.scale_pipeline);
            pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
            pass.set_bind_group(1, &chirp.params_bg, &[]);
            pass.dispatch_workgroups(postmul_total.div_ceil(256), 1, 1);
        }
    }

    /// Encode a pack pass (volume → workspace) for the given axis.
    fn dispatch_pack(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        axis: Axis,
        axis_len: u32,
        batch_count: u32,
    ) {
        let stage = match axis {
            Axis::X => &self.pack_bg_x,
            Axis::Y => &self.pack_bg_y,
            Axis::Z => &self.pack_bg_z,
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("apollo-fft-wgpu f16 pack pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pack_pipeline);
        pass.set_bind_group(0, &self.data_bg, &[]);
        pass.set_bind_group(1, &stage.fft_params_bg, &[]);
        pass.set_bind_group(2, &stage.bg, &[]);
        let total = axis_len * batch_count;
        pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
    }

    /// Encode an unpack pass (workspace → volume) for the given axis.
    fn dispatch_unpack(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        axis: Axis,
        axis_len: u32,
        batch_count: u32,
    ) {
        let stage = match axis {
            Axis::X => &self.pack_bg_x,
            Axis::Y => &self.pack_bg_y,
            Axis::Z => &self.pack_bg_z,
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("apollo-fft-wgpu f16 unpack pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.unpack_pipeline);
        pass.set_bind_group(0, &self.data_bg, &[]);
        pass.set_bind_group(1, &stage.fft_params_bg, &[]);
        pass.set_bind_group(2, &stage.bg, &[]);
        let total = axis_len * batch_count;
        pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
    }

    /// Encode a full radix-2 pass (bitrev + log₂M butterfly stages + optional scale).
    fn dispatch_radix2(&self, encoder: &mut wgpu::CommandEncoder, stages: &RadixStages) {
        if stages.fft_m == 0 {
            return;
        }
        let bitrev_total = stages.batch_count * stages.fft_m;
        let butterfly_total = stages.batch_count * (stages.fft_m / 2);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-fft-wgpu f16 bitrev pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bitrev_pipeline);
            pass.set_bind_group(0, &self.data_bg, &[]);
            pass.set_bind_group(1, &stages.bgs[0], &[]);
            pass.dispatch_workgroups(bitrev_total.div_ceil(256), 1, 1);
        }

        for stage_idx in 0..stages.fft_m.trailing_zeros() as usize {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-fft-wgpu f16 butterfly pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward_pipeline);
            pass.set_bind_group(0, &self.data_bg, &[]);
            pass.set_bind_group(1, &stages.bgs[1 + stage_idx], &[]);
            pass.dispatch_workgroups(butterfly_total.div_ceil(256), 1, 1);
        }

        // Optional scale stage (present only for inverse passes).
        if stages.bgs.len() > 1 + stages.fft_m.trailing_zeros() as usize {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-fft-wgpu f16 scale pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scale_pipeline);
            pass.set_bind_group(0, &self.data_bg, &[]);
            pass.set_bind_group(1, stages.bgs.last().unwrap(), &[]);
            pass.dispatch_workgroups(bitrev_total.div_ceil(256), 1, 1);
        }
    }

    /// Copy `src` to `staging`, map, read `n` f16 elements, and return them as f32.
    ///
    /// The mapping is synchronous (polls until complete). GPU→host byte order is
    /// little-endian IEEE 754 half-precision, matching the upload encoding.
    fn read_back_f16_as_f32(
        &self,
        src: &wgpu::Buffer,
        staging: &wgpu::Buffer,
        n: usize,
    ) -> Vec<f32> {
        let size = self.volume_buf_size();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-fft-wgpu f16 readback encoder"),
            });
        encoder.copy_buffer_to_buffer(src, 0, staging, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait);

        let result: Vec<f32> = {
            let mapped = slice.get_mapped_range();
            mapped
                .chunks_exact(2)
                .take(n)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    HalfF16::from_bits(bits).to_f32()
                })
                .collect()
        };
        staging.unmap();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that native f16 GPU forward FFT matches f32 GPU FFT within f16 numerical tolerance.
    ///
    /// # Mathematical justification
    ///
    /// f16 machine epsilon ε_f16 ≈ 9.77×10⁻⁴. For a 4×4×4 = 64-point 3D FFT,
    /// the radix-2 butterfly accumulates log₂(64)=6 stages per axis. Expected
    /// max absolute error per element:
    ///
    /// `err ≤ 3 · log₂(4) · ε_f16 · ‖input‖_∞ ≈ 3 · 2 · 1e-3 · 1.0 = 6e-3`
    ///
    /// The test allows 1e-2 (10× ε_f16) to account for inter-axis accumulation.
    #[test]
    fn native_f16_forward_matches_f32_within_f16_tolerance_when_device_exists() {
        let Ok(plan_f16) = GpuFft3dF16Native::try_new(4, 4, 4) else {
            // No SHADER_F16 device available; skip.
            return;
        };

        let instance = wgpu::Instance::default();
        let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .ok()
        else {
            return;
        };
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("apollo-fft-wgpu f16 test f32 ref"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            }))
        else {
            return;
        };

        let Ok(plan_f32) = crate::infrastructure::transport::gpu::infrastructure::gpu_fft::GpuFft3d::new(
            Arc::new(device),
            Arc::new(queue),
            4,
            4,
            4,
        ) else {
            return;
        };

        // Analytical test field: deterministic, non-trivial, ‖f‖_∞ ≤ 1.
        let field_f64 = leto::Array3::from_shape_fn((4, 4, 4), |(i, j, k)| {
            let x = (i + j * 3 + k * 7) as f64;
            (0.3 * x).sin() + 0.5 * (0.7 * x).cos()
        });
        let field_f32 = leto::Array3::from_shape_fn((4, 4, 4), |(i, j, k)| {
            let x = (i + j * 3 + k * 7) as f64;
            ((0.3 * x).sin() + 0.5 * (0.7 * x).cos()) as f32
        });

        let out_f32_ref = plan_f32.forward(&field_f64);
        let out_f16_native = plan_f16.forward_native_f16(&field_f32);

        assert_eq!(
            out_f32_ref.len(),
            out_f16_native.len(),
            "output length mismatch"
        );

        for (idx, (a, b)) in out_f32_ref.iter().zip(out_f16_native.iter()).enumerate() {
            let err = (a - b).abs();
            // 1e-2 = ~10 × ε_f16; derived from 3-axis log₂(4) accumulation bound.
            assert!(
                err < 1e-2,
                "f16 native vs f32 error {err:.2e} exceeds 1e-2 at index {idx} \
                 (f32_ref={a:.6}, f16_native={b:.6})"
            );
        }
    }

    /// Verify forward→inverse roundtrip for a non-power-of-two 3×3×3 volume.
    ///
    /// # Mathematical justification
    ///
    /// For each non-power-of-two axis of length N=3, the Bluestein reduction
    /// uses a padded length M = next_pow2(2·3−1) = 4.  Error per axis:
    ///   err_axis ≤ log₂(M) · ε_f16 · ‖input‖_∞ = 2 · 9.77×10⁻⁴ · 1.0 ≈ 2e-3
    ///
    /// For three independent axes the errors add, and the forward and inverse
    /// passes each contribute once, giving:
    ///   total ≤ 2 · 3 · 2e-3 ≈ 1.2e-2
    ///
    /// The test allows 0.05 (40× safety margin) to account for cross-axis
    /// accumulation and f16 intermediate rounding in the chirp premul/postmul
    /// stages.
    #[test]
    fn non_pow2_f16_forward_inverse_roundtrip_when_device_exists() {
        let Ok(plan) = GpuFft3dF16Native::try_new(3, 3, 3) else {
            // No SHADER_F16 device available; skip.
            return;
        };

        // Deterministic 3×3×3 field with values in [−1, 1].
        let field = leto::Array3::from_shape_fn((3, 3, 3), |(i, j, k)| {
            let x = (i + j * 3 + k * 7) as f32;
            (0.3 * x).sin() + 0.5 * (0.7 * x).cos()
        });

        let forward = plan.forward_native_f16(&field);
        let roundtrip = plan.inverse_native_f16(&forward);

        assert_eq!(
            roundtrip.len(),
            27,
            "roundtrip output length must equal nx·ny·nz=27"
        );

        let flat: Vec<f32> = field.iter().copied().collect();
        for (idx, (&orig, &rt)) in flat.iter().zip(roundtrip.iter()).enumerate() {
            let err = (orig - rt).abs();
            // 0.05 is the analytically derived upper bound with 40× safety margin.
            assert!(
                err < 0.05,
                "non-pow2 f16 roundtrip error {err:.4} exceeds 0.05 at index {idx} \
                 (original={orig:.6}, roundtrip={rt:.6})"
            );
        }
    }
}
