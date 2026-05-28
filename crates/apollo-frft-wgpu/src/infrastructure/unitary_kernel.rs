//! GPU execution kernel for the unitary discrete fractional Fourier transform.
//!
//! Implements DFrFT_a(x) = V · diag(exp(−i·a·k·π/2)) · V^T · x
//! using the Grünbaum eigenbasis (Candan 2000).
//!
//! All three passes are encoded in a single command encoder with sequential
//! compute passes. The implicit memory barrier at each `ComputePass` boundary
//! (WebGPU spec §3.4 sequential pass ordering) guarantees that writes from
//! pass k are globally visible when pass k+1 begins. A single `queue.submit`
//! and two `device.poll` calls replace the previous three-submission pattern,
//! reducing CPU–GPU round-trips from 4 submits + 5 polls to 1 submit + 2 polls.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use num_complex::Complex32;
use wgpu::util::DeviceExt;

use apollo_frft::GrunbaumBasis;

use crate::domain::error::{WgpuError, WgpuResult};

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ComplexPod {
    re: f32,
    im: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct UnitaryParams {
    len: u32,
    step: u32,
    order: f32,
    _pad: u32,
}

/// Cached WGPU pipeline and bind group layout for repeated unitary FrFT dispatches.
///
/// The unitary DFrFT is DFrFT_a(x) = V · diag(exp(−i·a·k·π/2)) · V^T · x where V is
/// the N×N real orthogonal eigenvector matrix from the Grünbaum commuting matrix.
/// All three compute passes are encoded in one command encoder and submitted once.
/// The implicit memory barrier at each `ComputePass` boundary (WebGPU spec §3.4)
/// guarantees cross-pass write visibility without separate submissions.
#[derive(Debug)]
pub struct UnitaryFrftGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl UnitaryFrftGpuKernel {
    /// Compile the unitary FrFT shader and cache the pipeline and bind group layout.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-frft-wgpu unitary shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/frft_unitary.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-frft-wgpu unitary bgl"),
            entries: &[
                // binding 0: input_data — read-only storage (complex signal)
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
                // binding 1: v_mat — read-only storage (f32 column-major eigenvectors)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: intermediate — read-write storage (eigenbasis coefficients)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: output_data — read-write storage (transform result)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 4: params — uniform (len, step, order)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(std::mem::size_of::<UnitaryParams>() as u64)
                                .expect("nonzero uniform size"),
                        ),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-frft-wgpu unitary pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-frft-wgpu unitary pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("unitary_step"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            bind_group_layout,
            pipeline,
        }
    }

    /// Execute the three-pass unitary DFrFT on GPU using a single command encoder.
    ///
    /// Pass 0: `c[k] = Σ_j V[j,k] · x[j]`          (V^T · x)
    /// Pass 1: `c[k] *= exp(−i · order · k · π/2)`   (phase multiply)
    /// Pass 2: `y[j] = Σ_k V[j,k] · c[k]`            (V · c)
    ///
    /// All three compute passes are encoded in a single command encoder followed
    /// by a buffer copy. WebGPU sequential pass ordering (spec §3.4) guarantees
    /// an implicit memory barrier at each pass boundary: writes from pass k are
    /// globally visible when pass k+1 begins. One `queue.submit` and two
    /// `device.poll` calls (one for work completion, one for map callback)
    /// replace the previous three-submission + five-poll pattern.
    ///
    /// `input` must be non-empty; validation is the caller's responsibility.
    pub fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &[Complex32],
        order: f32,
    ) -> WgpuResult<Vec<Complex32>> {
        let n = input.len();

        // Compute Grünbaum eigenbasis CPU-side (O(N³)).
        let basis = GrunbaumBasis::new(n);
        let v = basis.eigenvectors();

        // nalgebra DMatrix<f64> uses column-major storage.
        // as_slice() layout: [v[0,0], v[1,0], …, v[n-1,0], v[0,1], …]
        // which satisfies v_flat[row + col*n] = v[(row, col)]. ✓
        let v_flat: Vec<f32> = v.as_slice().iter().map(|&x| x as f32).collect();

        let complex_byte_len = (n * std::mem::size_of::<ComplexPod>()) as u64;

        // Build input pod buffer.
        let input_pods: Vec<ComplexPod> = input
            .iter()
            .map(|c| ComplexPod { re: c.re, im: c.im })
            .collect();

        let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-frft-wgpu unitary input"),
            contents: bytemuck::cast_slice(&input_pods),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let v_mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-frft-wgpu unitary v_mat"),
            contents: bytemuck::cast_slice(&v_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // intermediate: written by pass 0, updated in-place by pass 1, read by pass 2.
        let intermediate_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-frft-wgpu unitary intermediate"),
            size: complex_byte_len,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // output: written by pass 2, copied to staging for readback.
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-frft-wgpu unitary output"),
            size: complex_byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-frft-wgpu unitary staging"),
            size: complex_byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pre-create one params buffer per pass (step discriminant differs per pass).
        // All three are created before the command encoder so no resource creation
        // occurs inside an active encoding scope.
        let params_buf0 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-frft-wgpu unitary params step0"),
            contents: bytemuck::bytes_of(&UnitaryParams {
                len: n as u32,
                step: 0,
                order,
                _pad: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let params_buf1 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-frft-wgpu unitary params step1"),
            contents: bytemuck::bytes_of(&UnitaryParams {
                len: n as u32,
                step: 1,
                order,
                _pad: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let params_buf2 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-frft-wgpu unitary params step2"),
            contents: bytemuck::bytes_of(&UnitaryParams {
                len: n as u32,
                step: 2,
                order,
                _pad: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Pre-create one bind group per pass. Each bind group is identical except
        // for binding 4 (params_buf), which carries the per-pass step discriminant.
        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-frft-wgpu unitary bg0"),
            layout: &self.bind_group_layout,
            entries: &make_entries(
                &input_buf,
                &v_mat_buf,
                &intermediate_buf,
                &output_buf,
                &params_buf0,
            ),
        });
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-frft-wgpu unitary bg1"),
            layout: &self.bind_group_layout,
            entries: &make_entries(
                &input_buf,
                &v_mat_buf,
                &intermediate_buf,
                &output_buf,
                &params_buf1,
            ),
        });
        let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-frft-wgpu unitary bg2"),
            layout: &self.bind_group_layout,
            entries: &make_entries(
                &input_buf,
                &v_mat_buf,
                &intermediate_buf,
                &output_buf,
                &params_buf2,
            ),
        });

        // Single command encoder: three sequential compute passes followed by a copy.
        // WebGPU spec §3.4: compute passes within a command buffer execute in order;
        // each pass boundary enforces an implicit memory barrier so writes from pass k
        // are globally visible when pass k+1 reads `intermediate_buf`.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-frft-wgpu unitary encoder"),
        });

        // Pass 0: c[k] = Σ_j V[j,k] · x[j]  (V^T · x, writes intermediate_buf)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-frft-wgpu unitary pass0"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.dispatch_workgroups(dispatch_count(n as u32), 1, 1);
        } // pass dropped → ComputePass ends → implicit memory barrier

        // Pass 1: c[k] *= exp(−i·order·k·π/2)  (in-place phase, reads+writes intermediate_buf)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-frft-wgpu unitary pass1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg1, &[]);
            pass.dispatch_workgroups(dispatch_count(n as u32), 1, 1);
        } // pass dropped → ComputePass ends → implicit memory barrier

        // Pass 2: y[j] = Σ_k V[j,k] · c[k]  (V · c, reads intermediate_buf, writes output_buf)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-frft-wgpu unitary pass2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg2, &[]);
            pass.dispatch_workgroups(dispatch_count(n as u32), 1, 1);
        } // pass dropped → ComputePass ends

        // Copy output_buf → staging_buf encoded in the same command buffer.
        // WebGPU command ordering guarantees this executes after pass 2 completes.
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, complex_byte_len);

        // Single submit: all three passes and the copy complete before poll returns.
        queue.submit(std::iter::once(encoder.finish()));
        let _ = device.poll(wgpu::PollType::Wait);

        let slice = staging_buf.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // Second poll drives the map_async callback.
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
            let pods: &[ComplexPod] = bytemuck::cast_slice(&mapped);
            pods.iter().map(|p| Complex32::new(p.re, p.im)).collect()
        };
        staging_buf.unmap();
        Ok(output)
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}

/// Build the five bind group entries for one unitary FrFT compute pass.
///
/// All five buffers are borrowed under the same lifetime `'a`, ensuring the
/// returned array's `BindGroupEntry<'a>` elements are lifetime-consistent.
/// This is a module-level function rather than a closure because Rust's closure
/// lifetime inference cannot express `for<'a> Fn(&'a Buffer) -> [Entry<'a>; 5]`
/// when the closure also captures same-lifetime borrows from the enclosing scope.
fn make_entries<'a>(
    input_buf: &'a wgpu::Buffer,
    v_mat_buf: &'a wgpu::Buffer,
    intermediate_buf: &'a wgpu::Buffer,
    output_buf: &'a wgpu::Buffer,
    params: &'a wgpu::Buffer,
) -> [wgpu::BindGroupEntry<'a>; 5] {
    [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: input_buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: v_mat_buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: intermediate_buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: output_buf.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 4,
            resource: params.as_entire_binding(),
        },
    ]
}
