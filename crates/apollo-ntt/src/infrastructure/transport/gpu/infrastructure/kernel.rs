//! GPU execution for O(N log N) Cooley-Tukey DIT NTT.
//!
//! # Algorithm
//!
//! The forward NTT of length N is computed in `log₂(N)` butterfly passes,
//! each handled by the `ntt_butterfly` WGSL entry point.  The host applies the
//! standard bit-reversal permutation to the input before upload so that the
//! in-place DIT butterfly requires no reordering on the GPU.
//!
//! The inverse NTT appends one scaling pass (`ntt_scale`) that multiplies
//! every element by N⁻¹ mod m.
//!
//! # Twiddle precomputation
//!
//! A flat array `twiddles[k] = omega^k mod m` (k = 0 .. N/2 − 1) is
//! precomputed on the CPU and uploaded once per `NttGpuBuffers`.  At stage s,
//! the twiddle for butterfly offset j is `twiddles[j * (N >> (s+1))]`.
//! This flat layout is provably equivalent to the per-stage twiddle tables
//! used by the CPU implementation (see `NttPlan::calculate_twiddles`).
//!
//! # GPU submission strategy
//!
//! All `log₂(N)` butterfly passes (plus an optional scale pass) are encoded in
//! **one command encoder** and submitted in a single `queue.submit` call.
//! Per-stage parameters (stage index, modulus, N) are written up-front to a
//! UNIFORM buffer at stride-aligned offsets; each compute pass selects its
//! entry via a dynamic uniform offset.  A single `device.poll(Wait)` after
//! submission ensures host-side readback ordering.
//!
//! WebGPU guarantees that compute passes within the same command buffer execute
//! in program order, and that storage writes from one pass are visible to
//! subsequent passes in the same command buffer without explicit barriers.
//!
//! # References
//!
//! - Pollard, J. M. (1971). The fast Fourier transform in a finite field.
//!   *Mathematics of Computation*, 25(114), 365–374.
//! - Cooley, J. W. & Tukey, J. W. (1965). An algorithm for the machine
//!   calculation of complex Fourier series. *Mathematics of Computation*,
//!   19(90), 297–301.

use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use apollo_wgpu_helpers::hephaestus_wgpu::ComputeDevice;
use apollo_wgpu_helpers::hephaestus_wgpu::WgpuBuffer;
use apollo_wgpu_helpers::WgpuDevice;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 64;

/// Byte size of the `NttParams` uniform struct (4 × u32 = 16 bytes).
const NTT_PARAMS_BYTE_SIZE: u32 = 16;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Execution mode: forward (NTT) or inverse (INTT).
#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum NttMode {
    /// Forward NTT: `X[k] = Σ x[j] · ω^{jk} mod m`.
    Forward = 0,
    /// Inverse NTT: `x[j] = N⁻¹ · Σ X[k] · ω⁻^{jk} mod m`.
    Inverse = 1,
}

/// Per-stage uniform parameters for the NTT butterfly shader.
///
/// `stage_or_ninv` is dual-purpose:
/// - `ntt_butterfly` entry: butterfly stage index s (0 .. log₂N − 1).
/// - `ntt_scale` entry: N⁻¹ mod m.
///
/// Reusing one field avoids a second struct and keeps the buffer size at 16
/// bytes for all entries, simplifying dynamic-offset management.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct NttParams {
    n: u32,
    stage_or_ninv: u32,
    modulus: u32,
    _pad: u32,
}

// ---------------------------------------------------------------------------
// NttGpuKernel — compiled pipeline state (one per device)
// ---------------------------------------------------------------------------

/// Compiled WGPU pipeline state for repeated NTT dispatches.
///
/// One `NttGpuKernel` is shared across all `NttGpuBuffers` on the same device.
/// It holds the bind-group layout and two compiled compute pipelines
/// (`ntt_butterfly` and `ntt_scale`); all per-transform state lives in
/// [`NttGpuBuffers`].
#[derive(Debug)]
pub struct NttGpuKernel {
    bind_group_layout: wgpu::BindGroupLayout,
    butterfly_pipeline: wgpu::ComputePipeline,
    scale_pipeline: wgpu::ComputePipeline,
}

impl NttGpuKernel {
    /// Compile both shader entry points and allocate the bind-group layout.
    ///
    /// Compilation is performed once at construction; subsequent dispatches
    /// incur no shader-compilation cost.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("apollo-ntt-wgpu shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ntt.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apollo-ntt-wgpu bgl"),
            entries: &[
                storage_layout_entry(0, false),  // data buffer (read_write)
                storage_layout_entry(1, true),   // twiddles   (read)
                uniform_dynamic_layout_entry(2), // params     (dynamic uniform)
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apollo-ntt-wgpu pipeline layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let butterfly_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-ntt-wgpu butterfly pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("ntt_butterfly"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let scale_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apollo-ntt-wgpu scale pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("ntt_scale"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            bind_group_layout: bgl,
            butterfly_pipeline,
            scale_pipeline,
        }
    }

    // -----------------------------------------------------------------------
    // Buffer management
    // -----------------------------------------------------------------------

    /// Allocate reusable GPU buffers and precompute twiddle factors for one
    /// NTT configuration.
    ///
    /// # Parameters
    /// - `len`    — transform length N (must be a power of two, > 0).
    /// - `modulus` — prime modulus m (must satisfy `(m − 1) % N == 0`).
    /// - `omega`  — primitive N-th root of unity: `omega = g^{(m−1)/N} mod m`
    ///   where `g` is the primitive root of m.
    ///
    /// # Panics
    /// Returns `Err` for `len == 0`; other invariants (power-of-two, valid
    /// modulus) are validated by `NttWgpuBackend` before this call.
    pub fn create_buffers(
        &self,
        device: &WgpuDevice,
        len: usize,
        modulus: u64,
        omega: u64,
    ) -> WgpuResult<NttGpuBuffers> {
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: "invalid NTT GPU buffer length 0".to_string(),
            });
        }

        let n = len;
        let log2_n = n.trailing_zeros();
        let modulus32 = modulus as u32;

        let hep_device = device.hephaestus();

        // omega_inv = omega^{N−1} mod m  (since omega^N ≡ 1 mod m).
        // For N = 1: omega^0 = 1 is the correct inverse of 1.
        let omega_inv = if n > 1 {
            mod_pow_u64(omega, n as u64 - 1, modulus)
        } else {
            1u64
        };

        // N⁻¹ mod m via Fermat: m is prime, so N⁻¹ = N^{m−2} mod m.
        let n_inv = mod_pow_u64(n as u64, modulus - 2, modulus) as u32;

        // Precompute flat twiddle arrays of length max(N/2, 1).
        // twiddles[k] = omega^k mod m  (forward)
        // inv_twiddles[k] = omega_inv^k mod m  (inverse)
        // Length max(N/2, 1) avoids creating a zero-length GPU buffer for N=1.
        let twiddle_len = (n / 2).max(1);

        let fwd_twiddles = flat_twiddle_array(twiddle_len, omega, modulus);
        let inv_twiddles = flat_twiddle_array(twiddle_len, omega_inv, modulus);

        // Determine the stride between per-stage params entries.
        // Must be a multiple of min_uniform_buffer_offset_alignment.
        let alignment = device.inner().limits().min_uniform_buffer_offset_alignment;
        let params_stride = align_up(NTT_PARAMS_BYTE_SIZE, alignment);

        // Pre-write params for every butterfly stage (0 .. log₂N − 1) and
        // the scale stage at index log₂N.
        let num_entries = log2_n + 1;
        let params_total = (num_entries * params_stride) as usize;
        let mut params_raw = vec![0u8; params_total];

        for stage in 0..log2_n {
            let off = (stage * params_stride) as usize;
            let p = NttParams {
                n: n as u32,
                stage_or_ninv: stage,
                modulus: modulus32,
                _pad: 0,
            };
            params_raw[off..off + NTT_PARAMS_BYTE_SIZE as usize]
                .copy_from_slice(bytemuck::bytes_of(&p));
        }
        {
            // Scale entry at index log₂N: stage_or_ninv carries N⁻¹.
            let off = (log2_n * params_stride) as usize;
            let p = NttParams {
                n: n as u32,
                stage_or_ninv: n_inv,
                modulus: modulus32,
                _pad: 0,
            };
            params_raw[off..off + NTT_PARAMS_BYTE_SIZE as usize]
                .copy_from_slice(bytemuck::bytes_of(&p));
        }

        // ----- Allocate GPU buffers -----------------------------------------

        let data_buffer = hep_device.alloc_zeroed::<u32>(n).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        let fwd_twiddle_buffer = hep_device.upload(&fwd_twiddles).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        let inv_twiddle_buffer = hep_device.upload(&inv_twiddles).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        let params_buffer = device.inner().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-ntt-wgpu params"),
            contents: &params_raw,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // ----- Bind groups ---------------------------------------------------
        // Two bind groups sharing `data_buffer` and `params_buffer` but
        // differing in which twiddle buffer is bound (forward vs inverse).
        let params_size =
            NonZeroU64::new(NTT_PARAMS_BYTE_SIZE as u64).expect("nonzero params size");

        let fwd_bind_group = device.inner().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-ntt-wgpu fwd bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: fwd_twiddle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: Some(params_size),
                    }),
                },
            ],
        });

        let inv_bind_group = device.inner().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-ntt-wgpu inv bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: inv_twiddle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: Some(params_size),
                    }),
                },
            ],
        });

        Ok(NttGpuBuffers {
            len: n,
            log2_n,
            modulus: modulus32,
            params_stride,
            data_residues: vec![0u32; n],
            output_residues: vec![0u64; n],
            data_buffer,
            _fwd_twiddle_buffer: fwd_twiddle_buffer,
            _inv_twiddle_buffer: inv_twiddle_buffer,
            _params_buffer: params_buffer,
            fwd_bind_group,
            inv_bind_group,
        })
    }

    // -----------------------------------------------------------------------
    // Execution — allocating paths
    // -----------------------------------------------------------------------

    /// Allocate temporary buffers, execute, and return the output.
    ///
    /// For the reusable-buffer hot path use [`Self::execute_with_buffers`] instead.
    pub fn execute(
        &self,
        device: &WgpuDevice,
        input: &[u64],
        len: usize,
        modulus: u64,
        omega: u64,
        mode: NttMode,
    ) -> WgpuResult<Vec<u64>> {
        let mut bufs = self.create_buffers(device, len, modulus, omega)?;
        self.execute_with_buffers(device, input, mode, &mut bufs)?;
        Ok(bufs.output_residues.clone())
    }

    // -----------------------------------------------------------------------
    // Execution — reusable-buffer paths
    // -----------------------------------------------------------------------

    /// Execute an NTT with caller-owned reusable buffers (u64 input).
    ///
    /// Input values are reduced modulo `m` before upload; residues outside
    /// `[0, m)` are accepted and normalised automatically.
    pub fn execute_with_buffers(
        &self,
        device: &WgpuDevice,
        input: &[u64],
        mode: NttMode,
        bufs: &mut NttGpuBuffers,
    ) -> WgpuResult<()> {
        let len = bufs.len;
        if input.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input.len(),
            });
        }
        let m = u64::from(bufs.modulus);
        for (slot, &v) in bufs.data_residues.iter_mut().zip(input) {
            *slot = (v % m) as u32;
        }
        bit_reverse_permute(&mut bufs.data_residues);
        self.execute_from_residues(device, mode, bufs)
    }

    /// Execute an NTT with caller-owned reusable buffers (u32 residue input).
    ///
    /// Accepts exact `u32` residues already bounded by the modulus; values
    /// outside `[0, m)` are still reduced modulo `m` for safety.
    pub fn execute_quantized_with_buffers(
        &self,
        device: &WgpuDevice,
        input: &[u32],
        mode: NttMode,
        bufs: &mut NttGpuBuffers,
    ) -> WgpuResult<()> {
        let len = bufs.len;
        if input.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input.len(),
            });
        }
        let m = u64::from(bufs.modulus);
        for (slot, &v) in bufs.data_residues.iter_mut().zip(input) {
            *slot = (u64::from(v) % m) as u32;
        }
        bit_reverse_permute(&mut bufs.data_residues);
        self.execute_from_residues(device, mode, bufs)
    }

    /// Return a reference to the last readback output stored in `bufs`.
    #[must_use]
    pub fn buffer_output<'a>(&self, bufs: &'a NttGpuBuffers) -> &'a [u64] {
        &bufs.output_residues
    }

    // -----------------------------------------------------------------------
    // Internal — GPU dispatch
    // -----------------------------------------------------------------------

    /// Common GPU execution path after `data_residues` has been populated and
    /// bit-reversed by the caller.
    fn execute_from_residues(
        &self,
        device: &WgpuDevice,
        mode: NttMode,
        bufs: &mut NttGpuBuffers,
    ) -> WgpuResult<()> {
        let hep_device = device.hephaestus();

        // Upload bit-reversed input residues to the in-place data buffer.
        hep_device.write_buffer(&bufs.data_buffer, &bufs.data_residues).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        // Select the bind group (forward twiddles or inverse twiddles).
        let bind_group = match mode {
            NttMode::Forward => &bufs.fwd_bind_group,
            NttMode::Inverse => &bufs.inv_bind_group,
        };

        // Encode all butterfly passes and (optionally) the scale pass in one
        // command buffer.  WebGPU guarantees that compute passes in the same
        // command buffer execute in submission order and that storage writes
        // from earlier passes are visible to later passes without explicit
        // barriers.
        let mut encoder = device.inner().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-ntt-wgpu encoder"),
        });

        // Butterfly passes: one per stage, dispatching N/2 threads total.
        let half_n = ((bufs.len / 2).max(1)) as u32;
        let butterfly_wg = dispatch_count(half_n);

        for stage in 0..bufs.log2_n {
            let dynamic_offset = stage * bufs.params_stride;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-ntt-wgpu butterfly pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.butterfly_pipeline);
            pass.set_bind_group(0, bind_group, &[dynamic_offset]);
            pass.dispatch_workgroups(butterfly_wg, 1, 1);
        }

        // Scale pass for inverse NTT: multiply every element by N⁻¹ mod m.
        if matches!(mode, NttMode::Inverse) {
            let scale_offset = bufs.log2_n * bufs.params_stride;
            let scale_wg = dispatch_count(bufs.len as u32);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-ntt-wgpu scale pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scale_pipeline);
            pass.set_bind_group(0, bind_group, &[scale_offset]);
            pass.dispatch_workgroups(scale_wg, 1, 1);
        }

        // Single submission for all passes.
        device.queue().submit(std::iter::once(encoder.finish()));

        let mut output_u32 = vec![0u32; bufs.len];
        hep_device.download(&bufs.data_buffer, &mut output_u32).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        for (slot, &v) in bufs.output_residues.iter_mut().zip(&output_u32) {
            *slot = u64::from(v);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// NttGpuBuffers — per-plan reusable GPU state
// ---------------------------------------------------------------------------

/// Reusable GPU and host storage for repeated NTT dispatches at one (N, m, ω)
/// configuration.
///
/// Created by [`NttGpuKernel::create_buffers`]; reused across repeated calls
/// to [`NttGpuKernel::execute_with_buffers`] to avoid re-allocating GPU
/// buffers and re-computing twiddle factors.
#[derive(Debug)]
pub struct NttGpuBuffers {
    // ── Metadata ────────────────────────────────────────────────────────────
    len: usize,
    log2_n: u32,
    modulus: u32,
    params_stride: u32, // aligned byte stride between per-stage params entries

    // ── CPU scratch ─────────────────────────────────────────────────────────
    data_residues: Vec<u32>,   // bit-reversed input; reused each dispatch
    output_residues: Vec<u64>, // readback destination; reused each dispatch

    // ── GPU buffers ─────────────────────────────────────────────────────────
    data_buffer: WgpuBuffer<u32>, // in-place NTT data  (STORAGE rw)
    /// Forward twiddle buffer.  Kept alive so the GPU resource lives as long
    /// as the bind group that references it.  Not read directly from Rust.
    _fwd_twiddle_buffer: WgpuBuffer<u32>,
    /// Inverse twiddle buffer.  Same lifetime-keeper contract as above.
    _inv_twiddle_buffer: WgpuBuffer<u32>,
    /// Per-stage params uniform buffer.  Kept alive for the bind groups.
    _params_buffer: wgpu::Buffer,

    // ── Bind groups ─────────────────────────────────────────────────────────
    fwd_bind_group: wgpu::BindGroup, // data + fwd_twiddles + params (dynamic)
    inv_bind_group: wgpu::BindGroup, // data + inv_twiddles + params (dynamic)
}

impl NttGpuBuffers {
    /// Return the logical transform length these buffers support.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Return whether these buffers carry zero transform length.
    ///
    /// `NttGpuBuffers` with `len == 0` cannot be constructed by
    /// [`NttGpuKernel::create_buffers`]; this method always returns `false`
    /// for any successfully constructed instance.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Compute the flat twiddle array `[root^0, root^1, ..., root^{len-1}]` mod m.
fn flat_twiddle_array(len: usize, root: u64, modulus: u64) -> Vec<u32> {
    let mut t = Vec::with_capacity(len);
    let mut acc = 1u64;
    for _ in 0..len {
        t.push(acc as u32);
        acc = (acc * root) % modulus;
    }
    t
}

/// In-place bit-reversal permutation (Cooley-Tukey DIT prerequisite).
///
/// For N = 0 or N = 1 this is a no-op.  For a power-of-two N this performs
/// the standard log₂(N)-bit reversal by swapping every pair (i, j) with
/// j = bitrev(i) once (skipping i >= j to avoid double-swaps).
fn bit_reverse_permute(data: &mut [u32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = reverse_bits_n(i, bits);
        if j > i {
            data.swap(i, j);
        }
    }
}

/// Reverse the lowest `bits` bits of `x`.
fn reverse_bits_n(mut x: usize, bits: u32) -> usize {
    let mut r = 0usize;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

/// Modular exponentiation via binary squaring with 128-bit intermediate.
///
/// Computes `base^exp mod modulus` without overflow for `base, modulus < 2^62`.
fn mod_pow_u64(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
        exp >>= 1;
    }
    result
}

/// Round `value` up to the next multiple of `alignment`.
const fn align_up(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) / alignment * alignment
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

fn uniform_dynamic_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: Some(
                NonZeroU64::new(NTT_PARAMS_BYTE_SIZE as u64).expect("nonzero uniform binding size"),
            ),
        },
        count: None,
    }
}

fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}

