//! Hephaestus execution kernels for the O(N log N) Cooley-Tukey DIT NTT.
//!
//! The host bit-reverses the input, then records one butterfly dispatch per
//! stage.  Each dispatch transforms disjoint index pairs, and the ordered
//! Hephaestus command stream supplies the inter-stage visibility required by
//! the in-place algorithm.  The inverse appends multiplication by `N^-1`.
//!
//! Let `omega` be a primitive `N`-th root in the prime field `F_m`.  The
//! butterfly recurrence implements the factorization of
//! `X[k] = sum_j x[j] omega^(j k)`, and the inverse scale gives
//! `INTT(NTT(x)) = x` by finite-field character orthogonality.  Exact CPU and
//! The private GPU equality test suite is empirical evidence for this theorem
//! on the supported field and plan domain.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{
    Binding, BindingDecl, CommandStream, DispatchGrid, KernelDevice, KernelInterface, KernelSource,
    Wgsl,
};

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};

const WORKGROUP_SIZE: usize = 64;
const NTT_SOURCE: &str = include_str!("shaders/ntt.wgsl");

/// Execution direction selected before the accelerator dispatch boundary.
#[derive(Clone, Copy, Debug)]
pub(crate) enum NttMode {
    /// Evaluate the forward finite-field transform.
    Forward,
    /// Evaluate the inverse finite-field transform.
    Inverse,
}

/// Per-dispatch uniform parameters.  Its layout matches WGSL `NttParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct NttParams {
    n: u32,
    stage_or_ninv: u32,
    modulus: u32,
    _pad: u32,
}

const _: () = assert!(core::mem::size_of::<NttParams>() == 16);

impl NttParams {
    fn butterfly(len: usize, stage: u32, modulus: u32) -> WgpuResult<Self> {
        Ok(Self {
            n: u32::try_from(len).map_err(|_| WgpuError::InvalidPlan {
                message: format!("transform length {len} exceeds the accelerator parameter range"),
            })?,
            stage_or_ninv: stage,
            modulus,
            _pad: 0,
        })
    }

    fn scale(len: usize, inverse_len: u32, modulus: u32) -> WgpuResult<Self> {
        Ok(Self {
            n: u32::try_from(len).map_err(|_| WgpuError::InvalidPlan {
                message: format!("transform length {len} exceeds the accelerator parameter range"),
            })?,
            stage_or_ninv: inverse_len,
            modulus,
            _pad: 0,
        })
    }
}

/// Typed Hephaestus interface for one NTT butterfly stage.
pub(crate) struct NttButterflyKernel;

impl KernelInterface for NttButterflyKernel {
    type Params = NttParams;

    const LABEL: &'static str = "apollo-ntt-butterfly";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_write::<u32>(),
        BindingDecl::read_only::<u32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl KernelSource<Wgsl> for NttButterflyKernel {
    const ENTRY: &'static str = "ntt_butterfly";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(NTT_SOURCE)
    }
}

/// Typed Hephaestus interface for the inverse NTT normalization pass.
pub(crate) struct NttScaleKernel;

impl KernelInterface for NttScaleKernel {
    type Params = NttParams;

    const LABEL: &'static str = "apollo-ntt-scale";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_write::<u32>(),
        BindingDecl::read_only::<u32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl KernelSource<Wgsl> for NttScaleKernel {
    const ENTRY: &'static str = "ntt_scale";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(NTT_SOURCE)
    }
}

/// Zero-sized NTT orchestration over a Hephaestus kernel device.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NttGpuKernel;

impl NttGpuKernel {
    /// Construct reusable host-side transform state for one validated plan.
    pub(crate) fn create_buffers(
        len: usize,
        modulus: u64,
        omega: u64,
    ) -> WgpuResult<NttGpuBuffers> {
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: "invalid NTT buffer length 0".to_owned(),
            });
        }

        let omega_inverse = if len > 1 {
            mod_pow_u64(omega, len as u64 - 1, modulus)
        } else {
            1
        };
        let twiddle_len = (len / 2).max(1);
        Ok(NttGpuBuffers {
            len,
            log2_len: len.trailing_zeros(),
            modulus: u32::try_from(modulus).map_err(|_| WgpuError::InvalidPlan {
                message: format!("modulus {modulus} exceeds u32 accelerator storage"),
            })?,
            inverse_len: mod_pow_u64(len as u64, modulus - 2, modulus) as u32,
            data_residues: vec![0; len],
            output_residues: vec![0; len],
            forward_twiddles: flat_twiddle_array(twiddle_len, omega, modulus),
            inverse_twiddles: flat_twiddle_array(twiddle_len, omega_inverse, modulus),
        })
    }

    /// Execute with `u64` host values, reducing each to its field residue.
    pub(crate) fn execute_with_buffers<D>(
        device: &D,
        input: &[u64],
        mode: NttMode,
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        NttButterflyKernel: KernelSource<D::Dialect>,
        NttScaleKernel: KernelSource<D::Dialect>,
    {
        if input.len() != buffers.len {
            return Err(WgpuError::LengthMismatch {
                expected: buffers.len,
                actual: input.len(),
            });
        }
        let modulus = u64::from(buffers.modulus);
        for (slot, value) in buffers.data_residues.iter_mut().zip(input.iter().copied()) {
            *slot = (value % modulus) as u32;
        }
        bit_reverse_permute(&mut buffers.data_residues);
        Self::execute_from_residues(device, mode, buffers)
    }

    /// Execute with exact `u32` host residues.
    pub(crate) fn execute_quantized_with_buffers<D>(
        device: &D,
        input: &[u32],
        mode: NttMode,
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        NttButterflyKernel: KernelSource<D::Dialect>,
        NttScaleKernel: KernelSource<D::Dialect>,
    {
        if input.len() != buffers.len {
            return Err(WgpuError::LengthMismatch {
                expected: buffers.len,
                actual: input.len(),
            });
        }
        let modulus = u64::from(buffers.modulus);
        for (slot, value) in buffers.data_residues.iter_mut().zip(input.iter().copied()) {
            *slot = (u64::from(value) % modulus) as u32;
        }
        bit_reverse_permute(&mut buffers.data_residues);
        Self::execute_from_residues(device, mode, buffers)
    }

    fn execute_from_residues<D>(
        device: &D,
        mode: NttMode,
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        NttButterflyKernel: KernelSource<D::Dialect>,
        NttScaleKernel: KernelSource<D::Dialect>,
    {
        let data = device.upload(&buffers.data_residues)?;
        let twiddles = match mode {
            NttMode::Forward => device.upload(&buffers.forward_twiddles)?,
            NttMode::Inverse => device.upload(&buffers.inverse_twiddles)?,
        };
        let butterfly = device.prepare(&NttButterflyKernel)?;
        let scale = device.prepare(&NttScaleKernel)?;
        let bindings = [Binding::read_write(&data), Binding::read(&twiddles)];
        let mut stream = device.stream()?;

        let butterfly_grid = DispatchGrid::covering_domain(
            [(buffers.len / 2).max(1), 1, 1],
            [WORKGROUP_SIZE, 1, 1],
        )?;
        for stage in 0..buffers.log2_len {
            stream.encode(
                &butterfly,
                &bindings,
                &NttParams::butterfly(buffers.len, stage, buffers.modulus)?,
                butterfly_grid,
            )?;
        }

        if matches!(mode, NttMode::Inverse) || buffers.log2_len == 0 {
            let scale_grid =
                DispatchGrid::covering_domain([buffers.len, 1, 1], [WORKGROUP_SIZE, 1, 1])?;
            stream.encode(
                &scale,
                &bindings,
                &NttParams::scale(buffers.len, buffers.inverse_len, buffers.modulus)?,
                scale_grid,
            )?;
        }
        stream.submit()?;
        device.download(&data, &mut buffers.data_residues)?;
        for (output, residue) in buffers
            .output_residues
            .iter_mut()
            .zip(buffers.data_residues.iter().copied())
        {
            *output = u64::from(residue);
        }
        Ok(())
    }
}

/// Reusable host-side NTT state.  Device buffers and command resources remain
/// wholly owned by Hephaestus for each submitted transform.
#[derive(Debug)]
pub struct NttGpuBuffers {
    len: usize,
    log2_len: u32,
    modulus: u32,
    inverse_len: u32,
    data_residues: Vec<u32>,
    output_residues: Vec<u64>,
    forward_twiddles: Vec<u32>,
    inverse_twiddles: Vec<u32>,
}

impl NttGpuBuffers {
    /// Return the logical transform length these buffers support.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// A valid NTT buffer set always has a nonzero transform length.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Return the last host readback.
    #[must_use]
    pub(crate) fn output(&self) -> &[u64] {
        &self.output_residues
    }
}

fn flat_twiddle_array(len: usize, root: u64, modulus: u64) -> Vec<u32> {
    let mut twiddles = Vec::with_capacity(len);
    let mut value = 1_u64;
    for _ in 0..len {
        twiddles.push(value as u32);
        value = (value * root) % modulus;
    }
    twiddles
}

fn bit_reverse_permute(data: &mut [u32]) {
    let bits = data.len().trailing_zeros();
    for index in 0..data.len() {
        let reversed = reverse_bits_n(index, bits);
        if reversed > index {
            data.swap(index, reversed);
        }
    }
}

fn reverse_bits_n(mut value: usize, bits: u32) -> usize {
    let mut reversed = 0;
    for _ in 0..bits {
        reversed = (reversed << 1) | (value & 1);
        value >>= 1;
    }
    reversed
}

pub(crate) fn mod_pow_u64(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut result = 1_u64;
    base %= modulus;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
        exponent >>= 1;
    }
    result
}
