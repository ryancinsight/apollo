//! Hephaestus execution for the 1D Walsh-Hadamard butterfly network.
//!
//! Let `H_n` denote the Hadamard matrix for `n = 2^m`, with entries
//! `H_n[k, j] = (-1)^{popcount(k & j)}`. Its radix-2 factorization is the
//! ordered stage sequence `(a, b) -> (a + b, a - b)` at strides
//! `1, 2, 4, ..., n / 2`. Since `H_n² = nI`, inverse execution applies the
//! same stages followed by multiplication by `1 / n`.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{
    Binding, BindingDecl, CommandStream, DispatchGrid, KernelDevice, KernelInterface, KernelSource,
    Wgsl,
};

use apollo_fft::{GpuTransformExecutor, GpuTransformPlanner, WgpuError, WgpuResult};
use hephaestus_core::ComputeDevice;
use hephaestus_wgpu::WgpuDevice;

const WORKGROUP_SIZE: usize = 256;
const FWHT_SOURCE: &str = include_str!("shaders/fwht.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct FwhtParams {
    len: u32,
    stride: u32,
    padding: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<FwhtParams>() == 16);

pub(super) struct ButterflyKernel;

impl KernelInterface for ButterflyKernel {
    type Params = FwhtParams;

    const LABEL: &'static str = "apollo-fwht-butterfly";
    const BINDINGS: &'static [BindingDecl] = &[BindingDecl::read_write::<f32>()];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl KernelSource<Wgsl> for ButterflyKernel {
    const ENTRY: &'static str = "fwht_butterfly";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(FWHT_SOURCE)
    }
}

pub(super) struct InverseScaleKernel;

impl KernelInterface for InverseScaleKernel {
    type Params = FwhtParams;

    const LABEL: &'static str = "apollo-fwht-inverse-scale";
    const BINDINGS: &'static [BindingDecl] = &[BindingDecl::read_write::<f32>()];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl KernelSource<Wgsl> for InverseScaleKernel {
    const ENTRY: &'static str = "fwht_scale_inverse";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(FWHT_SOURCE)
    }
}

/// Zero-sized FWHT kernel orchestration over a Hephaestus kernel device.
#[derive(Clone, Copy, Debug, Default)]
pub struct FwhtGpuKernel;

impl GpuTransformPlanner for FwhtGpuKernel {
    type Plan = usize;

    fn input_len(plan: &usize) -> usize {
        *plan
    }

    /// The radix-2 butterfly factorization requires a power-of-two
    /// length; this is the FWHT's structural plan constraint.
    fn validate(plan: &usize) -> WgpuResult<()> {
        if !plan.is_power_of_two() {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {plan}: length must be a power of two"),
            });
        }
        Ok(())
    }
}

impl GpuTransformExecutor for FwhtGpuKernel {
    type Sample = f32;
    type Bin = f32;

    fn forward_into(
        device: &WgpuDevice,
        plan: &usize,
        input: &[f32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        Self::execute_into(device, plan, input, output, false)
    }

    fn inverse_into(
        device: &WgpuDevice,
        plan: &usize,
        input: &[f32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        Self::execute_into(device, plan, input, output, true)
    }
}

impl FwhtGpuKernel {
    /// Execute the forward or inverse 1D FWHT into caller-owned storage.
    fn execute_into(
        device: &WgpuDevice,
        _plan: &usize,
        input: &[f32],
        output: &mut [f32],
        inverse: bool,
    ) -> WgpuResult<()> {
        Self::run(device, input, output, inverse)
    }

    fn run(
        device: &WgpuDevice,
        input: &[f32],
        output: &mut [f32],
        inverse: bool,
    ) -> WgpuResult<()> {
        let len = input.len();
        let encoded_len = u32::try_from(len).map_err(|_| WgpuError::InvalidPlan {
            message: format!("length {len} exceeds the provider parameter range"),
        })?;
        let storage = device.upload(input)?;
        let butterfly = device.prepare(&ButterflyKernel)?;
        let inverse_scale = inverse
            .then(|| device.prepare(&InverseScaleKernel))
            .transpose()?;
        let mut stream = device.stream()?;

        let butterfly_grid =
            DispatchGrid::covering_domain([len / 2, 1, 1], [WORKGROUP_SIZE, 1, 1])?;
        let mut stride = 1usize;
        while stride < len {
            let encoded_stride = u32::try_from(stride).map_err(|_| WgpuError::InvalidPlan {
                message: format!("stride {stride} exceeds the provider parameter range"),
            })?;
            stream.encode(
                &butterfly,
                &[Binding::read_write(&storage)],
                &FwhtParams {
                    len: encoded_len,
                    stride: encoded_stride,
                    padding: [0; 2],
                },
                butterfly_grid,
            )?;
            stride <<= 1;
        }

        if let Some(inverse_scale) = inverse_scale {
            stream.encode(
                &inverse_scale,
                &[Binding::read_write(&storage)],
                &FwhtParams {
                    len: encoded_len,
                    stride: 0,
                    padding: [0; 2],
                },
                DispatchGrid::covering_domain([len, 1, 1], [WORKGROUP_SIZE, 1, 1])?,
            )?;
        }

        stream.submit()?;
        device.download(&storage, output)?;
        Ok(())
    }
}
