//! Hephaestus execution kernel for the direct fractional Fourier transform.
//!
//! The direct kernel evaluates the centred-coordinate FrFT. Integer quarter
//! rotations select their exact identity, DFT, reversal, or inverse-DFT
//! specializations; non-integer orders select the chirp formula. The CPU
//! differential suite is the executable evidence tier for the concrete
//! `Complex32` accelerator contract.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use eunomia::Complex32;
use hephaestus_core::{
    Binding, BindingDecl, CommandStream, DispatchGrid, KernelDevice, KernelInterface, KernelSource,
    Wgsl,
};

use apollo_fft::{GpuTransformExecutor, GpuTransformPlanner, WgpuError, WgpuResult};
use hephaestus_wgpu::WgpuDevice;

use crate::infrastructure::transport::gpu::OrderPlan;

const WORKGROUP_SIZE: usize = 64;
const FRFT_SOURCE: &str = include_str!("shaders/frft.wgsl");

/// Exact direct-FrFT mode selected before accelerator dispatch.
#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum FrftMode {
    /// Identity at orders congruent to zero modulo four.
    Identity = 0,
    /// Centered unitary DFT at orders congruent to one modulo four.
    CenteredDft = 1,
    /// Sample reversal at orders congruent to two modulo four.
    Reversal = 2,
    /// Centered unitary inverse DFT at orders congruent to three modulo four.
    CenteredInverseDft = 3,
    /// General centered-coordinate chirp kernel.
    Chirp = 4,
}

/// Uniform parameters matching WGSL `FrftParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FrftParams {
    len: u32,
    mode: u32,
    cot: f32,
    csc: f32,
    scale_re: f32,
    scale_im: f32,
    padding: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<FrftParams>() == 32);

impl FrftParams {
    fn new(
        len: usize,
        mode: FrftMode,
        cot: f32,
        csc: f32,
        scale_re: f32,
        scale_im: f32,
    ) -> WgpuResult<Self> {
        Ok(Self {
            len: u32::try_from(len).map_err(|_| WgpuError::InvalidPlan {
                message: format!("transform length {len} exceeds the accelerator parameter range"),
            })?,
            mode: mode as u32,
            cot,
            csc,
            scale_re,
            scale_im,
            padding: [0; 2],
        })
    }
}

/// Typed Hephaestus interface for the direct FrFT kernel.
pub(crate) struct FrftKernel;

impl KernelInterface for FrftKernel {
    type Params = FrftParams;

    const LABEL: &'static str = "apollo-frft-transform";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<Complex32>(),
        BindingDecl::read_write::<Complex32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl KernelSource<Wgsl> for FrftKernel {
    const ENTRY: &'static str = "frft_transform";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(FRFT_SOURCE)
    }
}

/// Zero-sized direct FrFT orchestration over a Hephaestus device.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrftGpuKernel;

impl GpuTransformPlanner for FrftGpuKernel {
    type Plan = OrderPlan;

    fn input_len(plan: &OrderPlan) -> usize {
        plan.len()
    }

    fn validate(plan: &OrderPlan) -> WgpuResult<()> {
        if !plan.order().is_finite() {
            return Err(WgpuError::NonFiniteParameter { parameter: "order" });
        }
        Ok(())
    }
}

impl GpuTransformExecutor for FrftGpuKernel {
    type Sample = Complex32;
    type Bin = Complex32;

    fn forward_into(
        device: &WgpuDevice,
        plan: &OrderPlan,
        input: &[Complex32],
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        let (mode, cot, csc, scale_re, scale_im) = mode_params(plan, false)?;
        Self::execute_into(device, input, output, mode, cot, csc, scale_re, scale_im)
    }

    fn inverse_into(
        device: &WgpuDevice,
        plan: &OrderPlan,
        input: &[Complex32],
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        let (mode, cot, csc, scale_re, scale_im) = mode_params(plan, true)?;
        Self::execute_into(device, input, output, mode, cot, csc, scale_re, scale_im)
    }
}

/// Determine the exact mode and chirp parameters for a direct FrFT dispatch.
fn mode_params(plan: &OrderPlan, inverse: bool) -> WgpuResult<(FrftMode, f32, f32, f32, f32)> {
    let order = if inverse { -plan.order() } else { plan.order() };
    if !order.is_finite() {
        return Err(WgpuError::NonFiniteParameter { parameter: "order" });
    }
    let reduced = ((order % 4.0_f32) + 4.0_f32) % 4.0_f32;
    let rounded = reduced.round();
    if (reduced - rounded).abs() < 1.0e-5_f32 {
        let mode = if reduced < 0.5_f32 || reduced > 3.5_f32 {
            FrftMode::Identity
        } else if reduced < 1.5_f32 {
            FrftMode::CenteredDft
        } else if reduced < 2.5_f32 {
            FrftMode::Reversal
        } else {
            FrftMode::CenteredInverseDft
        };
        return Ok((mode, 0.0, 0.0, 1.0, 0.0));
    }

    let alpha = reduced * core::f32::consts::FRAC_PI_2;
    let sin_alpha = alpha.sin();
    let cot = alpha.cos() / sin_alpha;
    let csc = sin_alpha.recip();
    let z_norm = (1.0_f32 + cot * cot).sqrt();
    let z_arg = (-cot).atan2(1.0_f32);
    let scale_radius = z_norm.sqrt() / (plan.len() as f32).sqrt();
    let scale_angle = z_arg * 0.5_f32;
    Ok((
        FrftMode::Chirp,
        cot,
        csc,
        scale_radius * scale_angle.cos(),
        scale_radius * scale_angle.sin(),
    ))
}

impl FrftGpuKernel {
    /// Execute one direct FrFT into caller-owned host storage.
    pub(crate) fn execute_into<D>(
        device: &D,
        input: &[Complex32],
        output: &mut [Complex32],
        mode: FrftMode,
        cot: f32,
        csc: f32,
        scale_re: f32,
        scale_im: f32,
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        FrftKernel: KernelSource<D::Dialect>,
    {
        let input_buffer = device.upload(input)?;
        let output_buffer = device.alloc_zeroed::<Complex32>(output.len())?;
        let kernel = device.prepare(&FrftKernel)?;
        let bindings = [
            Binding::read(&input_buffer),
            Binding::read_write(&output_buffer),
        ];
        let grid = DispatchGrid::covering_domain([output.len(), 1, 1], [WORKGROUP_SIZE, 1, 1])?;
        let params = FrftParams::new(input.len(), mode, cot, csc, scale_re, scale_im)?;
        let mut stream = device.stream()?;
        stream.encode(&kernel, &bindings, &params, grid)?;
        stream.submit()?;
        device.download(&output_buffer, output)?;
        Ok(())
    }
}
