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

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};

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
pub(crate) struct FrftGpuKernel;

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
