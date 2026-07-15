//! Typed Hephaestus direct-DFT kernel used by sparse Fourier execution.
//!
//! The sparse transform remains the mathematical owner. This module defines
//! one direction-parameterized kernel interface; Hephaestus owns device
//! allocation, pipeline preparation, binding, dispatch, submission, and
//! transfer.
//!
//! For `x in C^N`, the shader evaluates
//! `X[k] = sum_n x[n] exp(-2 pi i n k / N)` and its normalized inverse
//! `x[n] = (1/N) sum_k X[k] exp(2 pi i n k / N)`. Root-of-unity orthogonality
//! gives `sum_k exp(2 pi i k(n-m)/N) = N delta_nm`, proving that the inverse
//! recovers the dense input in exact arithmetic. Sparse support selection is
//! deliberately outside this kernel because it is the `apollo-sft` domain
//! contract rather than a device concern.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use eunomia::Complex32;
use hephaestus_core::{
    Binding, BindingDecl, CommandStream, DispatchGrid, KernelDevice, KernelInterface, KernelSource,
    Wgsl,
};

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};

const WORKGROUP_SIZE: usize = 64;
const SFT_SOURCE: &str = include_str!("shaders/sft.wgsl");

/// Execution mode for the direct dense transform.
#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum SftMode {
    /// Forward DFT with negative phase.
    Forward = 0,
    /// Normalized inverse DFT with positive phase.
    Inverse = 1,
}

/// Uniform parameters matching WGSL `SftParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct SftParams {
    len: u32,
    mode: u32,
    padding: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<SftParams>() == 16);

impl SftParams {
    fn new(len: usize, mode: SftMode) -> WgpuResult<Self> {
        Ok(Self {
            len: u32::try_from(len).map_err(|_| WgpuError::InvalidPlan {
                message: format!("transform length {len} exceeds the accelerator parameter range"),
            })?,
            mode: mode as u32,
            padding: [0; 2],
        })
    }
}

/// Zero-sized Hephaestus descriptor for the direction-parameterized direct DFT.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SftGpuKernel;

impl KernelInterface for SftGpuKernel {
    type Params = SftParams;

    const LABEL: &'static str = "apollo-sft-direct-dft";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<Complex32>(),
        BindingDecl::read_write::<Complex32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl KernelSource<Wgsl> for SftGpuKernel {
    const ENTRY: &'static str = "sft_direct_dft";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(SFT_SOURCE)
    }
}

impl SftGpuKernel {
    /// Execute the dense direct transform into caller-owned storage.
    pub(crate) fn execute_into<D>(
        device: &D,
        input: &[Complex32],
        output: &mut [Complex32],
        mode: SftMode,
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        SftGpuKernel: KernelSource<D::Dialect>,
    {
        if input.len() != output.len() {
            return Err(WgpuError::LengthMismatch {
                expected: input.len(),
                actual: output.len(),
            });
        }
        let params = SftParams::new(input.len(), mode)?;
        let input_buffer = device.upload(input)?;
        let output_buffer = device.alloc_zeroed::<Complex32>(output.len())?;
        let prepared = device.prepare(&Self)?;
        let bindings = [
            Binding::read(&input_buffer),
            Binding::read_write(&output_buffer),
        ];
        let grid = DispatchGrid::covering_domain([input.len(), 1, 1], [WORKGROUP_SIZE, 1, 1])?;
        let mut stream = device.stream()?;
        stream.encode(&prepared, &bindings, &params, grid)?;
        stream.submit()?;
        device.download(&output_buffer, output)?;
        Ok(())
    }
}
