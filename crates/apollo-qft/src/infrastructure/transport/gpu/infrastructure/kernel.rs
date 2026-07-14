//! Hephaestus execution kernel for the dense unitary quantum Fourier transform.
//!
//! The forward and inverse entries evaluate the conjugate pair
//! `N^(-1/2) sum_j x[j] exp(+-2*pi*i*j*k/N)`.  Their matrices are unitary, so
//! `QFT^-1(QFT(x)) = x`; the real-device differential and roundtrip suite is
//! the executable evidence tier for the concrete `f32` accelerator contract.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use eunomia::Complex32;
use hephaestus_core::{
    Binding, BindingDecl, CommandStream, DispatchGrid, KernelDevice, KernelInterface, KernelSource,
    Wgsl,
};

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};

const WORKGROUP_SIZE: usize = 64;
const QFT_SOURCE: &str = include_str!("shaders/qft.wgsl");

/// Direction selected before the accelerator boundary.
#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum QftMode {
    /// Forward QFT.
    Forward = 0,
    /// Inverse QFT.
    Inverse = 1,
}

/// Uniform parameters matching WGSL `QftParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct QftParams {
    len: u32,
    mode: u32,
    _padding: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<QftParams>() == 16);

impl QftParams {
    fn new(len: usize, mode: QftMode) -> WgpuResult<Self> {
        Ok(Self {
            len: u32::try_from(len).map_err(|_| WgpuError::InvalidPlan {
                message: format!("transform length {len} exceeds the accelerator parameter range"),
            })?,
            mode: mode as u32,
            _padding: [0; 2],
        })
    }
}

/// Typed Hephaestus interface for the direct unitary QFT kernel.
pub(crate) struct QftKernel;

impl KernelInterface for QftKernel {
    type Params = QftParams;

    const LABEL: &'static str = "apollo-qft-transform";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<Complex32>(),
        BindingDecl::read_write::<Complex32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl KernelSource<Wgsl> for QftKernel {
    const ENTRY: &'static str = "qft_transform";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(QFT_SOURCE)
    }
}

/// Zero-sized QFT orchestration over a Hephaestus device.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct QftGpuKernel;

impl QftGpuKernel {
    /// Execute one direct unitary transform into caller-owned host storage.
    pub(crate) fn execute_into<D>(
        device: &D,
        input: &[Complex32],
        output: &mut [Complex32],
        mode: QftMode,
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        QftKernel: KernelSource<D::Dialect>,
    {
        let input = device.upload(input)?;
        let output_buffer = device.alloc_zeroed::<Complex32>(output.len())?;
        let kernel = device.prepare(&QftKernel)?;
        let bindings = [Binding::read(&input), Binding::read_write(&output_buffer)];
        let grid = DispatchGrid::covering_domain([output.len(), 1, 1], [WORKGROUP_SIZE, 1, 1])?;
        let mut stream = device.stream()?;
        stream.encode(
            &kernel,
            &bindings,
            &QftParams::new(output.len(), mode)?,
            grid,
        )?;
        stream.submit()?;
        device.download(&output_buffer, output)?;
        Ok(())
    }
}
