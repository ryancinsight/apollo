//! Hephaestus execution kernel for the graph Fourier transform.
//!
//! Forward (mode 0): `X[k] = sum_i U[i,k] * x[i]`  (U^T x)
//! Inverse (mode 1): `x[i] = sum_k U[i,k] * X[k]`  (U X)
//!
//! The basis matrix U is column-major: `basis[i + k*N] = U[i,k]`.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{
    Binding, BindingDecl, CommandStream, DispatchGrid, KernelDevice, KernelInterface, KernelSource,
    Wgsl,
};

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};

const WORKGROUP_SIZE: usize = 64;
const GFT_SOURCE: &str = include_str!("shaders/gft.wgsl");

/// Direction selected before the device dispatch boundary.
#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum GftDirection {
    /// `U^T x`.
    Forward = 0,
    /// `U X`.
    Inverse = 1,
}

/// Uniform parameter block (16 bytes). Fields match WGSL GftParams exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GftParams {
    len: u32,
    mode: u32,
    _padding: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<GftParams>() == 16);

impl GftParams {
    fn new(len: usize, direction: GftDirection) -> WgpuResult<Self> {
        let len = u32::try_from(len).map_err(|_| WgpuError::InvalidPlan {
            message: format!("length {len} exceeds the provider parameter range"),
        })?;
        Ok(Self {
            len,
            mode: direction as u32,
            _padding: [0; 2],
        })
    }
}

pub(crate) struct GftKernel;

impl KernelInterface for GftKernel {
    type Params = GftParams;

    const LABEL: &'static str = "apollo-gft-transform";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<f32>(),
        BindingDecl::read_only::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl KernelSource<Wgsl> for GftKernel {
    const ENTRY: &'static str = "gft_transform";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(GFT_SOURCE)
    }
}

/// Zero-sized graph Fourier kernel orchestration over a Hephaestus device.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GftGpuKernel;

impl GftGpuKernel {
    /// Execute one GFT dispatch on the GPU.
    ///
    pub(crate) fn execute_into<D>(
        device: &D,
        input: &[f32],
        basis: &[f32],
        output: &mut [f32],
        direction: GftDirection,
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        GftKernel: KernelSource<D::Dialect>,
    {
        let input = device.upload(input)?;
        let output_buffer = device.alloc_zeroed::<f32>(output.len())?;
        let basis = device.upload(basis)?;
        let kernel = device.prepare(&GftKernel)?;
        let params = GftParams::new(output.len(), direction)?;
        let grid = DispatchGrid::covering_domain([output.len(), 1, 1], [WORKGROUP_SIZE, 1, 1])?;
        let bindings = [
            Binding::read(&input),
            Binding::read_write(&output_buffer),
            Binding::read(&basis),
        ];
        let mut stream = device.stream()?;
        stream.encode(&kernel, &bindings, &params, grid)?;
        stream.submit()?;
        device.download(&output_buffer, output)?;
        Ok(())
    }
}
