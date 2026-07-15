//! Hephaestus execution kernels for the discrete Hilbert transform.
//!
//! Forward analytic construction records DFT, analytic-mask, and inverse-DFT
//! passes. Inverse reconstruction records DFT, inverse-mask, and inverse-DFT
//! passes. Ordered command streams preserve each spectral dependency while
//! Hephaestus owns binding, pipeline, dispatch, and transfer mechanics.

use std::{borrow::Cow, marker::PhantomData};

use bytemuck::{Pod, Zeroable};
use eunomia::Complex32;
use hephaestus_core::{
    Binding, BindingDecl, CommandStream, DispatchGrid, KernelDevice, KernelInterface, KernelSource,
    Wgsl,
};

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};

const WORKGROUP_SIZE: usize = 64;
const HILBERT_SOURCE: &str = include_str!("shaders/hilbert.wgsl");

/// Uniform parameters matching WGSL `HilbertParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HilbertParams {
    len: u32,
    padding: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<HilbertParams>() == 16);

impl HilbertParams {
    fn new(len: usize) -> WgpuResult<Self> {
        Ok(Self {
            len: u32::try_from(len).map_err(|_| WgpuError::InvalidPlan {
                message: format!("transform length {len} exceeds the accelerator parameter range"),
            })?,
            padding: [0; 3],
        })
    }
}

/// Compile-time selection of one Hilbert shader entry point.
trait HilbertPass {
    /// Provider diagnostic label for the entry point.
    const LABEL: &'static str;
    /// WGSL entry point name.
    const ENTRY: &'static str;
}

/// Typed Hephaestus interface for one Hilbert shader pass.
pub(crate) struct HilbertKernel<P>(PhantomData<P>);

impl<P: HilbertPass> KernelInterface for HilbertKernel<P> {
    type Params = HilbertParams;

    const LABEL: &'static str = P::LABEL;
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<Complex32>(),
        BindingDecl::read_write::<Complex32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl<P: HilbertPass> KernelSource<Wgsl> for HilbertKernel<P> {
    const ENTRY: &'static str = P::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(HILBERT_SOURCE)
    }
}

/// Marker for the forward DFT pass.
pub(crate) struct ForwardDft;

impl HilbertPass for ForwardDft {
    const LABEL: &'static str = "apollo-hilbert-forward-dft";
    const ENTRY: &'static str = "hilbert_forward_dft";
}

/// Marker for the analytic-spectrum mask pass.
pub(crate) struct AnalyticMask;

impl HilbertPass for AnalyticMask {
    const LABEL: &'static str = "apollo-hilbert-analytic-mask";
    const ENTRY: &'static str = "hilbert_apply_mask";
}

/// Marker for the inverse DFT pass.
pub(crate) struct InverseDft;

impl HilbertPass for InverseDft {
    const LABEL: &'static str = "apollo-hilbert-inverse-dft";
    const ENTRY: &'static str = "hilbert_inverse_dft";
}

/// Marker for the inverse-spectrum mask pass.
pub(crate) struct InverseMask;

impl HilbertPass for InverseMask {
    const LABEL: &'static str = "apollo-hilbert-inverse-mask";
    const ENTRY: &'static str = "hilbert_inverse_mask";
}

/// Zero-sized Hilbert orchestration over a Hephaestus device.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct HilbertGpuKernel;

impl HilbertGpuKernel {
    /// Execute `x + i H{x}` into caller-owned complex output storage.
    pub(crate) fn execute_analytic_into<D>(
        device: &D,
        input: &[Complex32],
        output: &mut [Complex32],
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        HilbertKernel<ForwardDft>: KernelSource<D::Dialect>,
        HilbertKernel<AnalyticMask>: KernelSource<D::Dialect>,
        HilbertKernel<InverseDft>: KernelSource<D::Dialect>,
    {
        let len = input.len();
        let input_buffer = device.upload(input)?;
        let spectrum_buffer = device.alloc_zeroed::<Complex32>(len)?;
        let output_buffer = device.alloc_zeroed::<Complex32>(output.len())?;
        let forward = device.prepare(&HilbertKernel::<ForwardDft>(PhantomData))?;
        let mask = device.prepare(&HilbertKernel::<AnalyticMask>(PhantomData))?;
        let inverse = device.prepare(&HilbertKernel::<InverseDft>(PhantomData))?;
        let spectrum_bindings = [
            Binding::read(&input_buffer),
            Binding::read_write(&spectrum_buffer),
        ];
        let output_bindings = [
            Binding::read(&spectrum_buffer),
            Binding::read_write(&output_buffer),
        ];
        let grid = DispatchGrid::covering_domain([len, 1, 1], [WORKGROUP_SIZE, 1, 1])?;
        let params = HilbertParams::new(len)?;
        let mut stream = device.stream()?;
        stream.encode(&forward, &spectrum_bindings, &params, grid)?;
        stream.encode(&mask, &spectrum_bindings, &params, grid)?;
        stream.encode(&inverse, &output_bindings, &params, grid)?;
        stream.submit()?;
        device.download(&output_buffer, output)?;
        Ok(())
    }

    /// Execute inverse Hilbert reconstruction into caller-owned complex output storage.
    pub(crate) fn execute_inverse_into<D>(
        device: &D,
        quadrature: &[Complex32],
        output: &mut [Complex32],
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        HilbertKernel<ForwardDft>: KernelSource<D::Dialect>,
        HilbertKernel<InverseMask>: KernelSource<D::Dialect>,
        HilbertKernel<InverseDft>: KernelSource<D::Dialect>,
    {
        let len = quadrature.len();
        let input_buffer = device.upload(quadrature)?;
        let spectrum_buffer = device.alloc_zeroed::<Complex32>(len)?;
        let recovered_buffer = device.alloc_zeroed::<Complex32>(len)?;
        let output_buffer = device.alloc_zeroed::<Complex32>(output.len())?;
        let forward = device.prepare(&HilbertKernel::<ForwardDft>(PhantomData))?;
        let inverse_mask = device.prepare(&HilbertKernel::<InverseMask>(PhantomData))?;
        let inverse = device.prepare(&HilbertKernel::<InverseDft>(PhantomData))?;
        let spectrum_bindings = [
            Binding::read(&input_buffer),
            Binding::read_write(&spectrum_buffer),
        ];
        let recovered_bindings = [
            Binding::read(&spectrum_buffer),
            Binding::read_write(&recovered_buffer),
        ];
        let output_bindings = [
            Binding::read(&recovered_buffer),
            Binding::read_write(&output_buffer),
        ];
        let grid = DispatchGrid::covering_domain([len, 1, 1], [WORKGROUP_SIZE, 1, 1])?;
        let params = HilbertParams::new(len)?;
        let mut stream = device.stream()?;
        stream.encode(&forward, &spectrum_bindings, &params, grid)?;
        stream.encode(&inverse_mask, &recovered_bindings, &params, grid)?;
        stream.encode(&inverse, &output_bindings, &params, grid)?;
        stream.submit()?;
        device.download(&output_buffer, output)?;
        Ok(())
    }
}
