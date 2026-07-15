//! Hephaestus execution for the 1D Discrete Hartley Transform.
//!
//! The Hartley transform is
//! `H[k] = sum_{n=0}^{N-1} x[n] cas(2*pi*k*n/N)` with `cas(t) = cos(t) + sin(t)`.
//! The Hartley matrix is symmetric and satisfies `H_N^2 = N I`, so the inverse
//! reuses the same kernel followed by multiplication by `1 / N`.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{
    Binding, BindingDecl, CommandStream, DispatchGrid, KernelDevice, KernelInterface, KernelSource,
    Wgsl,
};

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};

const WORKGROUP_SIZE: usize = 64;
const DHT_SOURCE: &str = include_str!("shaders/dht.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct DhtParams {
    len: u32,
    padding: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<DhtParams>() == 16);

pub(super) struct TransformKernel;

impl KernelInterface for TransformKernel {
    type Params = DhtParams;

    const LABEL: &'static str = "apollo-dht-transform";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl KernelSource<Wgsl> for TransformKernel {
    const ENTRY: &'static str = "dht_transform";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(DHT_SOURCE)
    }
}

pub(super) struct InverseScaleKernel;

impl KernelInterface for InverseScaleKernel {
    type Params = DhtParams;

    const LABEL: &'static str = "apollo-dht-inverse-scale";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl KernelSource<Wgsl> for InverseScaleKernel {
    const ENTRY: &'static str = "dht_scale_inverse";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(DHT_SOURCE)
    }
}

/// Zero-sized DHT kernel orchestration over a Hephaestus kernel device.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct DhtGpuKernel;

impl DhtGpuKernel {
    /// Execute the forward or inverse 1D DHT into caller-owned storage.
    pub(super) fn execute_into<D>(
        device: &D,
        input: &[f32],
        output: &mut [f32],
        inverse: bool,
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        TransformKernel: KernelSource<D::Dialect>,
        InverseScaleKernel: KernelSource<D::Dialect>,
    {
        let len = input.len();
        let encoded_len = u32::try_from(len).map_err(|_| WgpuError::InvalidPlan {
            message: format!("length {len} exceeds the provider parameter range"),
        })?;
        let input_buffer = device.upload(input)?;
        let output_buffer = device.alloc_zeroed::<f32>(output.len())?;
        let transform = device.prepare(&TransformKernel)?;
        let inverse_scale = inverse
            .then(|| device.prepare(&InverseScaleKernel))
            .transpose()?;
        let params = DhtParams {
            len: encoded_len,
            padding: [0; 3],
        };
        let grid = DispatchGrid::covering_domain([len, 1, 1], [WORKGROUP_SIZE, 1, 1])?;
        let bindings = [
            Binding::read(&input_buffer),
            Binding::read_write(&output_buffer),
        ];
        let mut stream = device.stream()?;
        stream.encode(&transform, &bindings, &params, grid)?;
        if let Some(inverse_scale) = inverse_scale {
            stream.encode(&inverse_scale, &bindings, &params, grid)?;
        }
        stream.submit()?;
        device.download(&output_buffer, output)?;
        Ok(())
    }
}
