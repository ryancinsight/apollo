//! Hephaestus execution kernel for the Candan--Gr\u00fcnbaum unitary DFrFT.
//!
//! The ordered stream records `V^T x`, the diagonal phase, and `V c` as three
//! barrier-separated dispatches. The basis is constructed by the CPU-side Leto
//! eigensolver and uploaded as a column-major `f32` storage buffer, preserving
//! the shader's `V[row, column]` contract without a raw device API boundary.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use eunomia::Complex32;
use hephaestus_core::{
    Binding, BindingDecl, CommandStream, DispatchGrid, KernelDevice, KernelInterface, KernelSource,
    Wgsl,
};

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::GrunbaumBasis;

const WORKGROUP_SIZE: usize = 64;
const UNITARY_FRFT_SOURCE: &str = include_str!("shaders/frft_unitary.wgsl");

/// Ordered stage of the Candan--Gr\u00fcnbaum factorization.
#[repr(u32)]
#[derive(Clone, Copy, Debug)]
enum UnitaryStep {
    /// Project the input into the real orthonormal eigenbasis.
    Project = 0,
    /// Multiply eigenbasis coefficients by the unit-modulus phase.
    Phase = 1,
    /// Reconstruct the signal from phase-shifted coefficients.
    Reconstruct = 2,
}

/// Uniform parameters matching WGSL `UnitaryParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct UnitaryParams {
    len: u32,
    step: u32,
    order: f32,
    padding: u32,
}

const _: () = assert!(core::mem::size_of::<UnitaryParams>() == 16);

impl UnitaryParams {
    fn new(len: usize, step: UnitaryStep, order: f32) -> WgpuResult<Self> {
        Ok(Self {
            len: u32::try_from(len).map_err(|_| WgpuError::InvalidPlan {
                message: format!("transform length {len} exceeds the accelerator parameter range"),
            })?,
            step: step as u32,
            order,
            padding: 0,
        })
    }
}

/// Typed Hephaestus interface for all three unitary FrFT passes.
pub(crate) struct UnitaryFrftKernel;

impl KernelInterface for UnitaryFrftKernel {
    type Params = UnitaryParams;

    const LABEL: &'static str = "apollo-frft-unitary";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<Complex32>(),
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<Complex32>(),
        BindingDecl::read_write::<Complex32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl KernelSource<Wgsl> for UnitaryFrftKernel {
    const ENTRY: &'static str = "unitary_step";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(UNITARY_FRFT_SOURCE)
    }
}

/// Zero-sized orchestration for the unitary three-pass DFrFT.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct UnitaryFrftGpuKernel;

impl UnitaryFrftGpuKernel {
    /// Execute `V diag(exp(-i a k pi / 2)) V^T` into caller-owned storage.
    pub(crate) fn execute_into<D>(
        device: &D,
        input: &[Complex32],
        output: &mut [Complex32],
        order: f32,
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        UnitaryFrftKernel: KernelSource<D::Dialect>,
    {
        let len = input.len();
        let basis = GrunbaumBasis::new(len).eigenvectors_column_major_f32();
        let input_buffer = device.upload(input)?;
        let basis_buffer = device.upload(&basis)?;
        let coefficients = device.alloc_zeroed::<Complex32>(len)?;
        let output_buffer = device.alloc_zeroed::<Complex32>(output.len())?;
        let kernel = device.prepare(&UnitaryFrftKernel)?;
        let bindings = [
            Binding::read(&input_buffer),
            Binding::read(&basis_buffer),
            Binding::read_write(&coefficients),
            Binding::read_write(&output_buffer),
        ];
        let grid = DispatchGrid::covering_domain([len, 1, 1], [WORKGROUP_SIZE, 1, 1])?;
        let mut stream = device.stream()?;
        for step in [
            UnitaryStep::Project,
            UnitaryStep::Phase,
            UnitaryStep::Reconstruct,
        ] {
            let params = UnitaryParams::new(len, step, order)?;
            stream.encode(&kernel, &bindings, &params, grid)?;
        }
        stream.submit()?;
        device.download(&output_buffer, output)?;
        Ok(())
    }
}
