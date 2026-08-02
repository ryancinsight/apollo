//! Hephaestus execution for the direct complex chirp z-transform.
//!
//! For input `x[n]`, starting point `A`, and spiral ratio `W`, the CZT is
//! `X[k] = sum_{n=0}^{N-1} x[n] A^{-n} W^{nk}`. The authored kernels evaluate
//! the direct forward formula and its square-plan adjoint inverse in `f32`
//! complex arithmetic. Hephaestus owns buffers, preparation, dispatch,
//! synchronization, and transfer; Apollo owns these transform formulas.

use std::{borrow::Cow, marker::PhantomData};

use bytemuck::{Pod, Zeroable};
use eunomia::Complex32;
use hephaestus_core::{
    Binding, BindingDecl, CommandStream, DispatchGrid, KernelDevice, KernelInterface, KernelSource,
    Wgsl,
};

use apollo_fft::{GpuTransformExecutor, WgpuError, WgpuResult};
use hephaestus_wgpu::WgpuDevice;
/// Plan payload for the chirp-z transform: logical lengths and the
/// spiral parameters stored as IEEE bit patterns so the payload stays
/// `Eq`/`Hash`-clean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChirpPlan {
    input_len: usize,
    output_len: usize,
    a_re: u32,
    a_im: u32,
    w_re: u32,
    w_im: u32,
}

impl ChirpPlan {
    /// Create a chirp-z plan payload from the spiral parameters.
    #[must_use]
    pub fn new(input_len: usize, output_len: usize, a: Complex32, w: Complex32) -> Self {
        Self {
            input_len,
            output_len,
            a_re: a.re.to_bits(),
            a_im: a.im.to_bits(),
            w_re: w.re.to_bits(),
            w_im: w.im.to_bits(),
        }
    }

    /// Return the logical input length carried by this payload.
    #[must_use]
    pub const fn input_len(self) -> usize {
        self.input_len
    }

    /// Return the logical output length carried by this payload.
    #[must_use]
    pub const fn output_len(self) -> usize {
        self.output_len
    }

    /// Return the starting spiral point.
    #[must_use]
    pub fn a(self) -> Complex32 {
        Complex32::new(f32::from_bits(self.a_re), f32::from_bits(self.a_im))
    }

    /// Return the spiral step ratio.
    #[must_use]
    pub fn w(self) -> Complex32 {
        Complex32::new(f32::from_bits(self.w_re), f32::from_bits(self.w_im))
    }
}

const WORKGROUP_SIZE: usize = 64;
const CZT_SOURCE: &str = include_str!("shaders/czt.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct CztParams {
    input_len: u32,
    output_len: u32,
    a_re: f32,
    a_im: f32,
    w_re: f32,
    w_im: f32,
    padding: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<CztParams>() == 32);

pub(super) trait CztDirection {
    const LABEL: &'static str;
    const ENTRY: &'static str;
}

pub(super) struct Forward;

impl CztDirection for Forward {
    const LABEL: &'static str = "apollo-czt-forward";
    const ENTRY: &'static str = "czt_forward";
}

pub(super) struct Inverse;

impl CztDirection for Inverse {
    const LABEL: &'static str = "apollo-czt-inverse";
    const ENTRY: &'static str = "czt_inverse";
}

pub(super) struct CztKernel<M>(PhantomData<M>);

impl<M> CztKernel<M> {
    const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<M: CztDirection> KernelInterface for CztKernel<M> {
    type Params = CztParams;

    const LABEL: &'static str = M::LABEL;
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<Complex32>(),
        BindingDecl::read_write::<Complex32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl<M: CztDirection> KernelSource<Wgsl> for CztKernel<M> {
    const ENTRY: &'static str = M::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(CZT_SOURCE)
    }
}

/// Zero-sized CZT kernel orchestration over a Hephaestus kernel device.
#[derive(Clone, Copy, Debug, Default)]
pub struct CztGpuKernel;

impl GpuTransformExecutor for CztGpuKernel {
    type Plan = ChirpPlan;
    type Sample = Complex32;
    type Bin = Complex32;

    fn input_len(plan: &ChirpPlan) -> usize {
        plan.input_len()
    }

    fn output_len(plan: &ChirpPlan) -> usize {
        plan.output_len()
    }

    fn validate(plan: &ChirpPlan) -> WgpuResult<()> {
        let input_len = plan.input_len();
        let output_len = plan.output_len();
        if output_len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "CZT lengths input={input_len}, output={output_len} must be greater than zero"
                ),
            });
        }
        let a_norm = plan.a().norm();
        let w_norm = plan.w().norm();
        if !a_norm.is_finite() || !w_norm.is_finite() || a_norm == 0.0 || w_norm == 0.0 {
            return Err(WgpuError::InvalidPlan {
                message: "CZT spiral parameters must have finite non-zero magnitude".to_owned(),
            });
        }
        Ok(())
    }

    fn forward_into(
        device: &WgpuDevice,
        plan: &ChirpPlan,
        input: &[Complex32],
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        Self::execute_forward_into(device, plan, input, output)
    }

    fn inverse_into(
        device: &WgpuDevice,
        plan: &ChirpPlan,
        input: &[Complex32],
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        // The adjoint inverse is defined on square plans only.
        if plan.input_len() != plan.output_len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.input_len(),
                actual: plan.output_len(),
            });
        }
        Self::execute_inverse_into(device, plan, input, output)
    }
}

impl CztGpuKernel {
    /// Execute the direct forward CZT into caller-owned output storage.
    pub(super) fn execute_forward_into<D>(
        device: &D,
        plan: &ChirpPlan,
        input: &[Complex32],
        output: &mut [Complex32],
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        CztKernel<Forward>: KernelSource<D::Dialect>,
    {
        Self::execute::<D, Forward>(device, plan, input, output)
    }

    /// Execute the square-plan adjoint inverse CZT into caller-owned output storage.
    pub(super) fn execute_inverse_into<D>(
        device: &D,
        plan: &ChirpPlan,
        spectrum: &[Complex32],
        output: &mut [Complex32],
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        CztKernel<Inverse>: KernelSource<D::Dialect>,
    {
        Self::execute::<D, Inverse>(device, plan, spectrum, output)
    }

    fn execute<D, M>(
        device: &D,
        plan: &ChirpPlan,
        input: &[Complex32],
        output: &mut [Complex32],
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        M: CztDirection,
        CztKernel<M>: KernelSource<D::Dialect> + KernelInterface<Params = CztParams>,
    {
        let input_len = u32::try_from(plan.input_len()).map_err(|_| WgpuError::InvalidPlan {
            message: format!(
                "input length {} exceeds the provider parameter range",
                plan.input_len()
            ),
        })?;
        let output_len = u32::try_from(plan.output_len()).map_err(|_| WgpuError::InvalidPlan {
            message: format!(
                "output length {} exceeds the provider parameter range",
                plan.output_len()
            ),
        })?;
        let input_buffer = device.upload(input)?;
        let output_buffer = device.alloc_zeroed::<Complex32>(output.len())?;
        let kernel = CztKernel::<M>::new();
        let prepared = device.prepare(&kernel)?;
        let a = plan.a();
        let w = plan.w();
        let params = CztParams {
            input_len,
            output_len,
            a_re: a.re,
            a_im: a.im,
            w_re: w.re,
            w_im: w.im,
            padding: [0; 2],
        };
        let grid = DispatchGrid::covering_domain([output.len(), 1, 1], [WORKGROUP_SIZE, 1, 1])?;
        let mut stream = device.stream()?;
        stream.encode(
            &prepared,
            &[
                Binding::read(&input_buffer),
                Binding::read_write(&output_buffer),
            ],
            &params,
            grid,
        )?;
        stream.submit()?;
        device.download(&output_buffer, output)?;
        Ok(())
    }
}
