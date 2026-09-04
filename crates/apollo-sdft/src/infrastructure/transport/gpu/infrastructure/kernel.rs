//! Typed Hephaestus kernels for SDFT direct bins and complete-bin inversion.
//!
//! The forward descriptor evaluates `X[k] = sum_n x[n] exp(-2 pi i k n / N)`.
//! The inverse descriptor evaluates `x_hat[m] = (1/N) sum_k X[k] exp(2 pi i k
//! m / N)`. Root-of-unity orthogonality proves `x_hat = x` in exact arithmetic
//! when all `N` bins are supplied; the device boundary enforces that condition
//! before it selects the inverse descriptor.

use std::borrow::Cow;

use eunomia::Complex32;
use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    Binding, BindingDecl, CommandStream, DispatchGrid, KernelDevice, KernelInterface, KernelSource,
    Wgsl,
};

use apollo_fft::{GpuElement, GpuTransformExecutor, GpuTransformPlanner, WgpuError, WgpuResult};
use hephaestus_wgpu::WgpuDevice;

use crate::infrastructure::transport::gpu::WindowPlan;

const WORKGROUP_SIZE: usize = 64;
const FORWARD_SOURCE: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/forward.wgsl"),
);
const INVERSE_SOURCE: &str = concat!(
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/inverse.wgsl"),
);

/// Uniform parameters shared by both SDFT kernel descriptors.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct SdftParams {
    window_len: u32,
    bin_count: u32,
    padding: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<SdftParams>() == 16);

impl SdftParams {
    pub(crate) fn new(window_len: usize, bin_count: usize) -> WgpuResult<Self> {
        Ok(Self {
            window_len: u32::try_from(window_len).map_err(|_| WgpuError::InvalidPlan {
                message: format!(
                    "window length {window_len} exceeds the accelerator parameter range"
                ),
            })?,
            bin_count: u32::try_from(bin_count).map_err(|_| WgpuError::InvalidPlan {
                message: format!("bin count {bin_count} exceeds the accelerator parameter range"),
            })?,
            padding: [0; 2],
        })
    }
}

/// Zero-sized Hephaestus descriptor for real-window SDFT direct bins.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SdftForwardGpuKernel;

/// Zero-sized SDFT orchestration over the forward and inverse pass
/// kernels.
#[derive(Clone, Copy, Debug, Default)]
pub struct SdftGpuKernel;

impl GpuTransformPlanner for SdftGpuKernel {
    type Plan = WindowPlan;

    fn input_len(plan: &WindowPlan) -> usize {
        plan.window_len()
    }

    fn output_len(plan: &WindowPlan) -> usize {
        plan.bin_count()
    }

    fn validate(plan: &WindowPlan) -> WgpuResult<()> {
        if plan.window_len() == 0 || plan.bin_count() == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan window_len={}, bin_count={}: both values must be greater than zero",
                    plan.window_len(),
                    plan.bin_count()
                ),
            });
        }
        if plan.bin_count() > plan.window_len() {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan window_len={}, bin_count={}: bin count must not exceed window length",
                    plan.window_len(),
                    plan.bin_count()
                ),
            });
        }
        if u32::try_from(plan.window_len()).is_err() || u32::try_from(plan.bin_count()).is_err() {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan window_len={}, bin_count={}: values exceed the accelerator parameter range",
                    plan.window_len(),
                    plan.bin_count()
                ),
            });
        }
        Ok(())
    }
}

impl GpuTransformExecutor for SdftGpuKernel {
    type Sample = f32;
    type Bin = Complex32;

    fn forward_into(
        device: &WgpuDevice,
        _plan: &WindowPlan,
        input: &[f32],
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        SdftForwardGpuKernel::execute_into(device, input, output)
    }

    /// The inverse is defined only for `bin_count == window_len`; a
    /// partial spectrum is a projection, not an inverse of the original
    /// SDFT window. The pass kernel emits complex samples; the real
    /// component is the reconstructed signal.
    fn inverse_into(
        device: &WgpuDevice,
        plan: &WindowPlan,
        input: &[Complex32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        if plan.bin_count() != plan.window_len() {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "inverse SDFT requires a complete spectrum: bin_count {} must equal window_len {}",
                    plan.bin_count(),
                    plan.window_len()
                ),
            });
        }
        Complex32::with_output_scratch(output.len(), |computed| {
            SdftInverseGpuKernel::execute_into(device, input, computed)?;
            for (target, value) in output.iter_mut().zip(computed.iter().copied()) {
                *target = value.re;
            }
            Ok(())
        })
    }
}

impl KernelInterface for SdftForwardGpuKernel {
    type Params = SdftParams;

    const LABEL: &'static str = "apollo-sdft-direct-bins";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<Complex32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl KernelSource<Wgsl> for SdftForwardGpuKernel {
    const ENTRY: &'static str = "sdft_direct_bins";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(FORWARD_SOURCE)
    }
}

impl SdftForwardGpuKernel {
    /// Execute direct-bin initialization into caller-owned storage.
    pub(crate) fn execute_into<D>(
        device: &D,
        window: &[f32],
        output: &mut [Complex32],
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        Self: KernelSource<D::Dialect>,
    {
        let params = SdftParams::new(window.len(), output.len())?;
        let input_buffer = device.upload(window)?;
        let output_buffer = device.alloc_zeroed::<Complex32>(output.len())?;
        let prepared = device.prepare(&Self)?;
        let bindings = [
            Binding::read(&input_buffer),
            Binding::read_write(&output_buffer),
        ];
        let grid = DispatchGrid::covering_domain([output.len(), 1, 1], [WORKGROUP_SIZE, 1, 1])?;
        let mut stream = device.stream()?;
        stream.encode(&prepared, &bindings, &params, grid)?;
        stream.submit()?;
        device.download(&output_buffer, output)?;
        Ok(())
    }
}

/// Zero-sized Hephaestus descriptor for complete-spectrum SDFT inversion.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SdftInverseGpuKernel;

impl KernelInterface for SdftInverseGpuKernel {
    type Params = SdftParams;

    const LABEL: &'static str = "apollo-sdft-complete-bin-inverse";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<Complex32>(),
        BindingDecl::read_write::<Complex32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl KernelSource<Wgsl> for SdftInverseGpuKernel {
    const ENTRY: &'static str = "sdft_inverse_bins";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(INVERSE_SOURCE)
    }
}

impl SdftInverseGpuKernel {
    /// Execute complete-bin inversion into caller-owned complex sample storage.
    pub(crate) fn execute_into<D>(
        device: &D,
        bins: &[Complex32],
        output: &mut [Complex32],
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        Self: KernelSource<D::Dialect>,
    {
        let params = SdftParams::new(output.len(), bins.len())?;
        let input_buffer = device.upload(bins)?;
        let output_buffer = device.alloc_zeroed::<Complex32>(output.len())?;
        let prepared = device.prepare(&Self)?;
        let bindings = [
            Binding::read(&input_buffer),
            Binding::read_write(&output_buffer),
        ];
        let grid = DispatchGrid::covering_domain([output.len(), 1, 1], [WORKGROUP_SIZE, 1, 1])?;
        let mut stream = device.stream()?;
        stream.encode(&prepared, &bindings, &params, grid)?;
        stream.submit()?;
        device.download(&output_buffer, output)?;
        Ok(())
    }
}
