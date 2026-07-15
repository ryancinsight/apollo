//! Hephaestus execution kernels for the log-frequency Mellin transform.
//!
//! Forward execution records log-resampling followed by the spectrum pass.
//! Inverse execution records inverse spectrum followed by exponential
//! resampling. Ordered command streams preserve each materialized intermediate
//! while Hephaestus owns binding, pipeline, dispatch, and transfer mechanics.

use std::{borrow::Cow, marker::PhantomData};

use bytemuck::{Pod, Zeroable};
use eunomia::Complex32;
use hephaestus_core::{
    Binding, BindingDecl, CommandStream, DispatchGrid, KernelDevice, KernelInterface, KernelSource,
    Wgsl,
};

use crate::infrastructure::transport::gpu::application::plan::MellinWgpuPlan;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};

const WORKGROUP_SIZE: usize = 64;
const MELLIN_SOURCE: &str = include_str!("shaders/mellin.wgsl");

/// Uniform parameters used by the forward log-resample and spectrum passes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MellinParams {
    signal_len: u32,
    samples: u32,
    signal_min: f32,
    signal_max: f32,
    log_min: f32,
    log_max: f32,
    padding: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<MellinParams>() == 32);

impl MellinParams {
    fn forward(
        signal_len: usize,
        plan: &MellinWgpuPlan,
        signal_min: f32,
        signal_max: f32,
    ) -> WgpuResult<Self> {
        Ok(Self {
            signal_len: u32::try_from(signal_len).map_err(|_| WgpuError::InvalidPlan {
                message: format!(
                    "signal length {signal_len} exceeds the accelerator parameter range"
                ),
            })?,
            samples: u32::try_from(plan.samples()).map_err(|_| WgpuError::InvalidPlan {
                message: format!(
                    "Mellin sample count {} exceeds the accelerator parameter range",
                    plan.samples()
                ),
            })?,
            signal_min,
            signal_max,
            log_min: plan.min_scale().ln(),
            log_max: plan.max_scale().ln(),
            padding: [0; 2],
        })
    }
}

/// Uniform parameters used by the inverse spectrum and exponential-resample passes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct InverseMellinParams {
    samples: u32,
    out_len: u32,
    log_min: f32,
    log_max: f32,
    out_min: f32,
    out_max: f32,
    padding: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<InverseMellinParams>() == 32);

impl InverseMellinParams {
    fn new(plan: &MellinWgpuPlan, out_len: usize, out_min: f32, out_max: f32) -> WgpuResult<Self> {
        Ok(Self {
            samples: u32::try_from(plan.samples()).map_err(|_| WgpuError::InvalidPlan {
                message: format!(
                    "Mellin sample count {} exceeds the accelerator parameter range",
                    plan.samples()
                ),
            })?,
            out_len: u32::try_from(out_len).map_err(|_| WgpuError::InvalidPlan {
                message: format!("output length {out_len} exceeds the accelerator parameter range"),
            })?,
            log_min: plan.min_scale().ln(),
            log_max: plan.max_scale().ln(),
            out_min,
            out_max,
            padding: [0; 2],
        })
    }
}

/// Compile-time selection of one Mellin shader entry point and binding contract.
trait MellinPass {
    /// Provider diagnostic label for the entry point.
    const LABEL: &'static str;
    /// WGSL entry point name.
    const ENTRY: &'static str;
}

/// Typed Hephaestus interface for one Mellin shader pass.
pub(crate) struct MellinKernel<P, Input, Output, Params>(PhantomData<(P, Input, Output, Params)>);

impl<P: MellinPass, Input: Pod, Output: Pod, Params: Pod> KernelInterface
    for MellinKernel<P, Input, Output, Params>
{
    type Params = Params;

    const LABEL: &'static str = P::LABEL;
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<Input>(),
        BindingDecl::read_write::<Output>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl<P: MellinPass, Input: Pod, Output: Pod, Params: Pod> KernelSource<Wgsl>
    for MellinKernel<P, Input, Output, Params>
{
    const ENTRY: &'static str = P::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(MELLIN_SOURCE)
    }
}

/// Marker for linear-to-log resampling.
pub(crate) struct LogResample;

impl MellinPass for LogResample {
    const LABEL: &'static str = "apollo-mellin-log-resample";
    const ENTRY: &'static str = "mellin_resample";
}

/// Marker for the log-grid spectrum.
pub(crate) struct Spectrum;

impl MellinPass for Spectrum {
    const LABEL: &'static str = "apollo-mellin-spectrum";
    const ENTRY: &'static str = "mellin_spectrum";
}

/// Marker for inverse log-grid spectrum reconstruction.
pub(crate) struct InverseSpectrum;

impl MellinPass for InverseSpectrum {
    const LABEL: &'static str = "apollo-mellin-inverse-spectrum";
    const ENTRY: &'static str = "mellin_inverse_spectrum";
}

/// Marker for log-to-linear exponential resampling.
pub(crate) struct ExpResample;

impl MellinPass for ExpResample {
    const LABEL: &'static str = "apollo-mellin-exp-resample";
    const ENTRY: &'static str = "mellin_exp_resample";
}

/// Zero-sized Mellin orchestration over a Hephaestus device.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MellinGpuKernel;

impl MellinGpuKernel {
    /// Execute log-resampling and Mellin spectrum into caller-owned storage.
    pub(crate) fn execute_forward_into<D>(
        device: &D,
        plan: &MellinWgpuPlan,
        signal: &[f32],
        signal_min: f32,
        signal_max: f32,
        output: &mut [Complex32],
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        MellinKernel<LogResample, f32, f32, MellinParams>: KernelSource<D::Dialect>,
        MellinKernel<Spectrum, f32, Complex32, MellinParams>: KernelSource<D::Dialect>,
    {
        let params = MellinParams::forward(signal.len(), plan, signal_min, signal_max)?;
        let signal_buffer = device.upload(signal)?;
        let log_samples_buffer = device.alloc_zeroed::<f32>(plan.samples())?;
        let spectrum_buffer = device.alloc_zeroed::<Complex32>(output.len())?;
        let resample = device.prepare(&MellinKernel::<LogResample, f32, f32, MellinParams>(
            PhantomData,
        ))?;
        let spectrum = device.prepare(&MellinKernel::<Spectrum, f32, Complex32, MellinParams>(
            PhantomData,
        ))?;
        let resample_bindings = [
            Binding::read(&signal_buffer),
            Binding::read_write(&log_samples_buffer),
        ];
        let spectrum_bindings = [
            Binding::read(&log_samples_buffer),
            Binding::read_write(&spectrum_buffer),
        ];
        let grid = DispatchGrid::covering_domain([plan.samples(), 1, 1], [WORKGROUP_SIZE, 1, 1])?;
        let mut stream = device.stream()?;
        stream.encode(&resample, &resample_bindings, &params, grid)?;
        stream.encode(&spectrum, &spectrum_bindings, &params, grid)?;
        stream.submit()?;
        device.download(&spectrum_buffer, output)?;
        Ok(())
    }

    /// Execute inverse spectrum reconstruction and exponential resampling.
    pub(crate) fn execute_inverse_into<D>(
        device: &D,
        plan: &MellinWgpuPlan,
        spectrum: &[Complex32],
        out_min: f32,
        out_max: f32,
        output: &mut [f32],
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        MellinKernel<InverseSpectrum, Complex32, f32, InverseMellinParams>:
            KernelSource<D::Dialect>,
        MellinKernel<ExpResample, f32, f32, InverseMellinParams>: KernelSource<D::Dialect>,
    {
        let params = InverseMellinParams::new(plan, output.len(), out_min, out_max)?;
        let spectrum_buffer = device.upload(spectrum)?;
        let log_samples_buffer = device.alloc_zeroed::<f32>(plan.samples())?;
        let output_buffer = device.alloc_zeroed::<f32>(output.len())?;
        let inverse_spectrum = device.prepare(&MellinKernel::<
            InverseSpectrum,
            Complex32,
            f32,
            InverseMellinParams,
        >(PhantomData))?;
        let exp_resample =
            device.prepare(&MellinKernel::<ExpResample, f32, f32, InverseMellinParams>(
                PhantomData,
            ))?;
        let inverse_bindings = [
            Binding::read(&spectrum_buffer),
            Binding::read_write(&log_samples_buffer),
        ];
        let output_bindings = [
            Binding::read(&log_samples_buffer),
            Binding::read_write(&output_buffer),
        ];
        let inverse_grid =
            DispatchGrid::covering_domain([plan.samples(), 1, 1], [WORKGROUP_SIZE, 1, 1])?;
        let output_grid =
            DispatchGrid::covering_domain([output.len(), 1, 1], [WORKGROUP_SIZE, 1, 1])?;
        let mut stream = device.stream()?;
        stream.encode(&inverse_spectrum, &inverse_bindings, &params, inverse_grid)?;
        stream.encode(&exp_resample, &output_bindings, &params, output_grid)?;
        stream.submit()?;
        device.download(&output_buffer, output)?;
        Ok(())
    }
}
