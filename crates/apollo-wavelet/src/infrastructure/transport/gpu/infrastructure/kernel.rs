//! Hephaestus kernels for the multilevel orthonormal Haar DWT.
//!
//! Each Haar pass is orthonormal: its analysis matrix is the transpose of its
//! synthesis matrix. Thus the staged transform preserves energy and synthesis
//! in reverse level order reconstructs the input. Real-device differential,
//! Parseval, and roundtrip tests are the executable evidence tier.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{
    Binding, BindingDecl, CommandStream, DispatchGrid, KernelDevice, KernelInterface, KernelSource,
    Wgsl,
};

use apollo_fft::{GpuTransformExecutor, WgpuError, WgpuResult};
use hephaestus_wgpu::WgpuDevice;

const WORKGROUP_SIZE: usize = 256;
const WAVELET_SOURCE: &str = include_str!("shaders/wavelet.wgsl");

/// Uniform parameters matching WGSL `WaveletParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct WaveletParams {
    len: u32,
    _padding: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<WaveletParams>() == 16);

impl WaveletParams {
    fn new(len: usize) -> WgpuResult<Self> {
        Ok(Self {
            len: u32::try_from(len).map_err(|_| WgpuError::InvalidPlan {
                message: format!("transform length {len} exceeds the accelerator parameter range"),
            })?,
            _padding: [0; 3],
        })
    }
}

/// Typed Hephaestus interface for one Haar analysis pass.
pub(crate) struct HaarAnalysisKernel;

impl KernelInterface for HaarAnalysisKernel {
    type Params = WaveletParams;

    const LABEL: &'static str = "apollo-wavelet-haar-analysis";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl KernelSource<Wgsl> for HaarAnalysisKernel {
    const ENTRY: &'static str = "haar_analysis";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(WAVELET_SOURCE)
    }
}

/// Typed Hephaestus interface for one Haar synthesis pass.
pub(crate) struct HaarSynthesisKernel;

impl KernelInterface for HaarSynthesisKernel {
    type Params = WaveletParams;

    const LABEL: &'static str = "apollo-wavelet-haar-synthesis";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl KernelSource<Wgsl> for HaarSynthesisKernel {
    const ENTRY: &'static str = "haar_synthesis";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(WAVELET_SOURCE)
    }
}

/// Plan payload for the multilevel Haar DWT: signal length and
/// decomposition depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HaarDwtPlan {
    /// Signal length; must be a non-zero power of two.
    pub len: usize,
    /// Decomposition depth; must satisfy `0 < levels <= log2(len)`.
    pub levels: usize,
}

/// Zero-sized multilevel Haar orchestration over a Hephaestus device.
#[derive(Clone, Copy, Debug, Default)]
pub struct WaveletGpuKernel;

impl GpuTransformExecutor for WaveletGpuKernel {
    type Plan = HaarDwtPlan;

    fn input_len(plan: &HaarDwtPlan) -> usize {
        plan.len
    }

    fn validate(plan: &HaarDwtPlan) -> WgpuResult<()> {
        let HaarDwtPlan { len, levels } = *plan;
        if len == 0 || !len.is_power_of_two() {
            return Err(WgpuError::InvalidPlan {
                message: format!("length {len} must be a non-zero power of two"),
            });
        }
        if levels == 0 || levels >= usize::BITS as usize || (1usize << levels) > len {
            return Err(WgpuError::InvalidPlan {
                message: format!("levels {levels} must satisfy 0 < levels <= log2({len})"),
            });
        }
        Ok(())
    }

    fn execute_into(
        device: &WgpuDevice,
        plan: &HaarDwtPlan,
        input: &[f32],
        output: &mut [f32],
        inverse: bool,
    ) -> WgpuResult<()> {
        Self::execute_levels_into(device, input, output, plan.levels, inverse)
    }
}

impl WaveletGpuKernel {
    fn execute_levels_into<D>(
        device: &D,
        input: &[f32],
        output: &mut [f32],
        levels: usize,
        inverse: bool,
    ) -> WgpuResult<()>
    where
        D: KernelDevice<Dialect = Wgsl>,
        HaarAnalysisKernel: KernelSource<Wgsl>,
        HaarSynthesisKernel: KernelSource<Wgsl>,
    {
        let main = device.upload(input)?;
        let temporary = device.alloc_zeroed::<f32>(output.len())?;
        if inverse {
            for level in (0..levels).rev() {
                Self::execute_pass(
                    device,
                    &main,
                    &temporary,
                    input.len() >> level,
                    &HaarSynthesisKernel,
                )?;
            }
        } else {
            for level in 0..levels {
                Self::execute_pass(
                    device,
                    &main,
                    &temporary,
                    input.len() >> level,
                    &HaarAnalysisKernel,
                )?;
            }
        }
        device.download(&main, output)?;
        Ok(())
    }

    fn execute_pass<D, K>(
        device: &D,
        input: &D::Buffer<f32>,
        output: &D::Buffer<f32>,
        len: usize,
        kernel: &K,
    ) -> WgpuResult<()>
    where
        D: KernelDevice<Dialect = Wgsl>,
        K: KernelSource<Wgsl, Params = WaveletParams>,
    {
        let prepared = device.prepare(kernel)?;
        let bindings = [Binding::read(input), Binding::read_write(output)];
        let grid = DispatchGrid::covering_domain([len / 2, 1, 1], [WORKGROUP_SIZE, 1, 1])?;
        let mut stream = device.stream()?;
        stream.encode(&prepared, &bindings, &WaveletParams::new(len)?, grid)?;
        stream.copy_prefix(output, input, len)?;
        stream.submit()?;
        Ok(())
    }
}
