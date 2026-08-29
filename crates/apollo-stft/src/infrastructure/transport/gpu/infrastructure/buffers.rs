//! Reusable typed Hephaestus storage and prepared STFT dispatch state.

use apollo_fft::{WgpuError, WgpuResult};
use hephaestus_core::{
    ComputeDevice, FftDirection, FftOperands, FftOps, GroupedBinding, GroupedKernelDevice,
    StridedView,
};
use hephaestus_wgpu::{
    WgpuBoundGroupedDispatch, WgpuBuffer, WgpuDevice, WgpuFftOps, WgpuPreparedFft,
};
use leto::Layout;

use super::kernel::{
    dimension, fft_grid, ola_grid, ComplexPod, ForwardInterleaveKernel, ForwardPackKernel,
    InverseDeinterleaveKernel, InverseWindowKernel, OverlapAddKernel, StftParams,
};

/// Storage and prepared commands retained for one fixed STFT geometry.
///
/// Apollo owns framing, Hann windows, split/interleaved conversion, and
/// weighted overlap-add. The two selected-axis plans own all dense FFT and
/// Bluestein state in Hephaestus.
pub struct StftGpuBuffers {
    pub(crate) frame_count: usize,
    pub(crate) frame_len: usize,
    pub(crate) signal_len: usize,
    pub(crate) hop_len: usize,
    pub(crate) signal: WgpuBuffer<f32>,
    pub(crate) spectrum: WgpuBuffer<f32>,
    pub(crate) real_scratch: WgpuBuffer<f32>,
    pub(crate) imaginary_scratch: WgpuBuffer<f32>,
    pub(crate) forward_output: WgpuBuffer<ComplexPod>,
    pub(crate) frame_data: WgpuBuffer<f32>,
    pub(crate) reconstructed: WgpuBuffer<f32>,
    pub(crate) forward_host: Vec<ComplexPod>,
    pub(crate) spectrum_host: Vec<f32>,
    pub(crate) inverse_host: Vec<f32>,
    pub(crate) forward_fft: WgpuPreparedFft<2>,
    pub(crate) inverse_fft: WgpuPreparedFft<2>,
    pub(crate) forward_pack: WgpuBoundGroupedDispatch<ForwardPackKernel>,
    pub(crate) forward_interleave: WgpuBoundGroupedDispatch<ForwardInterleaveKernel>,
    pub(crate) inverse_deinterleave: WgpuBoundGroupedDispatch<InverseDeinterleaveKernel>,
    pub(crate) inverse_window: WgpuBoundGroupedDispatch<InverseWindowKernel>,
    pub(crate) overlap_add: WgpuBoundGroupedDispatch<OverlapAddKernel>,
}

impl core::fmt::Debug for StftGpuBuffers {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let Self {
            frame_count,
            frame_len,
            signal_len,
            hop_len,
            signal,
            spectrum,
            real_scratch,
            imaginary_scratch,
            forward_output,
            frame_data,
            reconstructed,
            forward_host,
            spectrum_host,
            inverse_host,
            forward_fft: _,
            inverse_fft: _,
            forward_pack: _,
            forward_interleave: _,
            inverse_deinterleave: _,
            inverse_window: _,
            overlap_add: _,
        } = self;
        formatter
            .debug_struct("StftGpuBuffers")
            .field("frame_count", frame_count)
            .field("frame_len", frame_len)
            .field("signal_len", signal_len)
            .field("hop_len", hop_len)
            .field("signal", signal)
            .field("spectrum", spectrum)
            .field("real_scratch", real_scratch)
            .field("imaginary_scratch", imaginary_scratch)
            .field("forward_output", forward_output)
            .field("frame_data", frame_data)
            .field("reconstructed", reconstructed)
            .field("forward_host_len", &forward_host.len())
            .field("spectrum_host_len", &spectrum_host.len())
            .field("inverse_host_len", &inverse_host.len())
            .field("prepared_fft_plans", &2)
            .field("bound_domain_dispatches", &5)
            .finish()
    }
}

impl StftGpuBuffers {
    /// Allocate and prepare all state for a fixed STFT geometry.
    pub(crate) fn new(
        device: &WgpuDevice,
        frame_count: usize,
        frame_len: usize,
        signal_len: usize,
        hop_len: usize,
    ) -> WgpuResult<Self> {
        let frame_elements =
            frame_count
                .checked_mul(frame_len)
                .ok_or_else(|| WgpuError::InvalidPlan {
                    message: "frame_count * frame_len overflows host address space".to_owned(),
                })?;
        let spectrum_elements =
            frame_elements
                .checked_mul(2)
                .ok_or_else(|| WgpuError::InvalidPlan {
                    message: "interleaved spectrum length overflows host address space".to_owned(),
                })?;
        let layout = Layout::c_contiguous([frame_count, frame_len]).map_err(|error| {
            WgpuError::InvalidPlan {
                message: format!("STFT frame-plane layout is invalid: {error}"),
            }
        })?;
        let params = StftParams {
            signal_len: dimension(signal_len, "signal_len")?,
            frame_len: dimension(frame_len, "frame_len")?,
            hop_len: dimension(hop_len, "hop_len")?,
            frame_count: dimension(frame_count, "frame_count")?,
        };

        let signal = device.alloc_zeroed(signal_len)?;
        let spectrum = device.alloc_zeroed(spectrum_elements)?;
        let real_scratch = device.alloc_zeroed(frame_elements)?;
        let imaginary_scratch = device.alloc_zeroed(frame_elements)?;
        let forward_output = device.alloc_zeroed(frame_elements)?;
        let frame_data = device.alloc_zeroed(frame_elements)?;
        let reconstructed = device.alloc_zeroed(signal_len)?;

        let operands = || FftOperands {
            real: StridedView::new(&real_scratch, &layout),
            imaginary: StridedView::new(&imaginary_scratch, &layout),
        };
        let fft = WgpuFftOps;
        let forward_fft = fft.prepare_fft_axes(device, operands(), FftDirection::Forward, &[1])?;
        let inverse_fft = fft.prepare_fft_axes(device, operands(), FftDirection::Inverse, &[1])?;

        let forward_bindings = [
            GroupedBinding::read(0, 0, &signal),
            GroupedBinding::read_write(0, 1, &real_scratch),
            GroupedBinding::read_write(0, 2, &imaginary_scratch),
            GroupedBinding::read_write(0, 3, &forward_output),
        ];
        let inverse_bindings = [
            GroupedBinding::read(0, 0, &spectrum),
            GroupedBinding::read_write(0, 1, &real_scratch),
            GroupedBinding::read_write(0, 2, &imaginary_scratch),
            GroupedBinding::read_write(0, 3, &frame_data),
        ];
        let overlap_bindings = [
            GroupedBinding::read(0, 0, &frame_data),
            GroupedBinding::read_write(0, 1, &reconstructed),
        ];
        let frame_grid = fft_grid(frame_elements)?;
        let forward_pack = device.bind_grouped_dispatch(
            &device.prepare_grouped(&ForwardPackKernel::new())?,
            &forward_bindings,
            &params,
            frame_grid,
        )?;
        let forward_interleave = device.bind_grouped_dispatch(
            &device.prepare_grouped(&ForwardInterleaveKernel::new())?,
            &forward_bindings,
            &params,
            frame_grid,
        )?;
        let inverse_deinterleave = device.bind_grouped_dispatch(
            &device.prepare_grouped(&InverseDeinterleaveKernel::new())?,
            &inverse_bindings,
            &params,
            frame_grid,
        )?;
        let inverse_window = device.bind_grouped_dispatch(
            &device.prepare_grouped(&InverseWindowKernel::new())?,
            &inverse_bindings,
            &params,
            frame_grid,
        )?;
        let overlap_add = device.bind_grouped_dispatch(
            &device.prepare_grouped(&OverlapAddKernel::new())?,
            &overlap_bindings,
            &params,
            ola_grid(signal_len)?,
        )?;

        Ok(Self {
            frame_count,
            frame_len,
            signal_len,
            hop_len,
            signal,
            spectrum,
            real_scratch,
            imaginary_scratch,
            forward_output,
            frame_data,
            reconstructed,
            forward_host: vec![ComplexPod { re: 0.0, im: 0.0 }; frame_elements],
            spectrum_host: vec![0.0; spectrum_elements],
            inverse_host: vec![0.0; signal_len],
            forward_fft,
            inverse_fft,
            forward_pack,
            forward_interleave,
            inverse_deinterleave,
            inverse_window,
            overlap_add,
        })
    }

    /// Return the frame count this storage represents.
    #[must_use]
    pub const fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Return the fixed frame length.
    #[must_use]
    pub const fn frame_len(&self) -> usize {
        self.frame_len
    }

    /// Return the fixed signal length.
    #[must_use]
    pub const fn signal_len(&self) -> usize {
        self.signal_len
    }

    /// Return the fixed hop length.
    #[must_use]
    pub const fn hop_len(&self) -> usize {
        self.hop_len
    }

    /// Return the forward transform from the most recent buffered execution.
    #[must_use]
    pub fn fwd_output(&self) -> &[eunomia::Complex32] {
        bytemuck::cast_slice(&self.forward_host)
    }

    /// Return the reconstructed signal from the most recent buffered execution.
    #[must_use]
    pub fn inv_output(&self) -> &[f32] {
        &self.inverse_host
    }
}
