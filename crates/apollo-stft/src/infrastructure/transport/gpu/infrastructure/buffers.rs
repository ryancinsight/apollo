//! Reusable typed Hephaestus storage for repeated radix-2 STFT dispatches.

use hephaestus_core::ComputeDevice;
use hephaestus_wgpu::{WgpuBuffer, WgpuDevice};

use super::kernel::ComplexPod;
use crate::infrastructure::transport::gpu::domain::error::WgpuResult;

/// Storage retained across repeated STFT executions with one fixed geometry.
///
/// The buffers are provider-owned device allocations.  Parameter blocks and
/// command streams remain per-dispatch because their values encode the
/// operation stage, while the signal, spectrum, scratch, and output storage
/// remains resident across invocations.
#[derive(Debug)]
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
    pub(crate) inverse_host: Vec<f32>,
}

impl StftGpuBuffers {
    /// Allocate provider-owned storage for a fixed STFT geometry.
    pub(crate) fn new(
        device: &WgpuDevice,
        frame_count: usize,
        frame_len: usize,
        signal_len: usize,
        hop_len: usize,
    ) -> WgpuResult<Self> {
        let frame_elements = frame_count.checked_mul(frame_len).ok_or_else(|| {
            crate::infrastructure::transport::gpu::domain::error::WgpuError::InvalidPlan {
                message: "frame_count * frame_len overflows host address space".to_owned(),
            }
        })?;
        let spectrum_elements = frame_elements.checked_mul(2).ok_or_else(|| {
            crate::infrastructure::transport::gpu::domain::error::WgpuError::InvalidPlan {
                message: "interleaved spectrum length overflows host address space".to_owned(),
            }
        })?;
        Ok(Self {
            frame_count,
            frame_len,
            signal_len,
            hop_len,
            signal: device.alloc_zeroed(signal_len)?,
            spectrum: device.alloc_zeroed(spectrum_elements)?,
            real_scratch: device.alloc_zeroed(frame_elements)?,
            imaginary_scratch: device.alloc_zeroed(frame_elements)?,
            forward_output: device.alloc_zeroed(frame_elements)?,
            frame_data: device.alloc_zeroed(frame_elements)?,
            reconstructed: device.alloc_zeroed(signal_len)?,
            forward_host: vec![ComplexPod { re: 0.0, im: 0.0 }; frame_elements],
            inverse_host: vec![0.0; signal_len],
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
