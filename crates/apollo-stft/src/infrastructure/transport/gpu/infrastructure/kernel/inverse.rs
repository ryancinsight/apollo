use apollo_fft::{WgpuError, WgpuResult};
use eunomia::Complex32;
use hephaestus_core::{ComputeDevice, GroupedCommandStream, GroupedKernelDevice};
use hephaestus_wgpu::WgpuDevice;

use super::StftGpuKernel;
use crate::infrastructure::transport::gpu::infrastructure::buffers::StftGpuBuffers;

impl StftGpuKernel {
    /// Execute one inverse STFT using retained provider state.
    pub(crate) fn execute_inverse_with_buffers(
        device: &WgpuDevice,
        spectrum: &[Complex32],
        signal_len: usize,
        buffers: &mut StftGpuBuffers,
    ) -> WgpuResult<()> {
        if signal_len != buffers.signal_len() {
            return Err(WgpuError::LengthMismatch {
                expected: buffers.signal_len(),
                actual: signal_len,
            });
        }
        if spectrum.len() != buffers.spectrum_host.len() / 2 {
            return Err(WgpuError::LengthMismatch {
                expected: buffers.spectrum_host.len() / 2,
                actual: spectrum.len(),
            });
        }
        buffers.inverse_fft.validate_device(device)?;

        for (destination, value) in buffers
            .spectrum_host
            .chunks_exact_mut(2)
            .zip(spectrum.iter())
        {
            destination[0] = value.re;
            destination[1] = value.im;
        }
        device.write_buffer(&buffers.spectrum, &buffers.spectrum_host)?;

        let mut stream = device.grouped_stream()?;
        stream.encode_grouped_sequence("apollo-stft-inverse", |sequence| {
            buffers.inverse_deinterleave.encode_in_sequence(sequence)?;
            buffers.inverse_fft.encode_in_sequence(sequence)?;
            buffers.inverse_window.encode_in_sequence(sequence)?;
            buffers.overlap_add.encode_in_sequence(sequence)
        })?;
        stream.submit_grouped()?;
        device.download(&buffers.reconstructed, &mut buffers.inverse_host)?;
        Ok(())
    }
}
