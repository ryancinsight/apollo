use apollo_fft::{WgpuError, WgpuResult};
use hephaestus_core::{ComputeDevice, GroupedCommandStream, GroupedKernelDevice};
use hephaestus_wgpu::WgpuDevice;

use super::StftGpuKernel;
use crate::infrastructure::transport::gpu::infrastructure::buffers::StftGpuBuffers;

impl StftGpuKernel {
    /// Execute one forward STFT using retained provider state.
    pub(crate) fn execute_forward_with_buffers(
        device: &WgpuDevice,
        signal: &[f32],
        buffers: &mut StftGpuBuffers,
    ) -> WgpuResult<()> {
        if signal.len() != buffers.signal_len() {
            return Err(WgpuError::LengthMismatch {
                expected: buffers.signal_len(),
                actual: signal.len(),
            });
        }
        buffers.forward_fft.validate_device(device)?;

        device.write_buffer(&buffers.signal, signal)?;
        let mut stream = device.grouped_stream()?;
        stream.encode_grouped_sequence("apollo-stft-forward", |sequence| {
            buffers.forward_pack.encode_in_sequence(sequence)?;
            buffers.forward_fft.encode_in_sequence(sequence)?;
            buffers.forward_interleave.encode_in_sequence(sequence)
        })?;
        stream.submit_grouped()?;
        device.download(&buffers.forward_output, &mut buffers.forward_host)?;
        Ok(())
    }
}
