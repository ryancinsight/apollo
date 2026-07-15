use eunomia::Complex32;
use hephaestus_core::{ComputeDevice, GroupedBinding, GroupedCommandStream, GroupedKernelDevice};
use hephaestus_wgpu::{WgpuBuffer, WgpuDevice};

use super::{
    dimension, fft_grid, ComplexPod, ForwardBitReverseKernel, ForwardButterflyKernel,
    ForwardInterleaveKernel, ForwardPackKernel, FwdFftStageParams, StftGpuKernel,
};
use crate::infrastructure::transport::gpu::{
    domain::error::{WgpuError, WgpuResult},
    infrastructure::buffers::StftGpuBuffers,
};

impl StftGpuKernel {
    /// Execute the forward STFT through the typed radix-2 or Bluestein pipeline.
    pub fn execute_forward_fft(
        device: &WgpuDevice,
        signal: &[f32],
        frame_len: usize,
        hop_len: usize,
        frame_count: usize,
    ) -> WgpuResult<Vec<Complex32>> {
        if !frame_len.is_power_of_two() {
            return Self::execute_forward_fft_chirp(
                device,
                signal,
                frame_len,
                hop_len,
                frame_count,
            );
        }
        let frame_elements =
            frame_count
                .checked_mul(frame_len)
                .ok_or_else(|| WgpuError::InvalidPlan {
                    message: "frame_count * frame_len overflows host address space".to_owned(),
                })?;
        let signal_buffer = device.upload(signal)?;
        let real_scratch = device.alloc_zeroed(frame_elements)?;
        let imaginary_scratch = device.alloc_zeroed(frame_elements)?;
        let output = device.alloc_zeroed(frame_elements)?;
        Self::execute_forward_radix(
            device,
            &signal_buffer,
            &real_scratch,
            &imaginary_scratch,
            &output,
            frame_len,
            hop_len,
            frame_count,
        )?;
        let mut values = vec![ComplexPod { re: 0.0, im: 0.0 }; frame_elements];
        device.download(&output, &mut values)?;
        Ok(values
            .into_iter()
            .map(|value| Complex32::new(value.re, value.im))
            .collect())
    }

    /// Execute a power-of-two forward STFT using retained provider storage.
    pub fn execute_forward_fft_with_buffers(
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
        device.write_buffer(&buffers.signal, signal)?;
        Self::execute_forward_radix(
            device,
            &buffers.signal,
            &buffers.real_scratch,
            &buffers.imaginary_scratch,
            &buffers.forward_output,
            buffers.frame_len(),
            buffers.hop_len(),
            buffers.frame_count(),
        )?;
        device.download(&buffers.forward_output, &mut buffers.forward_host)?;
        Ok(())
    }

    fn execute_forward_radix(
        device: &WgpuDevice,
        signal: &WgpuBuffer<f32>,
        real_scratch: &WgpuBuffer<f32>,
        imaginary_scratch: &WgpuBuffer<f32>,
        output: &WgpuBuffer<ComplexPod>,
        frame_len: usize,
        hop_len: usize,
        frame_count: usize,
    ) -> WgpuResult<()> {
        let elements =
            frame_count
                .checked_mul(frame_len)
                .ok_or_else(|| WgpuError::InvalidPlan {
                    message: "frame_count * frame_len overflows host address space".to_owned(),
                })?;
        let base = FwdFftStageParams {
            frame_count: dimension(frame_count, "frame_count")?,
            frame_len: dimension(frame_len, "frame_len")?,
            hop_len: dimension(hop_len, "hop_len")?,
            stage: 0,
        };
        let bindings = [
            GroupedBinding::read(0, 0, signal),
            GroupedBinding::read_write(0, 1, real_scratch),
            GroupedBinding::read_write(0, 2, imaginary_scratch),
            GroupedBinding::read_write(0, 3, output),
        ];
        let pack = device.prepare_grouped(&ForwardPackKernel::new())?;
        let bit_reverse = device.prepare_grouped(&ForwardBitReverseKernel::new())?;
        let butterfly = device.prepare_grouped(&ForwardButterflyKernel::new())?;
        let interleave = device.prepare_grouped(&ForwardInterleaveKernel::new())?;
        let mut stream = device.grouped_stream()?;
        stream.encode_grouped(&pack, &bindings, &base, fft_grid(elements)?)?;
        stream.encode_grouped(&bit_reverse, &bindings, &base, fft_grid(elements)?)?;
        for stage in 0..frame_len.trailing_zeros() {
            let params = FwdFftStageParams { stage, ..base };
            stream.encode_grouped(&butterfly, &bindings, &params, fft_grid(elements / 2)?)?;
        }
        stream.encode_grouped(&interleave, &bindings, &base, fft_grid(elements)?)?;
        stream.submit_grouped()?;
        Ok(())
    }
}
