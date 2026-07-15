use eunomia::Complex32;
use hephaestus_core::{
    Binding, ComputeDevice, DeviceBuffer, GroupedBinding, GroupedCommandStream,
    GroupedKernelDevice, KernelDevice,
};
use hephaestus_wgpu::{WgpuBuffer, WgpuDevice};

use super::{
    dimension, fft_grid, ola_grid, FftStageParams, InverseBitReverseKernel, InverseButterflyKernel,
    InverseDeinterleaveKernel, InverseScaleWindowKernel, OverlapAddKernel, StftGpuKernel,
    StftParams,
};
use crate::infrastructure::transport::gpu::{
    domain::error::{WgpuError, WgpuResult},
    infrastructure::buffers::StftGpuBuffers,
};

impl StftGpuKernel {
    /// Execute inverse STFT through the typed radix-2 or Bluestein pipeline.
    pub fn execute_inverse(
        device: &WgpuDevice,
        spectrum: &[Complex32],
        frame_len: usize,
        hop_len: usize,
        frame_count: usize,
        signal_len: usize,
    ) -> WgpuResult<Vec<f32>> {
        if !frame_len.is_power_of_two() {
            return Self::execute_inverse_chirp(
                device,
                spectrum,
                frame_len,
                hop_len,
                frame_count,
                signal_len,
            );
        }
        let elements =
            frame_count
                .checked_mul(frame_len)
                .ok_or_else(|| WgpuError::InvalidPlan {
                    message: "frame_count * frame_len overflows host address space".to_owned(),
                })?;
        let spectrum = interleave_spectrum(spectrum);
        let spectrum_buffer = device.upload(&spectrum)?;
        let real_scratch = device.alloc_zeroed(elements)?;
        let imaginary_scratch = device.alloc_zeroed(elements)?;
        let frame_data = device.alloc_zeroed(elements)?;
        let reconstructed = device.alloc_zeroed(signal_len)?;
        Self::execute_inverse_radix(
            device,
            &spectrum_buffer,
            &real_scratch,
            &imaginary_scratch,
            &frame_data,
            &reconstructed,
            frame_len,
            hop_len,
            frame_count,
            signal_len,
        )?;
        let mut output = vec![0.0; signal_len];
        device.download(&reconstructed, &mut output)?;
        Ok(output)
    }

    /// Execute a power-of-two inverse STFT using retained provider storage.
    pub fn execute_inverse_with_buffers(
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
        let spectrum = interleave_spectrum(spectrum);
        if spectrum.len() != buffers.spectrum.len() {
            return Err(WgpuError::LengthMismatch {
                expected: buffers.spectrum.len() / 2,
                actual: spectrum.len() / 2,
            });
        }
        device.write_buffer(&buffers.spectrum, &spectrum)?;
        Self::execute_inverse_radix(
            device,
            &buffers.spectrum,
            &buffers.real_scratch,
            &buffers.imaginary_scratch,
            &buffers.frame_data,
            &buffers.reconstructed,
            buffers.frame_len(),
            buffers.hop_len(),
            buffers.frame_count(),
            buffers.signal_len(),
        )?;
        device.download(&buffers.reconstructed, &mut buffers.inverse_host)?;
        Ok(())
    }

    fn execute_inverse_radix(
        device: &WgpuDevice,
        spectrum: &WgpuBuffer<f32>,
        real_scratch: &WgpuBuffer<f32>,
        imaginary_scratch: &WgpuBuffer<f32>,
        frame_data: &WgpuBuffer<f32>,
        reconstructed: &WgpuBuffer<f32>,
        frame_len: usize,
        hop_len: usize,
        frame_count: usize,
        signal_len: usize,
    ) -> WgpuResult<()> {
        let elements =
            frame_count
                .checked_mul(frame_len)
                .ok_or_else(|| WgpuError::InvalidPlan {
                    message: "frame_count * frame_len overflows host address space".to_owned(),
                })?;
        let params = FftStageParams {
            frame_count: dimension(frame_count, "frame_count")?,
            frame_len: dimension(frame_len, "frame_len")?,
            stage: 0,
            padding: 0,
        };
        let bindings = [
            GroupedBinding::read(0, 0, spectrum),
            GroupedBinding::read_write(0, 1, real_scratch),
            GroupedBinding::read_write(0, 2, imaginary_scratch),
            GroupedBinding::read_write(0, 3, frame_data),
        ];
        let deinterleave = device.prepare_grouped(&InverseDeinterleaveKernel::new())?;
        let bit_reverse = device.prepare_grouped(&InverseBitReverseKernel::new())?;
        let butterfly = device.prepare_grouped(&InverseButterflyKernel::new())?;
        let scale_window = device.prepare_grouped(&InverseScaleWindowKernel::new())?;
        let mut stream = device.grouped_stream()?;
        stream.encode_grouped(&deinterleave, &bindings, &params, fft_grid(elements)?)?;
        stream.encode_grouped(&bit_reverse, &bindings, &params, fft_grid(elements)?)?;
        for stage in 0..frame_len.trailing_zeros() {
            let stage_params = FftStageParams { stage, ..params };
            stream.encode_grouped(
                &butterfly,
                &bindings,
                &stage_params,
                fft_grid(elements / 2)?,
            )?;
        }
        stream.encode_grouped(&scale_window, &bindings, &params, fft_grid(elements)?)?;
        stream.submit_grouped()?;

        let overlap_add = device.prepare(&OverlapAddKernel::new())?;
        let ola_bindings = [
            Binding::read(frame_data),
            Binding::read_write(reconstructed),
        ];
        let ola_params = StftParams {
            signal_len: dimension(signal_len, "signal_len")?,
            frame_len: dimension(frame_len, "frame_len")?,
            hop_len: dimension(hop_len, "hop_len")?,
            frame_count: dimension(frame_count, "frame_count")?,
        };
        device.dispatch(
            &overlap_add,
            &ola_bindings,
            &ola_params,
            ola_grid(signal_len)?,
        )?;
        Ok(())
    }
}

fn interleave_spectrum(spectrum: &[Complex32]) -> Vec<f32> {
    spectrum
        .iter()
        .flat_map(|value| [value.re, value.im])
        .collect()
}
