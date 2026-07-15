use eunomia::Complex32;
use hephaestus_core::{ComputeDevice, GroupedBinding};
use hephaestus_wgpu::WgpuDevice;

use super::{
    chirp_frequency_kernel, chirp_padded_len, dimension, dispatch_chirp_radix, dispatch_grouped,
    fft_grid, ChirpForwardPointMultiplyKernel, ChirpForwardPostmultiplyKernel,
    ChirpForwardPremultiplyKernel, ComplexPod, StftChirpParams, StftGpuKernel,
};
use crate::infrastructure::transport::gpu::domain::error::WgpuResult;

impl StftGpuKernel {
    /// Execute the forward non-power-of-two STFT through Bluestein's identity.
    pub(crate) fn execute_forward_fft_chirp(
        device: &WgpuDevice,
        signal: &[f32],
        frame_len: usize,
        hop_len: usize,
        frame_count: usize,
    ) -> WgpuResult<Vec<Complex32>> {
        let chirp_len = chirp_padded_len(frame_len)?;
        let working_elements = frame_count.checked_mul(chirp_len).ok_or_else(|| {
            crate::infrastructure::transport::gpu::domain::error::WgpuError::InvalidPlan {
                message: "frame_count * chirp_len overflows host address space".to_owned(),
            }
        })?;
        let output_elements = frame_count.checked_mul(frame_len).ok_or_else(|| {
            crate::infrastructure::transport::gpu::domain::error::WgpuError::InvalidPlan {
                message: "frame_count * frame_len overflows host address space".to_owned(),
            }
        })?;
        let (kernel_real, kernel_imaginary) = chirp_frequency_kernel(frame_len, chirp_len);
        let signal_len = signal.len();
        let signal = device.upload(signal)?;
        let chirp_real = device.alloc_zeroed::<f32>(working_elements)?;
        let chirp_imaginary = device.alloc_zeroed::<f32>(working_elements)?;
        let kernel_real = device.upload(&kernel_real)?;
        let kernel_imaginary = device.upload(&kernel_imaginary)?;
        let output = device.alloc_zeroed::<ComplexPod>(output_elements)?;
        let params = StftChirpParams {
            frame_count: dimension(frame_count, "frame_count")?,
            frame_len: dimension(frame_len, "frame_len")?,
            chirp_len: dimension(chirp_len, "chirp_len")?,
            hop_len: dimension(hop_len, "hop_len")?,
            signal_len: dimension(signal_len, "signal_len")?,
            padding: [0; 3],
        };
        let bindings = [
            GroupedBinding::read_write(0, 0, &chirp_real),
            GroupedBinding::read_write(0, 1, &chirp_imaginary),
            GroupedBinding::read(0, 2, &kernel_real),
            GroupedBinding::read(0, 3, &kernel_imaginary),
            GroupedBinding::read(2, 0, &signal),
            GroupedBinding::read_write(2, 1, &output),
        ];
        dispatch_grouped(
            device,
            &ChirpForwardPremultiplyKernel::new(),
            &bindings,
            &params,
            fft_grid(working_elements)?,
        )?;
        dispatch_chirp_radix(device, &bindings[..4], frame_count, chirp_len, false)?;
        dispatch_grouped(
            device,
            &ChirpForwardPointMultiplyKernel::new(),
            &bindings,
            &params,
            fft_grid(working_elements)?,
        )?;
        dispatch_chirp_radix(device, &bindings[..4], frame_count, chirp_len, true)?;
        dispatch_grouped(
            device,
            &ChirpForwardPostmultiplyKernel::new(),
            &bindings,
            &params,
            fft_grid(output_elements)?,
        )?;
        let mut values = vec![ComplexPod { re: 0.0, im: 0.0 }; output_elements];
        device.download(&output, &mut values)?;
        Ok(values
            .into_iter()
            .map(|value| Complex32::new(value.re, value.im))
            .collect())
    }
}
