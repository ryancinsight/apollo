use eunomia::Complex32;
use hephaestus_core::{Binding, ComputeDevice, GroupedBinding, KernelDevice};
use hephaestus_wgpu::WgpuDevice;

use super::{
    chirp_frequency_kernel, chirp_padded_len, dimension, dispatch_chirp_radix, dispatch_grouped,
    fft_grid, ola_grid, ChirpInversePointMultiplyKernel, ChirpInversePostmultiplyKernel,
    ChirpInversePremultiplyKernel, OverlapAddKernel, StftChirpParams, StftGpuKernel, StftParams,
};
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};

impl StftGpuKernel {
    /// Execute the inverse non-power-of-two STFT through Bluestein's identity.
    pub(crate) fn execute_inverse_chirp(
        device: &WgpuDevice,
        spectrum: &[Complex32],
        frame_len: usize,
        hop_len: usize,
        frame_count: usize,
        signal_len: usize,
    ) -> WgpuResult<Vec<f32>> {
        let chirp_len = chirp_padded_len(frame_len)?;
        let working_elements =
            frame_count
                .checked_mul(chirp_len)
                .ok_or_else(|| WgpuError::InvalidPlan {
                    message: "frame_count * chirp_len overflows host address space".to_owned(),
                })?;
        let frame_elements =
            frame_count
                .checked_mul(frame_len)
                .ok_or_else(|| WgpuError::InvalidPlan {
                    message: "frame_count * frame_len overflows host address space".to_owned(),
                })?;
        let (kernel_real, kernel_imaginary) = chirp_frequency_kernel(frame_len, chirp_len);
        let spectrum: Vec<f32> = spectrum
            .iter()
            .flat_map(|value| [value.re, value.im])
            .collect();
        let spectrum = device.upload(&spectrum)?;
        let chirp_real = device.alloc_zeroed::<f32>(working_elements)?;
        let chirp_imaginary = device.alloc_zeroed::<f32>(working_elements)?;
        let kernel_real = device.upload(&kernel_real)?;
        let kernel_imaginary = device.upload(&kernel_imaginary)?;
        let frame_data = device.alloc_zeroed::<f32>(frame_elements)?;
        let output = device.alloc_zeroed::<f32>(signal_len)?;
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
            GroupedBinding::read(2, 0, &spectrum),
            GroupedBinding::read_write(2, 1, &frame_data),
        ];
        dispatch_grouped(
            device,
            &ChirpInversePremultiplyKernel::new(),
            &bindings,
            &params,
            fft_grid(working_elements)?,
        )?;
        dispatch_chirp_radix(device, &bindings[..4], frame_count, chirp_len, false)?;
        dispatch_grouped(
            device,
            &ChirpInversePointMultiplyKernel::new(),
            &bindings,
            &params,
            fft_grid(working_elements)?,
        )?;
        dispatch_chirp_radix(device, &bindings[..4], frame_count, chirp_len, true)?;
        dispatch_grouped(
            device,
            &ChirpInversePostmultiplyKernel::new(),
            &bindings,
            &params,
            fft_grid(frame_elements)?,
        )?;
        let overlap_add = device.prepare(&OverlapAddKernel::new())?;
        let ola_bindings = [Binding::read(&frame_data), Binding::read_write(&output)];
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
        let mut values = vec![0.0; signal_len];
        device.download(&output, &mut values)?;
        Ok(values)
    }
}
