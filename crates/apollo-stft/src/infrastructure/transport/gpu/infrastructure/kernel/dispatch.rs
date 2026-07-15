//! Shared STFT typed-dispatch and geometry operations.

use hephaestus_core::{
    DispatchGrid, GroupedBinding, GroupedKernelDevice, GroupedKernelSource, Wgsl,
};
use hephaestus_wgpu::WgpuDevice;

use super::{
    ChirpBitReverseKernel, ChirpFftParams, ChirpForwardButterflyKernel,
    ChirpInverseButterflyKernel, ChirpScaleKernel, FFT_WORKGROUP, OLA_WORKGROUP,
};
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};

/// Build a dispatch grid covering independent FFT elements.
pub(crate) fn fft_grid(elements: usize) -> WgpuResult<DispatchGrid> {
    DispatchGrid::covering_domain([elements, 1, 1], [FFT_WORKGROUP, 1, 1]).map_err(Into::into)
}

/// Build a dispatch grid covering overlap-add output samples.
pub(crate) fn ola_grid(elements: usize) -> WgpuResult<DispatchGrid> {
    DispatchGrid::covering_domain([elements, 1, 1], [OLA_WORKGROUP, 1, 1]).map_err(Into::into)
}

/// Validate that a host dimension has an accelerator representation.
pub(crate) fn dimension(value: usize, name: &'static str) -> WgpuResult<u32> {
    u32::try_from(value).map_err(|_| WgpuError::InvalidPlan {
        message: format!("{name} exceeds accelerator u32 range: {value}"),
    })
}

/// Return the power-of-two convolution length required by Bluestein's identity.
pub(crate) fn chirp_padded_len(frame_len: usize) -> WgpuResult<usize> {
    let minimum = frame_len
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| WgpuError::InvalidPlan {
            message: "2 * frame_len - 1 overflows host address space".to_owned(),
        })?;
    minimum
        .checked_next_power_of_two()
        .ok_or_else(|| WgpuError::InvalidPlan {
            message: "Bluestein padded frame length overflows host address space".to_owned(),
        })
}

/// Construct the host chirp frequency kernel through Apollo's FFT owner.
pub(crate) fn chirp_frequency_kernel(frame_len: usize, chirp_len: usize) -> (Vec<f32>, Vec<f32>) {
    let mut samples = leto::Array1::<eunomia::Complex64>::zeros([chirp_len]);
    for index in 0..frame_len {
        let sample_position = index as f64;
        let phase = core::f64::consts::PI * sample_position * sample_position / frame_len as f64;
        let value = eunomia::Complex64::new(phase.cos(), -phase.sin());
        samples[index] = value;
        if index != 0 {
            samples[chirp_len - index] = value;
        }
    }
    let frequency = apollo_fft::fft_1d_complex(&samples);
    let real = frequency.iter().map(|value| value.re as f32).collect();
    let imaginary = frequency.iter().map(|value| value.im as f32).collect();
    (real, imaginary)
}

/// Prepare and dispatch one grouped typed kernel.
pub(crate) fn dispatch_grouped<K>(
    device: &WgpuDevice,
    kernel: &K,
    bindings: &[GroupedBinding<'_, WgpuDevice>],
    params: &K::Params,
    grid: DispatchGrid,
) -> WgpuResult<()>
where
    K: GroupedKernelSource<Wgsl>,
{
    let prepared = device.prepare_grouped(kernel)?;
    device.dispatch_grouped(&prepared, bindings, params, grid)?;
    Ok(())
}

/// Record the ordered radix-2 stages for one Bluestein convolution transform.
pub(crate) fn dispatch_chirp_radix(
    device: &WgpuDevice,
    bindings: &[GroupedBinding<'_, WgpuDevice>],
    frame_count: usize,
    chirp_len: usize,
    inverse: bool,
) -> WgpuResult<()> {
    let frame_elements =
        frame_count
            .checked_mul(chirp_len)
            .ok_or_else(|| WgpuError::InvalidPlan {
                message: "frame_count * chirp_len overflows host address space".to_owned(),
            })?;
    let base = ChirpFftParams {
        fft_len: dimension(chirp_len, "chirp_len")?,
        stage: 0,
        inverse_flag: u32::from(inverse),
        batch_count: dimension(frame_count, "frame_count")?,
    };
    dispatch_grouped(
        device,
        &ChirpBitReverseKernel::new(),
        bindings,
        &base,
        fft_grid(frame_elements)?,
    )?;
    for stage in 0..chirp_len.trailing_zeros() {
        let params = ChirpFftParams { stage, ..base };
        if inverse {
            dispatch_grouped(
                device,
                &ChirpInverseButterflyKernel::new(),
                bindings,
                &params,
                fft_grid(frame_elements / 2)?,
            )?;
        } else {
            dispatch_grouped(
                device,
                &ChirpForwardButterflyKernel::new(),
                bindings,
                &params,
                fft_grid(frame_elements / 2)?,
            )?;
        }
    }
    if inverse {
        dispatch_grouped(
            device,
            &ChirpScaleKernel::new(),
            bindings,
            &base,
            fft_grid(frame_elements)?,
        )?;
    }
    Ok(())
}
