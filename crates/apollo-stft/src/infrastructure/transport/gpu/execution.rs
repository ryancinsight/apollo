use apollo_fft::{Complex32, GpuElement, GpuStorage, PrecisionProfile};
use hephaestus_core::DeviceLimits;
use hephaestus_wgpu::WgpuDevice;

use super::infrastructure::buffers::StftGpuBuffers;
use super::infrastructure::kernel::StftGpuKernel as Kernel;
use super::{FramedExecution, StftWgpuBackend, StftWgpuPlan, WgpuError, WgpuResult};

/// Return the Hephaestus limits required by STFT dispatch.
#[must_use]
pub fn required_device_limits() -> DeviceLimits {
    WgpuDevice::default_device_limits()
}

/// Return the spectrum length the forward STFT produces for a signal.
///
/// # Errors
///
/// Returns an invalid-plan or input-too-short rejection.
pub fn forward_output_len(plan: &StftWgpuPlan, signal_len: usize) -> WgpuResult<usize> {
    let (_, output_len) = checked_frame_geometry(plan, signal_len)?;
    if signal_len < plan.payload().frame_len() {
        return Err(WgpuError::InputTooShort {
            min: plan.payload().frame_len(),
            actual: signal_len,
        });
    }
    Ok(output_len)
}

fn checked_frame_geometry(plan: &StftWgpuPlan, signal_len: usize) -> WgpuResult<(usize, usize)> {
    plan.payload().validate_geometry()?;
    let frame_count = signal_len
        .div_ceil(plan.payload().hop_len())
        .checked_add(1)
        .ok_or_else(|| WgpuError::InvalidPlan {
            message: "1 + ceil(signal_len / hop_len) overflows host address space".to_owned(),
        })?;
    let output_len = frame_count
        .checked_mul(plan.payload().frame_len())
        .ok_or_else(|| WgpuError::InvalidPlan {
            message: "frame_count * frame_len overflows host address space".to_owned(),
        })?;
    Ok((frame_count, output_len))
}

impl FramedExecution for StftWgpuBackend {
    fn execute_forward(&self, plan: &StftWgpuPlan, signal: &[f32]) -> WgpuResult<Vec<Complex32>> {
        plan.payload().validate_geometry()?;
        if signal.len() < plan.payload().frame_len() {
            return Err(WgpuError::InputTooShort {
                min: plan.payload().frame_len(),
                actual: signal.len(),
            });
        }
        let mut buffers = self.make_buffers(plan, signal.len())?;
        self.execute_forward_with_buffers(plan, signal, &mut buffers)?;
        Ok(buffers.fwd_output().to_vec())
    }

    fn execute_forward_leto(
        &self,
        plan: &StftWgpuPlan,
        signal: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let signal = apollo_leto_interop::view_cow(&signal);
        let output = self.execute_forward(plan, &signal)?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(|| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto STFT output".to_owned(),
        })
    }

    fn execute_forward_typed_into<I, O>(
        &self,
        plan: &StftWgpuPlan,
        precision: PrecisionProfile,
        signal: &[I],
        output: &mut [O],
    ) -> WgpuResult<()>
    where
        I: GpuStorage,
        O: GpuStorage<Complex32>,
    {
        validate_profiles(precision, I::PROFILE, O::PROFILE)?;
        let expected_output = forward_output_len(plan, signal.len())?;
        if output.len() != expected_output {
            return Err(WgpuError::LengthMismatch {
                expected: expected_output,
                actual: output.len(),
            });
        }
        let computed = if let Some(represented) = I::as_element_slice(signal) {
            self.execute_forward(plan, represented)?
        } else {
            f32::with_input_scratch(signal.len(), |represented| {
                for (slot, value) in represented.iter_mut().zip(signal.iter().copied()) {
                    *slot = value.to_gpu();
                }
                self.execute_forward(plan, represented)
            })?
        };
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = O::from_gpu(value);
        }
        Ok(())
    }

    fn execute_forward_leto_typed<I, O>(
        &self,
        plan: &StftWgpuPlan,
        precision: PrecisionProfile,
        signal: leto::ArrayView1<'_, I>,
    ) -> WgpuResult<leto::Array<O, leto::MnemosyneStorage<O>, 1>>
    where
        I: GpuStorage,
        O: GpuStorage<Complex32> + Default,
    {
        let signal = apollo_leto_interop::view_cow(&signal);
        let output_len = forward_output_len(plan, signal.len())?;
        let mut output =
            leto::Array::<O, leto::MnemosyneStorage<O>, 1>::zeros_mnemosyne([output_len]);
        let output_slice = output
            .as_slice_mut()
            .expect("Mnemosyne STFT typed output must be contiguous");
        self.execute_forward_typed_into(plan, precision, &signal, output_slice)?;
        Ok(output)
    }

    fn execute_inverse(
        &self,
        plan: &StftWgpuPlan,
        spectrum: &[Complex32],
        signal_len: usize,
    ) -> WgpuResult<Vec<f32>> {
        if signal_len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan frame_len={}, hop_len={}: signal_len must be non-zero",
                    plan.payload().frame_len(),
                    plan.payload().hop_len()
                ),
            });
        }
        let (_, expected) = checked_frame_geometry(plan, signal_len)?;
        if spectrum.len() != expected {
            return Err(WgpuError::LengthMismatch {
                expected,
                actual: spectrum.len(),
            });
        }
        let mut buffers = self.make_buffers(plan, signal_len)?;
        self.execute_inverse_with_buffers(plan, spectrum, signal_len, &mut buffers)?;
        Ok(buffers.inv_output().to_vec())
    }

    fn execute_inverse_leto(
        &self,
        plan: &StftWgpuPlan,
        spectrum: leto::ArrayView1<'_, Complex32>,
        signal_len: usize,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let spectrum = apollo_leto_interop::view_cow(&spectrum);
        let output = self.execute_inverse(plan, &spectrum, signal_len)?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(|| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto STFT output".to_owned(),
        })
    }

    fn execute_inverse_typed_into<I, O>(
        &self,
        plan: &StftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: &[I],
        signal_len: usize,
        output: &mut [O],
    ) -> WgpuResult<()>
    where
        I: GpuStorage<Complex32>,
        O: GpuStorage,
    {
        validate_profiles(precision, I::PROFILE, O::PROFILE)?;
        if output.len() != signal_len {
            return Err(WgpuError::LengthMismatch {
                expected: signal_len,
                actual: output.len(),
            });
        }
        let computed = if let Some(promoted) = I::as_element_slice(spectrum) {
            self.execute_inverse(plan, promoted, signal_len)?
        } else {
            Complex32::with_input_scratch(spectrum.len(), |promoted| {
                for (slot, value) in promoted.iter_mut().zip(spectrum.iter().copied()) {
                    *slot = value.to_gpu();
                }
                self.execute_inverse(plan, promoted, signal_len)
            })?
        };
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = O::from_gpu(value);
        }
        Ok(())
    }

    fn execute_inverse_leto_typed<I, O>(
        &self,
        plan: &StftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: leto::ArrayView1<'_, I>,
        signal_len: usize,
    ) -> WgpuResult<leto::Array<O, leto::MnemosyneStorage<O>, 1>>
    where
        I: GpuStorage<Complex32>,
        O: GpuStorage + Default,
    {
        let spectrum = apollo_leto_interop::view_cow(&spectrum);
        let mut output =
            leto::Array::<O, leto::MnemosyneStorage<O>, 1>::zeros_mnemosyne([signal_len]);
        let output_slice = output
            .as_slice_mut()
            .expect("Mnemosyne STFT typed inverse output must be contiguous");
        self.execute_inverse_typed_into(plan, precision, &spectrum, signal_len, output_slice)?;
        Ok(output)
    }

    fn make_buffers(&self, plan: &StftWgpuPlan, signal_len: usize) -> WgpuResult<StftGpuBuffers> {
        let (frame_count, _) = checked_frame_geometry(plan, signal_len)?;
        StftGpuBuffers::new(
            self.device(),
            frame_count,
            plan.payload().frame_len(),
            signal_len,
            plan.payload().hop_len(),
        )
    }

    fn execute_forward_with_buffers(
        &self,
        plan: &StftWgpuPlan,
        signal: &[f32],
        buffers: &mut StftGpuBuffers,
    ) -> WgpuResult<()> {
        let expected_output = forward_output_len(plan, signal.len())?;
        if plan.payload().frame_len() != buffers.frame_len()
            || plan.payload().hop_len() != buffers.hop_len()
            || signal.len() != buffers.signal_len()
            || expected_output != buffers.forward_host.len()
        {
            return Err(WgpuError::InvalidPlan {
                message: "buffer geometry does not match the forward STFT plan and signal"
                    .to_owned(),
            });
        }
        Kernel::execute_forward_with_buffers(self.device(), signal, buffers)
    }

    fn execute_inverse_with_buffers(
        &self,
        plan: &StftWgpuPlan,
        spectrum: &[Complex32],
        signal_len: usize,
        buffers: &mut StftGpuBuffers,
    ) -> WgpuResult<()> {
        let (frame_count, expected) = checked_frame_geometry(plan, signal_len)?;
        if plan.payload().frame_len() != buffers.frame_len()
            || plan.payload().hop_len() != buffers.hop_len()
            || signal_len != buffers.signal_len()
            || frame_count != buffers.frame_count()
            || spectrum.len() != expected
        {
            return Err(WgpuError::InvalidPlan {
                message: "buffer geometry does not match the inverse STFT plan and spectrum"
                    .to_owned(),
            });
        }
        Kernel::execute_inverse_with_buffers(self.device(), spectrum, signal_len, buffers)
    }
}

fn validate_profiles(
    precision: PrecisionProfile,
    input_profile: PrecisionProfile,
    output_profile: PrecisionProfile,
) -> WgpuResult<()> {
    for expected in [input_profile, output_profile] {
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
    }
    Ok(())
}
