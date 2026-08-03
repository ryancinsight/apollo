//! NTT GPU execution backends.

#![warn(missing_docs)]
//! WGPU backend boundary for Apollo NTT.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns the NTT kernels, their domain names, and
//! the residue-field surface. The transform is exact modular arithmetic
//! over `u64`/`u32` residues — outside the scaffold's floating-point
//! element families — so the marker implements only the planner contract
//! and the surface lives on [`ModularExecution`].

/// Infrastructure boundary for the NTT kernels.
pub mod infrastructure;
use super::residue::ResiduePlan;

impl ModularExecution for NttWgpuBackend {
    fn create_buffers(&self, plan: &NttWgpuPlan) -> WgpuResult<NttGpuBuffers> {
        let omega = plan.payload().validate_field()?;
        Kernel::create_buffers(plan.len(), plan.payload().modulus(), omega)
    }

    fn execute_forward(&self, plan: &NttWgpuPlan, input: &[u64]) -> WgpuResult<Vec<u64>> {
        execute_allocating(self, plan, input, NttMode::Forward)
    }

    fn execute_forward_with_buffers(
        &self,
        plan: &NttWgpuPlan,
        input: &[u64],
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()> {
        execute_with_buffers(self, plan, input, buffers, NttMode::Forward)
    }

    fn execute_inverse(&self, plan: &NttWgpuPlan, input: &[u64]) -> WgpuResult<Vec<u64>> {
        execute_allocating(self, plan, input, NttMode::Inverse)
    }

    fn execute_inverse_with_buffers(
        &self,
        plan: &NttWgpuPlan,
        input: &[u64],
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()> {
        execute_with_buffers(self, plan, input, buffers, NttMode::Inverse)
    }

    fn execute_forward_quantized_into(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        output: &mut [u32],
    ) -> WgpuResult<()> {
        execute_quantized_into(self, plan, input, output, NttMode::Forward)
    }

    fn execute_inverse_quantized_into(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        output: &mut [u32],
    ) -> WgpuResult<()> {
        execute_quantized_into(self, plan, input, output, NttMode::Inverse)
    }

    fn execute_forward_quantized_with_buffers(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()> {
        execute_quantized_with_buffers(self, plan, input, buffers, NttMode::Forward)
    }

    fn execute_inverse_quantized_with_buffers(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()> {
        execute_quantized_with_buffers(self, plan, input, buffers, NttMode::Inverse)
    }

    fn execute_forward_leto(
        &self,
        plan: &NttWgpuPlan,
        input: leto::ArrayView1<'_, u64>,
    ) -> WgpuResult<leto::Array<u64, leto::MnemosyneStorage<u64>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        self.execute_forward(plan, &input).and_then(|output| {
            apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(|| {
                WgpuError::InvalidPlan {
                    message: "failed to allocate Mnemosyne-backed Leto NTT output".to_owned(),
                }
            })
        })
    }

    fn execute_inverse_leto(
        &self,
        plan: &NttWgpuPlan,
        input: leto::ArrayView1<'_, u64>,
    ) -> WgpuResult<leto::Array<u64, leto::MnemosyneStorage<u64>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        self.execute_inverse(plan, &input).and_then(|output| {
            apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(|| {
                WgpuError::InvalidPlan {
                    message: "failed to allocate Mnemosyne-backed Leto NTT output".to_owned(),
                }
            })
        })
    }

    fn execute_forward_quantized_leto(
        &self,
        plan: &NttWgpuPlan,
        input: leto::ArrayView1<'_, u32>,
    ) -> WgpuResult<leto::Array<u32, leto::MnemosyneStorage<u32>, 1>> {
        execute_quantized_leto(self, plan, input, NttMode::Forward)
    }

    fn execute_inverse_quantized_leto(
        &self,
        plan: &NttWgpuPlan,
        input: leto::ArrayView1<'_, u32>,
    ) -> WgpuResult<leto::Array<u32, leto::MnemosyneStorage<u32>, 1>> {
        execute_quantized_leto(self, plan, input, NttMode::Inverse)
    }
}

/// Return the last readback in reusable host state.
#[must_use]
pub fn buffer_output(buffers: &NttGpuBuffers) -> &[u64] {
    buffers.output()
}

fn execute_allocating(
    backend: &NttWgpuBackend,
    plan: &NttWgpuPlan,
    input: &[u64],
    mode: NttMode,
) -> WgpuResult<Vec<u64>> {
    let mut buffers = backend.create_buffers(plan)?;
    execute_with_buffers(backend, plan, input, &mut buffers, mode)?;
    Ok(buffers.output().to_vec())
}

fn execute_with_buffers(
    backend: &NttWgpuBackend,
    plan: &NttWgpuPlan,
    input: &[u64],
    buffers: &mut NttGpuBuffers,
    mode: NttMode,
) -> WgpuResult<()> {
    validate_run(plan, input.len(), buffers)?;
    Kernel::execute_with_buffers(backend.device(), input, mode, buffers)
}

fn execute_quantized_into(
    backend: &NttWgpuBackend,
    plan: &NttWgpuPlan,
    input: &[u32],
    output: &mut [u32],
    mode: NttMode,
) -> WgpuResult<()> {
    if output.len() != plan.len() {
        return Err(WgpuError::LengthMismatch {
            expected: plan.len(),
            actual: output.len(),
        });
    }
    let mut buffers = backend.create_buffers(plan)?;
    execute_quantized_with_buffers(backend, plan, input, &mut buffers, mode)?;
    for (target, value) in output.iter_mut().zip(buffers.output().iter().copied()) {
        *target = value as u32;
    }
    Ok(())
}

fn execute_quantized_with_buffers(
    backend: &NttWgpuBackend,
    plan: &NttWgpuPlan,
    input: &[u32],
    buffers: &mut NttGpuBuffers,
    mode: NttMode,
) -> WgpuResult<()> {
    validate_run(plan, input.len(), buffers)?;
    Kernel::execute_quantized_with_buffers(backend.device(), input, mode, buffers)
}

fn execute_quantized_leto(
    backend: &NttWgpuBackend,
    plan: &NttWgpuPlan,
    input: leto::ArrayView1<'_, u32>,
    mode: NttMode,
) -> WgpuResult<leto::Array<u32, leto::MnemosyneStorage<u32>, 1>> {
    let input = apollo_leto_interop::view_cow(&input);
    let mut output = vec![0; plan.len()];
    execute_quantized_into(backend, plan, &input, &mut output, mode)?;
    apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(|| WgpuError::InvalidPlan {
        message: "failed to allocate Mnemosyne-backed Leto NTT output".to_owned(),
    })
}

fn validate_run(plan: &NttWgpuPlan, input_len: usize, buffers: &NttGpuBuffers) -> WgpuResult<()> {
    plan.payload().validate_field()?;
    if input_len != plan.len() {
        return Err(WgpuError::LengthMismatch {
            expected: plan.len(),
            actual: input_len,
        });
    }
    if buffers.len() != plan.len() {
        return Err(WgpuError::LengthMismatch {
            expected: plan.len(),
            actual: buffers.len(),
        });
    }
    Ok(())
}
