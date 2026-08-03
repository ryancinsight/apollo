use apollo_fft::{GpuElement, GpuStorage, PrecisionProfile};

use super::infrastructure::kernel::GftDirection;
use super::surface::BasisTransform;
use super::{GftGpuKernel, GftWgpuBackend, GftWgpuPlan, WgpuError, WgpuResult};

impl BasisTransform for GftWgpuBackend {
    fn execute_forward(
        &self,
        plan: &GftWgpuPlan,
        signal: &[f32],
        basis: &[f32],
    ) -> WgpuResult<Vec<f32>> {
        let mut output = vec![0.0_f32; plan.len()];
        self.execute_forward_into(plan, signal, basis, &mut output)?;
        Ok(output)
    }

    fn execute_forward_into(
        &self,
        plan: &GftWgpuPlan,
        signal: &[f32],
        basis: &[f32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        validate_plan_input(plan, signal.len(), basis.len())?;
        validate_output(plan, output.len())?;
        GftGpuKernel::execute_into(self.device(), signal, basis, output, GftDirection::Forward)
    }

    fn execute_forward_leto(
        &self,
        plan: &GftWgpuPlan,
        signal: leto::ArrayView1<'_, f32>,
        basis: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let signal = apollo_leto_interop::view_cow(&signal);
        let basis = apollo_leto_interop::view_cow(&basis);
        let mut output =
            leto::Array::<f32, leto::MnemosyneStorage<f32>, 1>::zeros_mnemosyne([plan.len()]);
        self.execute_forward_into(
            plan,
            &signal,
            &basis,
            output
                .as_slice_mut()
                .expect("GFT Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    fn execute_inverse(
        &self,
        plan: &GftWgpuPlan,
        spectrum: &[f32],
        basis: &[f32],
    ) -> WgpuResult<Vec<f32>> {
        let mut output = vec![0.0_f32; plan.len()];
        self.execute_inverse_into(plan, spectrum, basis, &mut output)?;
        Ok(output)
    }

    fn execute_inverse_into(
        &self,
        plan: &GftWgpuPlan,
        spectrum: &[f32],
        basis: &[f32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        validate_plan_input(plan, spectrum.len(), basis.len())?;
        validate_output(plan, output.len())?;
        GftGpuKernel::execute_into(
            self.device(),
            spectrum,
            basis,
            output,
            GftDirection::Inverse,
        )
    }

    fn execute_inverse_leto(
        &self,
        plan: &GftWgpuPlan,
        spectrum: leto::ArrayView1<'_, f32>,
        basis: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let spectrum = apollo_leto_interop::view_cow(&spectrum);
        let basis = apollo_leto_interop::view_cow(&basis);
        let mut output =
            leto::Array::<f32, leto::MnemosyneStorage<f32>, 1>::zeros_mnemosyne([plan.len()]);
        self.execute_inverse_into(
            plan,
            &spectrum,
            &basis,
            output
                .as_slice_mut()
                .expect("GFT Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    fn execute_forward_typed_into<T: GpuStorage>(
        &self,
        plan: &GftWgpuPlan,
        precision: PrecisionProfile,
        signal: &[T],
        basis: &[f32],
        output: &mut [T],
    ) -> WgpuResult<()> {
        validate_typed_precision::<T>(precision)?;
        validate_plan_input(plan, signal.len(), basis.len())?;
        validate_output(plan, output.len())?;
        execute_typed_into(self, signal, basis, output, GftDirection::Forward)
    }

    fn execute_forward_leto_typed<T: GpuStorage + Default>(
        &self,
        plan: &GftWgpuPlan,
        precision: PrecisionProfile,
        signal: leto::ArrayView1<'_, T>,
        basis: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let signal = apollo_leto_interop::view_cow(&signal);
        let basis = apollo_leto_interop::view_cow(&basis);
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.len()]);
        self.execute_forward_typed_into(
            plan,
            precision,
            &signal,
            &basis,
            output
                .as_slice_mut()
                .expect("typed GFT Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    fn execute_inverse_typed_into<T: GpuStorage>(
        &self,
        plan: &GftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: &[T],
        basis: &[f32],
        output: &mut [T],
    ) -> WgpuResult<()> {
        validate_typed_precision::<T>(precision)?;
        validate_plan_input(plan, spectrum.len(), basis.len())?;
        validate_output(plan, output.len())?;
        execute_typed_into(self, spectrum, basis, output, GftDirection::Inverse)
    }

    fn execute_inverse_leto_typed<T: GpuStorage + Default>(
        &self,
        plan: &GftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: leto::ArrayView1<'_, T>,
        basis: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let spectrum = apollo_leto_interop::view_cow(&spectrum);
        let basis = apollo_leto_interop::view_cow(&basis);
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.len()]);
        self.execute_inverse_typed_into(
            plan,
            precision,
            &spectrum,
            &basis,
            output
                .as_slice_mut()
                .expect("typed GFT Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }
}

fn execute_typed_into<T: GpuStorage>(
    backend: &GftWgpuBackend,
    input: &[T],
    basis: &[f32],
    output: &mut [T],
    direction: GftDirection,
) -> WgpuResult<()> {
    if let (Some(input), Some(output)) =
        (T::as_element_slice(input), T::as_element_slice_mut(output))
    {
        return GftGpuKernel::execute_into(backend.device(), input, basis, output, direction);
    }
    f32::with_scratch(input.len(), output.len(), |represented, computed| {
        for (slot, value) in represented.iter_mut().zip(input.iter().copied()) {
            *slot = value.to_gpu();
        }
        GftGpuKernel::execute_into(backend.device(), represented, basis, computed, direction)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_gpu(value);
        }
        Ok(())
    })
}

fn validate_typed_precision<T: GpuStorage>(precision: PrecisionProfile) -> WgpuResult<()> {
    let expected = T::PROFILE;
    if precision.storage != expected.storage || precision.compute != expected.compute {
        return Err(WgpuError::InvalidPrecisionProfile);
    }
    Ok(())
}

fn validate_plan_input(plan: &GftWgpuPlan, input_len: usize, basis_len: usize) -> WgpuResult<()> {
    let n = plan.len();
    if n == 0 {
        return Err(WgpuError::InvalidPlan {
            message: "length must be greater than zero".to_owned(),
        });
    }
    if input_len != n {
        return Err(WgpuError::LengthMismatch {
            expected: n,
            actual: input_len,
        });
    }
    validate_basis_len(n, basis_len)
}

fn validate_output(plan: &GftWgpuPlan, output_len: usize) -> WgpuResult<()> {
    if output_len != plan.len() {
        return Err(WgpuError::LengthMismatch {
            expected: plan.len(),
            actual: output_len,
        });
    }
    Ok(())
}

fn validate_basis_len(n: usize, actual: usize) -> WgpuResult<()> {
    let expected = n.checked_mul(n).ok_or_else(|| WgpuError::InvalidPlan {
        message: format!("basis element count overflows usize for graph order {n}"),
    })?;
    if actual != expected {
        return Err(WgpuError::ShapeMismatch {
            message: format!("expected {expected} elements for a {n}x{n} basis, got {actual}"),
        });
    }
    Ok(())
}
