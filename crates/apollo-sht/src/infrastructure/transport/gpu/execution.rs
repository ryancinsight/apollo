use apollo_fft::{Complex32, GpuElement, GpuStorage, PrecisionProfile};
use eunomia::Complex64;
use leto::Array2;

use super::infrastructure::conversion::{
    array2_from_leto_view, coefficients_from_leto_view, coefficients_from_modes,
    exact_accelerator_component, grid_samples, mode_pairs, populate_modes,
    quantize_accelerator_component, validate_coefficient_shape, validate_forward,
    validate_sample_shape, validate_typed_input, validate_typed_output,
};
use super::infrastructure::kernel::ShtGpuKernel as Kernel;
use super::{ShtWgpuBackend, ShtWgpuPlan, WgpuError, WgpuResult};
use crate::SphericalHarmonicCoefficients;

/// Harmonic surface of the SHT backend.
///
/// The coefficient domain is `Complex64` while the GPU kernel is
/// `Complex32`: synthesis preserves caller values by rejecting
/// non-representable finite components with
/// [`WgpuError::PrecisionLoss`]; [`Self::quantize_coefficients`] is the
/// only lossy bridge, requested explicitly.
pub trait HarmonicExecution {
    /// Execute forward complex SHT by direct quadrature matrix summation.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, or provider failure.
    fn execute_forward(
        &self,
        plan: &ShtWgpuPlan,
        samples: &Array2<Complex32>,
    ) -> WgpuResult<SphericalHarmonicCoefficients>;

    /// Execute forward complex SHT from a Leto sample grid.
    ///
    /// The returned dense coefficient matrix has shape
    /// `(max_degree + 1, 2 * max_degree + 1)`. Strided sample views
    /// materialize once into logical order.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, or provider failure.
    fn execute_forward_leto(
        &self,
        plan: &ShtWgpuPlan,
        samples: leto::ArrayView2<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 2>>;

    /// Execute forward SHT from storage admitted by the concrete GPU
    /// contract.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, precision-profile, or
    /// provider failure.
    fn execute_forward_flat_typed<T: GpuStorage<Complex32>>(
        &self,
        plan: &ShtWgpuPlan,
        precision: PrecisionProfile,
        flat_samples: &[T],
    ) -> WgpuResult<SphericalHarmonicCoefficients>;

    /// Execute forward SHT from a flat typed Leto sample view.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, precision-profile, or
    /// provider failure.
    fn execute_forward_flat_leto_typed<T: GpuStorage<Complex32>>(
        &self,
        plan: &ShtWgpuPlan,
        precision: PrecisionProfile,
        flat_samples: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 2>>;

    /// Execute inverse complex SHT by direct synthesis matrix summation.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, precision-loss, or
    /// provider failure.
    fn execute_inverse(
        &self,
        plan: &ShtWgpuPlan,
        coefficients: &SphericalHarmonicCoefficients,
    ) -> WgpuResult<Array2<Complex64>>;

    /// Execute inverse complex SHT from a dense Leto coefficient matrix.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, precision-loss, or
    /// provider failure.
    fn execute_inverse_leto(
        &self,
        plan: &ShtWgpuPlan,
        coefficients: leto::ArrayView2<'_, Complex64>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 2>>;

    /// Execute inverse SHT and write the flat output to admitted typed
    /// storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, precision-profile,
    /// precision-loss, or provider failure.
    fn execute_inverse_flat_typed_into<T: GpuStorage<Complex32>>(
        &self,
        plan: &ShtWgpuPlan,
        precision: PrecisionProfile,
        coefficients: &SphericalHarmonicCoefficients,
        output: &mut [T],
    ) -> WgpuResult<()>;

    /// Execute inverse SHT from Leto coefficients into typed
    /// Mnemosyne-backed Leto storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, precision-profile,
    /// precision-loss, or provider failure.
    fn execute_inverse_flat_leto_typed<T: GpuStorage<Complex32> + Default>(
        &self,
        plan: &ShtWgpuPlan,
        precision: PrecisionProfile,
        coefficients: leto::ArrayView2<'_, Complex64>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>>;

    /// Explicitly quantize CPU-owned coefficients for concrete
    /// accelerator execution.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, or precision-loss
    /// failure.
    fn quantize_coefficients(
        &self,
        plan: &ShtWgpuPlan,
        coefficients: &SphericalHarmonicCoefficients,
    ) -> WgpuResult<SphericalHarmonicCoefficients>;
}

impl HarmonicExecution for ShtWgpuBackend {
    fn execute_forward(
        &self,
        plan: &ShtWgpuPlan,
        samples: &Array2<Complex32>,
    ) -> WgpuResult<SphericalHarmonicCoefficients> {
        plan.validate()?;
        validate_sample_shape(plan.payload(), samples.shape())?;
        let input = samples
            .as_slice()
            .expect("invariant: owned SHT input is contiguous");
        forward_accelerator(self, plan, input)
    }

    fn execute_forward_leto(
        &self,
        plan: &ShtWgpuPlan,
        samples: leto::ArrayView2<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 2>> {
        let samples = array2_from_leto_view(samples);
        let coefficients = self.execute_forward(plan, &samples)?;
        apollo_leto_interop::try_dense_from_array(coefficients.values()).ok_or_else(|| {
            WgpuError::InvalidPlan {
                message: "failed to allocate Mnemosyne-backed Leto SHT coefficients".to_owned(),
            }
        })
    }

    fn execute_forward_flat_typed<T: GpuStorage<Complex32>>(
        &self,
        plan: &ShtWgpuPlan,
        precision: PrecisionProfile,
        flat_samples: &[T],
    ) -> WgpuResult<SphericalHarmonicCoefficients> {
        plan.validate()?;
        validate_typed_input::<T>(plan.payload(), precision, flat_samples.len())?;
        if let Some(input) = T::as_element_slice(flat_samples) {
            return forward_accelerator(self, plan, input);
        }
        Complex32::with_input_scratch(flat_samples.len(), |represented| {
            for (target, value) in represented.iter_mut().zip(flat_samples.iter().copied()) {
                *target = value.to_gpu();
            }
            forward_accelerator(self, plan, represented)
        })
    }

    fn execute_forward_flat_leto_typed<T: GpuStorage<Complex32>>(
        &self,
        plan: &ShtWgpuPlan,
        precision: PrecisionProfile,
        flat_samples: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 2>> {
        let flat_samples = apollo_leto_interop::view_cow(&flat_samples);
        let coefficients = self.execute_forward_flat_typed(plan, precision, &flat_samples)?;
        apollo_leto_interop::try_dense_from_array(coefficients.values()).ok_or_else(|| {
            WgpuError::InvalidPlan {
                message: "failed to allocate Mnemosyne-backed Leto SHT typed coefficients"
                    .to_owned(),
            }
        })
    }

    fn execute_inverse(
        &self,
        plan: &ShtWgpuPlan,
        coefficients: &SphericalHarmonicCoefficients,
    ) -> WgpuResult<Array2<Complex64>> {
        plan.validate()?;
        validate_coefficient_shape(plan.payload(), coefficients)?;
        Complex32::with_input_scratch(plan.output_len(), |represented| {
            populate_modes(represented, coefficients)?;
            Complex32::with_output_scratch(plan.len(), |samples| {
                samples.fill(Complex32::new(0.0, 0.0));
                let grid = grid_samples(plan.payload())?;
                Kernel::execute_inverse_into(
                    self.device(),
                    plan.payload(),
                    represented,
                    &grid,
                    samples,
                )?;
                let mut output = Array2::from_elem(
                    [plan.payload().latitudes(), plan.payload().longitudes()],
                    Complex64::new(0.0, 0.0),
                );
                for (target, value) in output.iter_mut().zip(samples.iter().copied()) {
                    *target = Complex64::new(f64::from(value.re), f64::from(value.im));
                }
                Ok(output)
            })
        })
    }

    fn execute_inverse_leto(
        &self,
        plan: &ShtWgpuPlan,
        coefficients: leto::ArrayView2<'_, Complex64>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 2>> {
        let coefficients = coefficients_from_leto_view(plan.payload(), coefficients)?;
        let samples = self.execute_inverse(plan, &coefficients)?;
        apollo_leto_interop::try_dense_from_array(&samples).ok_or_else(|| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto SHT inverse samples".to_owned(),
        })
    }

    fn execute_inverse_flat_typed_into<T: GpuStorage<Complex32>>(
        &self,
        plan: &ShtWgpuPlan,
        precision: PrecisionProfile,
        coefficients: &SphericalHarmonicCoefficients,
        output: &mut [T],
    ) -> WgpuResult<()> {
        plan.validate()?;
        validate_typed_output::<T>(plan.payload(), precision, output.len())?;
        let samples = self.execute_inverse(plan, coefficients)?;
        for (target, value) in output.iter_mut().zip(samples.iter().copied()) {
            *target = T::from_gpu(Complex32::new(
                exact_accelerator_component(value.re, "real")?,
                exact_accelerator_component(value.im, "imaginary")?,
            ));
        }
        Ok(())
    }

    fn execute_inverse_flat_leto_typed<T: GpuStorage<Complex32> + Default>(
        &self,
        plan: &ShtWgpuPlan,
        precision: PrecisionProfile,
        coefficients: leto::ArrayView2<'_, Complex64>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let coefficients = coefficients_from_leto_view(plan.payload(), coefficients)?;
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.len()]);
        let output_slice = output
            .as_slice_mut()
            .expect("invariant: Mnemosyne-backed SHT output is contiguous");
        self.execute_inverse_flat_typed_into(plan, precision, &coefficients, output_slice)?;
        Ok(output)
    }

    fn quantize_coefficients(
        &self,
        plan: &ShtWgpuPlan,
        coefficients: &SphericalHarmonicCoefficients,
    ) -> WgpuResult<SphericalHarmonicCoefficients> {
        plan.validate()?;
        validate_coefficient_shape(plan.payload(), coefficients)?;
        let mut quantized = SphericalHarmonicCoefficients::zeros(plan.payload().max_degree());
        for (degree, order) in mode_pairs(plan.payload().max_degree()) {
            let value = coefficients.get(degree, order);
            quantized.set(
                degree,
                order,
                Complex64::new(
                    f64::from(quantize_accelerator_component(value.re, "real")?),
                    f64::from(quantize_accelerator_component(value.im, "imaginary")?),
                ),
            );
        }
        Ok(quantized)
    }
}

fn forward_accelerator(
    backend: &ShtWgpuBackend,
    plan: &ShtWgpuPlan,
    input: &[Complex32],
) -> WgpuResult<SphericalHarmonicCoefficients> {
    validate_forward(plan.payload(), input.len())?;
    let grid = grid_samples(plan.payload())?;
    Complex32::with_output_scratch(plan.output_len(), |output| {
        output.fill(Complex32::new(0.0, 0.0));
        Kernel::execute_forward_into(backend.device(), plan.payload(), input, &grid, output)?;
        Ok(coefficients_from_modes(plan.payload().max_degree(), output))
    })
}
