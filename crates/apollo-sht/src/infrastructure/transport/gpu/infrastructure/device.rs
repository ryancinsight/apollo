//! Hephaestus device acquisition and SHT host-boundary execution.
//!
//! The accelerator executes a concrete `Complex32` quadrature projection and
//! synthesis. Apollo preserves its `Complex64` coefficient SSOT by rejecting
//! inverse coefficients that cannot be represented exactly before provider
//! allocation or dispatch; callers that choose approximation cross the explicit
//! quantization boundary.

use apollo_fft::PrecisionProfile;
use eunomia::{Complex32, Complex64};
use hephaestus_wgpu::WgpuDevice;
use leto::Array2;
use mnemosyne::scratch::ScratchPool;

use crate::{
    infrastructure::transport::gpu::{
        application::plan::ShtWgpuPlan,
        domain::{
            capabilities::WgpuCapabilities,
            error::{WgpuError, WgpuResult},
            storage::ShtGpuStorage,
        },
        infrastructure::{
            conversion::{
                array2_from_leto_view, coefficients_from_leto_view, coefficients_from_modes,
                exact_accelerator_component, grid_samples, mode_pairs, populate_modes,
                quantize_accelerator_component, validate_coefficient_shape, validate_forward,
                validate_sample_shape, validate_typed_input, validate_typed_output,
            },
            kernel::ShtGpuKernel,
        },
    },
    SphericalHarmonicCoefficients,
};

thread_local! {
    static COMPLEX_SCRATCH: ScratchPool<Complex32> = const { ScratchPool::new() };
}

/// Hephaestus WGPU backend for direct spherical harmonic execution.
#[derive(Debug, Clone)]
pub struct ShtWgpuBackend {
    device: WgpuDevice,
}

impl ShtWgpuBackend {
    /// Create a backend from an acquired Hephaestus WGPU device.
    #[must_use]
    pub const fn new(device: WgpuDevice) -> Self {
        Self { device }
    }

    /// Acquire the default Hephaestus WGPU device.
    pub fn try_default() -> WgpuResult<Self> {
        Ok(Self::new(WgpuDevice::try_default("apollo-sht-wgpu")?))
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::direct_complex(true)
    }

    /// Return the acquired Hephaestus WGPU device implementation.
    #[must_use]
    pub const fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Create a metadata-only SHT plan descriptor.
    #[must_use]
    pub const fn plan(
        &self,
        latitudes: usize,
        longitudes: usize,
        max_degree: usize,
    ) -> ShtWgpuPlan {
        ShtWgpuPlan::new(latitudes, longitudes, max_degree)
    }

    /// Execute forward complex SHT by direct quadrature matrix summation.
    pub fn execute_forward(
        &self,
        plan: &ShtWgpuPlan,
        samples: &Array2<Complex32>,
    ) -> WgpuResult<SphericalHarmonicCoefficients> {
        validate_sample_shape(plan, samples.shape())?;
        let input = samples
            .as_slice()
            .expect("invariant: owned SHT input is contiguous");
        self.execute_forward_accelerator(plan, input)
    }

    /// Execute forward complex SHT from a Leto sample grid.
    ///
    /// The returned dense coefficient matrix has shape
    /// `(max_degree + 1, 2 * max_degree + 1)` and uses Mnemosyne-backed Leto
    /// storage. Strided sample views materialize once into logical order.
    pub fn execute_forward_leto(
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

    /// Execute inverse complex SHT by direct synthesis matrix summation.
    ///
    /// The coefficient domain is `Complex64`, while the GPU kernel is
    /// `Complex32`. This method preserves caller values: non-representable
    /// finite components return [`WgpuError::PrecisionLoss`] before provider
    /// allocation or dispatch. Use [`Self::quantize_coefficients`] to request
    /// the lossy boundary explicitly.
    pub fn execute_inverse(
        &self,
        plan: &ShtWgpuPlan,
        coefficients: &SphericalHarmonicCoefficients,
    ) -> WgpuResult<Array2<Complex64>> {
        validate_coefficient_shape(plan, coefficients)?;
        COMPLEX_SCRATCH.with(|pool| {
            pool.with_scratch(plan.mode_count(), |represented| {
                populate_modes(represented, coefficients)?;
                pool.with_scratch(plan.sample_count(), |samples| {
                    samples.fill(Complex32::new(0.0, 0.0));
                    let grid = grid_samples(plan)?;
                    ShtGpuKernel::execute_inverse_into(
                        &self.device,
                        plan,
                        represented,
                        &grid,
                        samples,
                    )?;
                    let mut output = Array2::from_elem(
                        [plan.latitudes(), plan.longitudes()],
                        Complex64::new(0.0, 0.0),
                    );
                    for (target, value) in output.iter_mut().zip(samples.iter().copied()) {
                        *target = Complex64::new(f64::from(value.re), f64::from(value.im));
                    }
                    Ok(output)
                })
            })
        })
    }

    /// Explicitly quantize CPU-owned coefficients for concrete accelerator execution.
    ///
    /// This is the only lossy bridge from the `Complex64` coefficient domain
    /// to `Complex32`. Passing the original values to [`Self::execute_inverse`]
    /// rejects a precision-changing conversion instead.
    pub fn quantize_coefficients(
        &self,
        plan: &ShtWgpuPlan,
        coefficients: &SphericalHarmonicCoefficients,
    ) -> WgpuResult<SphericalHarmonicCoefficients> {
        validate_coefficient_shape(plan, coefficients)?;
        let mut quantized = SphericalHarmonicCoefficients::zeros(plan.max_degree());
        for (degree, order) in mode_pairs(plan.max_degree()) {
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

    /// Execute inverse complex SHT from a dense Leto coefficient matrix.
    pub fn execute_inverse_leto(
        &self,
        plan: &ShtWgpuPlan,
        coefficients: leto::ArrayView2<'_, Complex64>,
    ) -> WgpuResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 2>> {
        let coefficients = coefficients_from_leto_view(plan, coefficients)?;
        let samples = self.execute_inverse(plan, &coefficients)?;
        apollo_leto_interop::try_dense_from_array(&samples).ok_or_else(|| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto SHT inverse samples".to_owned(),
        })
    }

    /// Execute forward SHT from storage admitted by the concrete GPU contract.
    pub fn execute_forward_flat_typed<T: ShtGpuStorage>(
        &self,
        plan: &ShtWgpuPlan,
        precision: PrecisionProfile,
        flat_samples: &[T],
    ) -> WgpuResult<SphericalHarmonicCoefficients> {
        validate_typed_input::<T>(plan, precision, flat_samples.len())?;
        if let Some(input) = T::as_gpu_slice(flat_samples) {
            return self.execute_forward_accelerator(plan, input);
        }
        COMPLEX_SCRATCH.with(|pool| {
            pool.with_scratch(flat_samples.len(), |represented| {
                for (target, value) in represented.iter_mut().zip(flat_samples.iter().copied()) {
                    *target = value.to_gpu();
                }
                self.execute_forward_accelerator(plan, represented)
            })
        })
    }

    /// Execute forward SHT from a flat typed Leto sample view.
    pub fn execute_forward_flat_leto_typed<T: ShtGpuStorage>(
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

    /// Execute inverse SHT and write the flat output to admitted typed storage.
    pub fn execute_inverse_flat_typed_into<T: ShtGpuStorage>(
        &self,
        plan: &ShtWgpuPlan,
        precision: PrecisionProfile,
        coefficients: &SphericalHarmonicCoefficients,
        output: &mut [T],
    ) -> WgpuResult<()> {
        validate_typed_output::<T>(plan, precision, output.len())?;
        let samples = self.execute_inverse(plan, coefficients)?;
        for (target, value) in output.iter_mut().zip(samples.iter().copied()) {
            *target = T::from_gpu(Complex32::new(
                exact_accelerator_component(value.re, "real")?,
                exact_accelerator_component(value.im, "imaginary")?,
            ));
        }
        Ok(())
    }

    /// Execute inverse SHT from Leto coefficients into typed Mnemosyne-backed Leto storage.
    pub fn execute_inverse_flat_leto_typed<T: ShtGpuStorage + Default>(
        &self,
        plan: &ShtWgpuPlan,
        precision: PrecisionProfile,
        coefficients: leto::ArrayView2<'_, Complex64>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let coefficients = coefficients_from_leto_view(plan, coefficients)?;
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.sample_count()]);
        let output_slice = output
            .as_slice_mut()
            .expect("invariant: Mnemosyne-backed SHT output is contiguous");
        self.execute_inverse_flat_typed_into(plan, precision, &coefficients, output_slice)?;
        Ok(output)
    }

    fn execute_forward_accelerator(
        &self,
        plan: &ShtWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<SphericalHarmonicCoefficients> {
        validate_forward(plan, input.len())?;
        let grid = grid_samples(plan)?;
        COMPLEX_SCRATCH.with(|pool| {
            pool.with_scratch(plan.mode_count(), |output| {
                output.fill(Complex32::new(0.0, 0.0));
                ShtGpuKernel::execute_forward_into(&self.device, plan, input, &grid, output)?;
                Ok(coefficients_from_modes(plan.max_degree(), output))
            })
        })
    }
}
