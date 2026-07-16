//! Hephaestus device acquisition and sparse Fourier execution boundary.
//!
//! The accelerator evaluates only the concrete `f32` dense DFT. Apollo owns
//! the sparse-domain projection and validates every host conversion before
//! device allocation, so a high-accuracy `SparseSpectrum` cannot silently
//! narrow during inverse reconstruction.

use apollo_fft::PrecisionProfile;
use eunomia::{Complex32, Complex64};
use hephaestus_wgpu::WgpuDevice;
use mnemosyne::scratch::ScratchPool;

use crate::infrastructure::transport::gpu::application::plan::SftWgpuPlan;
use crate::infrastructure::transport::gpu::domain::capabilities::WgpuCapabilities;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::kernel::{SftGpuKernel, SftMode};
use crate::{SftGpuStorage, SparseSpectrum};

thread_local! {
    static COMPLEX_SCRATCH: ScratchPool<Complex32> = const { ScratchPool::new() };
}

/// Hephaestus WGPU backend for direct dense SFT execution.
#[derive(Debug, Clone)]
pub struct SftWgpuBackend {
    device: WgpuDevice,
}

impl SftWgpuBackend {
    /// Create a backend from an acquired Hephaestus WGPU device.
    #[must_use]
    pub const fn new(device: WgpuDevice) -> Self {
        Self { device }
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::direct_dense_spectrum(true)
    }

    /// Return the acquired Hephaestus WGPU device implementation.
    #[must_use]
    pub const fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Create a metadata-only plan descriptor.
    #[must_use]
    pub const fn plan(&self, len: usize, sparsity: usize) -> SftWgpuPlan {
        SftWgpuPlan::new(len, sparsity)
    }

    /// Execute direct dense-spectrum SFT followed by deterministic top-k support selection.
    ///
    /// The GPU computes the full DFT. Host-side selection preserves the CPU
    /// crate's sparse-domain contract: largest magnitudes, lower index as the
    /// deterministic tie-breaker, and ascending stored support.
    pub fn execute_forward(
        &self,
        plan: &SftWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<SparseSpectrum> {
        Self::validate_forward(plan, input.len(), plan.len())?;
        let mut dense = vec![Complex32::new(0.0, 0.0); plan.len()];
        self.execute_forward_into(plan, input, &mut dense)?;
        select_top_k(plan.len(), plan.sparsity(), &dense)
    }

    /// Execute the forward dense DFT into caller-owned storage before sparse selection.
    pub fn execute_forward_into(
        &self,
        plan: &SftWgpuPlan,
        input: &[Complex32],
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        Self::validate_forward(plan, input.len(), output.len())?;
        SftGpuKernel::execute_into(&self.device, input, output, SftMode::Forward)
    }

    /// Execute direct dense-spectrum SFT from a Leto complex `f32` view.
    ///
    /// Contiguous Leto views borrow host storage directly; strided views copy
    /// once into logical order before provider dispatch.
    pub fn execute_forward_leto(
        &self,
        plan: &SftWgpuPlan,
        input: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<SparseSpectrum> {
        let input = apollo_leto_interop::view_cow(&input);
        self.execute_forward(plan, &input)
    }

    /// Execute inverse reconstruction from a sparse spectrum.
    ///
    /// `SparseSpectrum` is the CPU domain's `Complex64` SSOT. Each component
    /// must be exactly representable in the concrete `f32` accelerator before
    /// dispatch; otherwise this method returns [`WgpuError::PrecisionLoss`]
    /// rather than silently changing the requested reconstruction.
    pub fn execute_inverse(
        &self,
        plan: &SftWgpuPlan,
        spectrum: &SparseSpectrum,
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_inverse(plan, spectrum, plan.len())?;
        let mut output = vec![Complex32::new(0.0, 0.0); plan.len()];
        self.execute_inverse_into(plan, spectrum, &mut output)?;
        Ok(output)
    }

    /// Explicitly quantize a CPU-owned sparse spectrum for concrete `f32` execution.
    ///
    /// This is the only lossy bridge from the CPU `Complex64` domain into the
    /// accelerator representation. Callers that need exact value preservation
    /// should pass the original spectrum to [`Self::execute_inverse`], which
    /// rejects non-representable components instead of quantizing them.
    pub fn quantize_spectrum(
        &self,
        plan: &SftWgpuPlan,
        spectrum: &SparseSpectrum,
    ) -> WgpuResult<SparseSpectrum> {
        Self::validate_inverse(plan, spectrum, plan.len())?;
        let mut quantized = SparseSpectrum::new(plan.len());
        for (&frequency, &value) in spectrum.frequencies.iter().zip(spectrum.values.iter()) {
            quantized
                .insert(
                    frequency,
                    Complex64::new(
                        f64::from(quantize_accelerator_component(value.re, "real")?),
                        f64::from(quantize_accelerator_component(value.im, "imaginary")?),
                    ),
                )
                .map_err(|_| WgpuError::InvalidPlan {
                    message: format!(
                        "sparse frequency {frequency} is outside transform length {}",
                        plan.len()
                    ),
                })?;
        }
        Ok(quantized)
    }

    /// Execute inverse reconstruction into caller-owned concrete accelerator storage.
    pub fn execute_inverse_into(
        &self,
        plan: &SftWgpuPlan,
        spectrum: &SparseSpectrum,
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        Self::validate_inverse(plan, spectrum, output.len())?;
        COMPLEX_SCRATCH.with(|pool| {
            pool.with_scratch(plan.len(), |dense| {
                dense.fill(Complex32::new(0.0, 0.0));
                populate_dense_spectrum(dense, spectrum, plan.len())?;
                SftGpuKernel::execute_into(&self.device, dense, output, SftMode::Inverse)
            })
        })
    }

    /// Execute inverse reconstruction from a sparse spectrum into Leto storage.
    pub fn execute_inverse_leto(
        &self,
        plan: &SftWgpuPlan,
        spectrum: &SparseSpectrum,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        Self::validate_inverse(plan, spectrum, plan.len())?;
        let mut output = complex_output(plan.len());
        let output_slice = output
            .as_slice_mut()
            .expect("SFT Mnemosyne output must be contiguous");
        self.execute_inverse_into(plan, spectrum, output_slice)?;
        Ok(output)
    }

    /// Execute the forward SFT with storage admitted by the concrete GPU contract.
    pub fn execute_forward_typed<T: SftGpuStorage>(
        &self,
        plan: &SftWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
    ) -> WgpuResult<SparseSpectrum> {
        Self::validate_typed_input::<T>(plan, precision, input)?;
        if let Some(input) = T::as_gpu_slice(input) {
            return self.execute_forward(plan, input);
        }
        COMPLEX_SCRATCH.with(|pool| {
            pool.with_scratch(input.len(), |represented| {
                for (target, value) in represented.iter_mut().zip(input.iter().copied()) {
                    *target = value.to_gpu();
                }
                self.execute_forward(plan, represented)
            })
        })
    }

    /// Execute forward SFT from typed Leto complex storage.
    pub fn execute_forward_leto_typed<T: SftGpuStorage>(
        &self,
        plan: &SftWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<SparseSpectrum> {
        let input = apollo_leto_interop::view_cow(&input);
        self.execute_forward_typed(plan, precision, &input)
    }

    /// Execute the inverse SFT from a sparse spectrum with admitted output storage.
    pub fn execute_inverse_typed_into<T: SftGpuStorage>(
        &self,
        plan: &SftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: &SparseSpectrum,
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_typed_output::<T>(plan, precision, output.len())?;
        if let Some(output) = T::as_gpu_slice_mut(output) {
            return self.execute_inverse_into(plan, spectrum, output);
        }
        COMPLEX_SCRATCH.with(|pool| {
            pool.with_scratch(output.len(), |computed| {
                self.execute_inverse_into(plan, spectrum, computed)?;
                for (target, value) in output.iter_mut().zip(computed.iter().copied()) {
                    *target = T::from_gpu(value);
                }
                Ok(())
            })
        })
    }

    /// Execute inverse SFT from a sparse spectrum into typed Leto dense storage.
    pub fn execute_inverse_leto_typed<T: SftGpuStorage + Default>(
        &self,
        plan: &SftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: &SparseSpectrum,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        Self::validate_typed_output::<T>(plan, precision, plan.len())?;
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.len()]);
        let output_slice = output
            .as_slice_mut()
            .expect("SFT Mnemosyne typed output must be contiguous");
        self.execute_inverse_typed_into(plan, precision, spectrum, output_slice)?;
        Ok(output)
    }

    fn validate_plan(plan: &SftWgpuPlan) -> WgpuResult<()> {
        if plan.len() == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan len={}, sparsity={}: transform length must be greater than zero",
                    plan.len(),
                    plan.sparsity()
                ),
            });
        }
        if plan.sparsity() == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan len={}, sparsity={}: sparsity must be greater than zero",
                    plan.len(),
                    plan.sparsity()
                ),
            });
        }
        if plan.sparsity() > plan.len() {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan len={}, sparsity={}: sparsity must not exceed transform length",
                    plan.len(),
                    plan.sparsity()
                ),
            });
        }
        if u32::try_from(plan.len()).is_err() {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan len={}, sparsity={}: transform length exceeds the accelerator parameter range",
                    plan.len(),
                    plan.sparsity()
                ),
            });
        }
        Ok(())
    }

    fn validate_forward(plan: &SftWgpuPlan, input_len: usize, output_len: usize) -> WgpuResult<()> {
        Self::validate_plan(plan)?;
        validate_length(plan.len(), input_len)?;
        validate_length(plan.len(), output_len)
    }

    fn validate_inverse(
        plan: &SftWgpuPlan,
        spectrum: &SparseSpectrum,
        output_len: usize,
    ) -> WgpuResult<()> {
        Self::validate_plan(plan)?;
        validate_length(plan.len(), spectrum.n)?;
        validate_length(plan.len(), output_len)?;
        if spectrum.frequencies.len() != spectrum.values.len() {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "sparse spectrum frequency/value lengths differ: {} != {}",
                    spectrum.frequencies.len(),
                    spectrum.values.len()
                ),
            });
        }
        Ok(())
    }

    fn validate_typed_input<T: SftGpuStorage>(
        plan: &SftWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
    ) -> WgpuResult<()> {
        if precision != T::PROFILE {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Self::validate_forward(plan, input.len(), plan.len())
    }

    fn validate_typed_output<T: SftGpuStorage>(
        plan: &SftWgpuPlan,
        precision: PrecisionProfile,
        output_len: usize,
    ) -> WgpuResult<()> {
        if precision != T::PROFILE {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Self::validate_plan(plan)?;
        validate_length(plan.len(), output_len)
    }
}

fn validate_length(expected: usize, actual: usize) -> WgpuResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(WgpuError::LengthMismatch { expected, actual })
    }
}

fn populate_dense_spectrum(
    dense: &mut [Complex32],
    spectrum: &SparseSpectrum,
    len: usize,
) -> WgpuResult<()> {
    for (&frequency, &value) in spectrum.frequencies.iter().zip(spectrum.values.iter()) {
        let Some(slot) = dense.get_mut(frequency) else {
            return Err(WgpuError::InvalidPlan {
                message: format!("sparse frequency {frequency} is outside transform length {len}"),
            });
        };
        *slot = Complex32::new(
            exact_accelerator_component(value.re, "real")?,
            exact_accelerator_component(value.im, "imaginary")?,
        );
    }
    Ok(())
}

fn exact_accelerator_component(value: f64, component: &'static str) -> WgpuResult<f32> {
    let represented = quantize_accelerator_component(value, component)?;
    if value.is_finite() && f64::from(represented) == value {
        Ok(represented)
    } else {
        Err(WgpuError::PrecisionLoss { component, value })
    }
}

fn quantize_accelerator_component(value: f64, component: &'static str) -> WgpuResult<f32> {
    let represented = value as f32;
    if value.is_finite() && represented.is_finite() {
        Ok(represented)
    } else {
        Err(WgpuError::PrecisionLoss { component, value })
    }
}

fn complex_output(len: usize) -> leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1> {
    leto::Array::<Complex32, leto::MnemosyneStorage<Complex32>, 1>::zeros_mnemosyne([len])
}

fn select_top_k(len: usize, sparsity: usize, dense: &[Complex32]) -> WgpuResult<SparseSpectrum> {
    let mut ranked: Vec<(usize, Complex32, f32)> = dense
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (index, value, value.norm_sqr()))
        .filter(|(_, _, energy)| *energy > 0.0)
        .collect();
    ranked.sort_by(|left, right| {
        right
            .2
            .total_cmp(&left.2)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(sparsity);
    ranked.sort_by_key(|(index, _, _)| *index);

    let mut spectrum = SparseSpectrum::new(len);
    for (frequency, value, _) in ranked {
        spectrum
            .insert(
                frequency,
                Complex64::new(f64::from(value.re), f64::from(value.im)),
            )
            .map_err(|_| WgpuError::InvalidPlan {
                message: format!(
                    "invalid plan len={len}, sparsity={sparsity}: selected support violates sparse spectrum invariants"
                ),
            })?;
    }
    Ok(spectrum)
}
