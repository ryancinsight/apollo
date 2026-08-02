#![warn(missing_docs)]
//! WGPU backend boundary for Apollo SFT.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns the SFT kernel, its domain names, and
//! the sparse-domain surface. The dense DFT/IDFT instantiate the
//! scaffold directly; sparse selection, sparse reconstruction, and the
//! `Complex64` quantization boundary extend it through
//! [`SparseExecution`], since a [`SparseSpectrum`] is a domain structure,
//! not a slice.

/// Infrastructure boundary for the SFT kernel.
pub mod infrastructure;
#[cfg(test)]
pub(crate) mod verification;

use apollo_fft::{Complex32, GpuElement, GpuStorage, PrecisionProfile};
use eunomia::Complex64;

use crate::domain::spectrum::sparse::SparseSpectrum;

pub use apollo_fft::{WgpuCapabilities, WgpuError, WgpuResult};
pub use infrastructure::kernel::SftGpuKernel;

/// Plan payload for the sparse Fourier transform: dense length and
/// retained support size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparsityPlan {
    len: usize,
    sparsity: usize,
}

impl SparsityPlan {
    /// Create a sparsity-plan payload.
    ///
    /// Validation (`0 < sparsity <= len`, accelerator range) runs at
    /// dispatch.
    #[must_use]
    pub const fn new(len: usize, sparsity: usize) -> Self {
        Self { len, sparsity }
    }

    /// Return the dense transform length carried by this payload.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Return whether the payload carries zero dense length.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Return the retained support size carried by this payload.
    #[must_use]
    pub const fn sparsity(self) -> usize {
        self.sparsity
    }
}

/// Metadata-preserving WGPU plan descriptor.
pub type SftWgpuPlan = apollo_fft::WgpuTransformPlan<SftGpuKernel>;

/// WGPU backend descriptor.
pub type SftWgpuBackend = apollo_fft::WgpuTransformBackend<SftGpuKernel>;

/// Sparse-domain surface of the SFT backend.
///
/// The GPU computes dense spectra; host-side selection preserves the
/// CPU crate's sparse-domain contract (largest magnitudes, lower index
/// as the deterministic tie-breaker, ascending stored support), and
/// reconstruction densifies a [`SparseSpectrum`] before dispatch.
/// `SparseSpectrum` is the CPU domain's `Complex64` SSOT; components
/// must be exactly representable in the concrete `f32` accelerator or
/// the operation reports [`WgpuError::PrecisionLoss`] instead of
/// silently quantizing.
pub trait SparseExecution {
    /// Execute the forward transform and select the sparse support.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn sparse_forward(&self, plan: &SftWgpuPlan, input: &[Complex32])
        -> WgpuResult<SparseSpectrum>;

    /// Execute the forward transform from a Leto view and select the
    /// sparse support.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn sparse_forward_leto(
        &self,
        plan: &SftWgpuPlan,
        input: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<SparseSpectrum>;

    /// Execute the forward transform with admitted typed storage and
    /// select the sparse support.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, precision-profile, or
    /// provider failure.
    fn sparse_forward_typed<T: GpuStorage<Complex32>>(
        &self,
        plan: &SftWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
    ) -> WgpuResult<SparseSpectrum>;

    /// Execute the forward transform from typed Leto storage and select
    /// the sparse support.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, precision-profile, or
    /// provider failure.
    fn sparse_forward_leto_typed<T: GpuStorage<Complex32>>(
        &self,
        plan: &SftWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<SparseSpectrum>;

    /// Reconstruct the dense signal from a sparse spectrum.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, precision-loss, or
    /// provider failure.
    fn sparse_inverse(
        &self,
        plan: &SftWgpuPlan,
        spectrum: &SparseSpectrum,
    ) -> WgpuResult<Vec<Complex32>>;

    /// Reconstruct the dense signal into caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, precision-loss, or
    /// provider failure.
    fn sparse_inverse_into(
        &self,
        plan: &SftWgpuPlan,
        spectrum: &SparseSpectrum,
        output: &mut [Complex32],
    ) -> WgpuResult<()>;

    /// Reconstruct the dense signal into Mnemosyne-backed Leto storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, precision-loss, or
    /// provider failure.
    fn sparse_inverse_leto(
        &self,
        plan: &SftWgpuPlan,
        spectrum: &SparseSpectrum,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>>;

    /// Reconstruct the dense signal into admitted typed storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, precision-profile,
    /// precision-loss, or provider failure.
    fn sparse_inverse_typed_into<T: GpuStorage<Complex32>>(
        &self,
        plan: &SftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: &SparseSpectrum,
        output: &mut [T],
    ) -> WgpuResult<()>;

    /// Reconstruct the dense signal into typed Leto storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, precision-profile,
    /// precision-loss, or provider failure.
    fn sparse_inverse_leto_typed<T: GpuStorage<Complex32> + Default>(
        &self,
        plan: &SftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: &SparseSpectrum,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>>;

    /// Explicitly quantize a CPU-owned sparse spectrum for concrete
    /// `f32` execution.
    ///
    /// This is the only lossy bridge from the CPU `Complex64` domain
    /// into the accelerator representation; [`Self::sparse_inverse`]
    /// rejects non-representable components instead of quantizing them.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or precision-loss
    /// failure.
    fn quantize_spectrum(
        &self,
        plan: &SftWgpuPlan,
        spectrum: &SparseSpectrum,
    ) -> WgpuResult<SparseSpectrum>;
}

impl SparseExecution for SftWgpuBackend {
    fn sparse_forward(
        &self,
        plan: &SftWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<SparseSpectrum> {
        let dense = self.execute_forward(plan, input)?;
        select_top_k(plan.len(), plan.payload().sparsity(), &dense)
    }

    fn sparse_forward_leto(
        &self,
        plan: &SftWgpuPlan,
        input: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<SparseSpectrum> {
        let input = apollo_leto_interop::view_cow(&input);
        self.sparse_forward(plan, &input)
    }

    fn sparse_forward_typed<T: GpuStorage<Complex32>>(
        &self,
        plan: &SftWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
    ) -> WgpuResult<SparseSpectrum> {
        validate_profile::<T>(precision)?;
        if let Some(input) = T::as_element_slice(input) {
            return self.sparse_forward(plan, input);
        }
        Complex32::with_input_scratch(input.len(), |represented| {
            for (target, value) in represented.iter_mut().zip(input.iter().copied()) {
                *target = value.to_gpu();
            }
            self.sparse_forward(plan, represented)
        })
    }

    fn sparse_forward_leto_typed<T: GpuStorage<Complex32>>(
        &self,
        plan: &SftWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<SparseSpectrum> {
        let input = apollo_leto_interop::view_cow(&input);
        self.sparse_forward_typed(plan, precision, &input)
    }

    fn sparse_inverse(
        &self,
        plan: &SftWgpuPlan,
        spectrum: &SparseSpectrum,
    ) -> WgpuResult<Vec<Complex32>> {
        let mut output = vec![Complex32::new(0.0, 0.0); plan.len()];
        self.sparse_inverse_into(plan, spectrum, &mut output)?;
        Ok(output)
    }

    fn sparse_inverse_into(
        &self,
        plan: &SftWgpuPlan,
        spectrum: &SparseSpectrum,
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        validate_spectrum(plan, spectrum)?;
        Complex32::with_input_scratch(plan.len(), |dense| {
            dense.fill(Complex32::new(0.0, 0.0));
            populate_dense_spectrum(dense, spectrum, plan.len())?;
            self.execute_inverse_into(plan, dense, output)
        })
    }

    fn sparse_inverse_leto(
        &self,
        plan: &SftWgpuPlan,
        spectrum: &SparseSpectrum,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let mut output =
            leto::Array::<Complex32, leto::MnemosyneStorage<Complex32>, 1>::zeros_mnemosyne([
                plan.len()
            ]);
        let output_slice = output
            .as_slice_mut()
            .expect("SFT Mnemosyne output must be contiguous");
        self.sparse_inverse_into(plan, spectrum, output_slice)?;
        Ok(output)
    }

    fn sparse_inverse_typed_into<T: GpuStorage<Complex32>>(
        &self,
        plan: &SftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: &SparseSpectrum,
        output: &mut [T],
    ) -> WgpuResult<()> {
        validate_profile::<T>(precision)?;
        if let Some(output) = T::as_element_slice_mut(output) {
            return self.sparse_inverse_into(plan, spectrum, output);
        }
        Complex32::with_output_scratch(output.len(), |computed| {
            self.sparse_inverse_into(plan, spectrum, computed)?;
            for (target, value) in output.iter_mut().zip(computed.iter().copied()) {
                *target = T::from_gpu(value);
            }
            Ok(())
        })
    }

    fn sparse_inverse_leto_typed<T: GpuStorage<Complex32> + Default>(
        &self,
        plan: &SftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: &SparseSpectrum,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.len()]);
        let output_slice = output
            .as_slice_mut()
            .expect("SFT Mnemosyne typed output must be contiguous");
        self.sparse_inverse_typed_into(plan, precision, spectrum, output_slice)?;
        Ok(output)
    }

    fn quantize_spectrum(
        &self,
        plan: &SftWgpuPlan,
        spectrum: &SparseSpectrum,
    ) -> WgpuResult<SparseSpectrum> {
        validate_spectrum(plan, spectrum)?;
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
}

fn validate_profile<T: GpuStorage<Complex32>>(precision: PrecisionProfile) -> WgpuResult<()> {
    if precision != T::PROFILE {
        return Err(WgpuError::InvalidPrecisionProfile);
    }
    Ok(())
}

fn validate_spectrum(plan: &SftWgpuPlan, spectrum: &SparseSpectrum) -> WgpuResult<()> {
    if spectrum.n != plan.len() {
        return Err(WgpuError::LengthMismatch {
            expected: plan.len(),
            actual: spectrum.n,
        });
    }
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
