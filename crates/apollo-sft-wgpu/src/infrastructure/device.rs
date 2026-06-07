//! WGPU device acquisition for this transform backend.

use std::sync::Arc;

use apollo_fft::PrecisionProfile;
use apollo_sft::{SparseComplexStorage, SparseSpectrum};
use num_complex::{Complex32, Complex64};

use crate::application::plan::SftWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::{SftGpuKernel, SftMode};
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    SftWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct SftWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<SftGpuKernel>,
}

impl SftWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(SftGpuKernel::new(device.inner()));
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-sft-wgpu")?)
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::direct_dense_spectrum(true)
    }

    /// Return the acquired WGPU device.
    #[must_use]
    pub fn device(&self) -> &Arc<wgpu::Device> {
        self.device.device()
    }

    /// Return the acquired WGPU queue.
    #[must_use]
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        self.device.queue()
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
        Self::validate_plan(plan)?;
        if input.len() != plan.len() {
            return Err(WgpuError::InputLengthMismatch {
                expected: plan.len(),
                actual: input.len(),
            });
        }
        let dense = self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            input,
            plan.len(),
            SftMode::Forward,
        )?;
        select_top_k(plan.len(), plan.sparsity(), &dense)
    }

    /// Execute inverse reconstruction from a sparse spectrum.
    pub fn execute_inverse(
        &self,
        plan: &SftWgpuPlan,
        spectrum: &SparseSpectrum,
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_plan(plan)?;
        if spectrum.n != plan.len() {
            return Err(WgpuError::InputLengthMismatch {
                expected: plan.len(),
                actual: spectrum.n,
            });
        }
        let dense: Vec<Complex32> = spectrum
            .to_dense()
            .iter()
            .map(|value| Complex32::new(value.re as f32, value.im as f32))
            .collect();
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            &dense,
            plan.len(),
            SftMode::Inverse,
        )
    }

    /// Execute the forward SFT with typed `Complex64`, `Complex32`, or mixed `[f16; 2]` input storage.
    ///
    /// Promotes represented input once to `Complex32` before dispatch.
    /// Returns an allocated `SparseSpectrum` with `Complex64` internal representation.
    pub fn execute_forward_typed<T: SparseComplexStorage>(
        &self,
        plan: &SftWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
    ) -> WgpuResult<SparseSpectrum> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        let represented: Vec<Complex32> = input
            .iter()
            .map(|v| {
                let c = v.to_complex64();
                Complex32::new(c.re as f32, c.im as f32)
            })
            .collect();
        self.execute_forward(plan, &represented)
    }

    /// Execute the inverse SFT from a sparse spectrum with typed complex output storage.
    pub fn execute_inverse_typed_into<T: SparseComplexStorage>(
        &self,
        plan: &SftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: &SparseSpectrum,
        output: &mut [T],
    ) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        if output.len() != plan.len() {
            return Err(WgpuError::InputLengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        let computed = self.execute_inverse(plan, spectrum)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_complex64(Complex64::new(f64::from(value.re), f64::from(value.im)));
        }
        Ok(())
    }

    fn validate_plan(plan: &SftWgpuPlan) -> WgpuResult<()> {
        if plan.len() == 0 {
            return Err(WgpuError::InvalidPlan {
                len: plan.len(),
                sparsity: plan.sparsity(),
                message: "transform length must be greater than zero",
            });
        }
        if plan.sparsity() == 0 {
            return Err(WgpuError::InvalidPlan {
                len: plan.len(),
                sparsity: plan.sparsity(),
                message: "sparsity must be greater than zero",
            });
        }
        if plan.sparsity() > plan.len() {
            return Err(WgpuError::InvalidPlan {
                len: plan.len(),
                sparsity: plan.sparsity(),
                message: "sparsity must not exceed transform length",
            });
        }
        if plan.len() > u32::MAX as usize {
            return Err(WgpuError::InvalidPlan {
                len: plan.len(),
                sparsity: plan.sparsity(),
                message: "transform length must fit in u32 for WGPU dispatch",
            });
        }
        Ok(())
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
            .insert(frequency, Complex64::new(value.re as f64, value.im as f64))
            .map_err(|_| WgpuError::InvalidPlan {
                len,
                sparsity,
                message: "selected support violates sparse spectrum invariants",
            })?;
    }
    Ok(spectrum)
}
