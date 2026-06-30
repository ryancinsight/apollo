//! WGPU device acquisition for this transform backend.

use apollo_fft::application::utilities::leto_interop;
use std::borrow::Cow;
use std::sync::Arc;

use apollo_fft::PrecisionProfile;
use crate::{SparseComplexStorage, SparseSpectrum};
use eunomia::{Complex32, Complex64};

use crate::infrastructure::transport::gpu::application::plan::SftWgpuPlan;
use crate::infrastructure::transport::gpu::domain::capabilities::WgpuCapabilities;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::kernel::{SftGpuKernel, SftMode};
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
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: input.len(),
            });
        }
        let dense = self.kernel.execute(
            &self.device,
            input,
            plan.len(),
            SftMode::Forward,
        )?;
        select_top_k(plan.len(), plan.sparsity(), &dense)
    }

    /// Execute direct dense-spectrum SFT from a Leto complex `f32` view.
    ///
    /// Contiguous Leto views borrow host storage directly; strided views copy
    /// once into logical order before dispatching to the existing WGPU slice path.
    pub fn execute_forward_leto(
        &self,
        plan: &SftWgpuPlan,
        input: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<SparseSpectrum> {
        let input = leto_view1_cow(input);
        self.execute_forward(plan, &input)
    }

    /// Execute inverse reconstruction from a sparse spectrum.
    pub fn execute_inverse(
        &self,
        plan: &SftWgpuPlan,
        spectrum: &SparseSpectrum,
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_plan(plan)?;
        if spectrum.n != plan.len() {
            return Err(WgpuError::LengthMismatch {
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
            &self.device,
            &dense,
            plan.len(),
            SftMode::Inverse,
        )
    }

    /// Execute inverse reconstruction from a sparse spectrum into a Leto dense signal.
    pub fn execute_inverse_leto(
        &self,
        plan: &SftWgpuPlan,
        spectrum: &SparseSpectrum,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let output = self.execute_inverse(plan, spectrum)?;
        leto_array1_from_slice(&output)
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
        let represented = if let Some(slice_c32) = T::as_c32_slice(input) {
            Cow::Borrowed(slice_c32)
        } else {
            let vec: Vec<Complex32> = input
                .iter()
                .map(|v| {
                    let c = v.to_complex64();
                    Complex32::new(c.re as f32, c.im as f32)
                })
                .collect();
            Cow::Owned(vec)
        };
        self.execute_forward(plan, &represented)
    }

    /// Execute forward SFT from typed Leto complex storage.
    ///
    /// Precision-profile validation and host representation match
    /// [`Self::execute_forward_typed`].
    pub fn execute_forward_leto_typed<T: SparseComplexStorage>(
        &self,
        plan: &SftWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<SparseSpectrum> {
        let input = leto_view1_cow(input);
        self.execute_forward_typed(plan, precision, &input)
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
            return Err(WgpuError::LengthMismatch {
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

    /// Execute inverse SFT from a sparse spectrum into typed Leto dense storage.
    pub fn execute_inverse_leto_typed<T: SparseComplexStorage>(
        &self,
        plan: &SftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: &SparseSpectrum,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); plan.len()];
        self.execute_inverse_typed_into(plan, precision, spectrum, &mut output)?;
        leto_array1_from_slice(&output)
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
        if plan.len() > u32::MAX as usize {
            return Err(WgpuError::InvalidPlan {
                    message: format!("invalid plan len={}, sparsity={}: transform length must fit in u32 for WGPU dispatch", plan.len(), plan.sparsity()),
                });
        }
        Ok(())
    }
}

fn leto_view1_cow<T: Copy>(view: leto::ArrayView1<'_, T>) -> Cow<'_, [T]> {
    leto_interop::view1_cow(&view)
}
fn leto_array1_from_slice<T: Copy>(
    values: &[T],
) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    leto_interop::try_array1_from_slice(values).ok_or_else(|| WgpuError::InvalidPlan {
        message: "failed to allocate Mnemosyne-backed Leto output".to_string(),
    })
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
                    message: format!("invalid plan len={len}, sparsity={sparsity}: selected support violates sparse spectrum invariants"),
                })?;
    }
    Ok(spectrum)
}

#[cfg(test)]
mod tests {
    use eunomia::Complex32;

    use super::leto_view1_cow;

    #[test]
    fn leto_view1_cow_borrows_contiguous_views() {
        let input = leto::Array1::from_shape_vec(
            [2],
            vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)],
        )
        .expect("leto input");
        let cow = leto_view1_cow(input.view());
        assert!(matches!(cow, std::borrow::Cow::Borrowed(_)));
        assert_eq!(&*cow, &[Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)]);
    }
}
