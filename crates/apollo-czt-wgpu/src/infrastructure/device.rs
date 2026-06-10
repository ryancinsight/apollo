//! WGPU device acquisition for this transform backend.

use apollo_fft::application::utilities::leto_interop;
use std::{borrow::Cow, sync::Arc};

use apollo_czt::CztStorage;
use apollo_fft::PrecisionProfile;
use num_complex::{Complex32, Complex64};

use crate::application::plan::CztWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::CztGpuKernel;
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    CztWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct CztWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<CztGpuKernel>,
}

impl CztWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(CztGpuKernel::new(device.inner()));
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-czt-wgpu")?)
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::forward_inverse(true)
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
    pub fn plan(
        &self,
        input_len: usize,
        output_len: usize,
        a: Complex32,
        w: Complex32,
    ) -> CztWgpuPlan {
        CztWgpuPlan::new(
            input_len,
            output_len,
            [a.re.to_bits(), a.im.to_bits()],
            [w.re.to_bits(), w.im.to_bits()],
        )
    }

    /// Execute the direct forward CZT for a complex-valued `f32` signal.
    pub fn execute_forward(
        &self,
        plan: &CztWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_plan_input(plan, input)?;
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            plan,
            input,
        )
    }

    /// Execute the direct forward CZT from a Leto host view.
    ///
    /// Contiguous views are borrowed without copying. Strided views are
    /// materialized once into logical order before GPU upload.
    pub fn execute_forward_leto(
        &self,
        plan: &CztWgpuPlan,
        input: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let input = leto_view1_cow(input);
        let output = self.execute_forward(plan, &input)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the forward CZT with typed `Complex64`, `Complex32`, or mixed `[f16; 2]` storage.
    ///
    /// Promotes represented input once to `Complex32`, dispatches the GPU kernel,
    /// and quantizes the output back to the requested storage type.
    pub fn execute_forward_typed_into<T: CztStorage>(
        &self,
        plan: &CztWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_czt_typed_precision::<T>(precision)?;
        let output_len = plan.output_len();
        if output.len() != output_len {
            return Err(WgpuError::LengthMismatch {
                expected: output_len,
                actual: output.len(),
            });
        }
        let represented = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex32>() {
            // Safety: T is Complex32, so &[T] is layout-compatible with &[Complex32].
            let slice_c32 = unsafe {
                std::slice::from_raw_parts(input.as_ptr().cast::<Complex32>(), input.len())
            };
            std::borrow::Cow::Borrowed(slice_c32)
        } else {
            let vec: Vec<Complex32> = input
                .iter()
                .map(|v| {
                    let c = v.to_complex64();
                    Complex32::new(c.re as f32, c.im as f32)
                })
                .collect();
            std::borrow::Cow::Owned(vec)
        };
        let computed = self.execute_forward(plan, &represented)?;
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex32>() {
            // Safety: T is Complex32, so &mut [T] is layout-compatible with &mut [Complex32].
            let slice_c32 = unsafe {
                std::slice::from_raw_parts_mut(
                    output.as_mut_ptr().cast::<Complex32>(),
                    output.len(),
                )
            };
            slice_c32.copy_from_slice(&computed);
        } else {
            for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
                *slot = T::from_complex64(Complex64::new(f64::from(value.re), f64::from(value.im)));
            }
        }
        Ok(())
    }

    /// Execute typed forward CZT from a Leto host view into Mnemosyne-backed Leto storage.
    pub fn execute_forward_leto_typed<T: CztStorage>(
        &self,
        plan: &CztWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = leto_view1_cow(input);
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); plan.output_len()];
        self.execute_forward_typed_into(plan, precision, &input, &mut output)?;
        leto_array1_from_slice(&output)
    }

    fn validate_czt_typed_precision<T: CztStorage>(precision: PrecisionProfile) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Ok(())
    }

    /// Execute the adjoint inverse CZT.
    ///
    /// Computes `x[n] = (A^n / N) · Σ_k X[k] · W^{-nk}` on the GPU.
    ///
    /// **Exactness**: exact when |A| = 1, |W| = 1, and W is an N-th root of
    /// unity (DFT case).  For general spiral parameters this is the
    /// minimum-norm adjoint solution; the CPU crate's `CztPlan::inverse` uses
    /// the Björck–Pereyra Vandermonde solve for exact general inversion.
    ///
    /// # Errors
    ///
    /// Returns `WgpuError::NotSquare` when `plan.input_len() != plan.output_len()`.
    pub fn execute_inverse(
        &self,
        plan: &CztWgpuPlan,
        spectrum: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        if plan.input_len() != plan.output_len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.input_len(),
                actual: plan.output_len(),
            });
        }
        if spectrum.len() != plan.output_len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.output_len(),
                actual: spectrum.len(),
            });
        }
        let n = plan.input_len();
        if n == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("CZT lengths input={n}, output={n} must be greater than zero"),
            });
        }
        self.kernel.execute_inverse(
            self.device.inner(),
            self.device.queue().as_ref(),
            plan,
            spectrum,
        )
    }

    /// Execute the adjoint inverse CZT from a Leto host view.
    ///
    /// This preserves the existing WGPU inverse contract: exact for DFT
    /// parameters and adjoint/minimum-norm for general CZT spirals.
    pub fn execute_inverse_leto(
        &self,
        plan: &CztWgpuPlan,
        spectrum: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let spectrum = leto_view1_cow(spectrum);
        let output = self.execute_inverse(plan, &spectrum)?;
        leto_array1_from_slice(&output)
    }

    fn validate_plan_input(plan: &CztWgpuPlan, input: &[Complex32]) -> WgpuResult<()> {
        let input_len = plan.input_len();
        let output_len = plan.output_len();
        if input_len == 0 || output_len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!(
                    "CZT lengths input={input_len}, output={output_len} must be greater than zero"
                ),
            });
        }
        if input.len() != input_len {
            return Err(WgpuError::LengthMismatch {
                expected: input_len,
                actual: input.len(),
            });
        }
        let a = plan.a();
        let w = plan.w();
        let a_norm = a.norm();
        let w_norm = w.norm();
        if !a_norm.is_finite() || !w_norm.is_finite() || a_norm == 0.0 || w_norm == 0.0 {
            return Err(WgpuError::InvalidPlan {
                message: "CZT spiral parameters must have finite non-zero magnitude".to_owned(),
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
        message: "failed to allocate Mnemosyne-backed Leto CZT output".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use leto::SliceArg;

    use super::leto_view1_cow;

    #[test]
    fn leto_view1_cow_borrows_contiguous_views() {
        let input = leto::Array1::from_shape_vec([4], vec![1_u32, 2, 3, 4]).expect("input");
        let cow = leto_view1_cow(input.view());
        assert!(matches!(cow, Cow::Borrowed(_)));
        assert_eq!(cow.as_ref(), &[1, 2, 3, 4]);
    }

    #[test]
    fn leto_view1_cow_materializes_strided_views() {
        let input =
            leto::Array1::from_shape_vec([8], vec![1_u32, 99, 2, 99, 3, 99, 4, 99]).expect("input");
        let view = input
            .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
            .expect("strided view");
        let cow = leto_view1_cow(view);
        assert!(matches!(cow, Cow::Owned(_)));
        assert_eq!(cow.as_ref(), &[1, 2, 3, 4]);
    }
}
