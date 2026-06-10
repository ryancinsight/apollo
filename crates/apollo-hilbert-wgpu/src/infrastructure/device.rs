//! WGPU device acquisition for this transform backend.

use apollo_fft::application::utilities::leto_interop;
use std::{borrow::Cow, sync::Arc};

use num_complex::Complex32;

use apollo_fft::PrecisionProfile;
use apollo_hilbert::HilbertStorage;

use crate::application::plan::HilbertWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::HilbertGpuKernel;
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    HilbertWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct HilbertWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<HilbertGpuKernel>,
}

impl HilbertWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(HilbertGpuKernel::new(device.inner()));
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-hilbert-wgpu")?)
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::forward_and_inverse(true)
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
    pub const fn plan(&self, len: usize) -> HilbertWgpuPlan {
        HilbertWgpuPlan::new(len)
    }

    /// Execute the analytic signal `x + i H{x}` for a real-valued `f32` signal.
    pub fn execute_analytic_signal(
        &self,
        plan: &HilbertWgpuPlan,
        input: &[f32],
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_plan_input(plan, input)?;
        self.kernel
            .execute(self.device.inner(), self.device.queue().as_ref(), input)
    }

    /// Execute the analytic signal from a Leto real-valued host view.
    ///
    /// Contiguous views are borrowed without copying. Strided views are
    /// materialized once into logical order before GPU upload.
    pub fn execute_analytic_signal_leto(
        &self,
        plan: &HilbertWgpuPlan,
        input: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let input = leto_view1_cow(input);
        let output = self.execute_analytic_signal(plan, &input)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the forward Hilbert quadrature component `H{x}` for a real-valued `f32` signal.
    pub fn execute_forward(&self, plan: &HilbertWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        Ok(self
            .execute_analytic_signal(plan, input)?
            .into_iter()
            .map(|value| value.im)
            .collect())
    }

    /// Execute the forward Hilbert quadrature component from a Leto host view.
    pub fn execute_forward_leto(
        &self,
        plan: &HilbertWgpuPlan,
        input: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let input = leto_view1_cow(input);
        let output = self.execute_forward(plan, &input)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the forward Hilbert quadrature transform with typed `f64`, `f32`, or mixed `f16` storage.
    ///
    /// Promotes represented input once to `f32`, dispatches the GPU analytic-signal kernel,
    /// extracts the imaginary (quadrature) component, and quantizes output back to storage type.
    pub fn execute_forward_typed_into<T: HilbertStorage>(
        &self,
        plan: &HilbertWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_hilbert_typed_precision::<T>(precision)?;
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        let represented = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // Safety: T is f32, so &[T] is layout-compatible with &[f32].
            let slice_f32 =
                unsafe { std::slice::from_raw_parts(input.as_ptr().cast::<f32>(), input.len()) };
            std::borrow::Cow::Borrowed(slice_f32)
        } else {
            let vec: Vec<f32> = input.iter().map(|v| v.to_f64() as f32).collect();
            std::borrow::Cow::Owned(vec)
        };
        let computed = self.execute_forward(plan, &represented)?;
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // Safety: T is f32, so &mut [T] is layout-compatible with &mut [f32].
            let slice_f32 = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<f32>(), output.len())
            };
            slice_f32.copy_from_slice(&computed);
        } else {
            for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
                *slot = T::from_f64(f64::from(value));
            }
        }
        Ok(())
    }

    /// Execute typed forward Hilbert quadrature from a Leto host view.
    pub fn execute_forward_leto_typed<T: HilbertStorage>(
        &self,
        plan: &HilbertWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = leto_view1_cow(input);
        let mut output = vec![T::from_f64(0.0); plan.len()];
        self.execute_forward_typed_into(plan, precision, &input, &mut output)?;
        leto_array1_from_slice(&output)
    }

    fn validate_hilbert_typed_precision<T: HilbertStorage>(
        precision: PrecisionProfile,
    ) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Ok(())
    }

    /// Execute the inverse Hilbert transform: recover the original real signal from its quadrature component.
    ///
    /// By the Hilbert inversion theorem H(H(x)) = -I, the original signal is
    /// `x[n] = -H{H{x}[n]} = -H{quadrature[n]}`.
    pub fn execute_inverse(
        &self,
        plan: &HilbertWgpuPlan,
        quadrature: &[f32],
    ) -> WgpuResult<Vec<f32>> {
        Self::validate_plan_input(plan, quadrature)?;
        self.kernel.execute_inverse(
            self.device.inner(),
            self.device.queue().as_ref(),
            quadrature,
        )
    }

    /// Execute the inverse Hilbert transform from a Leto quadrature view.
    pub fn execute_inverse_leto(
        &self,
        plan: &HilbertWgpuPlan,
        quadrature: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let quadrature = leto_view1_cow(quadrature);
        let output = self.execute_inverse(plan, &quadrature)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the inverse Hilbert transform with typed storage.
    pub fn execute_inverse_typed_into<T: HilbertStorage>(
        &self,
        plan: &HilbertWgpuPlan,
        precision: PrecisionProfile,
        quadrature: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_hilbert_typed_precision::<T>(precision)?;
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        let represented = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // Safety: T is f32, so &[T] is layout-compatible with &[f32].
            let slice_f32 = unsafe {
                std::slice::from_raw_parts(quadrature.as_ptr().cast::<f32>(), quadrature.len())
            };
            std::borrow::Cow::Borrowed(slice_f32)
        } else {
            let vec: Vec<f32> = quadrature.iter().map(|v| v.to_f64() as f32).collect();
            std::borrow::Cow::Owned(vec)
        };
        let computed = self.execute_inverse(plan, &represented)?;
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // Safety: T is f32, so &mut [T] is layout-compatible with &mut [f32].
            let slice_f32 = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<f32>(), output.len())
            };
            slice_f32.copy_from_slice(&computed);
        } else {
            for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
                *slot = T::from_f64(f64::from(value));
            }
        }
        Ok(())
    }

    /// Execute typed inverse Hilbert transform from a Leto quadrature view.
    pub fn execute_inverse_leto_typed<T: HilbertStorage>(
        &self,
        plan: &HilbertWgpuPlan,
        precision: PrecisionProfile,
        quadrature: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let quadrature = leto_view1_cow(quadrature);
        let mut output = vec![T::from_f64(0.0); plan.len()];
        self.execute_inverse_typed_into(plan, precision, &quadrature, &mut output)?;
        leto_array1_from_slice(&output)
    }

    fn validate_plan_input(plan: &HilbertWgpuPlan, input: &[f32]) -> WgpuResult<()> {
        let len = plan.len();
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be greater than zero"),
            });
        }
        if input.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input.len(),
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
        message: "failed to allocate Mnemosyne-backed Leto Hilbert output".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use leto::SliceArg;

    use super::leto_view1_cow;

    #[test]
    fn leto_view1_cow_borrows_contiguous_views() {
        let input = leto::Array1::from_shape_vec([4], vec![1.0_f32, 2.0, 3.0, 4.0]).expect("input");
        let cow = leto_view1_cow(input.view());
        assert!(matches!(cow, Cow::Borrowed(_)));
        assert_eq!(cow.as_ref(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn leto_view1_cow_materializes_strided_views() {
        let input =
            leto::Array1::from_shape_vec([8], vec![1.0_f32, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0, 99.0])
                .expect("input");
        let view = input
            .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
            .expect("strided view");
        let cow = leto_view1_cow(view);
        assert!(matches!(cow, Cow::Owned(_)));
        assert_eq!(cow.as_ref(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
