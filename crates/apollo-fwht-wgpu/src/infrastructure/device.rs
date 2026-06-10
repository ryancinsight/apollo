//! WGPU device acquisition for this transform backend.

use std::borrow::Cow;
use std::sync::Arc;

use apollo_fft::PrecisionProfile;
use apollo_fwht::FwhtStorage;

use crate::application::plan::FwhtWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::FwhtGpuKernel;
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    FwhtWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct FwhtWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<FwhtGpuKernel>,
}

impl FwhtWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(FwhtGpuKernel::new(device.inner())?);
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-fwht-wgpu")?)
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::implemented(true)
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
    pub const fn plan(&self, len: usize) -> FwhtWgpuPlan {
        FwhtWgpuPlan::new(len)
    }

    /// Execute the unnormalized forward 1D FWHT for a real-valued `f32` signal.
    pub fn execute_forward(&self, plan: &FwhtWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        Self::validate_plan_input(plan, input)?;
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            input,
            false,
        )
    }

    /// Execute the unnormalized forward 1D FWHT from a Leto `f32` view.
    ///
    /// Contiguous Leto views borrow host storage directly; strided views copy
    /// once into logical order before dispatching to the existing WGPU slice path.
    pub fn execute_forward_leto(
        &self,
        plan: &FwhtWgpuPlan,
        input: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let input = leto_view1_cow(input)?;
        let output = self.execute_forward(plan, &input)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the unnormalized forward 1D FWHT with caller-owned typed storage.
    ///
    /// WGPU arithmetic remains `f32`; mixed `f16` storage is promoted once to
    /// represented `f32` before dispatch and quantized at the output boundary.
    pub fn execute_forward_typed_into<T: FwhtStorage>(
        &self,
        plan: &FwhtWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_typed_plan_input::<T>(plan, precision, input, output)?;
        let represented = typed_to_f32(input);
        let computed = self.execute_forward(plan, &represented)?;
        write_typed_output(&computed, output);
        Ok(())
    }

    /// Execute the unnormalized forward 1D FWHT from typed Leto storage.
    ///
    /// Precision-profile validation and host quantization match
    /// [`Self::execute_forward_typed_into`].
    pub fn execute_forward_leto_typed<T: FwhtStorage>(
        &self,
        plan: &FwhtWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = leto_view1_cow(input)?;
        let mut output = vec![T::from_f64(0.0); plan.len()];
        self.execute_forward_typed_into(plan, precision, &input, &mut output)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the normalized inverse 1D FWHT for a real-valued `f32` spectrum.
    pub fn execute_inverse(&self, plan: &FwhtWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        Self::validate_plan_input(plan, input)?;
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            input,
            true,
        )
    }

    /// Execute the normalized inverse 1D FWHT from a Leto `f32` view.
    ///
    /// Output storage is Mnemosyne-backed Leto host memory.
    pub fn execute_inverse_leto(
        &self,
        plan: &FwhtWgpuPlan,
        input: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let input = leto_view1_cow(input)?;
        let output = self.execute_inverse(plan, &input)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the normalized inverse 1D FWHT with caller-owned typed storage.
    pub fn execute_inverse_typed_into<T: FwhtStorage>(
        &self,
        plan: &FwhtWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_typed_plan_input::<T>(plan, precision, input, output)?;
        let represented = typed_to_f32(input);
        let computed = self.execute_inverse(plan, &represented)?;
        write_typed_output(&computed, output);
        Ok(())
    }

    /// Execute the normalized inverse 1D FWHT from typed Leto storage.
    pub fn execute_inverse_leto_typed<T: FwhtStorage>(
        &self,
        plan: &FwhtWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = leto_view1_cow(input)?;
        let mut output = vec![T::from_f64(0.0); plan.len()];
        self.execute_inverse_typed_into(plan, precision, &input, &mut output)?;
        leto_array1_from_slice(&output)
    }

    fn validate_plan_input(plan: &FwhtWgpuPlan, input: &[f32]) -> WgpuResult<()> {
        let len = plan.len();
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be greater than zero"),
            });
        }
        if !len.is_power_of_two() {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be a power of two"),
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

    fn validate_typed_plan_input<T: FwhtStorage>(
        plan: &FwhtWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &[T],
    ) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        let len = plan.len();
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be greater than zero"),
            });
        }
        if !len.is_power_of_two() {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be a power of two"),
            });
        }
        if input.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input.len(),
            });
        }
        if output.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: output.len(),
            });
        }
        Ok(())
    }
}

fn typed_to_f32<T: FwhtStorage>(input: &[T]) -> Cow<'_, [f32]> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        // Safety: T is f32, so &[T] is layout-compatible with &[f32].
        let slice_f32 =
            unsafe { std::slice::from_raw_parts(input.as_ptr().cast::<f32>(), input.len()) };
        Cow::Borrowed(slice_f32)
    } else {
        let vec: Vec<f32> = input.iter().map(|value| value.to_f64() as f32).collect();
        Cow::Owned(vec)
    }
}

fn write_typed_output<T: FwhtStorage>(source: &[f32], output: &mut [T]) {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        // Safety: T is f32, so &mut [T] is layout-compatible with &mut [f32].
        let slice_f32 = unsafe {
            std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<f32>(), output.len())
        };
        slice_f32.copy_from_slice(source);
    } else {
        for (slot, value) in output.iter_mut().zip(source.iter().copied()) {
            *slot = T::from_f64(f64::from(value));
        }
    }
}

fn leto_view1_cow<T: Copy>(view: leto::ArrayView1<'_, T>) -> WgpuResult<Cow<'_, [T]>> {
    if let Some(slice) = view.as_slice() {
        return Ok(Cow::Borrowed(slice));
    }

    let mut values = Vec::with_capacity(view.size());
    for index in 0..view.size() {
        let value = view.get([index]).map_err(|_| WgpuError::LengthMismatch {
            expected: view.size(),
            actual: index,
        })?;
        values.push(*value);
    }
    Ok(Cow::Owned(values))
}

fn leto_array1_from_slice<T: Copy>(
    values: &[T],
) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    leto::Array::from_mnemosyne_slice([values.len()], values).map_err(|err| {
        WgpuError::InvalidPlan {
            message: format!("failed to allocate Mnemosyne-backed Leto output: {err}"),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::leto_view1_cow;

    #[test]
    fn leto_view1_cow_borrows_contiguous_views() {
        let input =
            leto::Array1::from_shape_vec([4], vec![1.0_f32, 2.0, 3.0, 4.0]).expect("leto input");
        let cow = leto_view1_cow(input.view()).expect("contiguous view");
        assert!(matches!(cow, std::borrow::Cow::Borrowed(_)));
        assert_eq!(&*cow, &[1.0_f32, 2.0, 3.0, 4.0]);
    }
}
