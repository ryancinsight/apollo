//! WGPU device acquisition and GFT execution backend.

use std::{borrow::Cow, sync::Arc};

use apollo_fft::PrecisionProfile;
use apollo_gft::GftStorage;

use crate::application::plan::GftWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::GftGpuKernel;
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    GftWgpuBackend::try_default().is_ok()
}

/// WGPU backend for GFT execution.
#[derive(Debug, Clone)]
pub struct GftWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<GftGpuKernel>,
}

impl GftWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(GftGpuKernel::new(device.inner()));
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-gft-wgpu")?)
    }

    /// Return truthful current capabilities (forward and inverse both implemented).
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

    /// Create a metadata plan descriptor.
    #[must_use]
    pub const fn plan(&self, len: usize) -> GftWgpuPlan {
        GftWgpuPlan::new(len)
    }

    /// Execute the forward GFT: `X[k] = sum_i U[i,k] * signal[i]`  (U^T x).
    ///
    /// Requires signal.len() == plan.len() and basis.len() == len*len.
    pub fn execute_forward(
        &self,
        plan: &GftWgpuPlan,
        signal: &[f32],
        basis: &[f32],
    ) -> WgpuResult<Vec<f32>> {
        Self::validate(plan, signal, basis)?;
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            signal,
            basis,
            plan.len(),
            0,
        )
    }

    /// Execute the forward GFT from Leto host views.
    ///
    /// Contiguous views are borrowed without copying. Strided views are
    /// materialized once into logical order before GPU upload.
    pub fn execute_forward_leto(
        &self,
        plan: &GftWgpuPlan,
        signal: leto::ArrayView1<'_, f32>,
        basis: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let signal = leto_view1_cow(signal)?;
        let basis = leto_view1_cow(basis)?;
        let output = self.execute_forward(plan, &signal, &basis)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the inverse GFT: `x[i] = sum_k U[i,k] * spectrum[k]`  (U X).
    ///
    /// Requires spectrum.len() == plan.len() and basis.len() == len*len.
    pub fn execute_inverse(
        &self,
        plan: &GftWgpuPlan,
        spectrum: &[f32],
        basis: &[f32],
    ) -> WgpuResult<Vec<f32>> {
        Self::validate(plan, spectrum, basis)?;
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            spectrum,
            basis,
            plan.len(),
            1,
        )
    }

    /// Execute the inverse GFT from Leto host views.
    pub fn execute_inverse_leto(
        &self,
        plan: &GftWgpuPlan,
        spectrum: leto::ArrayView1<'_, f32>,
        basis: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let spectrum = leto_view1_cow(spectrum)?;
        let basis = leto_view1_cow(basis)?;
        let output = self.execute_inverse(plan, &spectrum, &basis)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the forward GFT with typed `f64`, `f32`, or mixed `f16` storage.
    ///
    /// The graph basis matrix must always be supplied as `f32`.
    pub fn execute_forward_typed_into<T: GftStorage>(
        &self,
        plan: &GftWgpuPlan,
        precision: PrecisionProfile,
        signal: &[T],
        basis: &[f32],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_gft_typed_precision::<T>(precision)?;
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        let represented = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // Safety: T is f32, so &[T] is layout-compatible with &[f32].
            let slice_f32 =
                unsafe { std::slice::from_raw_parts(signal.as_ptr().cast::<f32>(), signal.len()) };
            std::borrow::Cow::Borrowed(slice_f32)
        } else {
            let vec: Vec<f32> = signal.iter().map(|v| v.to_f64() as f32).collect();
            std::borrow::Cow::Owned(vec)
        };
        let computed = self.execute_forward(plan, &represented, basis)?;
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

    /// Execute typed forward GFT from Leto host views.
    ///
    /// The graph basis matrix must always be supplied as a logical `f32` Leto
    /// 1D view in row-major flattened order.
    pub fn execute_forward_leto_typed<T: GftStorage>(
        &self,
        plan: &GftWgpuPlan,
        precision: PrecisionProfile,
        signal: leto::ArrayView1<'_, T>,
        basis: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let signal = leto_view1_cow(signal)?;
        let basis = leto_view1_cow(basis)?;
        let mut output = vec![T::from_f64(0.0); plan.len()];
        self.execute_forward_typed_into(plan, precision, &signal, &basis, &mut output)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the inverse GFT with typed `f64`, `f32`, or mixed `f16` storage.
    pub fn execute_inverse_typed_into<T: GftStorage>(
        &self,
        plan: &GftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: &[T],
        basis: &[f32],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_gft_typed_precision::<T>(precision)?;
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        let represented = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // Safety: T is f32, so &[T] is layout-compatible with &[f32].
            let slice_f32 = unsafe {
                std::slice::from_raw_parts(spectrum.as_ptr().cast::<f32>(), spectrum.len())
            };
            std::borrow::Cow::Borrowed(slice_f32)
        } else {
            let vec: Vec<f32> = spectrum.iter().map(|v| v.to_f64() as f32).collect();
            std::borrow::Cow::Owned(vec)
        };
        let computed = self.execute_inverse(plan, &represented, basis)?;
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

    /// Execute typed inverse GFT from Leto host views.
    pub fn execute_inverse_leto_typed<T: GftStorage>(
        &self,
        plan: &GftWgpuPlan,
        precision: PrecisionProfile,
        spectrum: leto::ArrayView1<'_, T>,
        basis: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let spectrum = leto_view1_cow(spectrum)?;
        let basis = leto_view1_cow(basis)?;
        let mut output = vec![T::from_f64(0.0); plan.len()];
        self.execute_inverse_typed_into(plan, precision, &spectrum, &basis, &mut output)?;
        leto_array1_from_slice(&output)
    }

    fn validate_gft_typed_precision<T: GftStorage>(precision: PrecisionProfile) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Ok(())
    }

    fn validate(plan: &GftWgpuPlan, signal: &[f32], basis: &[f32]) -> WgpuResult<()> {
        let n = plan.len();
        if n == 0 {
            return Err(WgpuError::InvalidPlan {
                message: "invalid plan: length must be greater than zero".to_owned(),
            });
        }
        if signal.len() != n {
            return Err(WgpuError::LengthMismatch {
                expected: n,
                actual: signal.len(),
            });
        }
        if basis.len() != n * n {
            return Err(WgpuError::ShapeMismatch {
                message: format!(
                    "basis length mismatch: expected={}, actual={}",
                    n * n,
                    basis.len()
                ),
            });
        }
        Ok(())
    }
}

fn leto_view1_cow<T: Copy>(view: leto::ArrayView1<'_, T>) -> WgpuResult<Cow<'_, [T]>> {
    if let Some(slice) = view.as_slice() {
        return Ok(Cow::Borrowed(slice));
    }
    let len = view.shape()[0];
    let mut values = Vec::with_capacity(len);
    for index in 0..len {
        values.push(*view.get([index]).map_err(|err| WgpuError::ShapeMismatch {
            message: format!("invalid Leto GFT 1D view: {err:?}"),
        })?);
    }
    Ok(Cow::Owned(values))
}

fn leto_array1_from_slice<T: Copy>(
    values: &[T],
) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    leto::Array::from_mnemosyne_slice([values.len()], values).map_err(|err| {
        WgpuError::InvalidPlan {
            message: format!("failed to allocate Mnemosyne-backed Leto GFT output: {err:?}"),
        }
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
        let cow = leto_view1_cow(input.view()).expect("contiguous view");
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
        let cow = leto_view1_cow(view).expect("strided view");
        assert!(matches!(cow, Cow::Owned(_)));
        assert_eq!(cow.as_ref(), &[1, 2, 3, 4]);
    }
}
