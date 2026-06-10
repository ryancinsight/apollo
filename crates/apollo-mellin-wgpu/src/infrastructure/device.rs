//! WGPU device acquisition for this transform backend.

use std::{borrow::Cow, sync::Arc};

use apollo_fft::PrecisionProfile;
use apollo_mellin::MellinStorage;
use num_complex::Complex32;

use crate::application::plan::MellinWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::MellinGpuKernel;
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    MellinWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct MellinWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<MellinGpuKernel>,
}

impl MellinWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(MellinGpuKernel::new(device.inner()));
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-mellin-wgpu")?)
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
    pub fn plan(&self, samples: usize, min_scale: f64, max_scale: f64) -> MellinWgpuPlan {
        MellinWgpuPlan::new(samples, min_scale.to_bits(), max_scale.to_bits())
    }

    /// Execute the forward Mellin log-frequency spectrum for a real-valued `f32` signal.
    pub fn execute_forward(
        &self,
        plan: &MellinWgpuPlan,
        signal: &[f32],
        signal_min: f64,
        signal_max: f64,
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_inputs(plan, signal, signal_min, signal_max)?;
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            plan,
            signal,
            signal_min,
            signal_max,
        )
    }

    /// Execute the forward Mellin spectrum from a Leto real-valued host view.
    ///
    /// Contiguous views are borrowed without copying. Strided views are
    /// materialized once into logical order before GPU upload.
    pub fn execute_forward_leto(
        &self,
        plan: &MellinWgpuPlan,
        signal: leto::ArrayView1<'_, f32>,
        signal_min: f64,
        signal_max: f64,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let signal = leto_view1_cow(signal)?;
        let output = self.execute_forward(plan, &signal, signal_min, signal_max)?;
        leto_array1_from_slice(&output)
    }

    /// Inverse or adjoint execution is unsupported until the owning Mellin crate defines it.
    pub fn execute_inverse(
        &self,
        plan: &MellinWgpuPlan,
        spectrum: &[Complex32],
        out_min: f64,
        out_max: f64,
        out_len: usize,
    ) -> WgpuResult<Vec<f32>> {
        if plan.samples() == 0 || out_len == 0 {
            return Err(WgpuError::LengthMismatch {
                expected: 1,
                actual: 0,
            });
        }
        if spectrum.len() != plan.samples() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.samples(),
                actual: spectrum.len(),
            });
        }
        if !out_min.is_finite() || !out_max.is_finite() || out_min <= 0.0 || out_max <= 0.0 {
            return Err(WgpuError::InvalidSignalDomain {
                    message: format!("invalid signal domain min={out_min}, max={out_max}: output domain bounds must be finite and positive"),
                });
        }
        if out_min >= out_max {
            return Err(WgpuError::InvalidSignalDomain {
                    message: format!("invalid signal domain min={out_min}, max={out_max}: out_min must be less than out_max"),
                });
        }
        self.kernel.execute_inverse(
            self.device.inner(),
            self.device.queue().as_ref(),
            plan,
            spectrum,
            out_min,
            out_max,
            out_len,
        )
    }

    /// Execute the inverse Mellin path from a Leto spectrum host view.
    pub fn execute_inverse_leto(
        &self,
        plan: &MellinWgpuPlan,
        spectrum: leto::ArrayView1<'_, Complex32>,
        out_min: f64,
        out_max: f64,
        out_len: usize,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let spectrum = leto_view1_cow(spectrum)?;
        let output = self.execute_inverse(plan, &spectrum, out_min, out_max, out_len)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the forward Mellin transform with typed `f64`, `f32`, or mixed `f16` input storage.
    ///
    /// Promotes represented input once to `f32` before dispatch.
    /// Returns the log-frequency spectrum as `Vec<Complex32>`.
    pub fn execute_forward_typed<T: MellinStorage>(
        &self,
        plan: &MellinWgpuPlan,
        precision: PrecisionProfile,
        signal: &[T],
        signal_min: f64,
        signal_max: f64,
    ) -> WgpuResult<Vec<Complex32>> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        let represented: Vec<f32> = signal.iter().map(|v| v.to_f64() as f32).collect();
        self.execute_forward(plan, &represented, signal_min, signal_max)
    }

    /// Execute the forward Mellin spectrum from a typed Leto host view.
    pub fn execute_forward_leto_typed<T: MellinStorage>(
        &self,
        plan: &MellinWgpuPlan,
        precision: PrecisionProfile,
        signal: leto::ArrayView1<'_, T>,
        signal_min: f64,
        signal_max: f64,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let signal = leto_view1_cow(signal)?;
        let output =
            self.execute_forward_typed(plan, precision, &signal, signal_min, signal_max)?;
        leto_array1_from_slice(&output)
    }

    fn validate_inputs(
        plan: &MellinWgpuPlan,
        signal: &[f32],
        signal_min: f64,
        signal_max: f64,
    ) -> WgpuResult<()> {
        if plan.samples() == 0 {
            return Err(WgpuError::InvalidPlan {
                    message: format!("invalid plan samples={}, min_scale={}, max_scale={}: sample count must be greater than zero", plan.samples(), plan.min_scale(), plan.max_scale()),
                });
        }
        if !plan.min_scale().is_finite()
            || !plan.max_scale().is_finite()
            || plan.min_scale() <= 0.0
            || plan.max_scale() <= 0.0
        {
            return Err(WgpuError::InvalidPlan {
                    message: format!("invalid plan samples={}, min_scale={}, max_scale={}: plan scales must be finite and positive", plan.samples(), plan.min_scale(), plan.max_scale()),
                });
        }
        if plan.min_scale() >= plan.max_scale() {
            return Err(WgpuError::InvalidPlan {
                    message: format!("invalid plan samples={}, min_scale={}, max_scale={}: min_scale must be less than max_scale", plan.samples(), plan.min_scale(), plan.max_scale()),
                });
        }
        if signal.is_empty() {
            return Err(WgpuError::LengthMismatch {
                expected: 1,
                actual: 0,
            });
        }
        if !signal_min.is_finite()
            || !signal_max.is_finite()
            || signal_min <= 0.0
            || signal_max <= 0.0
        {
            return Err(WgpuError::InvalidSignalDomain {
                message: format!(
                    "signal bounds must be finite and positive: min={signal_min}, max={signal_max}"
                ),
            });
        }
        if signal_min >= signal_max {
            return Err(WgpuError::InvalidSignalDomain {
                message: format!(
                    "signal_min must be less than signal_max: min={signal_min}, max={signal_max}"
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
            message: format!("invalid Leto Mellin 1D view: {err:?}"),
        })?);
    }
    Ok(Cow::Owned(values))
}

fn leto_array1_from_slice<T: Copy>(
    values: &[T],
) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    leto::Array::from_mnemosyne_slice([values.len()], values).map_err(|err| {
        WgpuError::InvalidPlan {
            message: format!("failed to allocate Mnemosyne-backed Leto Mellin output: {err:?}"),
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
        let input = leto::Array1::from_shape_vec([4], vec![1.0_f32, 2.0, 3.0, 4.0]).expect("input");
        let cow = leto_view1_cow(input.view()).expect("contiguous view");
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
        let cow = leto_view1_cow(view).expect("strided view");
        assert!(matches!(cow, Cow::Owned(_)));
        assert_eq!(cow.as_ref(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
