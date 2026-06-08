//! WGPU device acquisition for this transform backend.

use std::sync::Arc;

use apollo_fft::PrecisionProfile;
use apollo_radon::RadonStorage;
use ndarray::Array2;

use crate::application::plan::RadonWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::RadonGpuKernel;
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    RadonWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct RadonWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<RadonGpuKernel>,
}

impl RadonWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(RadonGpuKernel::new(device.inner()));
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-radon-wgpu")?)
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::forward_inverse_and_fbp(true)
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
        rows: usize,
        cols: usize,
        angle_count: usize,
        detector_count: usize,
        detector_spacing: f64,
    ) -> RadonWgpuPlan {
        RadonWgpuPlan::new(
            rows,
            cols,
            angle_count,
            detector_count,
            detector_spacing.to_bits(),
        )
    }

    /// Execute the forward parallel-beam Radon projection.
    pub fn execute_forward(
        &self,
        plan: &RadonWgpuPlan,
        image: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>> {
        Self::validate_inputs(plan, image, angles)?;
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            plan,
            image,
            angles,
        )
    }

    /// Execute the GPU adjoint backprojection (Radon adjoint operator).
    ///
    /// Returns the backprojected image as `Array2<f32>` of shape `(rows, cols)`.
    /// This is the adjoint of `execute_forward`, not an exact inversion.
    /// For approximate CT reconstruction, apply filtered backprojection via the CPU plan.
    pub fn execute_inverse(
        &self,
        plan: &RadonWgpuPlan,
        sinogram: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>> {
        Self::validate_sinogram_inputs(plan, sinogram, angles)?;
        self.kernel.execute_backproject(
            self.device.inner(),
            self.device.queue().as_ref(),
            plan,
            sinogram,
            angles,
        )
    }

    /// Execute GPU adjoint backprojection from a flat typed sinogram slice.
    ///
    /// `flat_sinogram` must have exactly `plan.angle_count() * plan.detector_count()`
    /// elements in row-major order. Promotes represented input once to `f32`.
    pub fn execute_inverse_flat_typed<T: RadonStorage>(
        &self,
        plan: &RadonWgpuPlan,
        precision: PrecisionProfile,
        flat_sinogram: &[T],
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        let expected_len = plan.angle_count() * plan.detector_count();
        if flat_sinogram.len() != expected_len {
            return Err(WgpuError::ShapeMismatch {
                message: format!(
                    "sinogram shape mismatch: expected {}x{}, got {}x1",
                    plan.angle_count(),
                    plan.detector_count(),
                    flat_sinogram.len()
                ),
            });
        }
        let promoted: Vec<f32> = flat_sinogram.iter().map(|v| v.to_f64() as f32).collect();
        let sinogram_2d =
            Array2::from_shape_vec((plan.angle_count(), plan.detector_count()), promoted).map_err(
                |_| WgpuError::InvalidPlan {
                        message: format!("invalid plan rows={}, cols={}, angles={}, detectors={}, spacing={}: flat sinogram reshape failed", plan.rows(), plan.cols(), plan.angle_count(), plan.detector_count(), plan.detector_spacing()),
                    },
            )?;
        self.execute_inverse(plan, &sinogram_2d, angles)
    }

    /// Execute GPU ramp-filtered backprojection (FBP).
    ///
    /// Two-pass GPU execution:
    /// 1. Ram-Lak ramp filter applied to each sinogram projection row (circular convolution
    ///    with h = IFFT(R), `R[k] = 2π·|signed_k|/(N·Δ)`; Bracewell & Riddle 1967).
    /// 2. Adjoint backprojection of the filtered sinogram (Natterer 2001, §II.2).
    ///
    /// Result is scaled by π / angle_count to approximate the continuous FBP integral
    /// under uniform angular sampling (Fourier slice theorem limit).
    pub fn execute_filtered_backproject(
        &self,
        plan: &RadonWgpuPlan,
        sinogram: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>> {
        Self::validate_sinogram_inputs(plan, sinogram, angles)?;
        self.kernel.execute_filtered_backproject(
            self.device.inner(),
            self.device.queue().as_ref(),
            plan,
            sinogram,
            angles,
        )
    }

    /// Execute the forward Radon projection from a flat typed image slice.
    ///
    /// `flat_image` must have exactly `plan.rows() * plan.cols()` elements in
    /// row-major order. Promotes represented input once to `f32` before dispatch.
    pub fn execute_forward_flat_typed<T: RadonStorage>(
        &self,
        plan: &RadonWgpuPlan,
        precision: PrecisionProfile,
        flat_image: &[T],
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        let expected_len = plan.rows() * plan.cols();
        if flat_image.len() != expected_len {
            return Err(WgpuError::ShapeMismatch {
                message: format!(
                    "image shape mismatch: expected {}x{}, got {}x1",
                    plan.rows(),
                    plan.cols(),
                    flat_image.len()
                ),
            });
        }
        let promoted: Vec<f32> = flat_image.iter().map(|v| v.to_f64() as f32).collect();
        let image_2d =
            Array2::from_shape_vec((plan.rows(), plan.cols()), promoted).map_err(|_| {
                WgpuError::InvalidPlan {
                        message: format!("invalid plan rows={}, cols={}, angles={}, detectors={}, spacing={}: flat image reshape failed", plan.rows(), plan.cols(), plan.angle_count(), plan.detector_count(), plan.detector_spacing()),
                    }
            })?;
        self.execute_forward(plan, &image_2d, angles)
    }

    fn validate_sinogram_inputs(
        plan: &RadonWgpuPlan,
        sinogram: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<()> {
        if plan.rows() == 0
            || plan.cols() == 0
            || plan.angle_count() == 0
            || plan.detector_count() == 0
        {
            return Err(WgpuError::InvalidPlan {
                    message: format!("invalid plan rows={}, cols={}, angles={}, detectors={}, spacing={}: geometry dimensions must be greater than zero", plan.rows(), plan.cols(), plan.angle_count(), plan.detector_count(), plan.detector_spacing()),
                });
        }
        if !plan.detector_spacing().is_finite() || plan.detector_spacing() <= 0.0 {
            return Err(WgpuError::InvalidPlan {
                    message: format!("invalid plan rows={}, cols={}, angles={}, detectors={}, spacing={}: detector spacing must be finite and positive", plan.rows(), plan.cols(), plan.angle_count(), plan.detector_count(), plan.detector_spacing()),
                });
        }
        let (actual_angles, actual_detectors) = sinogram.dim();
        if (actual_angles, actual_detectors) != (plan.angle_count(), plan.detector_count()) {
            return Err(WgpuError::ShapeMismatch {
                message: format!(
                    "sinogram expected {}x{}, got {}x{}",
                    plan.angle_count(),
                    plan.detector_count(),
                    actual_angles,
                    actual_detectors
                ),
            });
        }
        if angles.len() != plan.angle_count() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.angle_count(),
                actual: angles.len(),
            });
        }
        Ok(())
    }

    fn validate_inputs(
        plan: &RadonWgpuPlan,
        image: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<()> {
        if plan.rows() == 0
            || plan.cols() == 0
            || plan.angle_count() == 0
            || plan.detector_count() == 0
        {
            return Err(WgpuError::InvalidPlan {
                    message: format!("invalid plan rows={}, cols={}, angles={}, detectors={}, spacing={}: geometry dimensions must be greater than zero", plan.rows(), plan.cols(), plan.angle_count(), plan.detector_count(), plan.detector_spacing()),
                });
        }
        if !plan.detector_spacing().is_finite() || plan.detector_spacing() <= 0.0 {
            return Err(WgpuError::InvalidPlan {
                    message: format!("invalid plan rows={}, cols={}, angles={}, detectors={}, spacing={}: detector spacing must be finite and positive", plan.rows(), plan.cols(), plan.angle_count(), plan.detector_count(), plan.detector_spacing()),
                });
        }
        let (actual_rows, actual_cols) = image.dim();
        if (actual_rows, actual_cols) != (plan.rows(), plan.cols()) {
            return Err(WgpuError::ShapeMismatch {
                message: format!(
                    "image expected {}x{}, got {}x{}",
                    plan.rows(),
                    plan.cols(),
                    actual_rows,
                    actual_cols
                ),
            });
        }
        if angles.len() != plan.angle_count() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.angle_count(),
                actual: angles.len(),
            });
        }
        Ok(())
    }
}
