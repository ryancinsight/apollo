use apollo_fft::{GpuElement, GpuStorage, PrecisionProfile};
use leto::Array2;

use super::infrastructure::kernel::RadonGpuKernel as Kernel;
use super::{RadonWgpuBackend, RadonWgpuPlan, WgpuError, WgpuResult};

/// Projection surface of the Radon backend.
///
/// The forward direction projects an image into a sinogram over the
/// supplied projection angles; the inverse is the adjoint
/// backprojection (not an exact inversion), and filtered backprojection
/// applies the Ram-Lak ramp filter before the adjoint for approximate
/// CT reconstruction. Angles are operands, not plan state, so this
/// surface extends the scaffold rather than instantiating its slice
/// contract.
pub trait ProjectionExecution {
    /// Execute the forward parallel-beam Radon projection.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, length-mismatch, or
    /// provider failure.
    fn execute_forward(
        &self,
        plan: &RadonWgpuPlan,
        image: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>>;

    /// Execute the forward Radon projection from Leto image and angle
    /// views.
    ///
    /// Contiguous angle views are borrowed. Strided image or angle views
    /// are materialized once into logical order before dispatch.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, length-mismatch, or
    /// provider failure.
    fn execute_forward_leto(
        &self,
        plan: &RadonWgpuPlan,
        image: leto::ArrayView2<'_, f32>,
        angles: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>>;

    /// Execute the forward Radon projection from a flat typed image
    /// slice of `rows * cols` row-major elements.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, length-mismatch,
    /// precision-profile, or provider failure.
    fn execute_forward_flat_typed<T: GpuStorage>(
        &self,
        plan: &RadonWgpuPlan,
        precision: PrecisionProfile,
        flat_image: &[T],
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>>;

    /// Execute the forward Radon projection from a typed Leto image view.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, length-mismatch,
    /// precision-profile, or provider failure.
    fn execute_forward_leto_typed<T: GpuStorage>(
        &self,
        plan: &RadonWgpuPlan,
        precision: PrecisionProfile,
        image: leto::ArrayView2<'_, T>,
        angles: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>>;

    /// Execute the GPU adjoint backprojection (Radon adjoint operator).
    ///
    /// This is the adjoint of the forward projection, not an exact
    /// inversion; for approximate CT reconstruction use
    /// [`Self::execute_filtered_backproject`].
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, length-mismatch, or
    /// provider failure.
    fn execute_inverse(
        &self,
        plan: &RadonWgpuPlan,
        sinogram: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>>;

    /// Execute the adjoint backprojection from Leto sinogram and angle
    /// views.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, length-mismatch, or
    /// provider failure.
    fn execute_inverse_leto(
        &self,
        plan: &RadonWgpuPlan,
        sinogram: leto::ArrayView2<'_, f32>,
        angles: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>>;

    /// Execute the adjoint backprojection from a flat typed sinogram
    /// slice of `angle_count * detector_count` row-major elements.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, length-mismatch,
    /// precision-profile, or provider failure.
    fn execute_inverse_flat_typed<T: GpuStorage>(
        &self,
        plan: &RadonWgpuPlan,
        precision: PrecisionProfile,
        flat_sinogram: &[T],
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>>;

    /// Execute the adjoint backprojection from a typed Leto sinogram
    /// view.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, length-mismatch,
    /// precision-profile, or provider failure.
    fn execute_inverse_leto_typed<T: GpuStorage>(
        &self,
        plan: &RadonWgpuPlan,
        precision: PrecisionProfile,
        sinogram: leto::ArrayView2<'_, T>,
        angles: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>>;

    /// Execute GPU ramp-filtered backprojection (FBP).
    ///
    /// Two-pass GPU execution: the Ram-Lak ramp filter applied per
    /// projection row (circular convolution with `h = IFFT(R)`,
    /// `R[k] = 2*pi*|signed_k|/(N*delta)`; Bracewell & Riddle 1967),
    /// then adjoint backprojection of the filtered sinogram (Natterer
    /// 2001, section II.2), scaled by `pi / angle_count` to approximate
    /// the continuous FBP integral under uniform angular sampling.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, length-mismatch, or
    /// provider failure.
    fn execute_filtered_backproject(
        &self,
        plan: &RadonWgpuPlan,
        sinogram: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>>;

    /// Execute ramp-filtered backprojection from Leto sinogram and angle
    /// views.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, length-mismatch, or
    /// provider failure.
    fn execute_filtered_backproject_leto(
        &self,
        plan: &RadonWgpuPlan,
        sinogram: leto::ArrayView2<'_, f32>,
        angles: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>>;
}

impl ProjectionExecution for RadonWgpuBackend {
    fn execute_forward(
        &self,
        plan: &RadonWgpuPlan,
        image: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>> {
        validate_image_inputs(plan, image, angles)?;
        Kernel::execute_forward(self.device(), *plan.payload(), image, angles)
    }

    fn execute_forward_leto(
        &self,
        plan: &RadonWgpuPlan,
        image: leto::ArrayView2<'_, f32>,
        angles: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>> {
        let image = image.to_contiguous();
        let angles = apollo_leto_interop::view_cow(&angles);
        let output = self.execute_forward(plan, &image, &angles)?;
        apollo_leto_interop::try_dense_from_array(&output).ok_or_else(|| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto Radon forward sinogram".to_owned(),
        })
    }

    fn execute_forward_flat_typed<T: GpuStorage>(
        &self,
        plan: &RadonWgpuPlan,
        precision: PrecisionProfile,
        flat_image: &[T],
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>> {
        validate_profile::<T>(precision)?;
        let expected_len = plan.len();
        if flat_image.len() != expected_len {
            return Err(WgpuError::ShapeMismatch {
                message: format!(
                    "image shape mismatch: expected {}x{}, got {}x1",
                    plan.payload().rows(),
                    plan.payload().cols(),
                    flat_image.len()
                ),
            });
        }
        let image = promote_to_matrix(
            flat_image,
            [plan.payload().rows(), plan.payload().cols()],
            "flat image reshape failed",
        )?;
        self.execute_forward(plan, &image, angles)
    }

    fn execute_forward_leto_typed<T: GpuStorage>(
        &self,
        plan: &RadonWgpuPlan,
        precision: PrecisionProfile,
        image: leto::ArrayView2<'_, T>,
        angles: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>> {
        let image = image.to_contiguous();
        let flat = image.iter().copied().collect::<Vec<_>>();
        let angles = apollo_leto_interop::view_cow(&angles);
        let output = self.execute_forward_flat_typed(plan, precision, &flat, &angles)?;
        apollo_leto_interop::try_dense_from_array(&output).ok_or_else(|| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto Radon typed forward sinogram"
                .to_owned(),
        })
    }

    fn execute_inverse(
        &self,
        plan: &RadonWgpuPlan,
        sinogram: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>> {
        validate_sinogram_inputs(plan, sinogram, angles)?;
        Kernel::execute_backproject(self.device(), *plan.payload(), sinogram, angles)
    }

    fn execute_inverse_leto(
        &self,
        plan: &RadonWgpuPlan,
        sinogram: leto::ArrayView2<'_, f32>,
        angles: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>> {
        let sinogram = sinogram.to_contiguous();
        let angles = apollo_leto_interop::view_cow(&angles);
        let output = self.execute_inverse(plan, &sinogram, &angles)?;
        apollo_leto_interop::try_dense_from_array(&output).ok_or_else(|| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto Radon backprojection image"
                .to_owned(),
        })
    }

    fn execute_inverse_flat_typed<T: GpuStorage>(
        &self,
        plan: &RadonWgpuPlan,
        precision: PrecisionProfile,
        flat_sinogram: &[T],
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>> {
        validate_profile::<T>(precision)?;
        let expected_len = plan.output_len();
        if flat_sinogram.len() != expected_len {
            return Err(WgpuError::ShapeMismatch {
                message: format!(
                    "sinogram shape mismatch: expected {}x{}, got {}x1",
                    plan.payload().angle_count(),
                    plan.payload().detector_count(),
                    flat_sinogram.len()
                ),
            });
        }
        let sinogram = promote_to_matrix(
            flat_sinogram,
            [
                plan.payload().angle_count(),
                plan.payload().detector_count(),
            ],
            "flat sinogram reshape failed",
        )?;
        self.execute_inverse(plan, &sinogram, angles)
    }

    fn execute_inverse_leto_typed<T: GpuStorage>(
        &self,
        plan: &RadonWgpuPlan,
        precision: PrecisionProfile,
        sinogram: leto::ArrayView2<'_, T>,
        angles: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>> {
        let sinogram = sinogram.to_contiguous();
        let flat = sinogram.iter().copied().collect::<Vec<_>>();
        let angles = apollo_leto_interop::view_cow(&angles);
        let output = self.execute_inverse_flat_typed(plan, precision, &flat, &angles)?;
        apollo_leto_interop::try_dense_from_array(&output).ok_or_else(|| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto Radon typed backprojection image"
                .to_owned(),
        })
    }

    fn execute_filtered_backproject(
        &self,
        plan: &RadonWgpuPlan,
        sinogram: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>> {
        validate_sinogram_inputs(plan, sinogram, angles)?;
        Kernel::execute_filtered_backproject(self.device(), *plan.payload(), sinogram, angles)
    }

    fn execute_filtered_backproject_leto(
        &self,
        plan: &RadonWgpuPlan,
        sinogram: leto::ArrayView2<'_, f32>,
        angles: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>> {
        let sinogram = sinogram.to_contiguous();
        let angles = apollo_leto_interop::view_cow(&angles);
        let output = self.execute_filtered_backproject(plan, &sinogram, &angles)?;
        apollo_leto_interop::try_dense_from_array(&output).ok_or_else(|| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto Radon filtered backprojection image"
                .to_owned(),
        })
    }
}

fn validate_profile<T: GpuStorage>(precision: PrecisionProfile) -> WgpuResult<()> {
    let expected = T::PROFILE;
    if precision.storage != expected.storage || precision.compute != expected.compute {
        return Err(WgpuError::InvalidPrecisionProfile);
    }
    Ok(())
}

/// Promote typed storage into an f32 matrix without the double-rounding
/// or per-call allocation the drifted copies carried.
fn promote_to_matrix<T: GpuStorage>(
    flat: &[T],
    shape: [usize; 2],
    context: &str,
) -> WgpuResult<Array2<f32>> {
    let promoted = f32::with_input_scratch(flat.len(), |represented| {
        for (slot, value) in represented.iter_mut().zip(flat.iter().copied()) {
            *slot = value.to_gpu();
        }
        represented.to_vec()
    });
    Array2::from_shape_vec(shape, promoted).map_err(|_| WgpuError::InvalidPlan {
        message: context.to_owned(),
    })
}

fn validate_image_inputs(
    plan: &RadonWgpuPlan,
    image: &Array2<f32>,
    angles: &[f32],
) -> WgpuResult<()> {
    plan.validate()?;
    let [actual_rows, actual_cols] = image.shape();
    if (actual_rows, actual_cols) != (plan.payload().rows(), plan.payload().cols()) {
        return Err(WgpuError::ShapeMismatch {
            message: format!(
                "image expected {}x{}, got {}x{}",
                plan.payload().rows(),
                plan.payload().cols(),
                actual_rows,
                actual_cols
            ),
        });
    }
    require_angles(plan, angles)
}

fn validate_sinogram_inputs(
    plan: &RadonWgpuPlan,
    sinogram: &Array2<f32>,
    angles: &[f32],
) -> WgpuResult<()> {
    plan.validate()?;
    let [actual_angles, actual_detectors] = sinogram.shape();
    if (actual_angles, actual_detectors)
        != (
            plan.payload().angle_count(),
            plan.payload().detector_count(),
        )
    {
        return Err(WgpuError::ShapeMismatch {
            message: format!(
                "sinogram expected {}x{}, got {}x{}",
                plan.payload().angle_count(),
                plan.payload().detector_count(),
                actual_angles,
                actual_detectors
            ),
        });
    }
    require_angles(plan, angles)
}

fn require_angles(plan: &RadonWgpuPlan, angles: &[f32]) -> WgpuResult<()> {
    if angles.len() != plan.payload().angle_count() {
        return Err(WgpuError::LengthMismatch {
            expected: plan.payload().angle_count(),
            actual: angles.len(),
        });
    }
    Ok(())
}
