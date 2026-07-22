//! Reusable Radon transform plan.

use crate::domain::contracts::error::{RadonError, RadonResult};
use crate::domain::geometry::parallel_beam::ParallelBeamGeometry;
use crate::domain::projection::sinogram::Sinogram;
use crate::infrastructure::kernel::direct::{
    adjoint_backproject, adjoint_backproject_into, forward_project, forward_project_into,
};
use crate::infrastructure::kernel::filter::ramp_filter_projection_into;
use apollo_fft::{f16, PrecisionProfile};
use leto::Array2;

/// Reusable 2D parallel-beam Radon plan.
#[derive(Debug, Clone, PartialEq)]
pub struct RadonPlan {
    geometry: ParallelBeamGeometry,
}

impl RadonPlan {
    /// Create a plan from validated geometry parameters.
    pub fn new(
        rows: usize,
        cols: usize,
        angles: Vec<f64>,
        detector_count: usize,
        detector_spacing: f64,
    ) -> RadonResult<Self> {
        Ok(Self {
            geometry: ParallelBeamGeometry::new(
                rows,
                cols,
                angles,
                detector_count,
                detector_spacing,
            )?,
        })
    }

    /// Create a plan from an existing geometry value.
    #[must_use]
    pub const fn from_geometry(geometry: ParallelBeamGeometry) -> Self {
        Self { geometry }
    }

    /// Borrow the plan geometry.
    #[must_use]
    pub const fn geometry(&self) -> &ParallelBeamGeometry {
        &self.geometry
    }

    /// Execute the forward Radon transform.
    pub fn forward(&self, image: &Array2<f64>) -> RadonResult<Sinogram> {
        if image.shape() != [self.geometry.rows(), self.geometry.cols()] {
            return Err(RadonError::ImageShapeMismatch);
        }
        Ok(Sinogram::new(forward_project(image, &self.geometry)))
    }

    /// Execute the forward Radon transform from a Leto 2D image view.
    ///
    /// The Leto view is copied once into logical row-major order before
    /// reusing the canonical Moirai projection kernel.
    pub fn forward_leto(
        &self,
        image: leto::ArrayView2<'_, f64>,
    ) -> RadonResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 2>> {
        let image = array2_from_leto_view(image);
        let sinogram = self.forward(&image)?;
        apollo_leto_interop::try_dense_from_array(sinogram.values())
            .ok_or(RadonError::ImageShapeMismatch)
    }

    /// Execute the forward Radon transform into caller-owned storage.
    pub fn forward_into(&self, image: &Array2<f64>, output: &mut Array2<f64>) -> RadonResult<()> {
        if image.shape() != [self.geometry.rows(), self.geometry.cols()] {
            return Err(RadonError::ImageShapeMismatch);
        }
        if output.shape() != [self.geometry.angle_count(), self.geometry.detector_count()] {
            return Err(RadonError::SinogramShapeMismatch);
        }
        forward_project_into(image, &self.geometry, output);
        Ok(())
    }

    /// Execute typed forward Radon projection into caller-owned storage.
    pub fn forward_typed_into<T: RadonStorage>(
        &self,
        image: &Array2<T>,
        output: &mut Array2<T>,
        profile: PrecisionProfile,
    ) -> RadonResult<()> {
        T::forward_into(self, image, output, profile)
    }

    /// Execute typed forward Radon projection from a Leto 2D image view.
    pub fn forward_leto_typed<T: RadonStorage>(
        &self,
        image: leto::ArrayView2<'_, T>,
        profile: PrecisionProfile,
    ) -> RadonResult<leto::Array<T, leto::MnemosyneStorage<T>, 2>> {
        let image = array2_from_leto_view(image);
        let mut output = Array2::<T>::from_elem(
            [self.geometry.angle_count(), self.geometry.detector_count()],
            T::from_f64(0.0),
        );
        self.forward_typed_into(&image, &mut output, profile)?;
        apollo_leto_interop::try_dense_from_array(&output).ok_or(RadonError::ImageShapeMismatch)
    }

    /// Execute adjoint backprojection.
    pub fn backproject(&self, sinogram: &Sinogram) -> RadonResult<Array2<f64>> {
        if sinogram.shape() != (self.geometry.angle_count(), self.geometry.detector_count()) {
            return Err(RadonError::SinogramShapeMismatch);
        }
        Ok(adjoint_backproject(sinogram.values(), &self.geometry))
    }

    /// Execute adjoint backprojection from a Leto 2D sinogram view.
    pub fn backproject_leto(
        &self,
        sinogram: leto::ArrayView2<'_, f64>,
    ) -> RadonResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 2>> {
        let sinogram = array2_from_leto_view(sinogram);
        let image = self.backproject(&Sinogram::new(sinogram))?;
        apollo_leto_interop::try_dense_from_array(&image).ok_or(RadonError::ImageShapeMismatch)
    }

    /// Execute adjoint backprojection into caller-owned image storage.
    pub fn backproject_into(
        &self,
        sinogram: &Sinogram,
        output: &mut Array2<f64>,
    ) -> RadonResult<()> {
        if sinogram.shape() != (self.geometry.angle_count(), self.geometry.detector_count()) {
            return Err(RadonError::SinogramShapeMismatch);
        }
        if output.shape() != [self.geometry.rows(), self.geometry.cols()] {
            return Err(RadonError::ImageShapeMismatch);
        }
        adjoint_backproject_into(sinogram.values(), &self.geometry, output);
        Ok(())
    }

    /// Execute typed adjoint backprojection into caller-owned image storage.
    pub fn backproject_typed_into<T: RadonStorage>(
        &self,
        sinogram: &Array2<T>,
        output: &mut Array2<T>,
        profile: PrecisionProfile,
    ) -> RadonResult<()> {
        T::backproject_into(self, sinogram, output, profile)
    }

    /// Execute typed adjoint backprojection from a Leto 2D sinogram view.
    pub fn backproject_leto_typed<T: RadonStorage>(
        &self,
        sinogram: leto::ArrayView2<'_, T>,
        profile: PrecisionProfile,
    ) -> RadonResult<leto::Array<T, leto::MnemosyneStorage<T>, 2>> {
        let sinogram = array2_from_leto_view(sinogram);
        let mut output = Array2::<T>::from_elem(
            [self.geometry.rows(), self.geometry.cols()],
            T::from_f64(0.0),
        );
        self.backproject_typed_into(&sinogram, &mut output, profile)?;
        apollo_leto_interop::try_dense_from_array(&output).ok_or(RadonError::ImageShapeMismatch)
    }

    /// Execute ramp-filtered backprojection.
    ///
    /// ## Normalization
    ///
    /// The backprojection scale factor PI / angle_count approximates the continuous FBP integral
    /// (1/2pi) integral_0^pi d_theta discretized with uniform angular step PI / angle_count.
    ///
    /// **Limitation**: this normalization is only correct for uniformly spaced angles.
    /// Non-uniform angle distributions require weighted integration (e.g., Parker weighting).
    /// The Fourier slice theorem guarantees exact reconstruction in the limit of dense uniform
    /// angular sampling and ideal ramp filtering.
    pub fn filtered_backprojection(&self, sinogram: &Sinogram) -> RadonResult<Array2<f64>> {
        if sinogram.shape() != (self.geometry.angle_count(), self.geometry.detector_count()) {
            return Err(RadonError::SinogramShapeMismatch);
        }
        let sinogram_values = sinogram.values();
        let mut filtered = Array2::zeros(sinogram_values.shape());
        let det_spacing = self.geometry.detector_spacing();
        let det_count = self.geometry.detector_count();
        // Parallel ramp filter per angle: each row is an independent, contiguous
        // chunk of the row-major `filtered` buffer; the source sinogram is read
        // row-by-row from its own contiguous backing slice.
        let ncols = filtered.shape()[1];
        let src_cols = sinogram_values.shape()[1];
        let src_flat = sinogram_values
            .as_slice()
            .expect("sinogram values must be contiguous (standard layout)");
        let flat = filtered
            .as_slice_mut()
            .expect("filtered must be contiguous (standard layout)");
        moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
            flat,
            ncols,
            |angle_idx, dest_row| {
                let start = angle_idx * src_cols;
                let projection: Vec<f64> = src_flat[start..start + src_cols].to_vec();
                let mut filtered_buf = vec![0.0_f64; det_count];
                ramp_filter_projection_into(&projection, det_spacing, &mut filtered_buf);
                dest_row.copy_from_slice(&filtered_buf);
            },
        );
        let mut image = adjoint_backproject(&filtered, &self.geometry);
        let scale = std::f64::consts::PI / self.geometry.angle_count() as f64;
        image
            .as_slice_mut()
            .expect("adjoint_backproject output must be contiguous")
            .iter_mut()
            .for_each(|value| *value *= scale);
        Ok(image)
    }

    /// Execute ramp-filtered backprojection from a Leto 2D sinogram view.
    pub fn filtered_backprojection_leto(
        &self,
        sinogram: leto::ArrayView2<'_, f64>,
    ) -> RadonResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 2>> {
        let sinogram = array2_from_leto_view(sinogram);
        let image = self.filtered_backprojection(&Sinogram::new(sinogram))?;
        apollo_leto_interop::try_dense_from_array(&image).ok_or(RadonError::ImageShapeMismatch)
    }
}

/// Real storage accepted by typed Radon forward and adjoint paths.
pub trait RadonStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage into the owner `f64` arithmetic path.
    fn to_f64(self) -> f64;

    /// Convert owner arithmetic result back to storage.
    fn from_f64(value: f64) -> Self;

    /// View slice as `f32` if layout is identical.
    #[inline]
    fn as_f32_slice(slice: &[Self]) -> Option<&[f32]> {
        let _ = slice;
        None
    }

    /// View mutable slice as `f32` if layout is identical.
    #[inline]
    fn as_f32_slice_mut(slice: &mut [Self]) -> Option<&mut [f32]> {
        let _ = slice;
        None
    }

    /// Execute forward projection into caller-owned typed storage.
    fn forward_into(
        plan: &RadonPlan,
        image: &Array2<Self>,
        output: &mut Array2<Self>,
        profile: PrecisionProfile,
    ) -> RadonResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if image.shape() != [plan.geometry.rows(), plan.geometry.cols()] {
            return Err(RadonError::ImageShapeMismatch);
        }
        if output.shape() != [plan.geometry.angle_count(), plan.geometry.detector_count()] {
            return Err(RadonError::SinogramShapeMismatch);
        }
        let input64 = image.mapv(Self::to_f64);
        let mut output64 =
            Array2::zeros([plan.geometry.angle_count(), plan.geometry.detector_count()]);
        plan.forward_into(&input64, &mut output64)?;
        for (slot, value) in output
            .as_slice_mut()
            .expect("contiguous output")
            .iter_mut()
            .zip(output64.iter().copied())
        {
            *slot = Self::from_f64(value);
        }
        Ok(())
    }

    /// Execute adjoint backprojection into caller-owned typed storage.
    fn backproject_into(
        plan: &RadonPlan,
        sinogram: &Array2<Self>,
        output: &mut Array2<Self>,
        profile: PrecisionProfile,
    ) -> RadonResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if sinogram.shape() != [plan.geometry.angle_count(), plan.geometry.detector_count()] {
            return Err(RadonError::SinogramShapeMismatch);
        }
        if output.shape() != [plan.geometry.rows(), plan.geometry.cols()] {
            return Err(RadonError::ImageShapeMismatch);
        }
        let sinogram64 = sinogram.mapv(Self::to_f64);
        let mut output64 = Array2::zeros([plan.geometry.rows(), plan.geometry.cols()]);
        let owner_sinogram = Sinogram::new(sinogram64);
        plan.backproject_into(&owner_sinogram, &mut output64)?;
        for (slot, value) in output
            .as_slice_mut()
            .expect("contiguous output")
            .iter_mut()
            .zip(output64.iter().copied())
        {
            *slot = Self::from_f64(value);
        }
        Ok(())
    }
}

impl RadonStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn forward_into(
        plan: &RadonPlan,
        image: &Array2<Self>,
        output: &mut Array2<Self>,
        profile: PrecisionProfile,
    ) -> RadonResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_into(image, output)
    }

    fn backproject_into(
        plan: &RadonPlan,
        sinogram: &Array2<Self>,
        output: &mut Array2<Self>,
        profile: PrecisionProfile,
    ) -> RadonResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if sinogram.shape() != [plan.geometry.angle_count(), plan.geometry.detector_count()] {
            return Err(RadonError::SinogramShapeMismatch);
        }
        if output.shape() != [plan.geometry.rows(), plan.geometry.cols()] {
            return Err(RadonError::ImageShapeMismatch);
        }
        adjoint_backproject_into(sinogram, &plan.geometry, output);
        Ok(())
    }
}

impl RadonStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }

    #[inline]
    fn as_f32_slice(slice: &[Self]) -> Option<&[f32]> {
        Some(slice)
    }

    #[inline]
    fn as_f32_slice_mut(slice: &mut [Self]) -> Option<&mut [f32]> {
        Some(slice)
    }
}

impl RadonStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    fn from_f64(value: f64) -> Self {
        f16::from_f32(value as f32)
    }
}

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> RadonResult<()> {
    if actual.matches_storage_and_compute(expected) {
        Ok(())
    } else {
        Err(RadonError::PrecisionMismatch)
    }
}

fn array2_from_leto_view<T: Copy>(view: leto::ArrayView2<'_, T>) -> Array2<T> {
    view.to_contiguous()
}
#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::assert_abs_diff_eq;

    fn plan() -> RadonPlan {
        RadonPlan::new(3, 3, vec![0.0, std::f64::consts::FRAC_PI_2], 3, 1.0).expect("valid plan")
    }

    fn image64() -> Array2<f64> {
        leto::Array2::from_shape_vec([3, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap()
    }

    #[test]
    fn caller_owned_forward_and_backproject_match_allocating_paths() {
        let plan = plan();
        let image = image64();
        let sinogram = plan.forward(&image).expect("forward");
        let mut forward_output =
            Array2::<f64>::zeros([plan.geometry.angle_count(), plan.geometry.detector_count()]);
        plan.forward_into(&image, &mut forward_output)
            .expect("caller-owned forward");
        for (actual, expected) in forward_output.iter().zip(sinogram.values().iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }

        let expected_backproject = plan.backproject(&sinogram).expect("backproject");
        let mut backproject_output =
            Array2::<f64>::zeros([plan.geometry.rows(), plan.geometry.cols()]);
        plan.backproject_into(&sinogram, &mut backproject_output)
            .expect("caller-owned backproject");
        for (actual, expected) in backproject_output.iter().zip(expected_backproject.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn typed_paths_support_f64_f32_and_mixed_f16_storage() {
        let plan = plan();
        let image = image64();
        let expected = plan.forward(&image).expect("forward");

        let mut out64 =
            Array2::<f64>::zeros([plan.geometry.angle_count(), plan.geometry.detector_count()]);
        plan.forward_typed_into(&image, &mut out64, PrecisionProfile::HIGH_ACCURACY_F64)
            .expect("typed f64 forward");
        for (actual, expected) in out64.iter().zip(expected.values().iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }

        let image32 = image.mapv(|value| value as f32);
        let represented32 = image32.mapv(f64::from);
        let expected32 = plan
            .forward(&represented32)
            .expect("represented f32 forward");
        let mut out32 =
            Array2::<f32>::zeros([plan.geometry.angle_count(), plan.geometry.detector_count()]);
        plan.forward_typed_into(&image32, &mut out32, PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed f32 forward");
        for (actual, expected) in out32.iter().zip(expected32.values().iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-5);
        }

        let mut recovered32 = Array2::<f32>::zeros([plan.geometry.rows(), plan.geometry.cols()]);
        plan.backproject_typed_into(
            &out32,
            &mut recovered32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed f32 backproject");
        let expected_backproject = plan
            .backproject(&Sinogram::new(expected32.into_values()))
            .expect("represented f32 backproject");
        for (actual, expected) in recovered32.iter().zip(expected_backproject.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-4);
        }

        let image16 = image.mapv(|value| f16::from_f32(value as f32));
        let represented16 = image16.mapv(|value| f64::from(value.to_f32()));
        let expected16 = plan
            .forward(&represented16)
            .expect("represented f16 forward");
        let mut out16 = Array2::from_elem(
            [plan.geometry.angle_count(), plan.geometry.detector_count()],
            f16::from_f32(0.0),
        );
        plan.forward_typed_into(
            &image16,
            &mut out16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed f16 forward");
        for (actual, expected) in out16.iter().zip(expected16.values().iter()) {
            let quantization_bound = expected.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!((f64::from(actual.to_f32()) - *expected).abs() <= quantization_bound);
        }
    }

    #[test]
    fn typed_path_rejects_profile_storage_mismatch() {
        let plan = plan();
        let image = Array2::from_elem([3, 3], 1.0_f32);
        let mut output =
            Array2::<f32>::zeros([plan.geometry.angle_count(), plan.geometry.detector_count()]);
        assert!(matches!(
            plan.forward_typed_into(&image, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
            Err(RadonError::PrecisionMismatch)
        ));
    }

    #[test]
    fn leto_forward_matches_leto_reference() {
        let plan = plan();
        let image = image64();
        let input = leto::Array2::from_shape_vec([3, 3], image.iter().copied().collect())
            .expect("leto image");
        let expected = plan.forward(&image).expect("leto forward");

        let actual = plan.forward_leto(input.view()).expect("leto forward");
        let actual = actual.view();
        let actual = actual.as_slice().expect("contiguous leto output");

        for (actual, expected) in actual.iter().zip(expected.values().iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_strided_forward_matches_leto_reference() {
        let plan = plan();
        let image = image64();
        let mut backing = Vec::with_capacity(18);
        for value in image.iter().copied() {
            backing.push(value);
            backing.push(99.0);
        }
        let input = leto::Array2::from_shape_vec([3, 6], backing).expect("leto image");
        let strided = input
            .view()
            .slice(&[(0, 3, 1), (0, 6, 2)])
            .expect("strided view");
        let expected = plan.forward(&image).expect("leto forward");

        let actual = plan.forward_leto(strided).expect("leto forward");
        let actual = actual.view();
        let actual = actual.as_slice().expect("contiguous leto output");

        for (actual, expected) in actual.iter().zip(expected.values().iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn typed_leto_forward_and_backproject_match_leto_reference() {
        let plan = plan();
        let image = image64().mapv(|value| value as f32);
        let input = leto::Array2::from_shape_vec([3, 3], image.iter().copied().collect())
            .expect("leto image");
        let mut expected_forward =
            Array2::<f32>::zeros([plan.geometry.angle_count(), plan.geometry.detector_count()]);
        plan.forward_typed_into(
            &image,
            &mut expected_forward,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed leto forward");

        let actual_forward = plan
            .forward_leto_typed::<f32>(input.view(), PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed leto forward");
        let actual_forward_view = actual_forward.view();
        let actual_forward_slice = actual_forward_view
            .as_slice()
            .expect("contiguous leto output");
        for (actual, expected) in actual_forward_slice.iter().zip(expected_forward.iter()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }

        let mut expected_backproject =
            Array2::<f32>::zeros([plan.geometry.rows(), plan.geometry.cols()]);
        plan.backproject_typed_into(
            &expected_forward,
            &mut expected_backproject,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed leto backproject");
        let values = leto::Array2::from_shape_vec(
            [plan.geometry.angle_count(), plan.geometry.detector_count()],
            expected_forward.iter().copied().collect(),
        )
        .expect("leto sinogram");

        let actual_backproject = plan
            .backproject_leto_typed::<f32>(values.view(), PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed leto backproject");
        let actual_backproject_view = actual_backproject.view();
        let actual_backproject_slice = actual_backproject_view
            .as_slice()
            .expect("contiguous leto output");
        for (actual, expected) in actual_backproject_slice
            .iter()
            .zip(expected_backproject.iter())
        {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn leto_backproject_and_filtered_backprojection_match_leto_reference() {
        let plan = plan();
        let image = image64();
        let sinogram = plan.forward(&image).expect("forward");
        let sinogram_array = leto::Array2::from_shape_vec(
            [plan.geometry.angle_count(), plan.geometry.detector_count()],
            sinogram.values().iter().copied().collect(),
        )
        .expect("leto sinogram");
        let expected_backproject = plan.backproject(&sinogram).expect("backproject");

        let actual_backproject = plan
            .backproject_leto(sinogram_array.view())
            .expect("leto backproject");
        let actual_backproject_view = actual_backproject.view();
        let actual_backproject_slice = actual_backproject_view
            .as_slice()
            .expect("contiguous leto output");
        for (actual, expected) in actual_backproject_slice
            .iter()
            .zip(expected_backproject.iter())
        {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }

        let expected_filtered = plan
            .filtered_backprojection(&sinogram)
            .expect("filtered backprojection");
        let actual_filtered = plan
            .filtered_backprojection_leto(sinogram_array.view())
            .expect("leto filtered backprojection");
        let actual_filtered_view = actual_filtered.view();
        let actual_filtered_slice = actual_filtered_view
            .as_slice()
            .expect("contiguous leto output");
        for (actual, expected) in actual_filtered_slice.iter().zip(expected_filtered.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }
}
