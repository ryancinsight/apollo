//! Reusable provider storage and host-boundary execution for dense FFT plans.

use leto::Array3;

use crate::{f16, ApolloError, ApolloResult};

use super::pipeline::GpuFft3d;

trait HostReal: Copy + Default {
    fn into_device(self) -> f32;
    fn from_device(value: f32) -> Self;
}

impl HostReal for f64 {
    fn into_device(self) -> f32 {
        self as f32
    }

    fn from_device(value: f32) -> Self {
        Self::from(value)
    }
}

impl HostReal for f16 {
    fn into_device(self) -> f32 {
        self.to_f32()
    }

    fn from_device(value: f32) -> Self {
        Self::from_f32(value)
    }
}

/// Reusable host buffers for repeated `GpuFft3d` dispatch.
///
/// The shape invariant is `len = nx * ny * nz`; each split component stores
/// exactly `len` f32 values and each interleaved spectrum stores `2 * len`.
/// Reuse removes per-call host staging allocation. The plan owns its
/// Hephaestus device buffers independently.
pub struct GpuFft3dBuffers {
    nx: usize,
    ny: usize,
    nz: usize,
    real_host: Vec<f32>,
    imaginary_host: Vec<f32>,
}

impl GpuFft3dBuffers {
    /// Allocate reusable typed buffers for `plan`.
    pub fn new(plan: &GpuFft3d) -> ApolloResult<Self> {
        let len = plan.element_count();
        Ok(Self {
            nx: plan.nx,
            ny: plan.ny,
            nz: plan.nz,
            real_host: vec![0.0; len],
            imaginary_host: vec![0.0; len],
        })
    }

    fn validate_for(&self, plan: &GpuFft3d) -> ApolloResult<()> {
        let actual = [self.nx, self.ny, self.nz];
        let expected = [plan.nx, plan.ny, plan.nz];
        if actual == expected {
            Ok(())
        } else {
            Err(ApolloError::ShapeMismatch {
                expected: format!("FFT reusable buffer shape {expected:?}"),
                actual: format!("FFT reusable buffer shape {actual:?}"),
            })
        }
    }

    fn len(&self) -> usize {
        self.nx * self.ny * self.nz
    }
}

impl GpuFft3d {
    /// Forward transform of a real field into an interleaved f32 spectrum.
    pub fn forward(&self, field: &Array3<f64>) -> ApolloResult<Vec<f32>> {
        let mut output = vec![0.0; 2 * self.element_count()];
        self.forward_into(field, &mut output)?;
        Ok(output)
    }

    /// Forward transform into caller-owned interleaved f32 storage.
    pub fn forward_into(&self, field: &Array3<f64>, output: &mut [f32]) -> ApolloResult<()> {
        let mut buffers = GpuFft3dBuffers::new(self)?;
        self.forward_into_with_buffers(field, output, &mut buffers)
    }

    /// Forward transform using caller-retained typed accelerator storage.
    pub fn forward_into_with_buffers(
        &self,
        field: &Array3<f64>,
        output: &mut [f32],
        buffers: &mut GpuFft3dBuffers,
    ) -> ApolloResult<()> {
        self.validate_field_shape(field.shape())?;
        self.forward_values(field.iter().copied(), output, buffers)
    }

    /// Forward transform from f16 host storage into an interleaved f32 spectrum.
    pub fn forward_f16(&self, field: &Array3<f16>) -> ApolloResult<Vec<f32>> {
        let mut output = vec![0.0; 2 * self.element_count()];
        let mut buffers = GpuFft3dBuffers::new(self)?;
        self.forward_f16_into_with_buffers(field, &mut output, &mut buffers)?;
        Ok(output)
    }

    /// Forward f16 transform using caller-retained typed accelerator storage.
    pub fn forward_f16_into_with_buffers(
        &self,
        field: &Array3<f16>,
        output: &mut [f32],
        buffers: &mut GpuFft3dBuffers,
    ) -> ApolloResult<()> {
        self.validate_field_shape(field.shape())?;
        self.forward_values(field.iter().copied(), output, buffers)
    }

    /// Forward transform from a Leto f64 view into Mnemosyne-backed spectrum storage.
    ///
    /// This allocating convenience path creates host staging for one dispatch.
    /// Repeated callers should retain [`GpuFft3dBuffers`] and call
    /// [`Self::forward_leto_with_buffers`].
    ///
    /// # Errors
    ///
    /// Returns a shape or provider execution error.
    pub fn forward_leto(
        &self,
        field: leto::ArrayView3<'_, f64>,
    ) -> ApolloResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let mut buffers = GpuFft3dBuffers::new(self)?;
        self.forward_leto_with_buffers(field, &mut buffers)
    }

    /// Forward transform from a Leto f64 view with caller-retained host staging.
    ///
    /// The returned spectrum owns one Mnemosyne allocation. Contiguous input is
    /// borrowed; only a strided input requires logical-order materialization.
    ///
    /// # Errors
    ///
    /// Returns a shape, reusable-buffer, or provider execution error.
    pub fn forward_leto_with_buffers(
        &self,
        field: leto::ArrayView3<'_, f64>,
        buffers: &mut GpuFft3dBuffers,
    ) -> ApolloResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        self.forward_leto_output(field, buffers)
    }

    /// Forward transform from a Leto f16 view into Mnemosyne-backed spectrum storage.
    ///
    /// This allocating convenience path creates host staging for one dispatch.
    /// Repeated callers should retain [`GpuFft3dBuffers`] and call
    /// [`Self::forward_half_leto_with_buffers`].
    ///
    /// # Errors
    ///
    /// Returns a shape or provider execution error.
    pub fn forward_f16_leto(
        &self,
        field: leto::ArrayView3<'_, f16>,
    ) -> ApolloResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let mut buffers = GpuFft3dBuffers::new(self)?;
        self.forward_half_leto_with_buffers(field, &mut buffers)
    }

    /// Forward transform from a Leto half-precision view with retained staging.
    ///
    /// The returned spectrum owns one Mnemosyne allocation. Contiguous input is
    /// borrowed; only a strided input requires logical-order materialization.
    ///
    /// # Errors
    ///
    /// Returns a shape, reusable-buffer, or provider execution error.
    pub fn forward_half_leto_with_buffers(
        &self,
        field: leto::ArrayView3<'_, f16>,
        buffers: &mut GpuFft3dBuffers,
    ) -> ApolloResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        self.forward_leto_output(field, buffers)
    }

    fn forward_leto_output<T: HostReal>(
        &self,
        field: leto::ArrayView3<'_, T>,
        buffers: &mut GpuFft3dBuffers,
    ) -> ApolloResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        self.validate_field_shape(field.shape())?;
        buffers.validate_for(self)?;
        let field = apollo_leto_interop::view_cow(&field);
        let output_len = self
            .element_count()
            .checked_mul(2)
            .expect("invariant: plan volume is validated before spectrum construction");
        self.stage_forward_values(field.iter().copied(), buffers)?;
        let mut real = buffers.real_host.iter().copied();
        let mut imaginary = buffers.imaginary_host.iter().copied();
        Ok(leto::Array::from_mnemosyne_shape_fn(
            [output_len],
            |[index]| {
                if index.is_multiple_of(2) {
                    real.next()
                        .expect("invariant: spectrum shape matches real staging length")
                } else {
                    imaginary
                        .next()
                        .expect("invariant: spectrum shape matches imaginary staging length")
                }
            },
        ))
    }

    /// Inverse transform from interleaved f32 spectrum into an f64 real field.
    pub fn inverse(&self, spectrum: &[f32], output: &mut Array3<f64>) -> ApolloResult<()> {
        let mut buffers = GpuFft3dBuffers::new(self)?;
        self.inverse_with_buffers(spectrum, output, &mut buffers)
    }

    /// Inverse transform using caller-retained typed accelerator storage.
    pub fn inverse_with_buffers(
        &self,
        spectrum: &[f32],
        output: &mut Array3<f64>,
        buffers: &mut GpuFft3dBuffers,
    ) -> ApolloResult<()> {
        self.validate_spectrum_len(spectrum.len())?;
        self.validate_field_shape(output.shape())?;
        buffers.validate_for(self)?;
        Self::split_spectrum(spectrum, buffers);
        self.execute_inverse(buffers)?;
        output
            .iter_mut()
            .zip(buffers.real_host.iter().copied())
            .for_each(|(destination, value)| *destination = f64::from(value));
        Ok(())
    }

    /// Inverse transform from interleaved f32 spectrum into an f16 real field.
    pub fn inverse_f16(&self, spectrum: &[f32], output: &mut Array3<f16>) -> ApolloResult<()> {
        let mut buffers = GpuFft3dBuffers::new(self)?;
        self.inverse_f16_with_buffers(spectrum, output, &mut buffers)
    }

    /// Inverse f16 transform using caller-retained typed accelerator storage.
    pub fn inverse_f16_with_buffers(
        &self,
        spectrum: &[f32],
        output: &mut Array3<f16>,
        buffers: &mut GpuFft3dBuffers,
    ) -> ApolloResult<()> {
        self.validate_spectrum_len(spectrum.len())?;
        self.validate_field_shape(output.shape())?;
        buffers.validate_for(self)?;
        Self::split_spectrum(spectrum, buffers);
        self.execute_inverse(buffers)?;
        output
            .iter_mut()
            .zip(buffers.real_host.iter().copied())
            .for_each(|(destination, value)| *destination = f16::from_f32(value));
        Ok(())
    }

    /// Inverse transform from Leto spectrum storage into Mnemosyne-backed f64 field storage.
    ///
    /// This allocating convenience path creates host staging for one dispatch.
    /// Repeated callers should retain [`GpuFft3dBuffers`] and call
    /// [`Self::inverse_leto_with_buffers`].
    ///
    /// # Errors
    ///
    /// Returns a shape or provider execution error.
    pub fn inverse_leto(
        &self,
        spectrum: leto::ArrayView1<'_, f32>,
    ) -> ApolloResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 3>> {
        let mut buffers = GpuFft3dBuffers::new(self)?;
        self.inverse_leto_with_buffers(spectrum, &mut buffers)
    }

    /// Inverse transform into an f64 Leto array with retained host staging.
    ///
    /// The returned field owns one Mnemosyne allocation. Contiguous spectrum
    /// input is borrowed; only a strided input is materialized.
    ///
    /// # Errors
    ///
    /// Returns a shape, reusable-buffer, or provider execution error.
    pub fn inverse_leto_with_buffers(
        &self,
        spectrum: leto::ArrayView1<'_, f32>,
        buffers: &mut GpuFft3dBuffers,
    ) -> ApolloResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 3>> {
        self.inverse_leto_output(spectrum, buffers)
    }

    /// Inverse transform from Leto spectrum storage into Mnemosyne-backed f16 field storage.
    ///
    /// This allocating convenience path creates host staging for one dispatch.
    /// Repeated callers should retain [`GpuFft3dBuffers`] and call
    /// [`Self::inverse_half_leto_with_buffers`].
    ///
    /// # Errors
    ///
    /// Returns a shape or provider execution error.
    pub fn inverse_f16_leto(
        &self,
        spectrum: leto::ArrayView1<'_, f32>,
    ) -> ApolloResult<leto::Array<f16, leto::MnemosyneStorage<f16>, 3>> {
        let mut buffers = GpuFft3dBuffers::new(self)?;
        self.inverse_half_leto_with_buffers(spectrum, &mut buffers)
    }

    /// Inverse transform into a half-precision Leto array with retained staging.
    ///
    /// The returned field owns one Mnemosyne allocation. Contiguous spectrum
    /// input is borrowed; only a strided input is materialized.
    ///
    /// # Errors
    ///
    /// Returns a shape, reusable-buffer, or provider execution error.
    pub fn inverse_half_leto_with_buffers(
        &self,
        spectrum: leto::ArrayView1<'_, f32>,
        buffers: &mut GpuFft3dBuffers,
    ) -> ApolloResult<leto::Array<f16, leto::MnemosyneStorage<f16>, 3>> {
        self.inverse_leto_output(spectrum, buffers)
    }

    fn inverse_leto_output<T: HostReal>(
        &self,
        spectrum: leto::ArrayView1<'_, f32>,
        buffers: &mut GpuFft3dBuffers,
    ) -> ApolloResult<leto::Array<T, leto::MnemosyneStorage<T>, 3>> {
        self.validate_spectrum_len(spectrum.shape()[0])?;
        buffers.validate_for(self)?;
        let spectrum = apollo_leto_interop::view_cow(&spectrum);
        Self::split_spectrum(&spectrum, buffers);
        self.execute_inverse(buffers)?;
        let mut values = buffers.real_host.iter().copied();
        Ok(leto::Array::from_mnemosyne_shape_fn(
            [self.nx, self.ny, self.nz],
            |_| {
                T::from_device(
                    values
                        .next()
                        .expect("invariant: field shape matches real staging length"),
                )
            },
        ))
    }

    fn validate_field_shape(&self, actual: [usize; 3]) -> ApolloResult<()> {
        let expected = [self.nx, self.ny, self.nz];
        if actual == expected {
            Ok(())
        } else {
            Err(ApolloError::ShapeMismatch {
                expected: format!("FFT field shape {expected:?}"),
                actual: format!("FFT field shape {actual:?}"),
            })
        }
    }

    fn validate_spectrum_len(&self, actual: usize) -> ApolloResult<()> {
        let expected = self
            .element_count()
            .checked_mul(2)
            .expect("invariant: plan volume is validated before interleaved length construction");
        if actual == expected {
            Ok(())
        } else {
            Err(ApolloError::ShapeMismatch {
                expected: format!("interleaved FFT spectrum length {expected}"),
                actual: format!("interleaved FFT spectrum length {actual}"),
            })
        }
    }

    fn forward_values<T: HostReal>(
        &self,
        field: impl ExactSizeIterator<Item = T>,
        output: &mut [f32],
        buffers: &mut GpuFft3dBuffers,
    ) -> ApolloResult<()> {
        self.validate_spectrum_len(output.len())?;
        self.stage_forward_values(field, buffers)?;
        for ((real, imaginary), destination) in buffers
            .real_host
            .iter()
            .zip(buffers.imaginary_host.iter())
            .zip(output.chunks_exact_mut(2))
        {
            destination[0] = *real;
            destination[1] = *imaginary;
        }
        Ok(())
    }

    fn stage_forward_values<T: HostReal>(
        &self,
        field: impl ExactSizeIterator<Item = T>,
        buffers: &mut GpuFft3dBuffers,
    ) -> ApolloResult<()> {
        buffers.validate_for(self)?;
        if field.len() != buffers.len() {
            return Err(ApolloError::ShapeMismatch {
                expected: format!("FFT real field length {}", buffers.len()),
                actual: format!("FFT real field length {}", field.len()),
            });
        }
        buffers.imaginary_host.fill(0.0);
        buffers
            .real_host
            .iter_mut()
            .zip(field)
            .for_each(|(destination, value)| *destination = value.into_device());
        self.execute_forward_in_place(&mut buffers.real_host, &mut buffers.imaginary_host)
    }

    fn execute_inverse(&self, buffers: &mut GpuFft3dBuffers) -> ApolloResult<()> {
        self.execute_inverse_in_place(&mut buffers.real_host, &mut buffers.imaginary_host)
    }

    fn split_spectrum(spectrum: &[f32], buffers: &mut GpuFft3dBuffers) {
        for (index, pair) in spectrum.chunks_exact(2).enumerate() {
            buffers.real_host[index] = pair[0];
            buffers.imaginary_host[index] = pair[1];
        }
    }
}

#[cfg(test)]
#[path = "verification/workspace.rs"]
mod verification;
