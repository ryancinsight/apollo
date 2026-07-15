//! Reusable provider storage and host-boundary execution for dense FFT plans.

use leto::Array3;

use crate::{f16, ApolloError, ApolloResult};

use super::pipeline::GpuFft3d;

/// Reusable host buffers for repeated `GpuFft3d` dispatch.
///
/// The shape invariant is `len = nx * ny * nz`; each split component stores
/// exactly `len` f32 values and each interleaved spectrum stores `2 * len`.
/// Reuse removes per-call provider allocation and host scratch allocation.
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
        buffers.validate_for(self)?;
        self.validate_spectrum_len(output.len())?;
        buffers.imaginary_host.fill(0.0);
        buffers
            .real_host
            .iter_mut()
            .zip(field.iter().copied())
            .for_each(|(destination, value)| *destination = value as f32);
        self.execute_forward(output, buffers)
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
        buffers.validate_for(self)?;
        self.validate_spectrum_len(output.len())?;
        buffers.imaginary_host.fill(0.0);
        buffers
            .real_host
            .iter_mut()
            .zip(field.iter().copied())
            .for_each(|(destination, value)| *destination = value.to_f32());
        self.execute_forward(output, buffers)
    }

    /// Forward transform from a Leto f64 view into Mnemosyne-backed spectrum storage.
    pub fn forward_leto(
        &self,
        field: leto::ArrayView3<'_, f64>,
    ) -> ApolloResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        self.validate_field_shape(field.shape())?;
        let field = apollo_leto_interop::view_cow(&field);
        let mut output = vec![0.0; 2 * self.element_count()];
        let mut buffers = GpuFft3dBuffers::new(self)?;
        self.forward_values(&field, &mut output, &mut buffers)?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(|| ApolloError::Wgpu {
            message: "failed to allocate Mnemosyne-backed Leto FFT forward Leto spectrum"
                .to_owned(),
        })
    }

    /// Forward transform from a Leto f16 view into Mnemosyne-backed spectrum storage.
    pub fn forward_f16_leto(
        &self,
        field: leto::ArrayView3<'_, f16>,
    ) -> ApolloResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        self.validate_field_shape(field.shape())?;
        let field = apollo_leto_interop::view_cow(&field);
        let mut output = vec![0.0; 2 * self.element_count()];
        let mut buffers = GpuFft3dBuffers::new(self)?;
        buffers.imaginary_host.fill(0.0);
        buffers
            .real_host
            .iter_mut()
            .zip(field.iter().copied())
            .for_each(|(destination, value)| *destination = value.to_f32());
        self.execute_forward(&mut output, &mut buffers)?;
        apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(|| ApolloError::Wgpu {
            message: "failed to allocate Mnemosyne-backed Leto FFT f16 forward Leto spectrum"
                .to_owned(),
        })
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
    pub fn inverse_leto(
        &self,
        spectrum: leto::ArrayView1<'_, f32>,
    ) -> ApolloResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 3>> {
        let spectrum = apollo_leto_interop::view_cow(&spectrum);
        self.validate_spectrum_len(spectrum.len())?;
        let mut buffers = GpuFft3dBuffers::new(self)?;
        Self::split_spectrum(&spectrum, &mut buffers);
        self.execute_inverse(&mut buffers)?;
        let output: Vec<f64> = buffers.real_host.iter().copied().map(f64::from).collect();
        apollo_leto_interop::try_dense_from_slice([self.nx, self.ny, self.nz], &output).ok_or_else(
            || ApolloError::Wgpu {
                message: "failed to allocate Mnemosyne-backed Leto FFT inverse field".to_owned(),
            },
        )
    }

    /// Inverse transform from Leto spectrum storage into Mnemosyne-backed f16 field storage.
    pub fn inverse_f16_leto(
        &self,
        spectrum: leto::ArrayView1<'_, f32>,
    ) -> ApolloResult<leto::Array<f16, leto::MnemosyneStorage<f16>, 3>> {
        let spectrum = apollo_leto_interop::view_cow(&spectrum);
        self.validate_spectrum_len(spectrum.len())?;
        let mut buffers = GpuFft3dBuffers::new(self)?;
        Self::split_spectrum(&spectrum, &mut buffers);
        self.execute_inverse(&mut buffers)?;
        let output: Vec<f16> = buffers
            .real_host
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect();
        apollo_leto_interop::try_dense_from_slice([self.nx, self.ny, self.nz], &output).ok_or_else(
            || ApolloError::Wgpu {
                message: "failed to allocate Mnemosyne-backed Leto FFT f16 inverse field"
                    .to_owned(),
            },
        )
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

    fn forward_values(
        &self,
        field: &[f64],
        output: &mut [f32],
        buffers: &mut GpuFft3dBuffers,
    ) -> ApolloResult<()> {
        buffers.validate_for(self)?;
        if field.len() != buffers.len() {
            return Err(ApolloError::ShapeMismatch {
                expected: format!("FFT real field length {}", buffers.len()),
                actual: format!("FFT real field length {}", field.len()),
            });
        }
        self.validate_spectrum_len(output.len())?;
        buffers.imaginary_host.fill(0.0);
        buffers
            .real_host
            .iter_mut()
            .zip(field.iter().copied())
            .for_each(|(destination, value)| *destination = value as f32);
        self.execute_forward(output, buffers)
    }

    fn execute_forward(
        &self,
        output: &mut [f32],
        buffers: &mut GpuFft3dBuffers,
    ) -> ApolloResult<()> {
        self.execute_forward_in_place(&mut buffers.real_host, &mut buffers.imaginary_host)?;
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
