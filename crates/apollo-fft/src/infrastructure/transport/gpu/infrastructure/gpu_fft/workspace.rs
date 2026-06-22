//! Buffer gathering, scatter operations, and forward/inverse routing for GPU FFT.

use crate::infrastructure::transport::gpu::infrastructure::gpu_fft::pipeline::GpuFft3d;
use crate::infrastructure::transport::gpu::infrastructure::gpu_fft::strategy::{Axis, AxisStrategy};
use crate::application::utilities::leto_interop;
use crate::{f16, ApolloError, ApolloResult};
use ndarray::Array3;
use std::borrow::Cow;

/// Reusable GPU and host buffers for repeated `GpuFft3d` dispatch.
///
/// The buffer shape is part of the value invariant:
/// `len = nx * ny * nz`, every split component buffer stores exactly `len`
/// `f32` values, and every interleaved complex buffer stores exactly `2 * len`
/// values. Reusing this object removes per-call device buffer allocation and
/// host-side scratch vector allocation for repeated forward/inverse transforms
/// with one fixed 3D shape.
pub struct GpuFft3dBuffers {
    nx: usize,
    ny: usize,
    nz: usize,
    re_buf: wgpu::Buffer,
    im_buf: wgpu::Buffer,
    re_staging: wgpu::Buffer,
    im_staging: wgpu::Buffer,
    re_host: Vec<f32>,
    im_host: Vec<f32>,
}

impl GpuFft3dBuffers {
    /// Allocate reusable buffers for a `GpuFft3d` plan.
    #[must_use]
    pub fn new(plan: &GpuFft3d) -> Self {
        let len = plan.element_count();
        let size = plan.component_buffer_size();
        let working_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let staging_usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
        let re_buf = plan.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-fft-wgpu reusable re"),
            size,
            usage: working_usage,
            mapped_at_creation: false,
        });
        let im_buf = plan.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-fft-wgpu reusable im"),
            size,
            usage: working_usage,
            mapped_at_creation: false,
        });
        let re_staging = plan.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-fft-wgpu reusable re staging"),
            size,
            usage: staging_usage,
            mapped_at_creation: false,
        });
        let im_staging = plan.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-fft-wgpu reusable im staging"),
            size,
            usage: staging_usage,
            mapped_at_creation: false,
        });
        Self {
            nx: plan.nx,
            ny: plan.ny,
            nz: plan.nz,
            re_buf,
            im_buf,
            re_staging,
            im_staging,
            re_host: vec![0.0; len],
            im_host: vec![0.0; len],
        }
    }

    #[inline]
    fn assert_matches(&self, plan: &GpuFft3d) {
        assert_eq!((self.nx, self.ny, self.nz), (plan.nx, plan.ny, plan.nz));
    }

    #[inline]
    fn len(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    fn read_split_into_host(&mut self, plan: &GpuFft3d) {
        let size = plan.component_buffer_size();
        let mut encoder = plan
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-fft-wgpu reusable readback encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.re_buf, 0, &self.re_staging, 0, size);
        encoder.copy_buffer_to_buffer(&self.im_buf, 0, &self.im_staging, 0, size);
        plan.queue.submit(std::iter::once(encoder.finish()));

        let re_slice = self.re_staging.slice(..size);
        let im_slice = self.im_staging.slice(..size);
        re_slice.map_async(wgpu::MapMode::Read, |_| {});
        im_slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = plan.device.poll(wgpu::PollType::Wait);

        {
            let mapped = re_slice.get_mapped_range();
            self.re_host.copy_from_slice(bytemuck::cast_slice(&mapped));
        }
        {
            let mapped = im_slice.get_mapped_range();
            self.im_host.copy_from_slice(bytemuck::cast_slice(&mapped));
        }
        self.re_staging.unmap();
        self.im_staging.unmap();
    }
}

impl GpuFft3d {
    /// Forward transform of a real field into an interleaved complex buffer.
    #[must_use]
    pub fn forward(&self, field: &Array3<f64>) -> Vec<f32> {
        let n = self.nx * self.ny * self.nz;
        let mut out = vec![0.0_f32; 2 * n];
        self.forward_into(field, &mut out);
        out
    }

    /// Forward transform of a real field into a caller-owned interleaved complex buffer.
    pub fn forward_into(&self, field: &Array3<f64>, out: &mut [f32]) {
        assert_eq!(field.dim(), (self.nx, self.ny, self.nz));
        assert_eq!(out.len(), 2 * self.nx * self.ny * self.nz);
        let n = self.nx * self.ny * self.nz;
        let mut re_data = vec![0.0_f32; n];
        let im_data = vec![0.0_f32; n];
        re_data
            .iter_mut()
            .zip(field.iter().copied())
            .for_each(|(dst, src)| *dst = src as f32);

        self.queue
            .write_buffer(&self.full_re_buf, 0, bytemuck::cast_slice(&re_data));
        self.queue
            .write_buffer(&self.full_im_buf, 0, bytemuck::cast_slice(&im_data));

        self.run_device_axis_fft(Axis::Z, false);
        self.run_device_axis_fft(Axis::Y, false);
        self.run_device_axis_fft(Axis::X, false);

        let (re_out, im_out) = self.read_back_full_buffers();

        for ((re, im), pair) in re_out
            .iter()
            .zip(im_out.iter())
            .zip(out.chunks_exact_mut(2))
        {
            pair[0] = *re;
            pair[1] = *im;
        }
    }

    /// Forward transform using caller-retained reusable GPU and host buffers.
    pub fn forward_into_with_buffers(
        &self,
        field: &Array3<f64>,
        out: &mut [f32],
        buffers: &mut GpuFft3dBuffers,
    ) {
        buffers.assert_matches(self);
        assert_eq!(field.dim(), (self.nx, self.ny, self.nz));
        assert_eq!(out.len(), 2 * buffers.len());

        buffers.im_host.fill(0.0);
        buffers
            .re_host
            .iter_mut()
            .zip(field.iter().copied())
            .for_each(|(dst, src)| *dst = src as f32);

        self.queue
            .write_buffer(&buffers.re_buf, 0, bytemuck::cast_slice(&buffers.re_host));
        self.queue
            .write_buffer(&buffers.im_buf, 0, bytemuck::cast_slice(&buffers.im_host));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-fft-wgpu reusable forward encoder"),
            });
        self.encode_forward_split(&mut encoder, &buffers.re_buf, &buffers.im_buf);
        self.queue.submit(std::iter::once(encoder.finish()));

        buffers.read_split_into_host(self);
        for ((re, im), pair) in buffers
            .re_host
            .iter()
            .zip(buffers.im_host.iter())
            .zip(out.chunks_exact_mut(2))
        {
            pair[0] = *re;
            pair[1] = *im;
        }
    }

    /// Forward mixed-precision transform from `f16` storage into an interleaved
    /// `f32` complex spectrum.
    ///
    /// The storage contract matches `PrecisionProfile::MIXED_PRECISION_F16_F32`:
    /// host inputs are stored as `f16`, promoted once to `f32` at the GPU buffer
    /// boundary, and the existing `f32` WGPU FFT kernels remain the single
    /// authoritative device implementation.
    #[must_use]
    pub fn forward_f16(&self, field: &Array3<f16>) -> Vec<f32> {
        let n = self.nx * self.ny * self.nz;
        let mut out = vec![0.0_f32; 2 * n];
        let mut buffers = GpuFft3dBuffers::new(self);
        self.forward_f16_into_with_buffers(field, &mut out, &mut buffers);
        out
    }

    /// Forward mixed-precision transform using caller-retained reusable GPU and
    /// host buffers.
    pub fn forward_f16_into_with_buffers(
        &self,
        field: &Array3<f16>,
        out: &mut [f32],
        buffers: &mut GpuFft3dBuffers,
    ) {
        buffers.assert_matches(self);
        assert_eq!(field.dim(), (self.nx, self.ny, self.nz));
        assert_eq!(out.len(), 2 * buffers.len());

        buffers.im_host.fill(0.0);
        buffers
            .re_host
            .iter_mut()
            .zip(field.iter().copied())
            .for_each(|(dst, src)| *dst = src.to_f32());

        self.queue
            .write_buffer(&buffers.re_buf, 0, bytemuck::cast_slice(&buffers.re_host));
        self.queue
            .write_buffer(&buffers.im_buf, 0, bytemuck::cast_slice(&buffers.im_host));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-fft-wgpu reusable f16 forward encoder"),
            });
        self.encode_forward_split(&mut encoder, &buffers.re_buf, &buffers.im_buf);
        self.queue.submit(std::iter::once(encoder.finish()));

        buffers.read_split_into_host(self);
        for ((re, im), pair) in buffers
            .re_host
            .iter()
            .zip(buffers.im_host.iter())
            .zip(out.chunks_exact_mut(2))
        {
            pair[0] = *re;
            pair[1] = *im;
        }
    }

    /// Forward transform of a Leto real field into a Mnemosyne-backed
    /// interleaved complex Leto spectrum.
    pub fn forward_leto(
        &self,
        field: leto::ArrayView3<'_, f64>,
    ) -> ApolloResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        self.validate_leto_3d_shape(field.shape())?;
        let field = leto_view3_cow(field)?;
        let mut out = vec![0.0_f32; 2 * self.element_count()];
        let mut buffers = GpuFft3dBuffers::new(self);
        self.forward_values_into_with_buffers(field.as_ref(), &mut out, &mut buffers)?;
        leto_array1_from_slice(&out, "FFT-WGPU forward Leto spectrum")
    }

    /// Forward mixed-precision transform of a Leto `f16` real field into a
    /// Mnemosyne-backed interleaved `f32` complex Leto spectrum.
    pub fn forward_f16_leto(
        &self,
        field: leto::ArrayView3<'_, f16>,
    ) -> ApolloResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        self.validate_leto_3d_shape(field.shape())?;
        let field = leto_view3_cow(field)?;
        let mut out = vec![0.0_f32; 2 * self.element_count()];
        let mut buffers = GpuFft3dBuffers::new(self);
        self.forward_f16_values_into_with_buffers(field.as_ref(), &mut out, &mut buffers)?;
        leto_array1_from_slice(&out, "FFT-WGPU f16 forward Leto spectrum")
    }

    /// Inverse transform of an interleaved complex buffer into a real field.
    pub fn inverse(&self, field_hat: &[f32], out: &mut Array3<f64>) {
        assert_eq!(field_hat.len(), 2 * self.nx * self.ny * self.nz);
        assert_eq!(out.dim(), (self.nx, self.ny, self.nz));

        let n = self.nx * self.ny * self.nz;
        let mut re_data = vec![0.0_f32; n];
        let mut im_data = vec![0.0_f32; n];
        for (idx, pair) in field_hat.chunks_exact(2).enumerate() {
            re_data[idx] = pair[0];
            im_data[idx] = pair[1];
        }

        self.queue
            .write_buffer(&self.full_re_buf, 0, bytemuck::cast_slice(&re_data));
        self.queue
            .write_buffer(&self.full_im_buf, 0, bytemuck::cast_slice(&im_data));

        self.run_device_axis_fft(Axis::X, true);
        self.run_device_axis_fft(Axis::Y, true);
        self.run_device_axis_fft(Axis::Z, true);

        let (re_out, _) = self.read_back_full_buffers();

        for (dst, &value) in out.iter_mut().zip(re_out.iter()) {
            *dst = value as f64;
        }
    }

    /// Inverse transform using caller-retained reusable GPU and host buffers.
    pub fn inverse_with_buffers(
        &self,
        field_hat: &[f32],
        out: &mut Array3<f64>,
        buffers: &mut GpuFft3dBuffers,
    ) {
        buffers.assert_matches(self);
        assert_eq!(field_hat.len(), 2 * buffers.len());
        assert_eq!(out.dim(), (self.nx, self.ny, self.nz));

        for (idx, pair) in field_hat.chunks_exact(2).enumerate() {
            buffers.re_host[idx] = pair[0];
            buffers.im_host[idx] = pair[1];
        }

        self.queue
            .write_buffer(&buffers.re_buf, 0, bytemuck::cast_slice(&buffers.re_host));
        self.queue
            .write_buffer(&buffers.im_buf, 0, bytemuck::cast_slice(&buffers.im_host));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-fft-wgpu reusable inverse encoder"),
            });
        self.encode_inverse_split(&mut encoder, &buffers.re_buf, &buffers.im_buf);
        self.queue.submit(std::iter::once(encoder.finish()));

        buffers.read_split_into_host(self);
        for (dst, &value) in out.iter_mut().zip(buffers.re_host.iter()) {
            *dst = value as f64;
        }
    }

    /// Inverse mixed-precision transform from an interleaved `f32` complex
    /// spectrum into `f16` real storage.
    pub fn inverse_f16(&self, field_hat: &[f32], out: &mut Array3<f16>) {
        let mut buffers = GpuFft3dBuffers::new(self);
        self.inverse_f16_with_buffers(field_hat, out, &mut buffers);
    }

    /// Inverse mixed-precision transform using caller-retained reusable GPU and
    /// host buffers.
    pub fn inverse_f16_with_buffers(
        &self,
        field_hat: &[f32],
        out: &mut Array3<f16>,
        buffers: &mut GpuFft3dBuffers,
    ) {
        buffers.assert_matches(self);
        assert_eq!(field_hat.len(), 2 * buffers.len());
        assert_eq!(out.dim(), (self.nx, self.ny, self.nz));

        for (idx, pair) in field_hat.chunks_exact(2).enumerate() {
            buffers.re_host[idx] = pair[0];
            buffers.im_host[idx] = pair[1];
        }

        self.queue
            .write_buffer(&buffers.re_buf, 0, bytemuck::cast_slice(&buffers.re_host));
        self.queue
            .write_buffer(&buffers.im_buf, 0, bytemuck::cast_slice(&buffers.im_host));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-fft-wgpu reusable f16 inverse encoder"),
            });
        self.encode_inverse_split(&mut encoder, &buffers.re_buf, &buffers.im_buf);
        self.queue.submit(std::iter::once(encoder.finish()));

        buffers.read_split_into_host(self);
        for (dst, &value) in out.iter_mut().zip(buffers.re_host.iter()) {
            *dst = f16::from_f32(value);
        }
    }

    /// Inverse transform of an interleaved complex Leto spectrum into a
    /// Mnemosyne-backed 3D Leto real field.
    pub fn inverse_leto(
        &self,
        field_hat: leto::ArrayView1<'_, f32>,
    ) -> ApolloResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 3>> {
        let field_hat = leto_view1_cow(field_hat);
        let mut out = vec![0.0_f64; self.element_count()];
        let mut buffers = GpuFft3dBuffers::new(self);
        self.inverse_values_into_with_buffers(field_hat.as_ref(), &mut out, &mut buffers)?;
        leto_array3_from_slice(
            [self.nx, self.ny, self.nz],
            &out,
            "FFT-WGPU inverse Leto field",
        )
    }

    /// Inverse mixed-precision transform of an interleaved complex Leto
    /// spectrum into a Mnemosyne-backed 3D Leto `f16` real field.
    pub fn inverse_f16_leto(
        &self,
        field_hat: leto::ArrayView1<'_, f32>,
    ) -> ApolloResult<leto::Array<f16, leto::MnemosyneStorage<f16>, 3>> {
        let field_hat = leto_view1_cow(field_hat);
        let mut out = vec![f16::from_f32(0.0); self.element_count()];
        let mut buffers = GpuFft3dBuffers::new(self);
        self.inverse_f16_values_into_with_buffers(field_hat.as_ref(), &mut out, &mut buffers)?;
        leto_array3_from_slice(
            [self.nx, self.ny, self.nz],
            &out,
            "FFT-WGPU f16 inverse Leto field",
        )
    }

    fn validate_leto_3d_shape(&self, shape: [usize; 3]) -> ApolloResult<()> {
        let expected = [self.nx, self.ny, self.nz];
        if shape == expected {
            Ok(())
        } else {
            Err(ApolloError::ShapeMismatch {
                expected: format!("{expected:?}"),
                actual: format!("{shape:?}"),
            })
        }
    }

    fn validate_interleaved_len(&self, len: usize) -> ApolloResult<()> {
        let expected = 2 * self.element_count();
        if len == expected {
            Ok(())
        } else {
            Err(ApolloError::ShapeMismatch {
                expected: format!("interleaved complex length {expected}"),
                actual: format!("interleaved complex length {len}"),
            })
        }
    }

    fn forward_values_into_with_buffers(
        &self,
        field: &[f64],
        out: &mut [f32],
        buffers: &mut GpuFft3dBuffers,
    ) -> ApolloResult<()> {
        buffers.assert_matches(self);
        if field.len() != buffers.len() {
            return Err(ApolloError::ShapeMismatch {
                expected: format!("real field length {}", buffers.len()),
                actual: format!("real field length {}", field.len()),
            });
        }
        self.validate_interleaved_len(out.len())?;

        buffers.im_host.fill(0.0);
        buffers
            .re_host
            .iter_mut()
            .zip(field.iter().copied())
            .for_each(|(dst, src)| *dst = src as f32);
        self.execute_forward_buffers(out, buffers);
        Ok(())
    }

    fn forward_f16_values_into_with_buffers(
        &self,
        field: &[f16],
        out: &mut [f32],
        buffers: &mut GpuFft3dBuffers,
    ) -> ApolloResult<()> {
        buffers.assert_matches(self);
        if field.len() != buffers.len() {
            return Err(ApolloError::ShapeMismatch {
                expected: format!("real field length {}", buffers.len()),
                actual: format!("real field length {}", field.len()),
            });
        }
        self.validate_interleaved_len(out.len())?;

        buffers.im_host.fill(0.0);
        buffers
            .re_host
            .iter_mut()
            .zip(field.iter().copied())
            .for_each(|(dst, src)| *dst = src.to_f32());
        self.execute_forward_buffers(out, buffers);
        Ok(())
    }

    fn inverse_values_into_with_buffers(
        &self,
        field_hat: &[f32],
        out: &mut [f64],
        buffers: &mut GpuFft3dBuffers,
    ) -> ApolloResult<()> {
        buffers.assert_matches(self);
        self.validate_interleaved_len(field_hat.len())?;
        if out.len() != buffers.len() {
            return Err(ApolloError::ShapeMismatch {
                expected: format!("real field length {}", buffers.len()),
                actual: format!("real field length {}", out.len()),
            });
        }

        Self::load_interleaved_spectrum(field_hat, buffers);
        self.execute_inverse_buffers(buffers);
        for (dst, &value) in out.iter_mut().zip(buffers.re_host.iter()) {
            *dst = value as f64;
        }
        Ok(())
    }

    fn inverse_f16_values_into_with_buffers(
        &self,
        field_hat: &[f32],
        out: &mut [f16],
        buffers: &mut GpuFft3dBuffers,
    ) -> ApolloResult<()> {
        buffers.assert_matches(self);
        self.validate_interleaved_len(field_hat.len())?;
        if out.len() != buffers.len() {
            return Err(ApolloError::ShapeMismatch {
                expected: format!("real field length {}", buffers.len()),
                actual: format!("real field length {}", out.len()),
            });
        }

        Self::load_interleaved_spectrum(field_hat, buffers);
        self.execute_inverse_buffers(buffers);
        for (dst, &value) in out.iter_mut().zip(buffers.re_host.iter()) {
            *dst = f16::from_f32(value);
        }
        Ok(())
    }

    fn execute_forward_buffers(&self, out: &mut [f32], buffers: &mut GpuFft3dBuffers) {
        self.queue
            .write_buffer(&buffers.re_buf, 0, bytemuck::cast_slice(&buffers.re_host));
        self.queue
            .write_buffer(&buffers.im_buf, 0, bytemuck::cast_slice(&buffers.im_host));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-fft-wgpu leto forward encoder"),
            });
        self.encode_forward_split(&mut encoder, &buffers.re_buf, &buffers.im_buf);
        self.queue.submit(std::iter::once(encoder.finish()));

        buffers.read_split_into_host(self);
        for ((re, im), pair) in buffers
            .re_host
            .iter()
            .zip(buffers.im_host.iter())
            .zip(out.chunks_exact_mut(2))
        {
            pair[0] = *re;
            pair[1] = *im;
        }
    }

    fn load_interleaved_spectrum(field_hat: &[f32], buffers: &mut GpuFft3dBuffers) {
        for (idx, pair) in field_hat.chunks_exact(2).enumerate() {
            buffers.re_host[idx] = pair[0];
            buffers.im_host[idx] = pair[1];
        }
    }

    fn execute_inverse_buffers(&self, buffers: &mut GpuFft3dBuffers) {
        self.queue
            .write_buffer(&buffers.re_buf, 0, bytemuck::cast_slice(&buffers.re_host));
        self.queue
            .write_buffer(&buffers.im_buf, 0, bytemuck::cast_slice(&buffers.im_host));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-fft-wgpu leto inverse encoder"),
            });
        self.encode_inverse_split(&mut encoder, &buffers.re_buf, &buffers.im_buf);
        self.queue.submit(std::iter::once(encoder.finish()));

        buffers.read_split_into_host(self);
    }

    pub(crate) fn run_device_axis_fft(&self, axis: Axis, inverse: bool) {
        let axis_len = axis.len(self.nx, self.ny, self.nz) as u32;
        let batch_count = axis.batch_count(self.nx, self.ny, self.nz) as u32;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-fft-wgpu device axis encoder"),
            });

        self.dispatch_pack(&mut encoder, axis, axis_len, batch_count);
        match axis {
            Axis::Z => match self.strategy_z {
                AxisStrategy::Radix2 => {
                    let stages = if inverse {
                        &self.axis_inv_z
                    } else {
                        &self.axis_fwd_z
                    };
                    self.dispatch_radix2(&mut encoder, stages);
                }
                AxisStrategy::ChirpZ { .. } => self.dispatch_chirp(
                    &mut encoder,
                    self.chirp_z.as_ref().expect("chirp_z exists for ChirpZ"),
                    inverse,
                ),
            },
            Axis::Y => match self.strategy_y {
                AxisStrategy::Radix2 => {
                    let stages = if inverse {
                        &self.axis_inv_y
                    } else {
                        &self.axis_fwd_y
                    };
                    self.dispatch_radix2(&mut encoder, stages);
                }
                AxisStrategy::ChirpZ { .. } => self.dispatch_chirp(
                    &mut encoder,
                    self.chirp_y.as_ref().expect("chirp_y exists for ChirpZ"),
                    inverse,
                ),
            },
            Axis::X => match self.strategy_x {
                AxisStrategy::Radix2 => {
                    let stages = if inverse {
                        &self.axis_inv_x
                    } else {
                        &self.axis_fwd_x
                    };
                    self.dispatch_radix2(&mut encoder, stages);
                }
                AxisStrategy::ChirpZ { .. } => self.dispatch_chirp(
                    &mut encoder,
                    self.chirp_x.as_ref().expect("chirp_x exists for ChirpZ"),
                    inverse,
                ),
            },
        }
        self.dispatch_unpack(&mut encoder, axis, axis_len, batch_count);
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    #[inline]
    pub(crate) fn element_count(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    #[inline]
    pub(crate) fn component_buffer_size(&self) -> u64 {
        (self.element_count() * std::mem::size_of::<f32>()) as u64
    }

    /// Encode a forward 3D FFT using external split real/imaginary wgpu buffers.
    ///
    /// **Theorem (Bluestein Chirp-Z Transform, Bluestein 1970, IEEE Trans. Audio
    /// Electroacoust. 18:451–455):**
    /// For an N-point DFT `X[k] = Σ_{n=0}^{N-1} x[n]·W^{nk}`, `W = e^{-2πi/N}`,
    /// using the identity `nk = [n² + k² − (k−n)²]/2`:
    /// ```text
    /// X[k] = W^{k²/2} · Σ_n [ x[n]·W^{n²/2} · W^{-(k−n)²/2} ]
    /// ```
    /// The inner sum is a length-N discrete convolution, computed via M-point radix-2
    /// FFT where `M = next_pow2(2N−1)`, giving O(N log N) complexity for arbitrary N.
    ///
    /// **Algorithm (forward separable 3D FFT):**
    /// 1. Copy `re_buf`/`im_buf` (caller-owned, COPY_SRC) → `full_re_buf`/`full_im_buf`
    /// 2. Z-axis: pack Z-slices into workspace, FFT (Radix2 or ChirpZ), unpack
    /// 3. Y-axis: pack Y-slices into workspace, FFT, unpack
    /// 4. X-axis: pack X-slices into workspace, FFT, unpack
    /// 5. Copy `full_re_buf`/`full_im_buf` → `re_buf`/`im_buf` (COPY_DST)
    ///
    /// All dispatches are encoded into `encoder`; the caller is responsible for
    /// submission. `re_buf` and `im_buf` must be created with
    /// `STORAGE | COPY_SRC | COPY_DST` usage flags and have
    /// `nx * ny * nz * 4` bytes each.
    pub fn encode_forward_split(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        re_buf: &wgpu::Buffer,
        im_buf: &wgpu::Buffer,
    ) {
        let buf_size = (self.nx * self.ny * self.nz * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(re_buf, 0, &self.full_re_buf, 0, buf_size);
        encoder.copy_buffer_to_buffer(im_buf, 0, &self.full_im_buf, 0, buf_size);
        self.encode_axis_fft(encoder, Axis::Z, false);
        self.encode_axis_fft(encoder, Axis::Y, false);
        self.encode_axis_fft(encoder, Axis::X, false);
        encoder.copy_buffer_to_buffer(&self.full_re_buf, 0, re_buf, 0, buf_size);
        encoder.copy_buffer_to_buffer(&self.full_im_buf, 0, im_buf, 0, buf_size);
    }

    /// Encode an inverse 3D FFT using external split real/imaginary wgpu buffers.
    ///
    /// **Theorem (Inverse DFT normalization):** The inverse DFT is
    /// `x[n] = (1/N) Σ_{k=0}^{N-1} X[k]·W^{-nk}`. For separable 3D, the
    /// normalization factor is `1/(Nx·Ny·Nz)`, applied axis-by-axis by the
    /// Radix2 inverse scale pipeline and the ChirpZ `chirp_scale` kernel.
    ///
    /// **Algorithm (inverse separable 3D FFT):**
    /// 1. Copy `re_buf`/`im_buf` → `full_re_buf`/`full_im_buf`
    /// 2. X-axis inverse FFT (including 1/Nx normalization)
    /// 3. Y-axis inverse FFT (including 1/Ny normalization)
    /// 4. Z-axis inverse FFT (including 1/Nz normalization)
    /// 5. Copy `full_re_buf`/`full_im_buf` → `re_buf`/`im_buf`
    ///
    /// Buffer requirements to `encode_forward_split`.
    ///
    /// Consumers that need the unnormalized IDFT must pre-scale their inputs
    /// or post-scale the outputs by the relevant axis length(s).
    pub fn encode_inverse_split(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        re_buf: &wgpu::Buffer,
        im_buf: &wgpu::Buffer,
    ) {
        let buf_size = (self.nx * self.ny * self.nz * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(re_buf, 0, &self.full_re_buf, 0, buf_size);
        encoder.copy_buffer_to_buffer(im_buf, 0, &self.full_im_buf, 0, buf_size);
        self.encode_axis_fft(encoder, Axis::X, true);
        self.encode_axis_fft(encoder, Axis::Y, true);
        self.encode_axis_fft(encoder, Axis::Z, true);
        encoder.copy_buffer_to_buffer(&self.full_re_buf, 0, re_buf, 0, buf_size);
        encoder.copy_buffer_to_buffer(&self.full_im_buf, 0, im_buf, 0, buf_size);
    }

    /// Encode one axis of the 3D FFT into the provided command encoder.
    ///
    /// Dispatches: pack (volume→workspace), FFT (Radix2 or ChirpZ), unpack (workspace→volume).
    /// This helper is the encoder-resident equivalent of `run_device_axis_fft`.
    fn encode_axis_fft(&self, encoder: &mut wgpu::CommandEncoder, axis: Axis, inverse: bool) {
        let axis_len = axis.len(self.nx, self.ny, self.nz) as u32;
        let batch_count = axis.batch_count(self.nx, self.ny, self.nz) as u32;
        self.dispatch_pack(encoder, axis, axis_len, batch_count);
        match axis {
            Axis::Z => match self.strategy_z {
                AxisStrategy::Radix2 => {
                    let stages = if inverse {
                        &self.axis_inv_z
                    } else {
                        &self.axis_fwd_z
                    };
                    self.dispatch_radix2(encoder, stages);
                }
                AxisStrategy::ChirpZ { .. } => {
                    self.dispatch_chirp(
                        encoder,
                        self.chirp_z
                            .as_ref()
                            .expect("chirp_z present for ChirpZ strategy"),
                        inverse,
                    );
                }
            },
            Axis::Y => match self.strategy_y {
                AxisStrategy::Radix2 => {
                    let stages = if inverse {
                        &self.axis_inv_y
                    } else {
                        &self.axis_fwd_y
                    };
                    self.dispatch_radix2(encoder, stages);
                }
                AxisStrategy::ChirpZ { .. } => {
                    self.dispatch_chirp(
                        encoder,
                        self.chirp_y
                            .as_ref()
                            .expect("chirp_y present for ChirpZ strategy"),
                        inverse,
                    );
                }
            },
            Axis::X => match self.strategy_x {
                AxisStrategy::Radix2 => {
                    let stages = if inverse {
                        &self.axis_inv_x
                    } else {
                        &self.axis_fwd_x
                    };
                    self.dispatch_radix2(encoder, stages);
                }
                AxisStrategy::ChirpZ { .. } => {
                    self.dispatch_chirp(
                        encoder,
                        self.chirp_x
                            .as_ref()
                            .expect("chirp_x present for ChirpZ strategy"),
                        inverse,
                    );
                }
            },
        }
        self.dispatch_unpack(encoder, axis, axis_len, batch_count);
    }

    pub(crate) fn read_back_full_buffers(&self) -> (Vec<f32>, Vec<f32>) {
        let n = self.nx * self.ny * self.nz;
        let buf_size = (n * std::mem::size_of::<f32>()) as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-fft-wgpu readback encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.full_re_buf, 0, &self.full_re_staging, 0, buf_size);
        encoder.copy_buffer_to_buffer(&self.full_im_buf, 0, &self.full_im_staging, 0, buf_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let re_slice = self.full_re_staging.slice(..buf_size);
        let im_slice = self.full_im_staging.slice(..buf_size);
        re_slice.map_async(wgpu::MapMode::Read, |_| {});
        im_slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait);

        let re_out = {
            let mapped = re_slice.get_mapped_range();
            bytemuck::cast_slice(&mapped).to_vec()
        };
        let im_out = {
            let mapped = im_slice.get_mapped_range();
            bytemuck::cast_slice(&mapped).to_vec()
        };
        self.full_re_staging.unmap();
        self.full_im_staging.unmap();
        (re_out, im_out)
    }
}

fn leto_view1_cow<T: Copy>(view: leto::ArrayView1<'_, T>) -> Cow<'_, [T]> {
    leto_interop::view1_cow(&view)
}
fn leto_view3_cow<T: Copy>(view: leto::ArrayView3<'_, T>) -> ApolloResult<Cow<'_, [T]>> {
    if let Some(slice) = view.as_slice() {
        return Ok(Cow::Borrowed(slice));
    }
    let shape = view.shape();
    let mut values = Vec::with_capacity(shape[0] * shape[1] * shape[2]);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                values.push(
                    *view
                        .get([i, j, k])
                        .map_err(|err| ApolloError::ShapeMismatch {
                            expected: "valid Leto FFT-WGPU 3D view".to_string(),
                            actual: format!("{err:?}"),
                        })?,
                );
            }
        }
    }
    Ok(Cow::Owned(values))
}

fn leto_array1_from_slice<T: Copy>(
    values: &[T],
    context: &str,
) -> ApolloResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    leto_interop::try_array1_from_slice(values).ok_or_else(|| ApolloError::Wgpu {
        message: format!("failed to allocate Mnemosyne-backed Leto {context}"),
    })
}

fn leto_array3_from_slice<T: Copy>(
    shape: [usize; 3],
    values: &[T],
    context: &str,
) -> ApolloResult<leto::Array<T, leto::MnemosyneStorage<T>, 3>> {
    leto::Array::from_mnemosyne_slice(shape, values).map_err(|err| ApolloError::Wgpu {
        message: format!("failed to allocate Mnemosyne-backed Leto {context}: {err:?}"),
    })
}

#[cfg(test)]
mod tests {
    use crate::infrastructure::transport::gpu::WgpuBackend;
    use crate::{f16, FftBackend, PrecisionProfile, Shape3D};
    use leto::{SliceArg, Storage};
    use std::borrow::Cow;

    use super::{leto_view3_cow, GpuFft3dBuffers};

    #[test]
    fn reusable_buffers_match_allocating_forward_and_inverse_when_device_exists() {
        let Ok(backend) = WgpuBackend::try_default() else {
            return;
        };
        let plan = backend
            .plan_3d(Shape3D::new(2, 2, 2).expect("shape"))
            .expect("gpu plan");
        let field = ndarray::Array3::from_shape_vec(
            (2, 2, 2),
            vec![1.0_f64, -2.0, 0.5, 3.0, -1.25, 0.75, 2.5, -0.5],
        )
        .expect("field");
        let mut buffers = GpuFft3dBuffers::new(&plan);

        let allocating_forward = plan.forward(&field);
        let mut reusable_forward = vec![0.0_f32; allocating_forward.len()];
        plan.forward_into_with_buffers(&field, &mut reusable_forward, &mut buffers);

        assert_eq!(reusable_forward.len(), allocating_forward.len());
        for (actual, expected) in reusable_forward.iter().zip(allocating_forward.iter()) {
            assert!(
                (actual - expected).abs() < 1.0e-5,
                "forward mismatch: actual={actual}, expected={expected}"
            );
        }

        let mut allocating_inverse = ndarray::Array3::<f64>::zeros((2, 2, 2));
        let mut reusable_inverse = ndarray::Array3::<f64>::zeros((2, 2, 2));
        plan.inverse(&allocating_forward, &mut allocating_inverse);
        plan.inverse_with_buffers(&reusable_forward, &mut reusable_inverse, &mut buffers);

        for (actual, expected) in reusable_inverse.iter().zip(allocating_inverse.iter()) {
            assert!(
                (actual - expected).abs() < 1.0e-5,
                "inverse mismatch: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn mixed_precision_f16_storage_matches_represented_f32_path_when_device_exists() {
        let Ok(backend) = WgpuBackend::try_default() else {
            return;
        };
        let capabilities = backend.capabilities();
        assert!(capabilities.supports_mixed_precision);
        assert!(capabilities
            .supported_precision_profiles
            .contains(&PrecisionProfile::MIXED_PRECISION_F16_F32));

        let plan = backend
            .plan_3d(Shape3D::new(2, 2, 2).expect("shape"))
            .expect("gpu plan");
        let values = [1.0_f32, -2.0, 0.5, 3.0, -1.25, 0.75, 2.5, -0.5];
        let field_f16 = ndarray::Array3::from_shape_vec(
            (2, 2, 2),
            values.iter().copied().map(f16::from_f32).collect(),
        )
        .expect("f16 field");
        let represented = ndarray::Array3::from_shape_vec(
            (2, 2, 2),
            field_f16
                .iter()
                .copied()
                .map(|value| value.to_f32() as f64)
                .collect(),
        )
        .expect("represented field");
        let mut buffers = GpuFft3dBuffers::new(&plan);

        let expected = plan.forward(&represented);
        let mut actual = vec![0.0_f32; expected.len()];
        plan.forward_f16_into_with_buffers(&field_f16, &mut actual, &mut buffers);

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < 1.0e-5,
                "mixed forward mismatch: actual={actual}, expected={expected}"
            );
        }

        let mut reconstructed = ndarray::Array3::from_elem((2, 2, 2), f16::from_f32(0.0));
        plan.inverse_f16_with_buffers(&actual, &mut reconstructed, &mut buffers);

        for (actual, expected) in reconstructed.iter().zip(field_f16.iter()) {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "mixed inverse mismatch: actual={}, expected={}",
                actual.to_f32(),
                expected.to_f32()
            );
        }
    }

    #[test]
    fn leto_view3_cow_borrows_contiguous_views() {
        let input = leto::Array3::from_shape_vec([2, 2, 2], vec![1_u32, 2, 3, 4, 5, 6, 7, 8])
            .expect("input");
        let cow = leto_view3_cow(input.view()).expect("contiguous view");
        assert!(matches!(cow, Cow::Borrowed(_)));
        assert_eq!(cow.as_ref(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn leto_view3_cow_materializes_strided_views() {
        let input = leto::Array3::from_shape_vec(
            [2, 2, 4],
            vec![1_u32, 99, 2, 99, 3, 99, 4, 99, 5, 99, 6, 99, 7, 99, 8, 99],
        )
        .expect("input");
        let view = input
            .slice_with::<3>(&[
                SliceArg::range(Some(0), None, 1),
                SliceArg::range(Some(0), None, 1),
                SliceArg::range(Some(0), None, 2),
            ])
            .expect("strided view");
        let cow = leto_view3_cow(view).expect("strided view");
        assert!(matches!(cow, Cow::Owned(_)));
        assert_eq!(cow.as_ref(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn leto_forward_and_inverse_match_ndarray_when_device_exists() {
        let Ok(backend) = WgpuBackend::try_default() else {
            return;
        };
        let plan = backend
            .plan_3d(Shape3D::new(2, 2, 2).expect("shape"))
            .expect("gpu plan");
        let values = vec![1.0_f64, -2.0, 0.5, 3.0, -1.25, 0.75, 2.5, -0.5];
        let field = ndarray::Array3::from_shape_vec((2, 2, 2), values.clone()).expect("field");
        let leto_field = leto::Array3::from_shape_vec([2, 2, 2], values).expect("leto field");

        let expected_forward = plan.forward(&field);
        let actual_forward = plan.forward_leto(leto_field.view()).expect("leto forward");
        assert_eq!(
            actual_forward.storage().as_slice(),
            expected_forward.as_slice()
        );

        let mut expected_inverse = ndarray::Array3::<f64>::zeros((2, 2, 2));
        plan.inverse(&expected_forward, &mut expected_inverse);
        let leto_spectrum =
            leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
                .expect("leto spectrum");
        let actual_inverse = plan
            .inverse_leto(leto_spectrum.view())
            .expect("leto inverse");
        for (actual, expected) in actual_inverse
            .storage()
            .as_slice()
            .iter()
            .zip(expected_inverse.iter())
        {
            assert!(
                (actual - expected).abs() < 1.0e-5,
                "inverse mismatch: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn leto_strided_forward_matches_logical_ndarray_when_device_exists() {
        let Ok(backend) = WgpuBackend::try_default() else {
            return;
        };
        let plan = backend
            .plan_3d(Shape3D::new(2, 2, 2).expect("shape"))
            .expect("gpu plan");
        let backing = vec![
            1.0_f64, 99.0, -2.0, 99.0, 0.5, 99.0, 3.0, 99.0, -1.25, 99.0, 0.75, 99.0, 2.5, 99.0,
            -0.5, 99.0,
        ];
        let logical = vec![1.0_f64, -2.0, 0.5, 3.0, -1.25, 0.75, 2.5, -0.5];
        let field = ndarray::Array3::from_shape_vec((2, 2, 2), logical).expect("field");
        let leto_field = leto::Array3::from_shape_vec([2, 2, 4], backing).expect("leto field");
        let view = leto_field
            .slice_with::<3>(&[
                SliceArg::range(Some(0), None, 1),
                SliceArg::range(Some(0), None, 1),
                SliceArg::range(Some(0), None, 2),
            ])
            .expect("strided view");

        let expected = plan.forward(&field);
        let actual = plan.forward_leto(view).expect("leto forward");
        assert_eq!(actual.storage().as_slice(), expected.as_slice());
    }

    #[test]
    fn typed_leto_forward_and_inverse_match_ndarray_f16_when_device_exists() {
        let Ok(backend) = WgpuBackend::try_default() else {
            return;
        };
        let plan = backend
            .plan_3d(Shape3D::new(2, 2, 2).expect("shape"))
            .expect("gpu plan");
        let values = [1.0_f32, -2.0, 0.5, 3.0, -1.25, 0.75, 2.5, -0.5];
        let field_f16 = ndarray::Array3::from_shape_vec(
            (2, 2, 2),
            values.iter().copied().map(f16::from_f32).collect(),
        )
        .expect("f16 field");
        let leto_field = leto::Array3::from_shape_vec(
            [2, 2, 2],
            values.iter().copied().map(f16::from_f32).collect(),
        )
        .expect("leto f16 field");

        let expected_forward = plan.forward_f16(&field_f16);
        let actual_forward = plan
            .forward_f16_leto(leto_field.view())
            .expect("leto f16 forward");
        assert_eq!(
            actual_forward.storage().as_slice(),
            expected_forward.as_slice()
        );

        let mut expected_inverse = ndarray::Array3::from_elem((2, 2, 2), f16::from_f32(0.0));
        plan.inverse_f16(&expected_forward, &mut expected_inverse);
        let leto_spectrum =
            leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
                .expect("leto spectrum");
        let actual_inverse = plan
            .inverse_f16_leto(leto_spectrum.view())
            .expect("leto f16 inverse");
        for (actual, expected) in actual_inverse
            .storage()
            .as_slice()
            .iter()
            .zip(expected_inverse.iter())
        {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "f16 inverse mismatch: actual={}, expected={}",
                actual.to_f32(),
                expected.to_f32()
            );
        }
    }
}
