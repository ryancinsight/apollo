//! Ordered typed-kernel dispatch for dense FFT plans.

use hephaestus_core::{
    Binding, CommandStream, ComputeDevice, DeviceBuffer, DispatchGrid, KernelDevice,
};
use hephaestus_wgpu::{WgpuBuffer, WgpuCommandStream};

use crate::infrastructure::transport::fft::{
    kernel::{BitReverse, Butterfly, FftKernel, RadixFourBitReverse, RadixFourButterfly, Scale},
    stages::RadixStages,
};
use crate::{ApolloError, ApolloResult};

use super::{
    kernel::{
        ChirpKernel, ChirpNegateImaginary, ChirpPointMultiply, ChirpPostmultiply, ChirpPremultiply,
        ChirpScale, FftStorage, Pack, PackKernel, Unpack,
    },
    pipeline::GpuFft3d,
    strategy::{Axis, AxisStrategy, ChirpData},
};

fn provider_error(error: impl core::fmt::Display) -> ApolloError {
    ApolloError::Wgpu {
        message: error.to_string(),
    }
}

fn grid(elements: usize) -> ApolloResult<DispatchGrid> {
    DispatchGrid::covering_domain([elements, 1, 1], [256, 1, 1]).map_err(provider_error)
}

fn product_grid(left: u32, right: u32) -> ApolloResult<DispatchGrid> {
    let elements =
        usize::try_from(u64::from(left) * u64::from(right)).map_err(|_| ApolloError::Wgpu {
            message: format!("FFT dispatch element count {left} * {right} exceeds usize"),
        })?;
    grid(elements)
}

impl<T: FftStorage> GpuFft3d<T> {
    fn encode_pack(&self, stream: &mut WgpuCommandStream<'_>, axis: Axis) -> ApolloResult<()> {
        let params = match axis {
            Axis::X => self.pack_x,
            Axis::Y => self.pack_y,
            Axis::Z => self.pack_z,
        };
        let kernel = self
            .device
            .prepare(&PackKernel::<T, Pack>::new())
            .map_err(provider_error)?;
        let bindings = [
            Binding::read_write(&self.workspace_real),
            Binding::read_write(&self.workspace_imaginary),
            Binding::read_write(&self.volume_real),
            Binding::read_write(&self.volume_imaginary),
        ];
        stream
            .encode(
                &kernel,
                &bindings,
                &params,
                product_grid(params.n, params.batch_count)?,
            )
            .map_err(provider_error)
    }

    fn encode_unpack(&self, stream: &mut WgpuCommandStream<'_>, axis: Axis) -> ApolloResult<()> {
        let params = match axis {
            Axis::X => self.pack_x,
            Axis::Y => self.pack_y,
            Axis::Z => self.pack_z,
        };
        let kernel = self
            .device
            .prepare(&PackKernel::<T, Unpack>::new())
            .map_err(provider_error)?;
        let bindings = [
            Binding::read_write(&self.workspace_real),
            Binding::read_write(&self.workspace_imaginary),
            Binding::read_write(&self.volume_real),
            Binding::read_write(&self.volume_imaginary),
        ];
        stream
            .encode(
                &kernel,
                &bindings,
                &params,
                product_grid(params.n, params.batch_count)?,
            )
            .map_err(provider_error)
    }

    fn encode_radix(
        &self,
        stream: &mut WgpuCommandStream<'_>,
        stages: &RadixStages,
    ) -> ApolloResult<()> {
        if stages.fft_len == 0 {
            return Ok(());
        }
        let bindings = [
            Binding::read_write(&self.workspace_real),
            Binding::read_write(&self.workspace_imaginary),
        ];
        let bit_reverse_grid = product_grid(stages.batch_count, stages.fft_len)?;
        if stages.radix_four {
            let bit_reverse = self
                .device
                .prepare(&FftKernel::<T, RadixFourBitReverse>::new())
                .map_err(provider_error)?;
            let butterfly = self
                .device
                .prepare(&FftKernel::<T, RadixFourButterfly>::new())
                .map_err(provider_error)?;
            stream
                .encode(
                    &bit_reverse,
                    &bindings,
                    &stages.bit_reverse,
                    bit_reverse_grid,
                )
                .map_err(provider_error)?;
            let butterfly_grid = product_grid(stages.batch_count, stages.fft_len / 4)?;
            for params in stages.butterflies.iter() {
                stream
                    .encode(&butterfly, &bindings, params, butterfly_grid)
                    .map_err(provider_error)?;
            }
        } else {
            let bit_reverse = self
                .device
                .prepare(&FftKernel::<T, BitReverse>::new())
                .map_err(provider_error)?;
            let butterfly = self
                .device
                .prepare(&FftKernel::<T, Butterfly>::new())
                .map_err(provider_error)?;
            stream
                .encode(
                    &bit_reverse,
                    &bindings,
                    &stages.bit_reverse,
                    bit_reverse_grid,
                )
                .map_err(provider_error)?;
            let butterfly_grid = product_grid(stages.batch_count, stages.fft_len / 2)?;
            for params in stages.butterflies.iter() {
                stream
                    .encode(&butterfly, &bindings, params, butterfly_grid)
                    .map_err(provider_error)?;
            }
        }
        if let Some(params) = stages.inverse_scale {
            let scale = self
                .device
                .prepare(&FftKernel::<T, Scale>::new())
                .map_err(provider_error)?;
            stream
                .encode(&scale, &bindings, &params, bit_reverse_grid)
                .map_err(provider_error)?;
        }
        Ok(())
    }

    fn encode_chirp(
        &self,
        stream: &mut WgpuCommandStream<'_>,
        chirp: &ChirpData<T>,
        inverse: bool,
    ) -> ApolloResult<()> {
        let bindings = [
            Binding::read_write(&self.workspace_real),
            Binding::read_write(&self.workspace_imaginary),
            Binding::read(&chirp.real_kernel),
            Binding::read(&chirp.imaginary_kernel),
        ];
        let padded_grid = product_grid(chirp.params.m, chirp.params.batch_count)?;
        let output_grid = product_grid(chirp.params.n, chirp.params.batch_count)?;
        let premultiply = self
            .device
            .prepare(&ChirpKernel::<T, ChirpPremultiply>::new())
            .map_err(provider_error)?;
        let point_multiply = self
            .device
            .prepare(&ChirpKernel::<T, ChirpPointMultiply>::new())
            .map_err(provider_error)?;
        let postmultiply = self
            .device
            .prepare(&ChirpKernel::<T, ChirpPostmultiply>::new())
            .map_err(provider_error)?;
        let negate_imaginary = self
            .device
            .prepare(&ChirpKernel::<T, ChirpNegateImaginary>::new())
            .map_err(provider_error)?;
        if inverse {
            stream
                .encode(&negate_imaginary, &bindings, &chirp.params, output_grid)
                .map_err(provider_error)?;
        }
        stream
            .encode(&premultiply, &bindings, &chirp.params, padded_grid)
            .map_err(provider_error)?;
        self.encode_radix(stream, &chirp.forward_radix)?;
        stream
            .encode(&point_multiply, &bindings, &chirp.params, padded_grid)
            .map_err(provider_error)?;
        self.encode_radix(stream, &chirp.inverse_radix)?;
        stream
            .encode(&postmultiply, &bindings, &chirp.params, output_grid)
            .map_err(provider_error)?;
        if inverse {
            let scale = self
                .device
                .prepare(&ChirpKernel::<T, ChirpScale>::new())
                .map_err(provider_error)?;
            stream
                .encode(&negate_imaginary, &bindings, &chirp.params, output_grid)
                .map_err(provider_error)?;
            stream
                .encode(&scale, &bindings, &chirp.params, output_grid)
                .map_err(provider_error)?;
        }
        Ok(())
    }

    fn encode_axis(
        &self,
        stream: &mut WgpuCommandStream<'_>,
        axis: Axis,
        inverse: bool,
    ) -> ApolloResult<()> {
        self.encode_pack(stream, axis)?;
        match axis {
            Axis::X => match self.strategy_x {
                AxisStrategy::Radix2 => self.encode_radix(
                    stream,
                    if inverse {
                        &self.axis_inverse_x
                    } else {
                        &self.axis_forward_x
                    },
                )?,
                AxisStrategy::ChirpZ { .. } => self.encode_chirp(
                    stream,
                    self.chirp_x
                        .as_ref()
                        .expect("invariant: Chirp-Z strategy owns X-axis chirp data"),
                    inverse,
                )?,
            },
            Axis::Y => match self.strategy_y {
                AxisStrategy::Radix2 => self.encode_radix(
                    stream,
                    if inverse {
                        &self.axis_inverse_y
                    } else {
                        &self.axis_forward_y
                    },
                )?,
                AxisStrategy::ChirpZ { .. } => self.encode_chirp(
                    stream,
                    self.chirp_y
                        .as_ref()
                        .expect("invariant: Chirp-Z strategy owns Y-axis chirp data"),
                    inverse,
                )?,
            },
            Axis::Z => match self.strategy_z {
                AxisStrategy::Radix2 => self.encode_radix(
                    stream,
                    if inverse {
                        &self.axis_inverse_z
                    } else {
                        &self.axis_forward_z
                    },
                )?,
                AxisStrategy::ChirpZ { .. } => self.encode_chirp(
                    stream,
                    self.chirp_z
                        .as_ref()
                        .expect("invariant: Chirp-Z strategy owns Z-axis chirp data"),
                    inverse,
                )?,
            },
        }
        self.encode_unpack(stream, axis)
    }

    /// Encode a forward transform over external typed split-complex buffers.
    ///
    /// The stream records external-to-plan copies, Z/Y/X axis transforms, and
    /// plan-to-external copies in that exact dependency order.  The caller
    /// submits the stream only after composing any adjacent typed kernels.
    pub(crate) fn encode_forward_split_typed(
        &self,
        stream: &mut WgpuCommandStream<'_>,
        real: &WgpuBuffer<T>,
        imaginary: &WgpuBuffer<T>,
    ) -> ApolloResult<()> {
        self.validate_external_buffers(real, imaginary)?;
        stream
            .copy(real, &self.volume_real)
            .map_err(provider_error)?;
        stream
            .copy(imaginary, &self.volume_imaginary)
            .map_err(provider_error)?;
        self.encode_axis(stream, Axis::Z, false)?;
        self.encode_axis(stream, Axis::Y, false)?;
        self.encode_axis(stream, Axis::X, false)?;
        stream
            .copy(&self.volume_real, real)
            .map_err(provider_error)?;
        stream
            .copy(&self.volume_imaginary, imaginary)
            .map_err(provider_error)
    }

    /// Encode an inverse transform over external typed split-complex buffers.
    ///
    /// The inverse records X/Y/Z axis transforms.  Each inverse axis applies
    /// its `1/N` scale, so exact arithmetic satisfies `F^{-1}(F(x)) = x`.
    pub(crate) fn encode_inverse_split_typed(
        &self,
        stream: &mut WgpuCommandStream<'_>,
        real: &WgpuBuffer<T>,
        imaginary: &WgpuBuffer<T>,
    ) -> ApolloResult<()> {
        self.validate_external_buffers(real, imaginary)?;
        stream
            .copy(real, &self.volume_real)
            .map_err(provider_error)?;
        stream
            .copy(imaginary, &self.volume_imaginary)
            .map_err(provider_error)?;
        self.encode_axis(stream, Axis::X, true)?;
        self.encode_axis(stream, Axis::Y, true)?;
        self.encode_axis(stream, Axis::Z, true)?;
        stream
            .copy(&self.volume_real, real)
            .map_err(provider_error)?;
        stream
            .copy(&self.volume_imaginary, imaginary)
            .map_err(provider_error)
    }

    fn validate_external_buffers(
        &self,
        real: &WgpuBuffer<T>,
        imaginary: &WgpuBuffer<T>,
    ) -> ApolloResult<()> {
        let expected = self.element_count();
        for (component, actual) in [("real", real.len()), ("imaginary", imaginary.len())] {
            if actual != expected {
                return Err(ApolloError::ShapeMismatch {
                    expected: format!("{expected} {component} FFT values"),
                    actual: format!("{actual} {component} FFT values"),
                });
            }
        }
        Ok(())
    }

    /// Submit a forward transform from and back into typed host components.
    pub(crate) fn execute_forward_in_place(
        &self,
        real: &mut [T],
        imaginary: &mut [T],
    ) -> ApolloResult<()> {
        self.execute_in_place(real, imaginary, false)
    }

    /// Submit an inverse transform from and back into typed host components.
    pub(crate) fn execute_inverse_in_place(
        &self,
        real: &mut [T],
        imaginary: &mut [T],
    ) -> ApolloResult<()> {
        self.execute_in_place(real, imaginary, true)
    }

    fn execute_in_place(
        &self,
        real: &mut [T],
        imaginary: &mut [T],
        inverse: bool,
    ) -> ApolloResult<()> {
        let expected = self.element_count();
        for (component, values) in [("real", real.len()), ("imaginary", imaginary.len())] {
            if values != expected {
                return Err(ApolloError::ShapeMismatch {
                    expected: format!("{expected} {component} FFT values"),
                    actual: format!("{values} {component} FFT values"),
                });
            }
        }
        self.device
            .write_buffer(&self.volume_real, real)
            .map_err(provider_error)?;
        self.device
            .write_buffer(&self.volume_imaginary, imaginary)
            .map_err(provider_error)?;
        let mut stream = self.device.stream().map_err(provider_error)?;
        if inverse {
            self.encode_axis(&mut stream, Axis::X, true)?;
            self.encode_axis(&mut stream, Axis::Y, true)?;
            self.encode_axis(&mut stream, Axis::Z, true)?;
        } else {
            self.encode_axis(&mut stream, Axis::Z, false)?;
            self.encode_axis(&mut stream, Axis::Y, false)?;
            self.encode_axis(&mut stream, Axis::X, false)?;
        }
        stream.submit().map_err(provider_error)?;
        self.device
            .download(&self.volume_real, real)
            .map_err(provider_error)?;
        self.device
            .download(&self.volume_imaginary, imaginary)
            .map_err(provider_error)
    }
}

impl GpuFft3d<f32> {
    /// Encode a forward transform over external typed split-complex buffers.
    ///
    /// The stream records external-to-plan copies, Z/Y/X axis transforms, and
    /// plan-to-external copies in that exact dependency order. The caller
    /// submits the stream only after composing any adjacent typed kernels.
    pub fn encode_forward_split(
        &self,
        stream: &mut WgpuCommandStream<'_>,
        real: &WgpuBuffer<f32>,
        imaginary: &WgpuBuffer<f32>,
    ) -> ApolloResult<()> {
        self.encode_forward_split_typed(stream, real, imaginary)
    }

    /// Encode an inverse transform over external typed split-complex buffers.
    ///
    /// The inverse records X/Y/Z axis transforms. Each inverse axis applies
    /// its `1/N` scale, so exact arithmetic satisfies `F^{-1}(F(x)) = x`.
    pub fn encode_inverse_split(
        &self,
        stream: &mut WgpuCommandStream<'_>,
        real: &WgpuBuffer<f32>,
        imaginary: &WgpuBuffer<f32>,
    ) -> ApolloResult<()> {
        self.encode_inverse_split_typed(stream, real, imaginary)
    }
}

#[cfg(test)]
mod tests {
    use hephaestus_core::{CommandStream, ComputeDevice, HephaestusError, KernelDevice};
    use hephaestus_wgpu::WgpuDevice;

    use super::GpuFft3d;

    fn device_or_skip(application_name: &str) -> Option<WgpuDevice> {
        match WgpuDevice::try_default(application_name) {
            Ok(device) => Some(device),
            Err(HephaestusError::AdapterUnavailable { .. }) => None,
            Err(error) => {
                panic!("typed FFT device-present regression requires a working provider: {error}");
            }
        }
    }

    #[test]
    fn typed_external_buffers_preserve_delta_roundtrip_when_device_exists() {
        let Some(device) = device_or_skip("apollo-fft-typed-stream-test") else {
            return;
        };
        let plan = GpuFft3d::new(device.clone(), 2, 2, 2)
            .expect("2x2x2 typed FFT plan must fit the acquired device");
        let input = [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let real = device
            .upload(&input)
            .expect("typed upload of the analytical delta field");
        let imaginary = device
            .alloc_zeroed(input.len())
            .expect("typed allocation of the imaginary component");

        let mut forward_stream = device.stream().expect("typed command stream");
        plan.encode_forward_split(&mut forward_stream, &real, &imaginary)
            .expect("typed forward encoding");
        forward_stream.submit().expect("typed forward submission");

        let mut forward_real = [0.0_f32; 8];
        let mut forward_imaginary = [0.0_f32; 8];
        device
            .download(&real, &mut forward_real)
            .expect("typed forward real readback");
        device
            .download(&imaginary, &mut forward_imaginary)
            .expect("typed forward imaginary readback");
        assert_eq!(forward_real, [1.0; 8]);
        assert_eq!(forward_imaginary, [0.0; 8]);

        let mut inverse_stream = device.stream().expect("typed command stream");
        plan.encode_inverse_split(&mut inverse_stream, &real, &imaginary)
            .expect("typed inverse encoding");
        inverse_stream.submit().expect("typed inverse submission");

        let mut reconstructed_real = [0.0_f32; 8];
        let mut reconstructed_imaginary = [0.0_f32; 8];
        device
            .download(&real, &mut reconstructed_real)
            .expect("typed inverse real readback");
        device
            .download(&imaginary, &mut reconstructed_imaginary)
            .expect("typed inverse imaginary readback");
        assert_eq!(reconstructed_real, input);
        assert_eq!(reconstructed_imaginary, [0.0; 8]);
    }

    #[test]
    fn typed_external_bluestein_delta_matches_dft_and_roundtrips_when_device_exists() {
        let Some(device) = device_or_skip("apollo-fft-typed-bluestein-test") else {
            return;
        };
        let plan = GpuFft3d::new(device.clone(), 2, 3, 2)
            .expect("2x3x2 typed FFT plan must fit the acquired device");
        let input = [
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let real = device
            .upload(&input)
            .expect("typed upload of the analytical delta field");
        let imaginary = device
            .alloc_zeroed(input.len())
            .expect("typed allocation of the imaginary component");

        let mut forward_stream = device.stream().expect("typed command stream");
        plan.encode_forward_split(&mut forward_stream, &real, &imaginary)
            .expect("typed Bluestein forward encoding");
        forward_stream
            .submit()
            .expect("typed Bluestein forward submission");

        let mut forward_real = [0.0_f32; 12];
        let mut forward_imaginary = [0.0_f32; 12];
        device
            .download(&real, &mut forward_real)
            .expect("typed Bluestein real readback");
        device
            .download(&imaginary, &mut forward_imaginary)
            .expect("typed Bluestein imaginary readback");

        // The 3-point axis uses Bluestein with M=8.  A transformed component
        // traverses fewer than 256 f32 rounding operations (two radix-2 axes,
        // chirp pre/post factors, two 8-point FFTs, point multiplication, and
        // normalization), so gamma_256 bounds the delta DFT error by gamma_256
        // times the input l1 norm.  The inverse consumes 12 unit-magnitude
        // coefficients, yielding gamma_256 * (1 + 12) for the roundtrip.
        let unit_roundoff = f32::EPSILON / 2.0;
        let gamma_256 = 256.0 * unit_roundoff / (1.0 - 256.0 * unit_roundoff);
        let forward_bound = gamma_256;
        let roundtrip_bound = gamma_256 * 13.0;
        for value in forward_real {
            assert!((value - 1.0).abs() <= forward_bound);
        }
        for value in forward_imaginary {
            assert!(value.abs() <= forward_bound);
        }

        let mut inverse_stream = device.stream().expect("typed command stream");
        plan.encode_inverse_split(&mut inverse_stream, &real, &imaginary)
            .expect("typed Bluestein inverse encoding");
        inverse_stream
            .submit()
            .expect("typed Bluestein inverse submission");

        let mut reconstructed_real = [0.0_f32; 12];
        let mut reconstructed_imaginary = [0.0_f32; 12];
        device
            .download(&real, &mut reconstructed_real)
            .expect("typed Bluestein inverse real readback");
        device
            .download(&imaginary, &mut reconstructed_imaginary)
            .expect("typed Bluestein inverse imaginary readback");
        for (actual, expected) in reconstructed_real.into_iter().zip(input) {
            assert!((actual - expected).abs() <= roundtrip_bound);
        }
        for value in reconstructed_imaginary {
            assert!(value.abs() <= roundtrip_bound);
        }
    }
}
