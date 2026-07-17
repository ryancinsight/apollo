//! One-dimensional CUDA FFT plan over Leto complex storage.

use hephaestus_core::{Binding, CommandStream, ComputeDevice, DispatchGrid, KernelDevice};
use hephaestus_cuda::{CudaBuffer, CudaDevice, CudaPrepared};
use leto::Array1;

use crate::infrastructure::transport::fft::{
    kernel::{BitReverse, Butterfly, FftKernel, Scale, WORKGROUP_SIZE},
    stages::RadixStages,
};
use crate::{ApolloError, ApolloResult, Complex32};

/// One-dimensional f32 split-complex FFT plan on a typed CUDA device.
///
/// The plan accepts the Atlas Leto array boundary and retains provider-native
/// split-complex buffers plus reusable host staging. It implements the
/// unnormalized forward DFT and the inverse normalized by `1 / N`.
pub struct CudaFft1d {
    device: CudaDevice,
    length: u32,
    real: CudaBuffer<f32>,
    imaginary: CudaBuffer<f32>,
    bit_reverse: CudaPrepared<FftKernel<f32, BitReverse>>,
    butterfly: CudaPrepared<FftKernel<f32, Butterfly>>,
    scale: CudaPrepared<FftKernel<f32, Scale>>,
    forward: RadixStages,
    inverse: RadixStages,
    host_real: Vec<f32>,
    host_imaginary: Vec<f32>,
}

fn cuda_error(error: impl core::fmt::Display) -> ApolloError {
    ApolloError::Cuda {
        message: error.to_string(),
    }
}

fn validate_length(length: usize) -> ApolloResult<u32> {
    if !length.is_power_of_two() {
        return Err(ApolloError::validation(
            "length",
            length.to_string(),
            "must be a non-zero power of two for the CUDA radix-two plan",
        ));
    }
    u32::try_from(length).map_err(|_| {
        ApolloError::validation(
            "length",
            length.to_string(),
            "must fit CUDA's u32 index domain",
        )
    })
}

fn grid(elements: u32) -> ApolloResult<DispatchGrid> {
    DispatchGrid::covering_domain(
        [
            usize::try_from(elements).expect("invariant: u32 fits usize"),
            1,
            1,
        ],
        [WORKGROUP_SIZE as usize, 1, 1],
    )
    .map_err(cuda_error)
}

impl CudaFft1d {
    /// Construct a radix-two CUDA plan from an existing typed provider device.
    ///
    /// # Errors
    /// Returns validation errors for zero, non-power-of-two, or unrepresentable
    /// lengths, and preserves typed Hephaestus allocation failures as CUDA
    /// backend errors.
    pub fn new(device: CudaDevice, length: usize) -> ApolloResult<Self> {
        let length_u32 = validate_length(length)?;
        let real = device.alloc_zeroed(length).map_err(cuda_error)?;
        let imaginary = device.alloc_zeroed(length).map_err(cuda_error)?;
        let bit_reverse = device
            .prepare(&FftKernel::<f32, BitReverse>::new())
            .map_err(cuda_error)?;
        let butterfly = device
            .prepare(&FftKernel::<f32, Butterfly>::new())
            .map_err(cuda_error)?;
        let scale = device
            .prepare(&FftKernel::<f32, Scale>::new())
            .map_err(cuda_error)?;
        Ok(Self {
            device,
            length: length_u32,
            real,
            imaginary,
            bit_reverse,
            butterfly,
            scale,
            forward: RadixStages::radix_two(length_u32, 1, false),
            inverse: RadixStages::radix_two(length_u32, 1, true),
            host_real: vec![0.0; length],
            host_imaginary: vec![0.0; length],
        })
    }

    /// Return the fixed logical FFT length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length as usize
    }

    /// Return whether the fixed FFT length is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Execute an unnormalized forward transform in place.
    ///
    /// # Errors
    /// Returns a shape error when `data` does not match the plan length and
    /// preserves any typed provider transfer, compile, dispatch, or readback
    /// failure.
    pub fn execute_forward_in_place(&mut self, data: &mut Array1<Complex32>) -> ApolloResult<()> {
        self.execute_in_place(data, false)
    }

    /// Execute an inverse transform normalized by the plan length in place.
    ///
    /// # Errors
    /// Returns a shape error when `data` does not match the plan length and
    /// preserves any typed provider transfer, compile, dispatch, or readback
    /// failure.
    pub fn execute_inverse_in_place(&mut self, data: &mut Array1<Complex32>) -> ApolloResult<()> {
        self.execute_in_place(data, true)
    }

    fn execute_in_place(
        &mut self,
        data: &mut Array1<Complex32>,
        inverse: bool,
    ) -> ApolloResult<()> {
        if data.size() != self.len() {
            return Err(ApolloError::ShapeMismatch {
                expected: format!("{} complex CUDA FFT values", self.len()),
                actual: format!("{} complex CUDA FFT values", data.size()),
            });
        }
        let input = data.as_slice().ok_or_else(|| ApolloError::NonContiguous {
            context: "CUDA FFT requires a contiguous Leto Array1".to_string(),
        })?;
        for ((real, imaginary), value) in self
            .host_real
            .iter_mut()
            .zip(self.host_imaginary.iter_mut())
            .zip(input.iter())
        {
            *real = value.re;
            *imaginary = value.im;
        }
        self.device
            .write_buffer(&self.real, &self.host_real)
            .map_err(cuda_error)?;
        self.device
            .write_buffer(&self.imaginary, &self.host_imaginary)
            .map_err(cuda_error)?;
        self.execute_radix(if inverse {
            &self.inverse
        } else {
            &self.forward
        })?;
        self.device
            .download(&self.real, &mut self.host_real)
            .map_err(cuda_error)?;
        self.device
            .download(&self.imaginary, &mut self.host_imaginary)
            .map_err(cuda_error)?;
        let output = data
            .as_slice_mut()
            .ok_or_else(|| ApolloError::NonContiguous {
                context: "CUDA FFT requires a contiguous Leto Array1".to_string(),
            })?;
        for (value, (&real, &imaginary)) in output
            .iter_mut()
            .zip(self.host_real.iter().zip(self.host_imaginary.iter()))
        {
            *value = Complex32::new(real, imaginary);
        }
        Ok(())
    }

    fn execute_radix(&self, stages: &RadixStages) -> ApolloResult<()> {
        let bindings = [
            Binding::read_write(&self.real),
            Binding::read_write(&self.imaginary),
        ];
        let mut stream = self.device.stream().map_err(cuda_error)?;
        stream
            .encode(
                &self.bit_reverse,
                &bindings,
                &stages.bit_reverse,
                grid(stages.fft_len)?,
            )
            .map_err(cuda_error)?;
        let butterfly_elements = stages
            .fft_len
            .checked_div(2)
            .expect("invariant: validated CUDA radix length is non-zero");
        for params in stages.butterflies.iter() {
            stream
                .encode(
                    &self.butterfly,
                    &bindings,
                    params,
                    grid(butterfly_elements)?,
                )
                .map_err(cuda_error)?;
        }
        if let Some(params) = stages.inverse_scale {
            stream
                .encode(&self.scale, &bindings, &params, grid(stages.fft_len)?)
                .map_err(cuda_error)?;
        }
        // The synchronizing typed downloads in `execute_in_place` observe this
        // stream's ordered work; do not add a redundant device-wide wait here.
        stream.submit().map_err(cuda_error)
    }
}

#[cfg(test)]
mod tests {
    use hephaestus_core::HephaestusError;
    use hephaestus_cuda::CudaDevice;
    use leto::Array1;

    use crate::{fft_1d_complex_typed, Complex32};

    use super::{validate_length, CudaFft1d};

    fn cuda_or_skip() -> Option<CudaDevice> {
        match CudaDevice::try_default() {
            Ok(device) => Some(device),
            Err(HephaestusError::AdapterUnavailable { .. }) => None,
            Err(error) => {
                panic!("CUDA FFT device-present regression requires a working provider: {error}");
            }
        }
    }

    #[cfg(feature = "wgpu")]
    fn wgpu_or_skip() -> Option<hephaestus_wgpu::WgpuDevice> {
        match hephaestus_wgpu::WgpuDevice::try_default("apollo-fft-cuda-wgpu-differential") {
            Ok(device) => Some(device),
            Err(HephaestusError::AdapterUnavailable { .. }) => None,
            Err(error) => {
                panic!("WGPU FFT differential requires a working provider: {error}");
            }
        }
    }

    fn signal() -> Array1<Complex32> {
        Array1::from_shape_vec(
            [8],
            (0..8)
                .map(|index| {
                    let value = index as f32;
                    Complex32::new((0.17 * value).sin(), (0.29 * value).cos())
                })
                .collect(),
        )
        .expect("fixed signal shape")
    }

    fn gamma_256() -> f32 {
        let roundoff = f32::EPSILON / 2.0;
        256.0 * roundoff / (1.0 - 256.0 * roundoff)
    }

    #[test]
    fn rejects_non_power_of_two_length() {
        let error = validate_length(6).expect_err("radix-two plan must reject length six");
        assert_eq!(
            error.to_string(),
            "validation failed for `length`: `6` violates `must be a non-zero power of two for the CUDA radix-two plan`"
        );
    }

    #[test]
    fn cuda_forward_matches_cpu_and_roundtrips_when_device_exists() {
        let Some(device) = cuda_or_skip() else {
            return;
        };
        let original = signal();
        let expected = fft_1d_complex_typed(&original);
        let norm_one: f32 = original
            .as_slice()
            .expect("owned Leto signal is contiguous")
            .iter()
            .map(|value| value.re.abs() + value.im.abs())
            .sum();
        let forward_bound = 2.0 * gamma_256() * norm_one;
        let roundtrip_bound = gamma_256() * (1.0 + norm_one);

        let mut actual = signal();
        let mut plan = CudaFft1d::new(device, actual.size()).expect("CUDA plan construction");
        plan.execute_forward_in_place(&mut actual)
            .expect("CUDA forward transform");

        for (index, (actual, expected)) in actual
            .as_slice()
            .expect("owned Leto signal is contiguous")
            .iter()
            .zip(
                expected
                    .as_slice()
                    .expect("CPU FFT output is contiguous")
                    .iter(),
            )
            .enumerate()
        {
            assert!(
                (actual.re - expected.re).abs() <= forward_bound,
                "real bin {index}: actual={}, expected={}, bound={forward_bound}",
                actual.re,
                expected.re
            );
            assert!(
                (actual.im - expected.im).abs() <= forward_bound,
                "imaginary bin {index}: actual={}, expected={}, bound={forward_bound}",
                actual.im,
                expected.im
            );
        }

        plan.execute_inverse_in_place(&mut actual)
            .expect("CUDA inverse transform");
        for (index, (actual, expected)) in actual
            .as_slice()
            .expect("owned Leto signal is contiguous")
            .iter()
            .zip(
                original
                    .as_slice()
                    .expect("owned Leto signal is contiguous")
                    .iter(),
            )
            .enumerate()
        {
            assert!(
                (actual.re - expected.re).abs() <= roundtrip_bound,
                "roundtrip real bin {index}: actual={}, expected={}, bound={roundtrip_bound}",
                actual.re,
                expected.re
            );
            assert!(
                (actual.im - expected.im).abs() <= roundtrip_bound,
                "roundtrip imaginary bin {index}: actual={}, expected={}, bound={roundtrip_bound}",
                actual.im,
                expected.im
            );
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn cuda_forward_matches_wgpu_when_devices_exist() {
        let Some(cuda_device) = cuda_or_skip() else {
            return;
        };
        let Some(wgpu_device) = wgpu_or_skip() else {
            return;
        };
        let original = signal();
        let norm_one: f32 = original
            .as_slice()
            .expect("owned Leto signal is contiguous")
            .iter()
            .map(|value| value.re.abs() + value.im.abs())
            .sum();
        let bound = 2.0 * gamma_256() * norm_one;

        let mut cuda_actual = signal();
        let mut cuda_plan =
            CudaFft1d::new(cuda_device, cuda_actual.size()).expect("CUDA plan construction");
        cuda_plan
            .execute_forward_in_place(&mut cuda_actual)
            .expect("CUDA forward transform");

        let input = original
            .as_slice()
            .expect("owned Leto signal is contiguous");
        let mut wgpu_real: Vec<f32> = input.iter().map(|value| value.re).collect();
        let mut wgpu_imaginary: Vec<f32> = input.iter().map(|value| value.im).collect();
        let wgpu_plan = crate::GpuFft3d::new(wgpu_device, original.size(), 1, 1)
            .expect("one-dimensional WGPU differential plan");
        wgpu_plan
            .execute_forward_in_place(&mut wgpu_real, &mut wgpu_imaginary)
            .expect("WGPU forward transform");

        for (index, (cuda, (&wgpu_real, &wgpu_imaginary))) in cuda_actual
            .as_slice()
            .expect("owned Leto signal is contiguous")
            .iter()
            .zip(wgpu_real.iter().zip(wgpu_imaginary.iter()))
            .enumerate()
        {
            assert!(
                (cuda.re - wgpu_real).abs() <= bound,
                "real bin {index}: CUDA={}, WGPU={}, bound={bound}",
                cuda.re,
                wgpu_real
            );
            assert!(
                (cuda.im - wgpu_imaginary).abs() <= bound,
                "imaginary bin {index}: CUDA={}, WGPU={}, bound={bound}",
                cuda.im,
                wgpu_imaginary
            );
        }
    }
}
