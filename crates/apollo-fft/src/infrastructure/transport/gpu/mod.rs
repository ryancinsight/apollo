#![warn(missing_docs)]
//! WGPU dense FFT backend surface for Apollo.
//!
//! The current adapter validates device availability and exposes the same
//! numerical contract as the CPU dense FFT backend. NUFFT-specific GPU
//! execution is intentionally owned by `apollo-nufft-wgpu`.

pub mod application;
pub mod domain;
pub mod infrastructure;

use crate::domain::contracts::backend::BackendCapabilities;
use crate::{ApolloError, ApolloResult, BackendKind, FftBackend, Shape1D, Shape2D, Shape3D};
use apollo_wgpu_helpers::{WgpuDevice, WgpuStorage};
use crate::coeus_core::{self, ComputeBackend, Scalar, Storage};
pub use infrastructure::gpu_fft::{gpu_fft_available, GpuFft3d, GpuFft3dBuffers};

#[cfg(feature = "native-f16")]
pub use infrastructure::gpu_fft::GpuFft3dF16Native;

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct WgpuBackend {
    device: WgpuDevice,
}

impl WgpuBackend {
    /// Create a backend from an existing device and queue.
    #[must_use]
    pub fn new(device: WgpuDevice) -> Self {
        Self { device }
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> ApolloResult<Self> {
        Ok(Self::new(
            WgpuDevice::try_default("apollo-fft-wgpu").map_err(|e| ApolloError::Wgpu {
                message: e.to_string(),
            })?,
        ))
    }
}

impl crate::coeus_core::backend::private::Sealed for WgpuBackend {}

impl ComputeBackend for WgpuBackend {
    type DeviceBuffer<T: Scalar> = WgpuStorage<T>;
    type KernelDescriptor = ();
    type DispatchFuture<T: Scalar> = std::future::Ready<T>;

    #[inline]
    fn name(&self) -> &'static str {
        "wgpu"
    }

    #[inline]
    fn num_threads(&self) -> usize {
        1
    }

    #[inline]
    fn allocate<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        WgpuStorage::new_on(&self.device, len)
    }

    #[inline]
    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) {
        let len = dst.len();
        let host = vec![val; len];
        self.copy_to_device(&host, dst);
    }

    #[inline]
    fn copy_to_device<T: Scalar>(&self, src: &[T], dst: &mut Self::DeviceBuffer<T>) {
        use hephaestus_wgpu::ComputeDevice;
        self.device.hephaestus().write_buffer(&dst.buffer, src).expect("Failed to write to device buffer");
    }

    #[inline]
    fn copy_to_host<T: Scalar>(&self, src: &Self::DeviceBuffer<T>, dst: &mut [T]) {
        use hephaestus_wgpu::ComputeDevice;
        self.device.hephaestus().download(&src.buffer, dst).expect("Failed to read from device buffer");
    }
}

impl<T: crate::coeus::FftScalar> crate::coeus::FftDeviceOps<T> for WgpuBackend {
    fn fft_1d(&self, signal: &Self::DeviceBuffer<T>) -> Self::DeviceBuffer<coeus_core::Complex<T>> {
        let len = signal.len();
        let mut host_signal = vec![T::zero(); len];
        self.copy_to_host(signal, &mut host_signal);
        let out_vec = T::fft_1d_impl(&host_signal);
        let mut out = self.allocate::<coeus_core::Complex<T>>(out_vec.len());
        self.copy_to_device(&out_vec, &mut out);
        out
    }

    fn ifft_1d(&self, spectrum: &Self::DeviceBuffer<coeus_core::Complex<T>>) -> Self::DeviceBuffer<T> {
        let len = spectrum.len();
        let mut host_spectrum = vec![coeus_core::Complex::new(T::zero(), T::zero()); len];
        self.copy_to_host(spectrum, &mut host_spectrum);
        let out_vec = T::ifft_1d_impl(&host_spectrum);
        let mut out = self.allocate::<T>(out_vec.len());
        self.copy_to_device(&out_vec, &mut out);
        out
    }
}

impl FftBackend for WgpuBackend {
    type Plan1D = ();
    type Plan2D = ();
    type Plan3D = GpuFft3d;

    fn backend_kind(&self) -> BackendKind {
        BackendKind::Wgpu
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::WGPU
    }

    fn plan_1d(&self, _shape: Shape1D) -> ApolloResult<Self::Plan1D> {
        Err(ApolloError::BackendUnavailable {
            backend: "wgpu 1D plans are not exposed in v1".to_string(),
        })
    }

    fn plan_2d(&self, _shape: Shape2D) -> ApolloResult<Self::Plan2D> {
        Err(ApolloError::BackendUnavailable {
            backend: "wgpu 2D plans are not exposed in v1".to_string(),
        })
    }

    fn plan_3d(&self, shape: Shape3D) -> ApolloResult<Self::Plan3D> {
        GpuFft3d::new(
            self.device.device().clone(),
            self.device.queue().clone(),
            shape.nx,
            shape.ny,
            shape.nz,
        )
        .map_err(|message| ApolloError::Wgpu { message })
    }
}
