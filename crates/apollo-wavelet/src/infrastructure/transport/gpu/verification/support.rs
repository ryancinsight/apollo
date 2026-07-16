//! Shared real-device acquisition for private Wavelet GPU verification.

use crate::infrastructure::transport::gpu::WaveletWgpuBackend;

pub(super) fn backend() -> Option<WaveletWgpuBackend> {
    match hephaestus_wgpu::WgpuDevice::try_default("apollo-wavelet-wgpu") {
        Ok(device) => Some(WaveletWgpuBackend::new(device)),
        Err(hephaestus_core::HephaestusError::AdapterUnavailable { .. }) => None,
        Err(error) => panic!("Wavelet GPU verification requires a working provider: {error}"),
    }
}
