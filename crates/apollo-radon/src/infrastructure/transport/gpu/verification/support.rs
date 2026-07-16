//! Shared real-device acquisition for private Radon GPU verification.

use crate::infrastructure::transport::gpu::RadonWgpuBackend;

pub(super) fn backend() -> Option<RadonWgpuBackend> {
    match hephaestus_wgpu::WgpuDevice::try_default("apollo-radon-wgpu") {
        Ok(device) => Some(RadonWgpuBackend::new(device)),
        Err(hephaestus_core::HephaestusError::AdapterUnavailable { .. }) => None,
        Err(error) => panic!("Radon GPU verification requires a working provider: {error}"),
    }
}
