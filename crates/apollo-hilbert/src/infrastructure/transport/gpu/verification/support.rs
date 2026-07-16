//! Shared real-device acquisition for private Hilbert GPU verification.

use crate::infrastructure::transport::gpu::HilbertWgpuBackend;

pub(super) fn backend() -> Option<HilbertWgpuBackend> {
    match hephaestus_wgpu::WgpuDevice::try_default("apollo-hilbert-wgpu") {
        Ok(device) => Some(HilbertWgpuBackend::new(device)),
        Err(hephaestus_core::HephaestusError::AdapterUnavailable { .. }) => None,
        Err(error) => panic!("Hilbert GPU verification requires a working provider: {error}"),
    }
}
