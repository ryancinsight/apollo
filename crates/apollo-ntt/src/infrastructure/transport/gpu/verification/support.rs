//! Shared real-device availability boundary for NTT transport verification.

use crate::infrastructure::transport::gpu::NttWgpuBackend;

pub(super) fn backend() -> Option<NttWgpuBackend> {
    match hephaestus_wgpu::WgpuDevice::try_default("apollo-ntt-wgpu") {
        Ok(device) => Some(NttWgpuBackend::new(device)),
        Err(hephaestus_core::HephaestusError::AdapterUnavailable { .. }) => None,
        Err(error) => panic!("NTT GPU verification requires a working provider: {error}"),
    }
}
