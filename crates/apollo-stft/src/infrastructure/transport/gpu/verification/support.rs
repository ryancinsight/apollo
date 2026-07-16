//! Shared real-device acquisition for STFT GPU verification.

use crate::infrastructure::transport::gpu::StftWgpuBackend;

pub(super) fn backend() -> Option<StftWgpuBackend> {
    match hephaestus_wgpu::WgpuDevice::try_with_device_preference_and_optional_device_features_and_limits(
        "apollo-stft-wgpu",
        hephaestus_core::DevicePreference::HighPerformance,
        &[],
        StftWgpuBackend::required_device_limits(),
    ) {
        Ok(device) => Some(StftWgpuBackend::new(device)),
        Err(hephaestus_core::HephaestusError::AdapterUnavailable { .. }) => None,
        Err(error) => panic!("STFT GPU verification requires a working provider: {error}"),
    }
}
