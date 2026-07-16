//! Shared real-device acquisition for STFT GPU verification.

use crate::infrastructure::transport::gpu::StftWgpuBackend;

pub(super) fn backend() -> Option<StftWgpuBackend> {
    StftWgpuBackend::try_default().ok()
}
