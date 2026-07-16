//! Shared real-device acquisition for private Radon GPU verification.

use crate::infrastructure::transport::gpu::RadonWgpuBackend;

pub(super) fn backend() -> Option<RadonWgpuBackend> {
    RadonWgpuBackend::try_default().ok()
}
