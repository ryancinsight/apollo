//! Shared real-device availability boundary for NTT transport verification.

use crate::infrastructure::transport::gpu::NttWgpuBackend;

pub(super) fn backend() -> Option<NttWgpuBackend> {
    NttWgpuBackend::try_default().ok()
}
