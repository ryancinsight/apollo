//! Shared real-device acquisition for private Hilbert GPU verification.

use crate::infrastructure::transport::gpu::HilbertWgpuBackend;

pub(super) fn backend() -> Option<HilbertWgpuBackend> {
    HilbertWgpuBackend::try_default().ok()
}
