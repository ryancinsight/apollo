//! Shared real-device acquisition for private Wavelet GPU verification.

use crate::infrastructure::transport::gpu::WaveletWgpuBackend;

pub(super) fn backend() -> Option<WaveletWgpuBackend> {
    WaveletWgpuBackend::try_default().ok()
}
