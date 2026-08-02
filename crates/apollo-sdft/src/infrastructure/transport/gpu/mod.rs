#![warn(missing_docs)]
//! WGPU backend boundary for Apollo SDFT.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns the SDFT pass kernels and their domain
//! names. The sliding DFT is asymmetric: real `f32` windows map to
//! [`apollo_fft::Complex32`] bins, so `Sample` and `Bin` differ and the
//! per-side typed dispatch carries reduced-precision storage on each.

/// Infrastructure boundary for the SDFT kernels.
pub mod infrastructure;
#[cfg(test)]
pub(crate) mod verification;

pub use apollo_fft::{WgpuCapabilities, WgpuError, WgpuResult};
pub use infrastructure::kernel::SdftGpuKernel;

/// Plan payload for the sliding DFT: window length and generated bin
/// count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowPlan {
    window_len: usize,
    bin_count: usize,
}

impl WindowPlan {
    /// Create a window-plan payload.
    ///
    /// Validation (`0 < bin_count <= window_len`, accelerator range) runs
    /// at dispatch.
    #[must_use]
    pub const fn new(window_len: usize, bin_count: usize) -> Self {
        Self {
            window_len,
            bin_count,
        }
    }

    /// Return the window length carried by this payload.
    #[must_use]
    pub const fn window_len(self) -> usize {
        self.window_len
    }

    /// Return the generated bin count carried by this payload.
    #[must_use]
    pub const fn bin_count(self) -> usize {
        self.bin_count
    }
}

/// Metadata-preserving WGPU plan descriptor.
pub type SdftWgpuPlan = apollo_fft::WgpuTransformPlan<SdftGpuKernel>;

/// WGPU backend descriptor.
pub type SdftWgpuBackend = apollo_fft::WgpuTransformBackend<SdftGpuKernel>;
