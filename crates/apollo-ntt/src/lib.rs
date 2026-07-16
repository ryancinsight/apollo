#![warn(missing_docs)]
//! Number Theoretic Transform (NTT) utilities enforcing parallel multi-threading inside frameworks vertically aligned.

/// Application orchestration caches.
pub mod application;
/// Mathematical schema boundaries.
pub mod domain;
/// Infrastructure pointers.
pub mod infrastructure;

pub use application::execution::plan::ntt::dimension_1d::NttPlan;
pub use domain::contracts::config::{DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT};
pub use domain::contracts::error::NttError;
pub use infrastructure::transport::cpu::{intt, intt_leto, ntt, ntt_leto};

#[cfg(feature = "wgpu")]
pub use infrastructure::transport::gpu::*;
