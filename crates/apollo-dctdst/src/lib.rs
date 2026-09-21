#![warn(missing_docs)]
//! DCT and DST real-to-real transform plans for Apollo.
//!
//! `apollo-dctdst` owns real-to-real cosine/sine transform metadata, direct
//! DCT-I/II/IV and DST-I/II/III/IV kernels, and value-semantic verification.
//! [`apollo_dctdst_core`] owns direct DCT-III mathematics.
//! These transforms encode even and odd boundary extensions used in image
//! compression, spectral methods, and compact real-domain analysis.

/// Application-layer DCT/DST plans.
pub mod application;
/// Domain contracts and metadata.
pub mod domain;
/// Infrastructure kernel namespace.
pub mod infrastructure;
#[cfg(test)]
mod verification;

pub use application::execution::plan::dctdst::{DctDstPlan, RealTransformStorage};
pub use domain::contracts::error::{DctDstError, DctDstResult};
pub use domain::metadata::kind::{RealTransformConfig, RealTransformKind};
pub use infrastructure::kernel::direct::{dct1, dct2, dct4, dst1, dst2, dst3, dst4};

#[cfg(feature = "wgpu")]
pub use infrastructure::transport::gpu::*;
