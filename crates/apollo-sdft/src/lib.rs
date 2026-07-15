#![warn(missing_docs)]
//! Sliding DFT streaming plans for Apollo.
//!
//! `apollo-sdft` owns streaming-window metadata, direct bin initialization,
//! recurrence updates, and value-semantic verification for continuous
//! fixed-window DFT tracking.

/// Application-layer SDFT plans.
pub mod application;
/// Domain contracts and metadata.
pub mod domain;
/// Infrastructure kernel namespace.
pub mod infrastructure;
/// Value-semantic verification.
pub mod verification;

pub use application::execution::plan::sdft::{
    SdftBinStorage, SdftPlan, SdftRealStorage, SdftState,
};
pub use domain::contracts::error::{SdftError, SdftResult};
pub use domain::metadata::window::SlidingDftConfig;

#[cfg(feature = "wgpu")]
pub use infrastructure::transport::gpu::*;
