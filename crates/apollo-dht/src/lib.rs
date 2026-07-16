#![warn(missing_docs)]
//! Discrete Hartley Transform plans for real-valued Apollo signals.
//!
//! The DHT maps a real signal `x[n]` to a real spectrum
//! `H[k] = sum_n x[n] cas(2 pi k n / N)`, where `cas(theta) = cos(theta) +
//! sin(theta)`. The transform is involutory up to scale:
//! `DHT(DHT(x)) = N x`, so the inverse reuses the same kernel and applies the
//! single normalization factor `1 / N`.
//!
//! This crate owns Hartley-domain contracts, validated plan metadata,
//! coefficient storage, real-valued kernels, and value-semantic verification.

/// Application-layer DHT plans.
pub mod application;
/// Domain contracts, metadata, and spectrum storage.
pub mod domain;
/// Infrastructure kernel namespace.
pub mod infrastructure;
#[cfg(test)]
mod verification;

pub use application::execution::plan::dht::{DhtPlan, HartleyGpuStorage, HartleyStorage};
pub use domain::contracts::error::{DhtError, DhtResult};
pub use domain::metadata::length::HartleyLength;
pub use domain::spectrum::coefficients::HartleySpectrum;
pub use infrastructure::kernel::direct::{hartley_cas, transform_real};
pub use leto::{Array2, Array3};

#[cfg(feature = "wgpu")]
pub use infrastructure::transport::gpu::*;
