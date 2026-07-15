#![warn(missing_docs)]
//! Hilbert transform and analytic-signal plans for Apollo.
//!
//! The discrete Hilbert transform shifts positive-frequency components by
//! `-pi / 2` and negative-frequency components by `+pi / 2`. Combining the
//! original real signal with this quadrature component yields the analytic
//! signal `z[n] = x[n] + i H{x}[n]`.
//!
//! For signals with zero DC and (for even lengths) zero Nyquist coefficient,
//! applying the transform twice gives `H(H(x)) = -x`: every retained DFT bin
//! has multiplier `(-i sign(k))^2 = -1`. The inverse accelerator path applies
//! `-H`, so it reconstructs that spectral subspace and deliberately omits DC
//! and Nyquist information.
//!
//! This crate owns Hilbert-domain contracts, analytic-signal storage,
//! frequency-domain masking kernels, and value-semantic verification. The
//! CPU implementation uses Apollo FFT plan execution. The optional accelerator
//! boundary uses Hephaestus typed kernels; Leto and Mnemosyne remain host-array
//! and scratch/output owners respectively.

/// Application-layer Hilbert plans.
pub mod application;
/// Domain contracts, metadata, and signal storage.
pub mod domain;
/// Infrastructure kernel namespace.
pub mod infrastructure;
/// Value-semantic verification.
pub mod verification;

pub use application::execution::plan::hilbert::{HilbertGpuStorage, HilbertPlan, HilbertStorage};
pub use domain::contracts::error::{HilbertError, HilbertResult};
pub use domain::metadata::length::SignalLength;
pub use domain::signal::analytic::AnalyticSignal;
pub use infrastructure::kernel::direct::{
    analytic_signal, analytic_signal_into, hilbert_transform,
};

/// GPU-accelerated backend using the Hephaestus WGPU provider.
#[cfg(feature = "wgpu")]
pub mod wgpu_backend {
    pub use crate::infrastructure::transport::gpu::*;
}
#[cfg(feature = "wgpu")]
pub use infrastructure::transport::gpu::*;
