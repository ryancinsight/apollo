#![warn(missing_docs)]
//! Discrete Radon transform and CT-style reconstruction plans for Apollo.
//!
//! The Radon transform maps a 2D image into projection lines indexed by angle
//! and detector coordinate. This crate defines a discrete image model where
//! each pixel value is a point mass located at its pixel center. Forward Radon
//! analysis deposits each mass onto detector bins with linear weights; adjoint
//! backprojection uses the same weights in reverse, preserving the inner-product
//! identity `<R f, p> = <f, R* p>`.
//!
//! Filtered backprojection applies a direct DFT ramp filter to each projection
//! before adjoint backprojection. This exposes the CT reconstruction primitive
//! while keeping the forward model, adjoint, and filter in one crate-owned
//! hierarchy.

/// Application-layer Radon plans.
pub mod application;
/// Domain contracts, geometry, and projection storage.
pub mod domain;
/// Infrastructure kernel namespace.
pub mod infrastructure;
/// Value-semantic verification.
pub mod verification;

pub use application::execution::plan::radon::{RadonPlan, RadonStorage};
pub use domain::contracts::error::{RadonError, RadonResult};
pub use domain::geometry::parallel_beam::ParallelBeamGeometry;
pub use domain::projection::sinogram::Sinogram;
pub use infrastructure::kernel::direct::{
    adjoint_backproject, adjoint_backproject_into, forward_project, forward_project_into,
};
pub use infrastructure::kernel::filter::{ramp_filter_projection, ramp_filter_projection_into};

/// GPU-accelerated backend using WGPU.
#[cfg(feature = "wgpu")]
pub mod wgpu_backend {
    pub use crate::infrastructure::transport::gpu::*;
}
#[cfg(feature = "wgpu")]
pub use infrastructure::transport::gpu::{*};
