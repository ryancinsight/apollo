#![warn(missing_docs)]
//! WGPU backend boundary for Apollo NTT.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns the NTT kernels, their domain names, and
//! the residue-field surface. The transform is exact modular arithmetic
//! over `u64`/`u32` residues — outside the scaffold's floating-point
//! element families — so the marker implements only the planner contract
//! and the surface lives on [`ModularExecution`].

/// Infrastructure boundary for the NTT kernels.
pub mod infrastructure;
#[cfg(test)]
pub(crate) mod verification;

pub use apollo_fft::{WgpuCapabilities, WgpuError, WgpuResult};
pub use infrastructure::kernel::{NttGpuBuffers, NttGpuKernel};

mod execution;
mod residue;

pub use execution::buffer_output;
pub use residue::{ModularExecution, NttWgpuBackend, NttWgpuPlan, ResiduePlan};
