#![deny(missing_docs)]
// ── Pedantic suppressions ────────────────────────────────────────────────────
// Only the lints the workspace lint table does *not* already allow live here.
// The cast, naming, length and doc suppressions this block used to repeat are
// configured once in the apollo `Cargo.toml` `[workspace.lints.clippy]` table
// and were redundant at crate level.
#![allow(
    clippy::many_single_char_names, // FFT/Rader formulas use standard n, m, j, k, w notation
    clippy::cast_ptr_alignment,     // loadu/storeu SIMD intrinsics intentionally accept unaligned lanes
    clippy::option_option,          // tri-state caches encode unknown/unsupported/supported distinctly
    clippy::approx_constant,        // generated tables preserve audited literal bit patterns
    clippy::excessive_precision,    // Winograd/codelet coefficients carry one guard digit past
                                    // f64 precision so the compiler selects the intended
                                    // nearest-representable value; trimming would alter
                                    // bit-exact differential-test results (e.g. -13/12 literal)
)]
//! Apollo core crate.
//!
//! This crate owns the reusable CPU FFT implementation, shared shape and error
//! contracts, backend abstractions, and cache-backed convenience helpers.

/// Application-layer execution and orchestration.
pub mod application;
pub mod domain;
/// Infrastructure adapters.
pub mod infrastructure;

/// Canonical public API functions.
pub mod api;

pub use application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;

pub use application::execution::kernel::scratch_hook::{
    release_thread_local_scratch, thread_local_scratch_hook_registered,
};

#[cfg(test)]
pub(crate) fn thread_local_scratch_capacity() -> usize {
    application::execution::kernel::mixed_radix::thread_local_scratch_capacity()
}

#[cfg(test)]
mod lib_tests;

pub use application::execution::plan::fft::{
    dimension_1d::{FftPlan1D, StaticFftPlan1D},
    dimension_2d::{FftPlan2D, StaticFftPlan2D},
    dimension_3d::{FftPlan3D, StaticFftPlan3D},
    real_storage::RealFftData,
};
pub use application::orchestration::cache::plans::{clear_plan_caches, PlanCacheProvider};
pub use domain::contracts::backend::{BackendCapabilities, FftBackend};
pub use domain::contracts::error::{ApolloError, ApolloResult};
pub use domain::metadata::precision::{
    BackendKind, ComputePrecision, Normalization, PrecisionMode, PrecisionProfile, StoragePrecision,
};
pub use domain::metadata::shape::{HalfSpectrum3D, Shape1D, Shape2D, Shape3D};
pub use domain::storage::scalar::{CpuElement, CpuStorage};
pub use eunomia::F16;
pub use infrastructure::transport::cpu::CpuBackend;
#[cfg(feature = "wgpu")]
pub use infrastructure::transport::transform::{
    GpuElement, GpuStorage, GpuTransformExecutor, GpuTransformPlanner, WgpuCapabilities, WgpuError,
    WgpuResult, WgpuTransformBackend, WgpuTransformPlan,
};

pub use eunomia::Complex32;
pub use eunomia::Complex64;

// Re-export the canonical API functions at the crate root.
pub use api::cfft::*;
pub use api::freq::*;
pub use api::icfft::*;
pub use api::irfft::*;
pub use api::rfft::*;
pub use api::shift::*;

#[cfg(feature = "cuda")]
pub use infrastructure::transport::cuda::*;
