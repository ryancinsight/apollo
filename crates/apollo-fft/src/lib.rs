#![warn(missing_docs)]
// ── Pedantic suppressions ────────────────────────────────────────────────────
// FFT math inherently uses index-to-float casts for normalisation factors and
// twiddle-factor computation. Grid sizes are bounded by available memory
// (< 2^52), so precision loss and truncation are hypothetical, not real.
// Naming conventions in signal processing (n_x / n_y, coeff_re / coeff_im)
// are standardised in the literature; renaming them reduces clarity.
// Complex FFT plans necessarily carry many boolean precision-mode flags;
// bitset refactors would add complexity without improving safety.
// These suppressions mirror those already configured in the apollo sub-workspace
// Cargo.toml (`similar_names = "allow"`, `too_many_lines = "allow"`, etc.).
#![allow(
    clippy::cast_possible_truncation, // grid sizes < 2^24 for f32, < 2^52 for f64
    clippy::cast_precision_loss,      // usize→f32/f64 normalisation, bounded by memory
    clippy::cast_sign_loss,           // non-negative index arithmetic
    clippy::cast_possible_wrap,       // modular butterfly arithmetic
    clippy::similar_names,            // n_x/n_y/n_z, coeff_re/coeff_im — math convention
    clippy::many_single_char_names,    // FFT/Rader formulas use standard n, m, j, k, w notation
    clippy::too_many_lines,           // FFT plan builders are inherently long
    clippy::missing_panics_doc,       // cache helpers panic only on logic error / OOM
    clippy::missing_errors_doc,       // error paths documented inline in struct fields
    clippy::missing_fields_in_debug,  // manual Debug omits large internal buffers by design
    clippy::struct_excessive_bools,   // PrecisionProfile flags are orthogonal bit fields
    clippy::cast_ptr_alignment,       // loadu/storeu SIMD intrinsics intentionally accept unaligned lanes
    clippy::option_option,             // tri-state caches encode unknown/unsupported/supported distinctly
    clippy::approx_constant,           // generated tables preserve audited literal bit patterns
    clippy::needless_pass_by_value,          // Copy-sized plan/shape types passed by value intentionally
    clippy::missing_const_for_thread_local,  // all thread_local! initializers already use const { }
    clippy::excessive_precision,             // Winograd/codelet coefficients carry one guard digit past
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

/// API wrappers.
pub mod api;

#[cfg(test)]
mod lib_tests;

pub use application::execution::plan::fft::{
    dimension_1d::{FftPlan1D, StaticFftPlan1D},
    dimension_2d::{FftPlan2D, StaticFftPlan2D},
    dimension_3d::{FftPlan3D, StaticFftPlan3D},
    real_storage::RealFftData,
};
pub use application::orchestration::cache::plans::PlanCacheProvider;
pub use domain::contracts::backend::FftBackend;
pub use domain::contracts::error::{ApolloError, ApolloResult};
pub use domain::metadata::precision::{
    BackendKind, ComputePrecision, Normalization, PrecisionMode, PrecisionProfile, StoragePrecision,
};
pub use domain::metadata::shape::{HalfSpectrum3D, Shape1D, Shape2D, Shape3D};
pub use half::f16;
pub use infrastructure::transport::cpu::CpuBackend;

pub use num_complex::Complex32;
pub use num_complex::Complex64;

// Re-export all API functions directly from crate root for backwards compatibility.
pub use api::cfft::*;
pub use api::icfft::*;
pub use api::irfft::*;
pub use api::rfft::*;
pub use api::utils::*;
