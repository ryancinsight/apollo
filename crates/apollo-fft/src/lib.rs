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

/// Canonical public API functions.
pub mod api;

pub use application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;

static THREAD_LOCAL_SCRATCH_HOOK: std::sync::OnceLock<()> = std::sync::OnceLock::new();
#[cfg(test)]
static THREAD_LOCAL_SCRATCH_RELEASES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

pub(crate) fn ensure_thread_local_scratch_hook_registered() {
    THREAD_LOCAL_SCRATCH_HOOK.get_or_init(|| {
        moirai::register_idle_hook(release_thread_local_scratch)
            .expect("invariant: Apollo's consolidated idle hook fits Moirai's registry");
    });
}

/// Releases idle FFT scratch capacity held by the current thread.
///
/// Call this at a quiescent boundary on each long-lived worker after its FFT
/// workload has completed. The function affects only the current thread's
/// thread-local banks; calling it from a coordinator does not reach worker
/// banks. It does not invalidate a live scratch borrow, and it is not intended
/// for the per-transform hot path. Dynamic FFT plan construction registers this
/// release with Moirai automatically. For const-constructed static plans or
/// direct kernel entry points, call this once at a runtime boundary before work
/// is submitted so the worker-idle hook is installed.
pub fn release_thread_local_scratch() {
    #[cfg(test)]
    THREAD_LOCAL_SCRATCH_RELEASES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ensure_thread_local_scratch_hook_registered();
    application::execution::kernel::mixed_radix::release_thread_local_scratch();
}

#[cfg(test)]
pub(crate) fn thread_local_scratch_release_count() -> usize {
    THREAD_LOCAL_SCRATCH_RELEASES.load(std::sync::atomic::Ordering::Relaxed)
}

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
pub use application::orchestration::cache::plans::PlanCacheProvider;
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
