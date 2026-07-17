//! Validation suite public surface.
//!
//! Each concern owns one private leaf. This manifest retains only the shared
//! numerical limits and curated public suite paths.

mod benchmark;
mod environment;
mod external;
mod fft;
mod fixtures;
mod metrics;
mod nufft;
mod orchestration;
#[cfg(test)]
mod tests;

pub use benchmark::run_benchmark_suite;
pub use external::{run_external_comparison_suite, run_published_reference_suite};
pub use fft::{run_fft_cpu_suite, run_fft_gpu_suite};
pub use nufft::run_nufft_suite;
pub use orchestration::{run_full_suite, run_smoke_suite, run_validation_suite};

use std::error::Error;

pub(super) const CPU_ROUNDTRIP_LIMIT: f64 = 1.0e-10;
pub(super) const CPU_PARSEVAL_LIMIT: f64 = 1.0e-10;
pub(super) const CPU_STABILITY_LIMIT: f64 = 1.0e-12;
pub(super) const EXTERNAL_FFT_LIMIT: f64 = 1.0e-9;
pub(super) const NUFFT_FAST_RELATIVE_LIMIT: f64 = 1.0e-5;

pub(super) type SuiteResult<T> = Result<T, Box<dyn Error>>;
