#![deny(missing_docs)]
//! Validation and benchmark helpers for Apollo FFT.

pub mod application;
pub mod domain;
/// Infrastructure hooks for validation tooling.
pub mod infrastructure;

pub use application::suite::{
    run_benchmark_suite, run_external_comparison_suite, run_fft_cpu_suite, run_fft_gpu_suite,
    run_full_suite, run_nufft_suite, run_smoke_suite, run_validation_suite,
};
pub use domain::report::ValidationReport;
