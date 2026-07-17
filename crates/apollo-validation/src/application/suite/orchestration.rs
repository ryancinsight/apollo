use crate::domain::report::ValidationReport;
use crate::infrastructure::numpy::probe_python_environment;

use super::benchmark::run_benchmark_suite;
use super::environment::environment_report;
use super::external::run_external_comparison_suite;
use super::fft::{run_fft_cpu_suite, run_fft_gpu_suite};
use super::nufft::run_nufft_suite;
use super::SuiteResult;

/// Run the full validation and benchmark suite.
pub fn run_full_suite() -> SuiteResult<ValidationReport> {
    run_validation_suite()
}

/// Run all validation suites and benchmarks.
pub fn run_validation_suite() -> SuiteResult<ValidationReport> {
    let environment_probe = probe_python_environment().ok();
    let fft_cpu = run_fft_cpu_suite()?;
    let fft_gpu = run_fft_gpu_suite()?;
    let nufft = run_nufft_suite()?;
    let external = run_external_comparison_suite()?;
    let benchmarks = run_benchmark_suite()?;
    let environment = environment_report(environment_probe.as_ref());
    Ok(ValidationReport {
        fft_cpu,
        fft_gpu,
        nufft,
        external,
        benchmarks,
        environment,
    })
}

/// Run the lightweight smoke suite.
pub fn run_smoke_suite() -> SuiteResult<ValidationReport> {
    run_validation_suite()
}
