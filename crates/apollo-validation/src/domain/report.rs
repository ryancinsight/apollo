//! Validation report structures.

use serde::{Deserialize, Serialize};

/// Structured summary of the Apollo validation matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// CPU FFT validation results.
    pub fft_cpu: CpuFftReport,
    /// GPU FFT validation results.
    pub fft_gpu: GpuFftReport,
    /// NUFFT validation results.
    pub nufft: NufftReport,
    /// External comparison results.
    pub external: ExternalComparisonReport,
    /// Benchmark timings collected during validation.
    pub benchmarks: BenchmarkReport,
    /// Execution environment metadata.
    pub environment: EnvironmentReport,
}

/// CPU FFT validation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuFftReport {
    /// Maximum absolute error for a 3D roundtrip.
    pub roundtrip_max_abs_error: f64,
    /// Relative Parseval error for a representative 1D signal.
    pub parseval_relative_error: f64,
    /// Maximum deviation across repeated forward evaluations of the same signal.
    pub stability_max_abs_delta: f64,
    /// Whether non-finite input produced non-finite spectral output without panicking.
    pub non_finite_input_propagates: bool,
    /// Whether the CPU validation thresholds passed.
    pub passed: bool,
    /// Precision-profile validation results.
    pub precision_profiles: Vec<PrecisionRunReport>,
}

/// GPU FFT validation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuFftReport {
    /// Whether an adapter/device-backed validation run was attempted.
    pub attempted: bool,
    /// Whether the attempted validation run succeeded.
    pub passed: bool,
    /// Forward comparison max absolute error against Apollo CPU.
    pub forward_max_abs_error: Option<f64>,
    /// Inverse comparison max absolute error against Apollo CPU.
    pub inverse_max_abs_error: Option<f64>,
    /// Optional note explaining skip/failure details.
    pub note: Option<String>,
    /// Precision-profile validation results.
    pub precision_profiles: Vec<PrecisionRunReport>,
}

/// NUFFT validation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NufftReport {
    /// Relative error for fast 1D type-1 against exact direct evaluation.
    pub type1_1d_max_relative_error: f64,
    /// Relative error for fast 1D type-2 against exact direct evaluation.
    pub type2_1d_max_relative_error: f64,
    /// Relative error for fast 3D type-1 against exact direct evaluation.
    pub type1_3d_max_relative_error: f64,
    /// Relative error for an irrational-position stress case.
    pub irrational_positions_max_relative_error: f64,
    /// Relative error for a clustered near-boundary stress case.
    pub clustered_positions_max_relative_error: f64,
    /// Whether the NUFFT validation thresholds passed.
    pub passed: bool,
}

/// Comparison with one external FFT backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalBackendReport {
    /// Human-readable backend name.
    pub backend: String,
    /// Whether the backend is installed and callable on this host.
    pub available: bool,
    /// Whether a parity run was attempted.
    pub attempted: bool,
    /// 1D max absolute error for a representative power-of-two signal.
    pub fft1_max_abs_error: Option<f64>,
    /// 1D max absolute error for a representative prime-sized signal.
    pub fft1_prime_max_abs_error: Option<f64>,
    /// 3D max absolute error for a representative real field.
    pub fft3_max_abs_error: Option<f64>,
    /// Maximum repeated-run deviation reported by the backend.
    pub stability_max_abs_delta: Option<f64>,
    /// Optional version string when known.
    pub version: Option<String>,
    /// Optional note explaining skip/failure details.
    pub note: Option<String>,
}

/// One deterministic published-reference fixture comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishedFixtureReport {
    /// Transform or theorem surface under validation.
    pub transform: String,
    /// Fixture identifier.
    pub fixture: String,
    /// Published or canonical source for the expected values.
    pub reference: String,
    /// Maximum absolute error against the expected values.
    pub max_abs_error: f64,
    /// Threshold used for this fixture.
    pub threshold: f64,
    /// Whether this fixture passed.
    pub passed: bool,
}

/// Published-reference fixture validation summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishedReferenceReport {
    /// Whether all fixture comparisons passed.
    pub passed: bool,
    /// Number of fixture comparisons attempted.
    pub attempted: usize,
    /// Per-fixture value-semantic reports.
    pub fixtures: Vec<PublishedFixtureReport>,
}

/// Comparison with external FFT implementations or assets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalComparisonReport {
    /// Whether the external comparison thresholds passed.
    pub passed: bool,
    /// Whether an external `pyfftw` checkout exists under the workspace.
    pub pyfftw_checkout_present: bool,
    /// Direct O(N²) DFT parity report (always available, no external dep).
    pub dft: ExternalBackendReport,
    /// NumPy parity report.
    pub numpy: ExternalBackendReport,
    /// Optional `pyfftw` parity report.
    pub pyfftw: ExternalBackendReport,
    /// Whether adversarial robustness probes passed.
    pub robustness_passed: bool,
    /// Precision-specific comparison results.
    pub precision_comparisons: Vec<PrecisionRunReport>,
    /// Published-reference fixture comparisons.
    pub published_references: PublishedReferenceReport,
    /// Optional note explaining skipped comparisons.
    pub note: Option<String>,
}

/// Benchmark timings collected during validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Apollo CPU 1D FFT wall time in milliseconds.
    pub apollo_fft1_ms: f64,
    /// Apollo CPU 3D FFT forward wall time in milliseconds.
    pub apollo_fft3_forward_ms: f64,
    /// Apollo CPU 3D FFT inverse wall time in milliseconds.
    pub apollo_fft3_inverse_ms: f64,
    /// Direct O(N²) DFT 1D wall time in milliseconds.
    pub dft_fft1_ms: f64,
    /// Direct O(N²) DFT 3D wall time in milliseconds.
    pub dft_fft3_ms: f64,
    /// NumPy 1D FFT wall time in milliseconds when available.
    pub numpy_fft1_ms: Option<f64>,
    /// NumPy 3D FFT wall time in milliseconds when available.
    pub numpy_fft3_ms: Option<f64>,
    /// Optional `pyfftw` 1D FFT wall time in milliseconds.
    pub pyfftw_fft1_ms: Option<f64>,
    /// Optional `pyfftw` 3D FFT wall time in milliseconds.
    pub pyfftw_fft3_ms: Option<f64>,
    /// Exact direct 1D type-1 NUFFT wall time in milliseconds.
    pub nufft_exact_type1_1d_ms: f64,
    /// Fast 1D type-1 NUFFT wall time in milliseconds.
    pub nufft_fast_type1_1d_ms: f64,
    /// Fast 3D type-1 NUFFT wall time in milliseconds.
    pub nufft_fast_type1_3d_ms: f64,
    /// GPU FFT forward wall time in milliseconds when attempted.
    pub gpu_fft_forward_ms: Option<f64>,
    /// GPU FFT inverse wall time in milliseconds when attempted.
    pub gpu_fft_inverse_ms: Option<f64>,
    /// Precision-specific benchmark timings.
    pub precision_benchmarks: Vec<PrecisionBenchmarkReport>,
}

/// Precision-specific validation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionRunReport {
    /// Precision profile name.
    pub profile: String,
    /// Whether the run was attempted.
    pub attempted: bool,
    /// Whether the profile-specific thresholds passed.
    pub passed: bool,
    /// Forward max absolute error for this profile.
    pub forward_max_abs_error: Option<f64>,
    /// Inverse max absolute error for this profile.
    pub inverse_max_abs_error: Option<f64>,
    /// Relative roundtrip error or reference-relative error.
    pub relative_error: Option<f64>,
    /// Optional note explaining skip or behavior.
    pub note: Option<String>,
}

/// Precision-specific benchmark timing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionBenchmarkReport {
    /// Precision profile name.
    pub profile: String,
    /// Representative forward timing in milliseconds.
    pub forward_ms: Option<f64>,
    /// Representative inverse timing in milliseconds.
    pub inverse_ms: Option<f64>,
    /// Optional note explaining skipped timings.
    pub note: Option<String>,
}

/// Environment metadata captured during a validation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentReport {
    /// Host operating system.
    pub os: String,
    /// Host architecture.
    pub arch: String,
    /// Whether debug assertions were enabled for the build.
    pub debug_build: bool,
    /// Python version when a compatible interpreter was found.
    pub python_version: Option<String>,
    /// NumPy version when available.
    pub numpy_version: Option<String>,
    /// `pyfftw` version when available.
    pub pyfftw_version: Option<String>,
}
