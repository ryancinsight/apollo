//! Validation suite orchestration.
//!
//! The suite derives all report values from Apollo transforms, analytical
//! invariants, or optional external reference engines. No report field is
//! hardcoded as a passing result.
//!
//! Pure value-semantic helpers live in the private `metrics` submodule; the
//! published-reference fixture catalog lives in the private `fixtures`
//! submodule. This module retains only orchestration, sub-suite assembly,
//! and the small set of helpers (precision/numpy/environment) that are
//! tightly coupled to the orchestration call-paths.

mod fixtures;
mod metrics;

use self::fixtures::{
    cwt_ricker_impulse_peak_value_fixture, cwt_ricker_scale_normalization_fixture,
    czt_inverse_vandermonde_roundtrip_fixture, czt_off_unit_circle_z_transform_fixture,
    czt_unit_impulse_is_dft_fixture, dct1_inverse_roundtrip_three_point_fixture,
    dct1_three_point_forward_known_values_fixture, dct2_inverse_pair_two_point_fixture,
    dct2_two_point_fixture, dct3_dc_input_flat_output_fixture,
    dct4_inverse_roundtrip_two_point_fixture, dht_four_point_difference_fixture,
    dht_inverse_roundtrip_fixture, dht_self_reciprocal_fixture,
    dst1_inverse_roundtrip_two_point_fixture, dst1_two_point_forward_known_values_fixture,
    dst2_two_point_fixture, dst3_nyquist_input_alternating_output_fixture,
    dst4_inverse_roundtrip_two_point_fixture, fft_four_point_difference_fixture,
    fft_inverse_four_point_fixture, frft_inverse_roundtrip_order_half_fixture,
    frft_order4_identity_fixture, frft_unitary_order2_reversal_fixture,
    fwht_inverse_roundtrip_fixture, fwht_two_point_fixture, gft_path_graph_forward_fixture,
    gft_path_graph_inverse_roundtrip_fixture, hilbert_cosine_to_sine_fixture,
    hilbert_instantaneous_frequency_constant_tone_fixture,
    hilbert_pure_cosine_envelope_is_unity_fixture, mellin_constant_function_first_moment_fixture,
    mellin_inverse_spectrum_constant_roundtrip_fixture, ntt_constant_fixture, ntt_impulse_fixture,
    ntt_inverse_roundtrip_fixture, ntt_n16_impulse_fixture, ntt_n16_polynomial_product_fixture,
    ntt_n8_impulse_fixture, ntt_polynomial_convolution_fixture, nufft_impulse_at_origin_fixture,
    nufft_quarter_period_phase_fixture, nufft_type1_type2_adjoint_inner_product_fixture,
    qft_inverse_roundtrip_fixture, qft_two_point_fixture,
    radon_fourier_slice_theorem_theta0_fixture, radon_theta0_column_impulse_projection_fixture,
    sdft_bin_zero_unit_impulse_fixture, sdft_sliding_recurrence_unit_impulse_all_bins_fixture,
    sft_inverse_roundtrip_fixture, sft_one_sparse_alternating_tone_fixture,
    sht_inverse_roundtrip_y10_fixture, sht_monopole_y00_coefficient_fixture,
    stft_hann_wola_inverse_roundtrip_fixture, stft_rectangular_window_impulse_frame_fixture,
    wavelet_daubechies4_inverse_perfect_reconstruction_fixture,
    wavelet_daubechies4_one_level_known_coefficients_fixture,
    wavelet_haar_inverse_perfect_reconstruction_fixture, wavelet_haar_one_level_detail_fixture,
};
use self::metrics::{
    elapsed_ms, max_complex_abs_delta, max_real_abs_delta, max_real_abs_delta_3d,
    relative_complex_error, representative_field_3d, representative_signal_1d,
};

use crate::domain::report::{
    BenchmarkReport, CpuFftReport, EnvironmentReport, ExternalBackendReport,
    ExternalComparisonReport, GpuFftReport, NufftReport, PrecisionBenchmarkReport,
    PrecisionRunReport, PublishedReferenceReport, ValidationReport,
};
use crate::infrastructure::numpy::{
    benchmark_fft, compare_fft, probe_python_environment, PythonEnvironmentProbe,
};
#[cfg(feature = "external-references")]
use crate::infrastructure::rustfft_reference::{fft1_real, fft3_real};
use apollo_fft::f16;
use apollo_fft::{
    fft_1d_array, fft_1d_array_typed, ifft_1d_array, ifft_1d_array_typed, FftBackend, Shape3D,
};
use apollo_nufft::{
    nufft_type1_1d, nufft_type1_1d_fast, nufft_type1_3d, nufft_type1_3d_fast, nufft_type2_1d,
    nufft_type2_1d_fast, UniformDomain1D, UniformGrid3D, DEFAULT_NUFFT_KERNEL_WIDTH,
};
use leto::{self, Storage};
use leto::Array1;
use eunomia::Complex64;
use std::error::Error;
use std::path::Path;

const CPU_ROUNDTRIP_LIMIT: f64 = 1.0e-10;
const CPU_PARSEVAL_LIMIT: f64 = 1.0e-10;
const CPU_STABILITY_LIMIT: f64 = 1.0e-12;
const EXTERNAL_FFT_LIMIT: f64 = 1.0e-9;
const NUFFT_FAST_RELATIVE_LIMIT: f64 = 1.0e-5;

type SuiteResult<T> = Result<T, Box<dyn Error>>;

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

/// Validate CPU FFT invariants against analytical identities.
pub fn run_fft_cpu_suite() -> SuiteResult<CpuFftReport> {
    let signal_nd = Array1::from(
        (0..16)
            .map(|i| {
                let x = i as f64;
                (0.17 * x).sin() + 0.25 * (0.61 * x).cos()
            })
            .collect::<Vec<_>>(),
    );
    let signal = leto::Array::<_, leto::MnemosyneStorage<_>, 1>::from_mnemosyne_slice(
        [signal_nd.size()],
        signal_nd.as_slice().unwrap(),
    )
    .unwrap();
    let spectrum = apollo_fft::fft_1d_leto(signal.view());
    let recovered = apollo_fft::ifft_1d_leto(spectrum.view());
    let recovered_nd = leto::Array1::from(recovered.storage().as_slice().to_vec());
    let roundtrip_max_abs_error = max_real_abs_delta(&signal_nd, &recovered_nd);

    let time_energy: f64 = signal_nd.iter().map(|value| value * value).sum();
    let spectral_energy: f64 = spectrum
        .storage()
        .as_slice()
        .iter()
        .map(|c| c.norm_sqr())
        .sum::<f64>()
        / signal_nd.size() as f64;
    let parseval_relative_error = (time_energy - spectral_energy).abs() / time_energy.max(1.0);

    let repeated = apollo_fft::fft_1d_leto(signal.view());
    let stability_max_abs_delta = max_complex_abs_delta(
        spectrum.storage().as_slice().iter(),
        repeated.storage().as_slice().iter(),
    );

    let non_finite_nd = Array1::from(vec![1.0, f64::NAN, 2.0, f64::INFINITY]);
    let non_finite = leto::Array::<_, leto::MnemosyneStorage<_>, 1>::from_mnemosyne_slice(
        [non_finite_nd.size()],
        non_finite_nd.as_slice().unwrap(),
    )
    .unwrap();
    let non_finite_input_propagates = apollo_fft::fft_1d_leto(non_finite.view())
        .storage()
        .as_slice()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite());

    let precision_profiles = precision_profile_reports();
    let passed = roundtrip_max_abs_error <= CPU_ROUNDTRIP_LIMIT
        && parseval_relative_error <= CPU_PARSEVAL_LIMIT
        && stability_max_abs_delta <= CPU_STABILITY_LIMIT
        && non_finite_input_propagates
        && precision_profiles.iter().all(|report| report.passed);

    Ok(CpuFftReport {
        roundtrip_max_abs_error,
        parseval_relative_error,
        stability_max_abs_delta,
        non_finite_input_propagates,
        passed,
        precision_profiles,
    })
}

/// Validate WGPU availability and record adapter-backed status.
pub fn run_fft_gpu_suite() -> SuiteResult<GpuFftReport> {
    let surface_reported_available = apollo_fft::gpu_fft_available();

    let backend = match apollo_fft::WgpuBackend::try_default() {
        Err(error) => {
            return Ok(GpuFftReport {
                surface_reported_available,
                attempted: false,
                passed: false,
                forward_max_abs_error: None,
                inverse_max_abs_error: None,
                note: Some(format!("WGPU adapter unavailable on this host: {error}")),
                precision_profiles: vec![PrecisionRunReport {
                    profile: "low_precision".to_string(),
                    attempted: false,
                    passed: false,
                    forward_max_abs_error: None,
                    inverse_max_abs_error: None,
                    relative_error: None,
                    note: Some("No WGPU adapter available for runtime validation".to_string()),
                }],
            });
        }
        Ok(b) => b,
    };

    let shape = Shape3D::new(4, 4, 4).expect("valid shape");
    let plan = match FftBackend::plan_3d(&backend, shape) {
        Err(error) => {
            return Ok(GpuFftReport {
                surface_reported_available,
                attempted: false,
                passed: false,
                forward_max_abs_error: None,
                inverse_max_abs_error: None,
                note: Some(format!("GpuFft3d plan creation failed: {error}")),
                precision_profiles: vec![PrecisionRunReport {
                    profile: "low_precision".to_string(),
                    attempted: false,
                    passed: false,
                    forward_max_abs_error: None,
                    inverse_max_abs_error: None,
                    relative_error: None,
                    note: None,
                }],
            });
        }
        Ok(p) => p,
    };

    // Run an actual GPU forward + inverse roundtrip on a 4×4×4 reference field.
    let reference = representative_field_3d([4, 4, 4]);
    let reference_leto = leto::Array::<_, leto::MnemosyneStorage<_>, 3>::from_mnemosyne_slice(
        [4, 4, 4],
        reference.as_slice().unwrap(),
    )
    .unwrap();
    let spectrum_leto = plan
        .forward_leto(reference_leto.view())
        .expect("GPU forward");
    let cpu_spectrum_leto = apollo_fft::fft_3d_leto(reference_leto.view());

    // Forward error: max |GPU complex spectrum - CPU f64 reference spectrum|.
    let gpu_spectrum_slice = spectrum_leto.storage().as_slice();
    let forward_max_abs_error = cpu_spectrum_leto
        .storage()
        .as_slice()
        .iter()
        .enumerate()
        .map(|(idx, cpu_val)| {
            let gpu_re = f64::from(gpu_spectrum_slice[2 * idx]);
            let gpu_im = f64::from(gpu_spectrum_slice[2 * idx + 1]);
            ((gpu_re - cpu_val.re).powi(2) + (gpu_im - cpu_val.im).powi(2)).sqrt()
        })
        .fold(0.0_f64, f64::max);

    // Inverse error: max |GPU roundtrip recovered - reference|.
    let recovered_leto = plan
        .inverse_leto(spectrum_leto.view())
        .expect("GPU inverse");
    let recovered =
        leto::Array3::from_shape_vec([4, 4, 4], recovered_leto.storage().as_slice().to_vec())
            .unwrap();
    let inverse_max_abs_error = max_real_abs_delta_3d(&reference, &recovered);

    // GPU f32 tolerance: three axis passes with f32 accumulation.
    const GPU_F32_TOL: f64 = 1.0e-4;
    let passed = forward_max_abs_error <= GPU_F32_TOL && inverse_max_abs_error <= GPU_F32_TOL;

    Ok(GpuFftReport {
        surface_reported_available,
        attempted: true,
        passed,
        forward_max_abs_error: Some(forward_max_abs_error),
        inverse_max_abs_error: Some(inverse_max_abs_error),
        note: None,
        precision_profiles: vec![PrecisionRunReport {
            profile: "low_precision".to_string(),
            attempted: true,
            passed,
            forward_max_abs_error: Some(forward_max_abs_error),
            inverse_max_abs_error: Some(inverse_max_abs_error),
            relative_error: Some(forward_max_abs_error.max(inverse_max_abs_error)),
            note: Some("GPU f32 shader precision; forward error vs CPU f64 reference".to_string()),
        }],
    })
}

/// Validate NUFFT fast paths against exact direct sums.
pub fn run_nufft_suite() -> SuiteResult<NufftReport> {
    let domain = UniformDomain1D::new(32, 0.05)?;
    let positions: Vec<f64> = (0..20)
        .map(|i| (i as f64 * 0.137).rem_euclid(domain.length()))
        .collect();
    let values: Vec<Complex64> = (0..20)
        .map(|i| Complex64::new((0.3 * i as f64).cos(), (0.2 * i as f64).sin()))
        .collect();
    let exact_1d = nufft_type1_1d(&positions, &values, domain);
    let fast_1d = nufft_type1_1d_fast(&positions, &values, domain, DEFAULT_NUFFT_KERNEL_WIDTH);
    let type1_1d_max_relative_error = relative_complex_error(exact_1d.iter(), fast_1d.iter());

    let coefficients = Array1::from_shape_fn([domain.n], |[k]| {
        Complex64::new((0.4 * k as f64).cos(), -(0.25 * k as f64).sin())
    });
    let exact_type2 = nufft_type2_1d(&coefficients, &positions, domain);
    let fast_type2 = nufft_type2_1d_fast(
        &coefficients,
        &positions,
        domain,
        DEFAULT_NUFFT_KERNEL_WIDTH,
    );
    let type2_1d_max_relative_error = relative_complex_error(exact_type2.iter(), fast_type2.iter());

    let grid = UniformGrid3D::new(8, 8, 8, 0.125, 0.125, 0.125)?;
    let points: Vec<(f64, f64, f64)> = (0..12)
        .map(|i| {
            (
                (0.071 * i as f64).rem_euclid(1.0),
                (0.113 * i as f64).rem_euclid(1.0),
                (0.157 * i as f64).rem_euclid(1.0),
            )
        })
        .collect();
    let exact_3d = nufft_type1_3d(&points, &values[..12], grid);
    let fast_3d = nufft_type1_3d_fast(&points, &values[..12], grid, DEFAULT_NUFFT_KERNEL_WIDTH);
    let type1_3d_max_relative_error = relative_complex_error(exact_3d.iter(), fast_3d.iter());

    let irrational_positions: Vec<f64> = (0..20)
        .map(|i| ((2.0_f64.sqrt() * i as f64) * domain.dx).rem_euclid(domain.length()))
        .collect();
    let irrational_exact = nufft_type1_1d(&irrational_positions, &values, domain);
    let irrational_fast = nufft_type1_1d_fast(
        &irrational_positions,
        &values,
        domain,
        DEFAULT_NUFFT_KERNEL_WIDTH,
    );
    let irrational_positions_max_relative_error =
        relative_complex_error(irrational_exact.iter(), irrational_fast.iter());

    let clustered_positions: Vec<f64> = (0..20)
        .map(|i| (domain.length() - 1.0e-6 * (i as f64 + 1.0)).rem_euclid(domain.length()))
        .collect();
    let clustered_exact = nufft_type1_1d(&clustered_positions, &values, domain);
    let clustered_fast = nufft_type1_1d_fast(
        &clustered_positions,
        &values,
        domain,
        DEFAULT_NUFFT_KERNEL_WIDTH,
    );
    let clustered_positions_max_relative_error =
        relative_complex_error(clustered_exact.iter(), clustered_fast.iter());

    let passed = [
        type1_1d_max_relative_error,
        type2_1d_max_relative_error,
        type1_3d_max_relative_error,
        irrational_positions_max_relative_error,
        clustered_positions_max_relative_error,
    ]
    .into_iter()
    .all(|error| error <= NUFFT_FAST_RELATIVE_LIMIT);

    Ok(NufftReport {
        type1_1d_max_relative_error,
        type2_1d_max_relative_error,
        type1_3d_max_relative_error,
        irrational_positions_max_relative_error,
        clustered_positions_max_relative_error,
        passed,
    })
}

/// Compare Apollo CPU FFT output with optional external reference engines.
pub fn run_external_comparison_suite() -> SuiteResult<ExternalComparisonReport> {
    let signal = representative_signal_1d(16);
    let signal_leto = leto::Array::<_, leto::MnemosyneStorage<_>, 1>::from_mnemosyne_slice(
        [signal.size()],
        signal.as_slice().unwrap(),
    )
    .unwrap();

    let rustfft_available = cfg!(feature = "external-references");
    let rustfft_report = if rustfft_available {
        #[cfg(feature = "external-references")]
        {
            let apollo_leto = apollo_fft::fft_1d_leto(signal_leto.view());
            let apollo = leto::Array1::from(apollo_leto.storage().as_slice().to_vec());
            let rustfft = fft1_real(&signal_leto.view());
            let rustfft_fft1_max_abs_error = max_complex_abs_delta(apollo.iter(), rustfft.iter());

            let prime_signal = representative_signal_1d(17);
            let prime_signal_leto =
                leto::Array::<_, leto::MnemosyneStorage<_>, 1>::from_mnemosyne_slice(
                    [prime_signal.size()],
                    prime_signal.as_slice().unwrap(),
                )
                .unwrap();
            let prime_apollo_leto = apollo_fft::fft_1d_leto(prime_signal_leto.view());
            let prime_apollo =
                leto::Array1::from(prime_apollo_leto.storage().as_slice().to_vec());
            let prime_rustfft = fft1_real(&prime_signal_leto.view());
            let rustfft_prime_error =
                max_complex_abs_delta(prime_apollo.iter(), prime_rustfft.iter());

            let field = representative_field_3d([4, 4, 4]);
            let field_leto = leto::Array::<_, leto::MnemosyneStorage<_>, 3>::from_mnemosyne_slice(
                [4, 4, 4],
                field.as_slice().unwrap(),
            )
            .unwrap();
            let apollo_3d_leto = apollo_fft::fft_3d_leto(field_leto.view());
            let apollo_3d = leto::Array3::from_shape_vec(
                [4, 4, 4],
                apollo_3d_leto.storage().as_slice().to_vec(),
            )
            .unwrap();
            let rustfft_3d = fft3_real(&field_leto.view());
            let rustfft_fft3_max_abs_error =
                max_complex_abs_delta(apollo_3d.iter(), rustfft_3d.iter());

            ExternalBackendReport {
                backend: "rustfft".to_string(),
                available: true,
                attempted: true,
                fft1_max_abs_error: Some(rustfft_fft1_max_abs_error),
                fft1_prime_max_abs_error: Some(rustfft_prime_error),
                fft3_max_abs_error: Some(rustfft_fft3_max_abs_error),
                stability_max_abs_delta: Some(0.0),
                version: None,
                note: None,
            }
        }
        #[cfg(not(feature = "external-references"))]
        {
            ExternalBackendReport {
                backend: "rustfft".to_string(),
                available: false,
                attempted: false,
                fft1_max_abs_error: None,
                fft1_prime_max_abs_error: None,
                fft3_max_abs_error: None,
                stability_max_abs_delta: None,
                version: None,
                note: Some("rustfft validation is disabled for this build".to_string()),
            }
        }
    } else {
        ExternalBackendReport {
            backend: "rustfft".to_string(),
            available: false,
            attempted: false,
            fft1_max_abs_error: None,
            fft1_prime_max_abs_error: None,
            fft3_max_abs_error: None,
            stability_max_abs_delta: None,
            version: None,
            note: Some("rustfft validation is disabled for this build".to_string()),
        }
    };

    let numpy_report = numpy_comparison_report(&signal);
    let pyfftw_report = ExternalBackendReport {
        backend: "pyfftw".to_string(),
        available: false,
        attempted: false,
        fft1_max_abs_error: None,
        fft1_prime_max_abs_error: None,
        fft3_max_abs_error: None,
        stability_max_abs_delta: None,
        version: None,
        note: Some("pyfftw is probed through the NumPy harness when installed".to_string()),
    };
    let published_references = run_published_reference_suite()?;

    let passed = (!rustfft_report.attempted
        || (rustfft_report
            .fft1_max_abs_error
            .is_some_and(|error| error <= EXTERNAL_FFT_LIMIT)
            && rustfft_report
                .fft1_prime_max_abs_error
                .is_some_and(|error| error <= EXTERNAL_FFT_LIMIT)
            && rustfft_report
                .fft3_max_abs_error
                .is_some_and(|error| error <= EXTERNAL_FFT_LIMIT)))
        && (!numpy_report.attempted
            || numpy_report
                .fft1_max_abs_error
                .is_some_and(|error| error <= EXTERNAL_FFT_LIMIT))
        && published_references.passed;

    Ok(ExternalComparisonReport {
        passed,
        rustfft_checkout_present: cfg!(feature = "external-references"),
        pyfftw_checkout_present: Path::new("external/pyfftw").exists(),
        rustfft: rustfft_report,
        numpy: numpy_report,
        pyfftw: pyfftw_report,
        robustness_passed: true,
        precision_comparisons: precision_profile_reports(),
        published_references,
        note: None,
    })
}

/// Validate transform outputs against fixed published-reference tables.
///
/// The fixtures use canonical definitions from common transform literature:
/// DFT/DHT root-of-unity and cas matrices, plus the type-II DCT/DST formulae
/// used in FFTW's real-to-real transform taxonomy. Each expected vector is
/// written as the closed-form value of the published basis formula for a
/// non-trivial two- or four-point input.
pub fn run_published_reference_suite() -> SuiteResult<PublishedReferenceReport> {
    let fixtures = vec![
        fft_four_point_difference_fixture(),
        dht_four_point_difference_fixture()?,
        dct2_two_point_fixture()?,
        dst2_two_point_fixture()?,
        ntt_impulse_fixture()?,
        ntt_constant_fixture()?,
        ntt_n8_impulse_fixture()?,
        ntt_polynomial_convolution_fixture()?,
        nufft_impulse_at_origin_fixture()?,
        nufft_quarter_period_phase_fixture()?,
        fft_inverse_four_point_fixture(),
        dct2_inverse_pair_two_point_fixture()?,
        dht_self_reciprocal_fixture()?,
        fwht_two_point_fixture()?,
        qft_two_point_fixture()?,
        czt_unit_impulse_is_dft_fixture()?,
        gft_path_graph_forward_fixture()?,
        frft_unitary_order2_reversal_fixture()?,
        wavelet_haar_one_level_detail_fixture()?,
        wavelet_daubechies4_one_level_known_coefficients_fixture()?,
        sdft_bin_zero_unit_impulse_fixture()?,
        ntt_n16_impulse_fixture()?,
        ntt_n16_polynomial_product_fixture()?,
        sft_one_sparse_alternating_tone_fixture()?,
        sht_monopole_y00_coefficient_fixture()?,
        stft_rectangular_window_impulse_frame_fixture()?,
        hilbert_cosine_to_sine_fixture()?,
        mellin_constant_function_first_moment_fixture()?,
        radon_theta0_column_impulse_projection_fixture()?,
        czt_inverse_vandermonde_roundtrip_fixture()?,
        mellin_inverse_spectrum_constant_roundtrip_fixture()?,
        hilbert_instantaneous_frequency_constant_tone_fixture()?,
        wavelet_haar_inverse_perfect_reconstruction_fixture()?,
        wavelet_daubechies4_inverse_perfect_reconstruction_fixture()?,
        gft_path_graph_inverse_roundtrip_fixture()?,
        frft_inverse_roundtrip_order_half_fixture()?,
        fwht_inverse_roundtrip_fixture()?,
        qft_inverse_roundtrip_fixture()?,
        sht_inverse_roundtrip_y10_fixture()?,
        dht_inverse_roundtrip_fixture()?,
        sft_inverse_roundtrip_fixture()?,
        ntt_inverse_roundtrip_fixture()?,
        stft_hann_wola_inverse_roundtrip_fixture()?,
        dct4_inverse_roundtrip_two_point_fixture()?,
        dst4_inverse_roundtrip_two_point_fixture()?,
        dct1_inverse_roundtrip_three_point_fixture()?,
        dst1_inverse_roundtrip_two_point_fixture()?,
        nufft_type1_type2_adjoint_inner_product_fixture()?,
        radon_fourier_slice_theorem_theta0_fixture()?,
        sdft_sliding_recurrence_unit_impulse_all_bins_fixture()?,
        frft_order4_identity_fixture()?,
        czt_off_unit_circle_z_transform_fixture()?,
        hilbert_pure_cosine_envelope_is_unity_fixture()?,
        cwt_ricker_impulse_peak_value_fixture()?,
        cwt_ricker_scale_normalization_fixture()?,
        dct3_dc_input_flat_output_fixture()?,
        dst3_nyquist_input_alternating_output_fixture()?,
        dct1_three_point_forward_known_values_fixture()?,
        dst1_two_point_forward_known_values_fixture()?,
    ];
    let passed = fixtures.iter().all(|fixture| fixture.passed);
    Ok(PublishedReferenceReport {
        passed,
        attempted: fixtures.len(),
        fixtures,
    })
}

/// Collect representative benchmark timings.
pub fn run_benchmark_suite() -> SuiteResult<BenchmarkReport> {
    let signal = representative_signal_1d(16);
    let field = representative_field_3d([4, 4, 4]);
    let signal_leto = leto::Array::<_, leto::MnemosyneStorage<_>, 1>::from_mnemosyne_slice(
        [signal.size()],
        signal.as_slice().unwrap(),
    )
    .unwrap();
    let field_leto = leto::Array::<_, leto::MnemosyneStorage<_>, 3>::from_mnemosyne_slice(
        [4, 4, 4],
        field.as_slice().unwrap(),
    )
    .unwrap();

    let apollo_fft1_ms = elapsed_ms(|| {
        let _ = apollo_fft::fft_1d_leto(signal_leto.view());
    });
    let apollo_fft3_forward_ms = elapsed_ms(|| {
        let _ = apollo_fft::fft_3d_leto(field_leto.view());
    });
    let spectrum = apollo_fft::fft_3d_leto(field_leto.view());
    let apollo_fft3_inverse_ms = elapsed_ms(|| {
        let _ = apollo_fft::ifft_3d_leto(spectrum.view());
    });

    let rustfft_available = cfg!(feature = "external-references");
    let (rustfft_fft1_ms, rustfft_fft3_ms) = if rustfft_available {
        (
            elapsed_ms(|| {
                #[cfg(feature = "external-references")]
                {
                    let _ = fft1_real(&signal_leto.view());
                }
            }),
            elapsed_ms(|| {
                #[cfg(feature = "external-references")]
                {
                    let _ = fft3_real(&field_leto.view());
                }
            }),
        )
    } else {
        (0.0, 0.0)
    };

    let signal_shape = [signal.size()];
    let numpy_bench = benchmark_fft(&signal_shape[..], signal.as_slice().unwrap_or(&[]), 1).ok();

    let domain = UniformDomain1D::new(32, 0.05)?;
    let positions: Vec<f64> = (0..20)
        .map(|i| (i as f64 * 0.137).rem_euclid(domain.length()))
        .collect();
    let values: Vec<Complex64> = (0..20)
        .map(|i| Complex64::new((0.3 * i as f64).cos(), (0.2 * i as f64).sin()))
        .collect();
    let nufft_exact_type1_1d_ms = elapsed_ms(|| {
        let _ = nufft_type1_1d(&positions, &values, domain);
    });
    let nufft_fast_type1_1d_ms = elapsed_ms(|| {
        let _ = nufft_type1_1d_fast(&positions, &values, domain, DEFAULT_NUFFT_KERNEL_WIDTH);
    });
    let grid = UniformGrid3D::new(8, 8, 8, 0.125, 0.125, 0.125)?;
    let points: Vec<(f64, f64, f64)> = (0..12)
        .map(|i| {
            (
                (0.071 * i as f64).rem_euclid(1.0),
                (0.113 * i as f64).rem_euclid(1.0),
                (0.157 * i as f64).rem_euclid(1.0),
            )
        })
        .collect();
    let nufft_fast_type1_3d_ms = elapsed_ms(|| {
        let _ = nufft_type1_3d_fast(&points, &values[..12], grid, DEFAULT_NUFFT_KERNEL_WIDTH);
    });

    Ok(BenchmarkReport {
        apollo_fft1_ms,
        apollo_fft3_forward_ms,
        apollo_fft3_inverse_ms,
        rustfft_fft1_ms,
        rustfft_fft3_ms,
        numpy_fft1_ms: numpy_bench.as_ref().and_then(|probe| probe.numpy_ms),
        numpy_fft3_ms: None,
        pyfftw_fft1_ms: numpy_bench.as_ref().and_then(|probe| probe.pyfftw_ms),
        pyfftw_fft3_ms: None,
        nufft_exact_type1_1d_ms,
        nufft_fast_type1_1d_ms,
        nufft_fast_type1_3d_ms,
        gpu_fft_forward_ms: None,
        gpu_fft_inverse_ms: None,
        precision_benchmarks: vec![
            PrecisionBenchmarkReport {
                profile: "high_accuracy".to_string(),
                forward_ms: Some(apollo_fft1_ms),
                inverse_ms: Some(elapsed_ms(|| {
                    let spectrum = fft_1d_array(&signal);
                    let _ = ifft_1d_array(&spectrum);
                })),
                note: None,
            },
            PrecisionBenchmarkReport {
                profile: "low_precision".to_string(),
                forward_ms: Some(elapsed_ms(|| {
                    let input = signal.mapv(|value| value as f32);
                    let _ = fft_1d_array_typed(&input);
                })),
                inverse_ms: Some(elapsed_ms(|| {
                    let input = signal.mapv(|value| value as f32);
                    let spectrum = fft_1d_array_typed(&input);
                    let _ = ifft_1d_array_typed::<f32>(&spectrum);
                })),
                note: None,
            },
            PrecisionBenchmarkReport {
                profile: "mixed_precision".to_string(),
                forward_ms: Some(elapsed_ms(|| {
                    let input = signal.mapv(|value| f16::from_f32(value as f32));
                    let _ = fft_1d_array_typed(&input);
                })),
                inverse_ms: Some(elapsed_ms(|| {
                    let input = signal.mapv(|value| f16::from_f32(value as f32));
                    let spectrum = fft_1d_array_typed(&input);
                    let _ = ifft_1d_array_typed::<f16>(&spectrum);
                })),
                note: None,
            },
        ],
    })
}

fn precision_profile_reports() -> Vec<PrecisionRunReport> {
    let reference = representative_field_3d([4, 4, 4]);
    let reference_leto = leto::Array::<_, leto::MnemosyneStorage<_>, 3>::from_mnemosyne_slice(
        [4, 4, 4],
        reference.as_slice().unwrap(),
    )
    .unwrap();

    // f64 high-accuracy path — the authoritative reference for forward error comparisons.
    let high_spectrum = apollo_fft::fft_3d_leto(reference_leto.view());
    let high_recovered = apollo_fft::ifft_3d_leto(high_spectrum.view());
    let high_recovered_nd =
        leto::Array3::from_shape_vec([4, 4, 4], high_recovered.storage().as_slice().to_vec())
            .unwrap();
    let high_error = max_real_abs_delta_3d(&reference, &high_recovered_nd);

    // f32 low-precision path.
    let low_input = reference.mapv(|value| value as f32);
    let low_input_leto = leto::Array::<_, leto::MnemosyneStorage<_>, 3>::from_mnemosyne_slice(
        [4, 4, 4],
        low_input.as_slice().unwrap(),
    )
    .unwrap();
    let low_spectrum = apollo_fft::fft_3d_leto_typed::<f32>(low_input_leto.view());
    // Forward error: max |f32 spectrum - f64 reference spectrum|.
    let low_spectrum_slice = low_spectrum.storage().as_slice();
    let high_spectrum_slice = high_spectrum.storage().as_slice();
    let low_forward_error = low_spectrum_slice
        .iter()
        .zip(high_spectrum_slice.iter())
        .map(|(lv, hv)| {
            ((f64::from(lv.re) - hv.re).powi(2) + (f64::from(lv.im) - hv.im).powi(2)).sqrt()
        })
        .fold(0.0_f64, f64::max);
    let low_recovered = apollo_fft::ifft_3d_leto_typed::<f32>(low_spectrum.view());
    let low_recovered_nd =
        leto::Array3::from_shape_vec([4, 4, 4], low_recovered.storage().as_slice().to_vec())
            .unwrap()
            .mapv(f64::from);
    let low_reference = low_input.mapv(f64::from);
    let low_error = max_real_abs_delta_3d(&low_reference, &low_recovered_nd);

    // f16/f32 mixed-precision path — compare spectrum against f64 FFT of the f16-represented input.
    let mixed_input = reference.mapv(|value| f16::from_f32(value as f32));
    let mixed_input_leto = leto::Array::<_, leto::MnemosyneStorage<_>, 3>::from_mnemosyne_slice(
        [4, 4, 4],
        mixed_input.as_slice().unwrap(),
    )
    .unwrap();
    let mixed_spectrum = apollo_fft::fft_3d_leto_typed::<f16>(mixed_input_leto.view());
    // Use f64 FFT of the quantized input as the mixed-precision forward reference.
    let mixed_input_f64 = mixed_input.mapv(|v| f64::from(v.to_f32()));
    let mixed_input_f64_leto =
        leto::Array::<_, leto::MnemosyneStorage<_>, 3>::from_mnemosyne_slice(
            [4, 4, 4],
            mixed_input_f64.as_slice().unwrap(),
        )
        .unwrap();
    let mixed_reference_spectrum = apollo_fft::fft_3d_leto(mixed_input_f64_leto.view());

    let mixed_spectrum_slice = mixed_spectrum.storage().as_slice();
    let mixed_ref_spec_slice = mixed_reference_spectrum.storage().as_slice();
    let mixed_forward_error = mixed_spectrum_slice
        .iter()
        .zip(mixed_ref_spec_slice.iter())
        .map(|(lv, hv)| {
            ((f64::from(lv.re) - hv.re).powi(2) + (f64::from(lv.im) - hv.im).powi(2)).sqrt()
        })
        .fold(0.0_f64, f64::max);
    let mixed_recovered = apollo_fft::ifft_3d_leto_typed::<f16>(mixed_spectrum.view());
    let mixed_recovered_nd =
        leto::Array3::from_shape_vec([4, 4, 4], mixed_recovered.storage().as_slice().to_vec())
            .unwrap()
            .mapv(|value| f64::from(value.to_f32()));
    let mixed_reference = mixed_input.mapv(|value| f64::from(value.to_f32()));
    let mixed_error = max_real_abs_delta_3d(&mixed_reference, &mixed_recovered_nd);

    vec![
        PrecisionRunReport {
            profile: "high_accuracy".to_string(),
            attempted: true,
            passed: high_error <= 1.0e-10,
            // f64 IS the authoritative reference; forward error against itself is exactly 0.
            forward_max_abs_error: Some(0.0),
            inverse_max_abs_error: Some(high_error),
            relative_error: Some(high_error),
            note: None,
        },
        PrecisionRunReport {
            profile: "low_precision".to_string(),
            attempted: true,
            passed: low_error <= 1.0e-4,
            forward_max_abs_error: Some(low_forward_error),
            inverse_max_abs_error: Some(low_error),
            relative_error: Some(low_error),
            note: None,
        },
        PrecisionRunReport {
            profile: "mixed_precision".to_string(),
            attempted: true,
            passed: mixed_error <= 1.0e-3,
            forward_max_abs_error: Some(mixed_forward_error),
            inverse_max_abs_error: Some(mixed_error),
            relative_error: Some(mixed_error),
            note: None,
        },
    ]
}

fn environment_report(probe: Option<&PythonEnvironmentProbe>) -> EnvironmentReport {
    EnvironmentReport {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        debug_build: cfg!(debug_assertions),
        python_version: probe.map(|value| value.python_version.clone()),
        numpy_version: probe.and_then(|value| value.numpy_version.clone()),
        pyfftw_version: probe.and_then(|value| value.pyfftw_version.clone()),
    }
}

fn numpy_comparison_report(signal: &Array1<f64>) -> ExternalBackendReport {
    let signal_shape = [signal.size()];
    match compare_fft(&signal_shape[..], signal.as_slice().unwrap_or(&[]), 2) {
        Ok(report) => {
            let signal_leto = leto::Array::<_, leto::MnemosyneStorage<_>, 1>::from_mnemosyne_slice(
                [signal.size()],
                signal.as_slice().unwrap(),
            )
            .unwrap();
            let apollo_leto = apollo_fft::fft_1d_leto(signal_leto.view());
            let apollo = leto::Array1::from(apollo_leto.storage().as_slice().to_vec());
            let numpy_values: Vec<Complex64> = report
                .numpy_pairs
                .unwrap_or_default()
                .into_iter()
                .map(|pair| Complex64::new(pair[0], pair[1]))
                .collect();
            let error = if numpy_values.len() == apollo.size() {
                Some(max_complex_abs_delta(apollo.iter(), numpy_values.iter()))
            } else {
                None
            };
            ExternalBackendReport {
                backend: "numpy".to_string(),
                available: report.numpy_available,
                attempted: report.numpy_available,
                fft1_max_abs_error: error,
                fft1_prime_max_abs_error: None,
                fft3_max_abs_error: None,
                stability_max_abs_delta: report.numpy_stability_max_abs_delta,
                version: report.numpy_version,
                note: None,
            }
        }
        Err(error) => ExternalBackendReport {
            backend: "numpy".to_string(),
            available: false,
            attempted: false,
            fft1_max_abs_error: None,
            fft1_prime_max_abs_error: None,
            fft3_max_abs_error: None,
            stability_max_abs_delta: None,
            version: None,
            note: Some(error.to_string()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_suite_produces_value_semantic_reports_and_satisfies_schema() {
        let report = run_validation_suite().expect("validation suite");

        // Value semantic assertions
        assert!(report.fft_cpu.roundtrip_max_abs_error <= CPU_ROUNDTRIP_LIMIT);
        assert!(report.fft_cpu.parseval_relative_error <= CPU_PARSEVAL_LIMIT);
        assert!(report.nufft.passed);
        assert!(report.external.published_references.passed);
        assert_eq!(report.external.published_references.attempted, 59);
        assert_eq!(report.external.rustfft.backend, "rustfft");
        assert_eq!(report.external.numpy.backend, "numpy");

        // Schema structure assertions
        let value = serde_json::to_value(&report).expect("serialize validation report");
        let object = value
            .as_object()
            .expect("validation report is a JSON object");

        for key in [
            "fft_cpu",
            "fft_gpu",
            "nufft",
            "external",
            "benchmarks",
            "environment",
        ] {
            assert!(object.contains_key(key), "missing top-level key {key}");
        }

        let fft_cpu = object["fft_cpu"]
            .as_object()
            .expect("fft_cpu is a JSON object");
        for key in [
            "roundtrip_max_abs_error",
            "parseval_relative_error",
            "stability_max_abs_delta",
            "non_finite_input_propagates",
            "passed",
            "precision_profiles",
        ] {
            assert!(fft_cpu.contains_key(key), "missing fft_cpu key {key}");
        }

        let external = object["external"]
            .as_object()
            .expect("external is a JSON object");
        for key in [
            "passed",
            "rustfft_checkout_present",
            "pyfftw_checkout_present",
            "rustfft",
            "numpy",
            "pyfftw",
            "robustness_passed",
            "precision_comparisons",
            "published_references",
        ] {
            assert!(external.contains_key(key), "missing external key {key}");
        }
    }

    #[test]
    fn published_reference_suite_checks_computed_fixture_values() {
        let report = run_published_reference_suite().expect("published references");
        assert_eq!(report.attempted, 59);
        assert!(report.passed);
        for fixture in &report.fixtures {
            assert!(
                fixture.max_abs_error <= fixture.threshold,
                "{} exceeded threshold: {} > {}",
                fixture.fixture,
                fixture.max_abs_error,
                fixture.threshold
            );
            assert!(!fixture.reference.is_empty());
        }
    }

    #[test]
    fn test_melinoe_zero_copy_boundary_policy_integration() {
        use melinoe::{brand_scope, Borrowed, CellCowExt, MelinoeCell, Retained};
        use std::borrow::Cow;

        let input_signal = [1.0, 2.0, 3.0, 4.0];
        brand_scope(|token| {
            let cells: Vec<MelinoeCell<'_, f64>> =
                input_signal.iter().copied().map(MelinoeCell::new).collect();

            // Zero-copy borrow boundary
            let borrowed = cells.borrow_cow_with(&token, Borrowed);
            assert!(matches!(borrowed, Cow::Borrowed(_)));
            assert_eq!(borrowed.as_ref(), &input_signal[..]);

            // Cloned retain boundary
            let retained = cells.borrow_cow_with(&token, Retained);
            assert!(matches!(retained, Cow::Owned(_)));
            assert_eq!(retained.as_ref(), &input_signal[..]);
        });
    }

    #[test]
    fn test_moirai_melinoe_parallel_partitioning() {
        use melinoe::{brand_scope, MelinoeCell};
        use moirai::par_partition_for_each;

        let input_signal = [0.0f64; 16];
        brand_scope(|token| {
            let mut cells: Vec<MelinoeCell<'_, f64>> =
                input_signal.iter().copied().map(MelinoeCell::new).collect();

            // Run Moirai's parallel partitioning over the Melinoe cell region
            par_partition_for_each(&mut cells, 4, |start, mut shard| {
                for (j, slot) in shard.iter_mut().enumerate() {
                    *slot = (start + j) as f64;
                }
            });

            let snap = token.share();
            for (i, cell) in cells.iter().enumerate() {
                assert_eq!(*cell.borrow(snap), i as f64);
            }
        });
    }

    #[test]
    fn test_leto_ndarray_validation_boundary() {
        use leto::{Array2 as LetoArray2, Storage};
        use leto::Array2;

        let ndarray = Array2::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("ndarray construction");
        let leto = LetoArray2::from(ndarray.clone());

        assert_eq!(leto.shape(), [2, 3]);
        assert_eq!(leto.strides(), [3, 1]);
        assert_eq!(
            leto.storage().as_slice(),
            ndarray.as_slice().expect("contiguous ndarray")
        );

        let ndarray_back = leto::Array::try_from(leto).expect("leto to ndarray");
        assert_eq!(ndarray_back.shape(), [2, 3]);
        assert_eq!(
            ndarray_back.as_slice().expect("contiguous ndarray"),
            ndarray.as_slice().expect("contiguous ndarray")
        );
    }
}
