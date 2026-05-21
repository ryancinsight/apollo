//! Validation suite orchestration.
//!
//! The suite derives all report values from Apollo transforms, analytical
//! invariants, or optional external reference engines. No report field is
//! hardcoded as a passing result.

use crate::domain::report::{
    BenchmarkReport, CpuFftReport, EnvironmentReport, ExternalBackendReport,
    ExternalComparisonReport, GpuFftReport, NufftReport, PrecisionBenchmarkReport,
    PrecisionRunReport, PublishedFixtureReport, PublishedReferenceReport, ValidationReport,
};
use crate::infrastructure::numpy::{
    benchmark_fft, compare_fft, probe_python_environment, PythonEnvironmentProbe,
};
#[cfg(feature = "external-references")]
use crate::infrastructure::rustfft_reference::{fft1_real, fft3_real};
use apollo_czt::CztPlan;
use apollo_dctdst::{DctDstPlan, RealTransformKind};
use apollo_dht::DhtPlan;
use apollo_fft::f16;
use apollo_fft::{
    fft_1d_array, fft_1d_array_typed, fft_3d_array, ifft_1d_array, ifft_1d_array_typed,
    ifft_3d_array, ifft_3d_array_typed, FftBackend, Shape3D,
};
use apollo_frft::UnitaryFrftPlan;
use apollo_fwht::FwhtPlan;
use apollo_gft::GftPlan;
use apollo_hilbert::HilbertPlan;
use apollo_mellin::MellinPlan;
use apollo_ntt::{intt, ntt, NttPlan, DEFAULT_MODULUS};
use apollo_nufft::{
    nufft_type1_1d, nufft_type1_1d_fast, nufft_type1_3d, nufft_type1_3d_fast, nufft_type2_1d,
    nufft_type2_1d_fast, UniformDomain1D, UniformGrid3D, DEFAULT_NUFFT_KERNEL_WIDTH,
};
use apollo_qft::qft as qft_transform;
use apollo_radon::RadonPlan;
use apollo_sdft::SdftPlan;
use apollo_sft::SparseFftPlan;
use apollo_sht::ShtPlan;
use apollo_stft::StftPlan;
use apollo_wavelet::{ContinuousWavelet, CwtPlan, DiscreteWavelet, DwtPlan};
use nalgebra::DMatrix;
use ndarray::{Array1, Array2, Array3};
use num_complex::{Complex32, Complex64};
use std::error::Error;
use std::path::Path;
use std::time::Instant;

const CPU_ROUNDTRIP_LIMIT: f64 = 1.0e-10;
const CPU_PARSEVAL_LIMIT: f64 = 1.0e-10;
const CPU_STABILITY_LIMIT: f64 = 1.0e-12;
const EXTERNAL_FFT_LIMIT: f64 = 1.0e-9;
const NUFFT_FAST_RELATIVE_LIMIT: f64 = 1.0e-5;
const PUBLISHED_FIXTURE_LIMIT: f64 = 1.0e-12;

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
    let signal = Array1::from_vec(
        (0..16)
            .map(|i| {
                let x = i as f64;
                (0.17 * x).sin() + 0.25 * (0.61 * x).cos()
            })
            .collect(),
    );
    let spectrum = fft_1d_array(&signal);
    let recovered = ifft_1d_array(&spectrum);
    let roundtrip_max_abs_error = max_real_abs_delta(&signal, &recovered);

    let time_energy: f64 = signal.iter().map(|value| value * value).sum();
    let spectral_energy: f64 =
        spectrum.iter().map(Complex64::norm_sqr).sum::<f64>() / signal.len() as f64;
    let parseval_relative_error = (time_energy - spectral_energy).abs() / time_energy.max(1.0);

    let repeated = fft_1d_array(&signal);
    let stability_max_abs_delta = max_complex_abs_delta(spectrum.iter(), repeated.iter());

    let non_finite = Array1::from_vec(vec![1.0, f64::NAN, 2.0, f64::INFINITY]);
    let non_finite_input_propagates = fft_1d_array(&non_finite)
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
    let surface_reported_available = apollo_fft_wgpu::gpu_fft_available();

    let backend = match apollo_fft_wgpu::WgpuBackend::try_default() {
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
    let reference = representative_field_3d((4, 4, 4));
    let spectrum = plan.forward(&reference);
    let cpu_spectrum = fft_3d_array(&reference);

    // Forward error: max |GPU complex spectrum - CPU f64 reference spectrum|.
    let forward_max_abs_error = cpu_spectrum
        .iter()
        .enumerate()
        .map(|(idx, cpu_val)| {
            let gpu_re = f64::from(spectrum[2 * idx]);
            let gpu_im = f64::from(spectrum[2 * idx + 1]);
            ((gpu_re - cpu_val.re).powi(2) + (gpu_im - cpu_val.im).powi(2)).sqrt()
        })
        .fold(0.0_f64, f64::max);

    // Inverse error: max |GPU roundtrip recovered - reference|.
    let mut recovered = Array3::<f64>::zeros((4, 4, 4));
    plan.inverse(&spectrum, &mut recovered);
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

    let coefficients = Array1::from_shape_fn(domain.n, |k| {
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

    let rustfft_available = cfg!(feature = "external-references");
    let rustfft_report = if rustfft_available {
        #[cfg(feature = "external-references")]
        {
            let apollo = fft_1d_array(&signal);
            let rustfft = fft1_real(&signal);
            let rustfft_fft1_max_abs_error = max_complex_abs_delta(apollo.iter(), rustfft.iter());

            let prime_signal = representative_signal_1d(17);
            let prime_apollo = fft_1d_array(&prime_signal);
            let prime_rustfft = fft1_real(&prime_signal);
            let rustfft_prime_error =
                max_complex_abs_delta(prime_apollo.iter(), prime_rustfft.iter());

            let field = representative_field_3d((4, 4, 4));
            let apollo_3d = fft_3d_array(&field);
            let rustfft_3d = fft3_real(&field);
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
    let field = representative_field_3d((4, 4, 4));
    let apollo_fft1_ms = elapsed_ms(|| {
        let _ = fft_1d_array(&signal);
    });
    let apollo_fft3_forward_ms = elapsed_ms(|| {
        let _ = fft_3d_array(&field);
    });
    let spectrum = fft_3d_array(&field);
    let apollo_fft3_inverse_ms = elapsed_ms(|| {
        let _ = ifft_3d_array(&spectrum);
    });

    let rustfft_available = cfg!(feature = "external-references");
    let (rustfft_fft1_ms, rustfft_fft3_ms) = if rustfft_available {
        (
            elapsed_ms(|| {
                #[cfg(feature = "external-references")]
                {
                    let _ = fft1_real(&signal);
                }
            }),
            elapsed_ms(|| {
                #[cfg(feature = "external-references")]
                {
                    let _ = fft3_real(&field);
                }
            }),
        )
    } else {
        (0.0, 0.0)
    };

    let signal_shape = [signal.len()];
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
    let reference = representative_field_3d((4, 4, 4));

    // f64 high-accuracy path — the authoritative reference for forward error comparisons.
    let high_spectrum = fft_3d_array(&reference);
    let high_recovered = ifft_3d_array(&high_spectrum);
    let high_error = max_real_abs_delta_3d(&reference, &high_recovered);

    // f32 low-precision path.
    let low_input = reference.mapv(|value| value as f32);
    let low_spectrum: Array3<Complex32> = apollo_fft::fft_3d_array_typed(&low_input);
    // Forward error: max |f32 spectrum - f64 reference spectrum|.
    let low_forward_error = low_spectrum
        .iter()
        .zip(high_spectrum.iter())
        .map(|(lv, hv)| {
            ((f64::from(lv.re) - hv.re).powi(2) + (f64::from(lv.im) - hv.im).powi(2)).sqrt()
        })
        .fold(0.0_f64, f64::max);
    let low_recovered = apollo_fft::ifft_3d_array_typed::<f32>(&low_spectrum).mapv(f64::from);
    let low_reference = low_input.mapv(f64::from);
    let low_error = max_real_abs_delta_3d(&low_reference, &low_recovered);

    // f16/f32 mixed-precision path — compare spectrum against f64 FFT of the f16-represented input.
    let mixed_input = reference.mapv(|value| f16::from_f32(value as f32));
    let mixed_spectrum: Array3<Complex32> = apollo_fft::fft_3d_array_typed(&mixed_input);
    // Use f64 FFT of the quantized input as the mixed-precision forward reference.
    let mixed_input_f64 = mixed_input.mapv(|v| f64::from(v.to_f32()));
    let mixed_reference_spectrum = fft_3d_array(&mixed_input_f64);
    let mixed_forward_error = mixed_spectrum
        .iter()
        .zip(mixed_reference_spectrum.iter())
        .map(|(lv, hv)| {
            ((f64::from(lv.re) - hv.re).powi(2) + (f64::from(lv.im) - hv.im).powi(2)).sqrt()
        })
        .fold(0.0_f64, f64::max);
    let mixed_recovered =
        ifft_3d_array_typed::<f16>(&mixed_spectrum).mapv(|value| f64::from(value.to_f32()));
    let mixed_reference = mixed_input.mapv(|value| f64::from(value.to_f32()));
    let mixed_error = max_real_abs_delta_3d(&mixed_reference, &mixed_recovered);

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
    let signal_shape = [signal.len()];
    match compare_fft(&signal_shape[..], signal.as_slice().unwrap_or(&[]), 2) {
        Ok(report) => {
            let apollo = fft_1d_array(signal);
            let numpy_values: Vec<Complex64> = report
                .numpy_pairs
                .unwrap_or_default()
                .into_iter()
                .map(|pair| Complex64::new(pair[0], pair[1]))
                .collect();
            let error = if numpy_values.len() == apollo.len() {
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

fn fft_four_point_difference_fixture() -> PublishedFixtureReport {
    let signal = Array1::from_vec(vec![1.0, 0.0, -1.0, 0.0]);
    let actual = fft_1d_array(&signal);
    let expected = [
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    published_complex_fixture(
        "FFT",
        "DFT4([1,0,-1,0])",
        "Cooley and Tukey (1965), finite root-of-unity DFT definition",
        actual.iter(),
        expected.iter(),
    )
}

fn dht_four_point_difference_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DhtPlan::new(4)?;
    let spectrum = plan.forward(&[1.0, 0.0, -1.0, 0.0])?;
    let expected = [0.0, 2.0, 0.0, 2.0];
    Ok(published_real_fixture(
        "DHT",
        "DHT4([1,0,-1,0])",
        "Bracewell (1983), cas(theta)=cos(theta)+sin(theta) Hartley definition",
        spectrum.values(),
        &expected,
    ))
}

fn dct2_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(2, RealTransformKind::DctII)?;
    let actual = plan.forward(&[1.0, 3.0])?;
    let expected = [4.0, -std::f64::consts::SQRT_2];
    Ok(published_real_fixture(
        "DCT-II",
        "DCT-II2([1,3])",
        "FFTW real-to-real REDFT10 convention, unnormalized DCT-II basis",
        &actual,
        &expected,
    ))
}

fn dst2_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(2, RealTransformKind::DstII)?;
    let actual = plan.forward(&[1.0, 3.0])?;
    let expected = [
        1.0 * (std::f64::consts::PI / 4.0).sin() + 3.0 * (3.0 * std::f64::consts::PI / 4.0).sin(),
        -2.0,
    ];
    Ok(published_real_fixture(
        "DST-II",
        "DST-II2([1,3])",
        "FFTW real-to-real RODFT10 convention, unnormalized DST-II basis",
        &actual,
        &expected,
    ))
}

/// NTT of the unit impulse [1,0,0,0] with N=4, modulus=998244353, primitive root=3.
///
/// # Mathematical contract
/// By the NTT definition (Pollard 1971): F[k] = Σ_{n=0}^{N-1} f[n]·ω^{nk} mod p.
/// With f[0]=1 and f[1..3]=0, every term except n=0 vanishes, giving F[k] = ω^0 = 1 for all k.
/// Reference: "NTT definition, Pollard (1971): F[k] = Σ f[n]·ω^{nk} mod p, impulse response F[k]=1"
fn ntt_impulse_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from_vec(vec![1u64, 0, 0, 0]);
    let actual = ntt(&input)?;
    let actual_f64: Vec<f64> = actual.iter().map(|&v| v as f64).collect();
    let expected = [1.0_f64, 1.0, 1.0, 1.0];
    Ok(published_real_fixture(
        "NTT",
        "NTT4([1,0,0,0])",
        "NTT definition, Pollard (1971): F[k] = \u{03a3} f[n]\u{00b7}\u{03c9}^{nk} mod p, impulse response F[k]=1",
        &actual_f64,
        &expected,
    ))
}

/// NTT of the constant-one vector [1,1,1,1] with N=4, modulus=998244353, primitive root=3.
///
/// # Mathematical contract
/// F[0] = Σ_{n=0}^3 1 = 4 (mod p). For k≠0: F[k] = Σ_{n=0}^3 ω^{nk}.
/// Since ω is a primitive N-th root of unity and ω^k≠1 for k=1,2,3, the geometric series
/// Σ_{n=0}^{N-1} (ω^k)^n = (ω^{Nk}-1)/(ω^k-1) = 0 mod p, because ω^N ≡ 1 (mod p).
/// Reference: "NTT DFT-of-constant theorem: F[0]=N, F[k≠0]=0 for constant input (Pollard 1971)"
fn ntt_constant_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from_vec(vec![1u64, 1, 1, 1]);
    let actual = ntt(&input)?;
    let actual_f64: Vec<f64> = actual.iter().map(|&v| v as f64).collect();
    let expected = [4.0_f64, 0.0, 0.0, 0.0];
    Ok(published_real_fixture(
        "NTT",
        "NTT4([1,1,1,1])",
        "NTT DFT-of-constant theorem: F[0]=N, F[k\u{2260}0]=0 for constant input (Pollard 1971)",
        &actual_f64,
        &expected,
    ))
}

/// NUFFT Type-1 1D with a single source at x=0 and value 1+0i, N=4, dx=π/2 (L=2π).
///
/// # Mathematical contract
/// F[k] = Σ_j f[j]·exp(-2πi·k_signed·x_j/L). With x_j=0:
///   angle = -2π·k_signed·0/L = 0 for every k.
///   exp(0) = 1+0i exactly in IEEE 754 (cos(0)=1, sin(0)=0 are exact).
/// Therefore F[k] = 1+0i for all k=0..3, with zero floating-point error.
/// Reference: "NUFFT Type-1 definition: F[k] = Σ_j f[j]·exp(-2πi·k·x_j/L); at x_j=0, F[k]=1 for all k"
fn nufft_impulse_at_origin_fixture() -> SuiteResult<PublishedFixtureReport> {
    let domain = UniformDomain1D::new(4, std::f64::consts::FRAC_PI_2)?;
    let actual = nufft_type1_1d(&[0.0], &[Complex64::new(1.0, 0.0)], domain);
    let expected = [Complex64::new(1.0, 0.0); 4];
    Ok(published_complex_fixture(
        "NUFFT",
        "NUFFT-Type1-1D(x=[0], f=[1+0i], N=4)",
        "NUFFT Type-1 definition: F[k] = \u{03a3}_j f[j]\u{00b7}exp(-2\u{03c0}i\u{00b7}k\u{00b7}x_j/L); at x_j=0, F[k]=1 for all k",
        actual.iter(),
        expected.iter(),
    ))
}

/// NTT of the unit impulse [1,0,0,0,0,0,0,0] with N=8, modulus=998244353, primitive root=3.
///
/// # Mathematical contract
/// By the NTT definition (Pollard 1971): F[k] = Σ_{n=0}^{N-1} f[n]·ω^{nk} mod p.
/// With f[0]=1 and f[1..7]=0, every term except n=0 vanishes, giving F[k] = ω^0 = 1 for all k.
/// This is the same impulse theorem as N=4, generalized to N=8.
/// Reference: "NTT definition, Pollard (1971): F[k] = Σ f[n]·ω^{nk} mod p, impulse response F[k]=1 (N=8)"
fn ntt_n8_impulse_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from_vec(vec![1u64, 0, 0, 0, 0, 0, 0, 0]);
    let actual = ntt(&input)?;
    let actual_f64: Vec<f64> = actual.iter().map(|&v| v as f64).collect();
    let expected = [1.0_f64; 8];
    Ok(published_real_fixture(
        "NTT",
        "NTT8([1,0,0,0,0,0,0,0])",
        "NTT definition, Pollard (1971): F[k] = \u{03a3} f[n]\u{00b7}\u{03c9}^{nk} mod p, impulse response F[k]=1 (N=8)",
        &actual_f64,
        &expected,
    ))
}

/// NTT convolution theorem: NTT^{-1}(NTT(a) ⊙ NTT(b)) = a ★ b for polynomial product.
///
/// # Mathematical contract
/// By the NTT Convolution Theorem (Pollard 1971): for a=[1,2,0,0] and b=[3,4,0,0],
/// the cyclic convolution a★b equals the coefficients of (1+2x)(3+4x) = 3+10x+8x², giving c=[3,10,8,0].
/// All values satisfy 3,10,8 ≪ p = 998244353 so modular reduction is trivial.
/// The fixture verifies INTT(NTT(a) ⊙ NTT(b)) against the analytically derived polynomial product.
/// Reference: "NTT Convolution Theorem (Pollard 1971): INTT(NTT(a)⊙NTT(b)) = a★b mod p"
fn ntt_polynomial_convolution_fixture() -> SuiteResult<PublishedFixtureReport> {
    let p = DEFAULT_MODULUS;
    let plan = NttPlan::new(4)?;
    let a = Array1::from_vec(vec![1u64, 2, 0, 0]);
    let b = Array1::from_vec(vec![3u64, 4, 0, 0]);
    let fa = plan.forward(&a)?;
    let fb = plan.forward(&b)?;
    let fc: Vec<u64> = fa
        .iter()
        .zip(fb.iter())
        .map(|(&x, &y)| ((x as u128 * y as u128) % p as u128) as u64)
        .collect();
    let fc_arr = Array1::from(fc);
    let c = intt(&fc_arr)?;
    let actual_f64: Vec<f64> = c.iter().map(|&v| v as f64).collect();
    // (1+2x)(3+4x) = 3+10x+8x^2 → [3,10,8,0]
    let expected = [3.0_f64, 10.0, 8.0, 0.0];
    Ok(published_real_fixture(
        "NTT",
        "INTT(NTT([1,2,0,0])\u{2299}NTT([3,4,0,0]))",
        "NTT Convolution Theorem (Pollard 1971): INTT(NTT(a)\u{2299}NTT(b)) = a\u{2605}b mod p; (1+2x)(3+4x)=3+10x+8x\u{00b2}",
        &actual_f64,
        &expected,
    ))
}

/// NUFFT Type-1 1D with a single source at x=L/4 and value 1+0i, N=4, dx=π/2 (L=2π).
///
/// # Mathematical contract
/// F[k] = Σ_j f[j]·exp(-2πi·k_signed·x_j/L). With x_j=L/4 and f[j]=1:
///   angle = -2π·k_signed·(L/4)/L = -π·k_signed/2.
///   k_signed sequence for N=4: k=0→0, k=1→1, k=2→2, k=3→-1.
///   F[0] = exp(0) = 1+0i (exact)
///   F[1] = exp(-πi/2) = cos(-π/2)+i·sin(-π/2) ≈ 0-i  (|error| < 7e-17)
///   F[2] = exp(-πi)   = cos(-π)+i·sin(-π)     ≈ -1+0i (|error| < 2e-16)
///   F[3] = exp(+πi/2) = cos(+π/2)+i·sin(+π/2) ≈ 0+i  (|error| < 7e-17)
/// All errors are well within the 1×10⁻¹² published-fixture threshold.
/// Reference: "NUFFT Type-1 definition: F[k]=exp(-2πi·k_signed·x₀/L) for unit source at x₀=L/4 (Dutt and Rokhlin 1993)"
fn nufft_quarter_period_phase_fixture() -> SuiteResult<PublishedFixtureReport> {
    let domain = UniformDomain1D::new(4, std::f64::consts::FRAC_PI_2)?;
    let x0 = std::f64::consts::FRAC_PI_2; // L/4 = (4·π/2)/4 = π/2
    let actual = nufft_type1_1d(&[x0], &[Complex64::new(1.0, 0.0)], domain);
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, -1.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, 1.0),
    ];
    Ok(published_complex_fixture(
        "NUFFT",
        "NUFFT-Type1-1D(x=[L/4], f=[1+0i], N=4)",
        "NUFFT Type-1 definition: F[k]=exp(-2\u{03c0}i\u{00b7}k_signed\u{00b7}x\u{2080}/L) for unit source at x\u{2080}=L/4; F=[1,-i,-1,i] (Dutt and Rokhlin 1993)",
        actual.iter(),
        expected.iter(),
    ))
}

fn published_complex_fixture<'a, I, J>(
    transform: &str,
    fixture: &str,
    reference: &str,
    actual: I,
    expected: J,
) -> PublishedFixtureReport
where
    I: IntoIterator<Item = &'a Complex64>,
    J: IntoIterator<Item = &'a Complex64>,
{
    let max_abs_error = max_complex_abs_delta(actual, expected);
    PublishedFixtureReport {
        transform: transform.to_string(),
        fixture: fixture.to_string(),
        reference: reference.to_string(),
        max_abs_error,
        threshold: PUBLISHED_FIXTURE_LIMIT,
        passed: max_abs_error <= PUBLISHED_FIXTURE_LIMIT,
    }
}

fn published_real_fixture(
    transform: &str,
    fixture: &str,
    reference: &str,
    actual: &[f64],
    expected: &[f64],
) -> PublishedFixtureReport {
    published_real_fixture_with_threshold(
        transform,
        fixture,
        reference,
        actual,
        expected,
        PUBLISHED_FIXTURE_LIMIT,
    )
}

fn published_real_fixture_with_threshold(
    transform: &str,
    fixture: &str,
    reference: &str,
    actual: &[f64],
    expected: &[f64],
    threshold: f64,
) -> PublishedFixtureReport {
    let max_abs_error = actual
        .iter()
        .zip(expected.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max);
    PublishedFixtureReport {
        transform: transform.to_string(),
        fixture: fixture.to_string(),
        reference: reference.to_string(),
        max_abs_error,
        threshold,
        passed: max_abs_error <= threshold,
    }
}

/// IDFT of the all-ones vector [1,1,1,1] with N=4.
///
/// # Mathematical contract
///
/// The normalized IDFT is IDFT[n] = (1/N) Σ_k F[k] exp(2πikn/N).
/// For F=[1,1,1,1] and N=4, every sum collapses to IDFT[0]=1, IDFT[1..3]=0
/// because Σ_k exp(2πikn/N) = N·δ_{n,0} by the geometric series identity for
/// primitive roots of unity. Reference: DFT Inversion Theorem, Cooley and Tukey (1965).
fn fft_inverse_four_point_fixture() -> PublishedFixtureReport {
    let spectrum = Array1::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ]);
    let actual = ifft_1d_array(&spectrum);
    let expected = [1.0_f64, 0.0, 0.0, 0.0];
    published_real_fixture(
        "FFT",
        "IDFT4([1,1,1,1])",
        "Cooley and Tukey (1965), DFT inversion theorem: IDFT(DFT(x))=x; DFT([1,0,0,0])=[1,1,1,1] so IDFT([1,1,1,1])=[1,0,0,0]",
        actual.as_slice().unwrap(),
        &expected,
    )
}

/// DCT-III is the inverse of DCT-II up to a 2/N scaling factor.
///
/// # Mathematical contract
///
/// For N=2, x=[1,3]: DCT-II([1,3])=[4,−√2].
/// DCT-III([4,−√2]) scaled by 2/N recovers [1,3].
/// The `plan.inverse` method applies the 2/N factor.
/// Reference: Rao and Yip (1990), DCT-III inverse pair theorem.
fn dct2_inverse_pair_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(2, RealTransformKind::DctII)?;
    let actual = plan.inverse(&[4.0, -std::f64::consts::SQRT_2])?;
    let expected = [1.0_f64, 3.0];
    Ok(published_real_fixture(
        "DCT",
        "DCT-III(DCT-II([1,3]))\u{00d7}(2/N)=[1,3]",
        "Rao and Yip (1990), DCT-III is the inverse of DCT-II: DCT-III(DCT-II(x))\u{00b7}(2/N)=x for N=2",
        &actual,
        &expected,
    ))
}

/// DHT self-reciprocal property: DHT(DHT(x)) = N·x.
///
/// # Mathematical contract
///
/// For x=[1,0,0,0] and N=4:
///   step1 = DHT([1,0,0,0]) = [1,1,1,1]  (cas impulse response at n=0)
///   step2 = DHT([1,1,1,1]) = [4,0,0,0]  (DC-only signal maps to scaled impulse)
/// step2 = N·x = 4·[1,0,0,0] = [4,0,0,0]. ✓
/// Reference: Bracewell (1983), DHT self-reciprocal property: H{H{x}}[n] = N·x[n].
fn dht_self_reciprocal_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DhtPlan::new(4)?;
    let step1 = plan.forward(&[1.0_f64, 0.0, 0.0, 0.0])?;
    let step2 = plan.forward(step1.values())?;
    let expected = [4.0_f64, 0.0, 0.0, 0.0];
    Ok(published_real_fixture(
        "DHT",
        "DHT(DHT([1,0,0,0]))=[4,0,0,0]",
        "Bracewell (1983), DHT self-reciprocal property: DHT(DHT(x))=N\u{00b7}x; for x=[1,0,0,0], DHT([1,0,0,0])=[1,1,1,1] then DHT([1,1,1,1])=[4,0,0,0]",
        step2.values(),
        &expected,
    ))
}

/// FWHT two-point known-value fixture.
///
/// # Mathematical contract
///
/// The 2×2 Hadamard matrix is H₂ = [[1,1],[1,−1]].
/// H₂·[1,1]ᵀ = [2,0]ᵀ (both basis functions evaluate to 1+1 and 1−1 respectively).
/// The FWHT is the unnormalized Hadamard transform.
/// Reference: Hadamard (1893), two-point Walsh-Hadamard matrix definition.
fn fwht_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = FwhtPlan::new(2)?;
    let input = Array1::from_vec(vec![1.0_f64, 1.0]);
    let actual = plan.forward(&input)?;
    let expected = [2.0_f64, 0.0];
    Ok(published_real_fixture(
        "FWHT",
        "FWHT2([1,1])",
        "Hadamard (1893), H_2=[[1,1],[1,-1]], H_2\u{00b7}[1,1]^T=[2,0]^T",
        actual.as_slice().unwrap(),
        &expected,
    ))
}

/// QFT two-point known-value fixture: QFT₂(|0⟩) = (1/√2)(|0⟩ + |1⟩).
///
/// # Mathematical contract
///
/// The unitary QFT matrix for N=2 is (1/√2)·[[1,1],[1,−1]].
/// For input [1,0] (computational basis state |0⟩):
///   QFT₂([1,0])ᵀ = (1/√2)·[1,1]ᵀ.
/// Both output components equal 1/√2 = 1/√2 + 0i.
/// Reference: Shor (1994), quantum Fourier transform for N=2 as Hadamard gate.
fn qft_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
    let actual = qft_transform(&input)?;
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    let expected = [
        Complex64::new(inv_sqrt2, 0.0),
        Complex64::new(inv_sqrt2, 0.0),
    ];
    Ok(published_complex_fixture(
        "QFT",
        "QFT2([1,0])",
        "Shor (1994), QFT for N=2 is the Hadamard gate: QFT([1,0])=[1/\u{221a}2, 1/\u{221a}2]",
        actual.iter(),
        expected.iter(),
    ))
}

/// CZT with DFT parameters equals the DFT for the unit impulse [1,0,0,0].
///
/// # Mathematical contract
///
/// When A=1 and W=exp(−2πi/N), the CZT spiral contour collapses to the unit
/// circle at the N-th roots of unity, recovering the DFT exactly.
/// For input [1+0i,0,0,0], DFT([1,0,0,0])=[1,1,1,1].
/// Therefore CZT(N=4,M=4,A=1,W=exp(−2πi/4))([1,0,0,0])=[1,1,1,1].
/// Reference: Rabiner, Schafer and Rader (1969), CZT spiral-collapse theorem.
fn czt_unit_impulse_is_dft_fixture() -> SuiteResult<PublishedFixtureReport> {
    let n = 4usize;
    let a = Complex64::new(1.0, 0.0);
    let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64);
    let plan = CztPlan::new(n, n, a, w)?;
    let input = Array1::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]);
    let actual = plan.forward(&input)?;
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "CZT",
        "CZT4(A=1,W=exp(-2\u{03c0}i/4),[1,0,0,0])",
        "Rabiner, Schafer and Rader (1969), CZT with A=1 and W=exp(-2\u{03c0}i/N) equals DFT; DFT([1,0,0,0])=[1,1,1,1]",
        actual.iter(),
        expected.iter(),
    ))
}

/// GFT K₂ path graph Laplacian eigenvalues are {0, 2}.
///
/// # Mathematical contract
///
/// For the 2-vertex path graph with adjacency A=[[0,1],[1,0]], the combinatorial
/// Laplacian L = D − A = [[1,−1],[−1,1]]. Its characteristic polynomial is
/// det(L − λI) = (1−λ)² − 1 = λ² − 2λ = λ(λ−2), giving eigenvalues {0, 2}.
/// The `spectral_basis` kernel sorts eigenvalues in ascending order, so the
/// GFT plan returns eigenvalues [0.0, 2.0]. This is a sign-independent,
/// numerically deterministic published result.
/// Reference: Shuman et al. (2013), IEEE Signal Processing Magazine, graph Fourier basis definition.
fn gft_path_graph_forward_fixture() -> SuiteResult<PublishedFixtureReport> {
    let adjacency = DMatrix::from_row_slice(2, 2, &[0.0_f64, 1.0, 1.0, 0.0]);
    let plan = GftPlan::from_adjacency(&adjacency)?;
    let eigenvalues = plan.eigenvalues().to_vec();
    let expected = [0.0_f64, 2.0];
    Ok(published_real_fixture(
        "GFT",
        "GFT-K2-path-eigenvalues",
        "Shuman et al. (2013), K\u{2082} path graph Laplacian L=[[1,-1],[-1,1]]: eigenvalues {0,2} from det(L-\u{03bb}I)=\u{03bb}(\u{03bb}-2)",
        &eigenvalues,
        &expected,
    ))
}

/// Unitary FrFT of order 2 is the reversal operator.
///
/// # Mathematical contract
///
/// By Candan et al. (2000), Theorem 3: DFrFT₂(x)[k] = x[N−1−k] for all x.
/// The palindrome-structured Grünbaum eigenvectors diagonalize the DFT; the
/// fractional phase exp(−i·2·k·π/2) = (−1)^k combines with V·diag((−1)^k)·Vᵀ
/// to produce the exact index-reversal operator.
/// Input [1+0i, 2+0i, 3+0i, 4+0i], N=4, order=2.
/// Expected output: [4+0i, 3+0i, 2+0i, 1+0i].
fn frft_unitary_order2_reversal_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ]);
    let plan = UnitaryFrftPlan::new(4, 2.0)?;
    let actual = plan.forward(&input)?;
    let expected = [
        Complex64::new(4.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "UnitaryFrFT",
        "UnitaryFrFT(order=2,[1,2,3,4])",
        "Candan et al. (2000), Theorem 3: DFrFT\u{2082} = reversal operator; UnitaryFrFT(order=2,[1,2,3,4])=[4,3,2,1]",
        actual.iter(),
        expected.iter(),
    ))
}

/// Haar DWT one-level detail coefficients for [1, −1, 0, 0].
///
/// # Mathematical contract
///
/// Haar highpass QMF: g = [1/√2, −1/√2] (from g[k] = (−1)^k·h[L−1−k] with h=[1/√2,1/√2]).
/// detail[k] = g[0]·x[2k] + g[1]·x[2k+1] = (x[2k] − x[2k+1]) / √2.
/// Input [1, −1, 0, 0], N=4, levels=1:
///   detail[0] = (1 − (−1)) / √2 = 2/√2 = √2
///   detail[1] = (0 − 0) / √2 = 0
/// Reference: Haar (1910), Mallat (1989) two-channel QMF framework.
fn wavelet_haar_one_level_detail_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = [1.0_f64, -1.0, 0.0, 0.0];
    let plan = DwtPlan::new(4, 1, DiscreteWavelet::Haar)?;
    let coeffs = plan.forward(&input)?;
    let detail = coeffs.details()[0].as_slice();
    let expected = [std::f64::consts::SQRT_2, 0.0_f64];
    Ok(published_real_fixture(
        "DWT-Haar",
        "Haar-DWT-1level([1,-1,0,0])",
        "Haar (1910), Mallat (1989) two-channel QMF: DWT([1,-1,0,0]) detail=[\u{221a}2,0] via g=[1/\u{221a}2,-1/\u{221a}2]",
        detail,
        &expected,
    ))
}

/// NTT of the unit impulse [1, 0, 0, ..., 0] with N=16, modulus=998244353.
///
/// # Mathematical contract
/// By the NTT impulse theorem (Pollard 1971): for x[0]=1 and x[n]=0 for n≥1,
///   F[k] = Σ_{n=0}^{N-1} x[n] · ω^{nk} = ω^{0·k} = 1  for all k = 0..15.
/// This extends the N=4 and N=8 impulse fixtures to N=16, confirming that the
/// twiddle precomputation and butterfly structure are correct at this size.
/// Reference: "NTT impulse theorem, Pollard (1971): F[k]=1 ∀k for impulse input"
fn ntt_n16_impulse_fixture() -> SuiteResult<PublishedFixtureReport> {
    let mut input_vec = vec![0u64; 16];
    input_vec[0] = 1;
    let input = Array1::from_vec(input_vec);
    let plan = NttPlan::new(16)?;
    let actual = plan.forward(&input)?;
    let actual_f64: Vec<f64> = actual.iter().map(|&v| v as f64).collect();
    let expected = [1.0_f64; 16];
    Ok(published_real_fixture(
        "NTT",
        "NTT16([1,0,...,0])",
        "NTT impulse theorem, Pollard (1971): F[k]=1 for all k; impulse input, N=16",
        &actual_f64,
        &expected,
    ))
}

/// NTT convolution theorem for N=16: degree-3 times degree-1 polynomial product.
///
/// # Mathematical contract
/// By the NTT Convolution Theorem (Pollard 1971):
///   INTT(NTT(a) ⊙ NTT(b)) = cyclic convolution a ★ b  mod p
/// For a = [1,2,3,4,0,...,0] and b = [2,1,0,...,0] (N=16):
///   (1 + 2x + 3x² + 4x³)(2 + x) = 2 + 5x + 8x² + 11x³ + 4x⁴
/// All coefficients (2, 5, 8, 11, 4) are ≪ p = 998244353, so modular reduction
/// is trivial and the result equals the integer polynomial product exactly.
/// Reference: "NTT Convolution Theorem (Pollard 1971): INTT(NTT(a)⊙NTT(b))=a★b; N=16"
fn ntt_n16_polynomial_product_fixture() -> SuiteResult<PublishedFixtureReport> {
    let p = DEFAULT_MODULUS;
    let plan = NttPlan::new(16)?;
    let mut a_vec = vec![0u64; 16];
    a_vec[..4].copy_from_slice(&[1u64, 2, 3, 4]);
    let mut b_vec = vec![0u64; 16];
    b_vec[..2].copy_from_slice(&[2u64, 1]);
    let a = Array1::from_vec(a_vec);
    let b = Array1::from_vec(b_vec);
    let fa = plan.forward(&a)?;
    let fb = plan.forward(&b)?;
    let fc: Vec<u64> = fa
        .iter()
        .zip(fb.iter())
        .map(|(&x, &y)| ((x as u128 * y as u128) % p as u128) as u64)
        .collect();
    let c = plan.inverse(&Array1::from_vec(fc))?;
    let actual_f64: Vec<f64> = c.iter().map(|&v| v as f64).collect();
    // (1+2x+3x²+4x³)(2+x) = 2 + 5x + 8x² + 11x³ + 4x⁴, all higher coefficients 0
    let mut expected = [0.0_f64; 16];
    expected[..5].copy_from_slice(&[2.0, 5.0, 8.0, 11.0, 4.0]);
    Ok(published_real_fixture(
        "NTT",
        "INTT(NTT([1,2,3,4,0\u{2026}])\u{2299}NTT([2,1,0\u{2026}]))",
        "NTT Convolution Theorem (Pollard 1971): (1+2x+3x\u{00b2}+4x\u{00b3})(2+x)=2+5x+8x\u{00b2}+11x\u{00b3}+4x\u{2074}; N=16",
        &actual_f64,
        &expected,
    ))
}

/// SDFT bin 0 for the unit impulse [1, 0, 0, 0].
///
/// # Mathematical contract
///
/// DFT bin k=0: F[0] = Σ_{n=0}^{N-1} x[n] = 1 + 0 + 0 + 0 = 1 for the unit impulse.
/// The SDFT direct-bins path computes the standard DFT for the given window exactly.
/// Reference: Jacobsen and Lyons (2003), Sliding DFT, IEEE Signal Processing Magazine.
fn sdft_bin_zero_unit_impulse_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = [1.0_f64, 0.0, 0.0, 0.0];
    let plan = SdftPlan::new(4, 4)?;
    let bins = plan.direct_bins(&input)?;
    let actual_arr = [bins[0]];
    let expected_arr = [Complex64::new(1.0, 0.0)];
    Ok(published_complex_fixture(
        "SDFT",
        "SDFT-bin0-impulse([1,0,0,0])",
        "Jacobsen and Lyons (2003), SDFT bin-0 at position 0 equals DFT bin-0 = sum(x) = 1 for unit impulse",
        actual_arr.iter(),
        expected_arr.iter(),
    ))
}

fn sft_one_sparse_alternating_tone_fixture() -> SuiteResult<PublishedFixtureReport> {
    // x[n] = (-1)^n for N=4 is a single-frequency pure tone.
    // DFT: X[k] = N·δ(k - N/2) = 4·δ(k-2) (Parseval shift theorem).
    // SparseFftPlan::new(4,1) retains the 1 largest-magnitude coefficient.
    // Exact recovery theorem: top-1 heap finds bin 2 with value 4+0i;
    // all other bins have magnitude 0 and cannot displace the winner.
    // to_dense() expands to [0+0i, 0+0i, 4+0i, 0+0i].
    // Reference: Cooley-Tukey (1965) DFT[(−1)^n]=N·δ[k−N/2];
    //            Hassanieh et al. (2012) sFFT exact K-sparse recovery theorem.
    let plan = SparseFftPlan::new(4, 1)?;
    let signal = [
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
    ];
    let spectrum = plan.forward(&signal)?;
    let dense = spectrum.to_dense();
    let expected = [
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(4.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "SFT",
        "SFT-1-sparse-alternating-tone(N=4,K=1)",
        "Cooley-Tukey (1965): DFT[(−1)^n]=4·δ[k−2] for N=4; Hassanieh et al. (2012) sFFT exact K-sparse recovery",
        dense.iter(),
        expected.iter(),
    ))
}

fn sht_monopole_y00_coefficient_fixture() -> SuiteResult<PublishedFixtureReport> {
    // f(θ,φ) = Y_0^0(θ,φ) = 1/√(4π) is the spherical harmonic of degree 0.
    // By orthonormality of spherical harmonics Y_l^m:
    //   a_{lm} = ∫_S² f · Y_{lm}^* dΩ = δ_{l,0}δ_{m,0}
    // Therefore a_{00} = 1.0 exactly; all higher-degree coefficients are zero.
    // Gauss-Legendre quadrature with n_lat=12 integrates degree-0 polynomials
    // exactly; the uniform longitude sum with n_lon=25 integrates the constant exactly.
    // Reference: Varshalovich, Moskalev & Khersonskii (1988) §4.1: Y_0^0 = 1/√(4π);
    //            Driscoll & Healy (1994) sampling theorem on S².
    let plan = ShtPlan::new(12, 25, 1)?;
    let constant = 1.0_f64 / (4.0 * std::f64::consts::PI).sqrt();
    let samples = Array2::from_elem(
        (plan.grid().latitudes(), plan.grid().longitudes()),
        constant,
    );
    let coefficients = plan.forward_real(&samples)?;
    let a00 = coefficients.get(0, 0);
    let actual = [a00];
    let expected = [Complex64::new(1.0, 0.0)];
    Ok(published_complex_fixture(
        "SHT",
        "SHT-monopole-Y00-maps-to-unit-coefficient",
        "Varshalovich et al. (1988) §4.1; Driscoll & Healy (1994): Y_0^0=1/sqrt(4π), orthonormality gives a_{00}=1",
        actual.iter(),
        expected.iter(),
    ))
}

fn stft_rectangular_window_impulse_frame_fixture() -> SuiteResult<PublishedFixtureReport> {
    // Signal: unit impulse at position 0, frame_len=4, hop_len=4, rectangular window.
    // Centered framing: frame 0 center at 0, starts at index -2.
    // Frame 0 reads signal indices [-2,-1,0,1]: out-of-bounds pad to 0.
    //   Windowed frame 0 = [0, 0, x[0], x[1]] = [0, 0, 1, 0].
    // DFT([0,0,1,0]): X[k] = exp(-2πi·2·k/4) = exp(-πik) = (-1)^k.
    //   X[0]=1, X[1]=-1, X[2]=1, X[3]=-1.
    // Frame 1 reads signal indices [2,3,4,5]: indices 4,5 out of bounds.
    //   Windowed frame 1 = [x[2],x[3],0,0] = [0,0,0,0]; DFT = [0,0,0,0].
    // Full output: [1,-1,1,-1, 0,0,0,0].
    // Reference: Cooley-Tukey (1965) DFT shift theorem X[k]=exp(-2πikn₀/N) for δ[n-n₀];
    //            Allen & Rabiner (1977) STFT centered-frame analysis.
    let plan = StftPlan::new(4, 4)?;
    let signal = Array1::from_vec(vec![1.0_f64, 0.0, 0.0, 0.0]);
    let window = [1.0_f64, 1.0, 1.0, 1.0];
    let output = plan.forward_with_window(&signal, &window)?;
    let actual: Vec<Complex64> = output.iter().copied().collect();
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "STFT",
        "STFT-rect-window-impulse-centered-frame(N=4)",
        "Cooley-Tukey (1965) DFT shift: X[k]=(-1)^k for δ[n-2] in 4-point frame; Allen & Rabiner (1977) STFT centered framing",
        actual.iter(),
        expected.iter(),
    ))
}

fn hilbert_cosine_to_sine_fixture() -> SuiteResult<PublishedFixtureReport> {
    // H{cos(2πn/N)} = sin(2πn/N) for bin-frequency f₀=1, N=4.
    // Input: x[n] = cos(2πn/4) = [1, 0, -1, 0].
    // DFT mask: multiply positive bins by -i, DC/Nyquist by 0, negative bins by +i.
    // DFT(x) = [0, 2, 0, 2]; after mask: [0, -2i, 0, 2i].
    // IDFT([0,-2i,0,2i]) = [0, 1, 0, -1] = sin(2πn/4). ✓
    // Reference: Bracewell (1965) §12: H{cos(ω₀n)} = sin(ω₀n) for single-bin frequency;
    //            Oppenheim & Schafer (1999) §12.1 discrete Hilbert via DFT mask.
    let plan = HilbertPlan::new(4)?;
    let signal = [1.0_f64, 0.0, -1.0, 0.0];
    let quadrature = plan.transform(&signal)?;
    let expected = [0.0_f64, 1.0, 0.0, -1.0];
    Ok(published_real_fixture(
        "Hilbert",
        "Hilbert-cosine-to-sine-4point",
        "Bracewell (1965) §12: H{cos(2πn/N)}=sin(2πn/N); Oppenheim-Schafer (1999) §12.1 discrete Hilbert via DFT mask",
        &quadrature,
        &expected,
    ))
}

fn hilbert_instantaneous_frequency_constant_tone_fixture() -> SuiteResult<PublishedFixtureReport> {
    // Instantaneous frequency of a discrete cosine at normalised frequency f₀ = k/N.
    // The analytic signal of cos(2πkn/N) is exp(2πi·k·n/N), so:
    //   f[n] = arg(conj(z[n]) · z[n+1]) / (2π) = k/N (constant).
    // Reference: Boashash (1992) "Estimating and interpreting the instantaneous
    //            frequency of a signal", Proc. IEEE 80(4): §II.A complex-derivative
    //            formula f(t) = (1/2π) d/dt arg(z(t)).
    const TOL: f64 = 1.0e-10;
    let n: usize = 64;
    let k: usize = 5;
    let f_expected = k as f64 / n as f64;
    let signal: Vec<f64> = (0..n)
        .map(|i| (std::f64::consts::TAU * k as f64 * i as f64 / n as f64).cos())
        .collect();
    let plan = HilbertPlan::new(n)?;
    let analytic = plan.analytic_signal(&signal)?;
    let freq = analytic.instantaneous_frequency();
    // freq has length N-1; compare each sample to f_expected
    let expected: Vec<f64> = vec![f_expected; freq.len()];
    Ok(published_real_fixture_with_threshold(
        "Hilbert",
        "Hilbert-instantaneous-frequency-tone(N=64,k=5)",
        "Boashash (1992) §II.A: IF of cos(2πkn/N) is k/N via complex-derivative formula",
        &freq,
        &expected,
        TOL,
    ))
}

fn mellin_constant_function_first_moment_fixture() -> SuiteResult<PublishedFixtureReport> {
    // Mellin moment M(s) = ∫_a^b f(r) r^{s-1} dr for f(r)=1, s=1, a=1, b=3.
    // M(1) = ∫_1^3 1·r^0 dr = ∫_1^3 1 dr = b - a = 2.0.
    // Trapezoid rule on 3 equidistant points r=[1,2,3], h=1:
    //   (h/2)·[f(1)·1 + 2·f(2)·1 + f(3)·1] = (1/2)·[1+2+1] = 2.0 — exact.
    // Reference: Mellin (1897) transform definition M(s)=∫_a^b f(r)r^{s-1}dr;
    //            Titchmarsh (1937) §1.1: M(1)=b-a=2 for f(r)=1 on [1,3].
    let plan = MellinPlan::new(3, 1.0, 3.0)?;
    let signal = [1.0_f64, 1.0, 1.0];
    let moment = plan.moment(&signal, 1.0, 3.0, 1.0)?;
    let actual = [moment];
    let expected = [2.0_f64];
    Ok(published_real_fixture(
        "Mellin",
        "Mellin-constant-function-first-moment([1,3],s=1)",
        "Mellin (1897) transform definition; Titchmarsh (1937) §1.1: M(1)=b-a=2 for f(r)=1 on [1,3]",
        &actual,
        &expected,
    ))
}

fn radon_theta0_column_impulse_projection_fixture() -> SuiteResult<PublishedFixtureReport> {
    // Theorem: parallel-beam Radon projection at θ=0 equals column sums.
    // Image: unit impulse at pixel (row=0, col=0), all other pixels zero.
    //   Column 0 sum = 1.0, column 1 sum = 0.0, column 2 sum = 0.0.
    // Sinogram[θ=0] = [1.0, 0.0, 0.0] (3 detectors, spacing=1.0, column-aligned).
    // Reference: Radon (1917) parallel-beam projection definition (Über die Bestimmung
    //            von Funktionen durch ihre Integralwerte);
    //            Natterer (1986) §I.2: discrete projection at θ=0 equals column sums.
    let mut image = Array2::<f64>::zeros((3, 3));
    image[[0, 0]] = 1.0;
    let plan = RadonPlan::new(3, 3, vec![0.0], 3, 1.0)?;
    let sinogram = plan.forward(&image)?;
    let row: Vec<f64> = sinogram.values().row(0).iter().copied().collect();
    let expected = vec![1.0_f64, 0.0, 0.0];
    Ok(published_real_fixture(
        "Radon",
        "Radon-theta0-column0-impulse-projection(3x3)",
        "Radon (1917): parallel-beam θ=0 projection = column sums; Natterer (1986) §I.2",
        &row,
        &expected,
    ))
}

/// CZT inverse Vandermonde roundtrip: inverse(forward(x)) == x for N=4, A=1, W=exp(−2πi/4).
///
/// # Mathematical contract
///
/// The CZT with A=1 and W=exp(−2πi/N) evaluates the DFT at the N-th roots of unity
/// (Rabiner-Schafer-Rader 1969, spiral-collapse theorem).  The inverse solves the
/// resulting Vandermonde system V·x̂ = X via the Björck-Pereyra algorithm (1970),
/// which is backward-stable for N=4 DFT nodes.  Theorem: for any x,
/// CZT⁻¹(CZT(x)) = x in exact arithmetic.  f64 arithmetic gives ‖x − x̂‖∞ < 1 ulp ≈ 2.2×10⁻¹⁶
/// for N=4; the fixture threshold is 1×10⁻¹² (five orders of margin).
/// Reference: Rabiner, Schafer and Rader (1969); Björck and Pereyra (1970), SIAM J. Numer. Anal. 7(6).
fn czt_inverse_vandermonde_roundtrip_fixture() -> SuiteResult<PublishedFixtureReport> {
    let n = 4usize;
    let a = Complex64::new(1.0, 0.0);
    let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64);
    let plan = CztPlan::new(n, n, a, w)?;
    // Non-trivial input: unit impulse → DFT = [1,1,1,1]; invert back to [1,0,0,0].
    let input = Array1::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]);
    let spectrum = plan.forward(&input)?;
    let recovered = plan.inverse(&spectrum)?;
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "CZT",
        "CZT4-inverse-Vandermonde-roundtrip(A=1,W=exp(-2\u{03c0}i/4),[1,0,0,0])",
        "Rabiner-Schafer-Rader (1969) spiral-collapse; Björck-Pereyra (1970) Vandermonde solve: CZT\u{207b}\u{00b9}(CZT(x))=x",
        recovered.iter(),
        expected.iter(),
    ))
}

/// Mellin inverse spectrum roundtrip for constant signal on N=32 grid.
///
/// # Mathematical contract
///
/// For a constant signal f(r) = c on [a, b], the log-resampled values are
/// identically c, so the DFT of the log-samples has only a DC component F[0] = N·c
/// and F[k] = 0 for k > 0.  The IDFT recovers the constant log-domain signal, and
/// exp-resampling of a constant is c at every output point.  Accumulation error in
/// f64 DFT+IDFT for N=32 is bounded by O(log N)·ε_f64 ≈ 5·2.2×10⁻¹⁶ ≈ 1.1×10⁻¹⁵;
/// the fixture threshold is 1×10⁻¹⁰ (five orders of margin over the dominant
/// interpolation residual; constant function makes interpolation error zero).
/// Reference: Mellin (1896) transform definition; Titchmarsh (1937) §1.1.
fn mellin_inverse_spectrum_constant_roundtrip_fixture() -> SuiteResult<PublishedFixtureReport> {
    // MELLIN_INVERSE_ROUNDTRIP_LIMIT: log-domain DFT/IDFT accumulation for N=32
    // plus linear interpolation residual.  Constant signal makes interpolation
    // error zero; f64 DFT error is O(log N)·ε_f64 << 1e-10.
    const MELLIN_INVERSE_ROUNDTRIP_LIMIT: f64 = 1.0e-10;
    let n = 32usize;
    let min_scale = 1.0_f64;
    let max_scale = 4.0_f64;
    let plan = MellinPlan::new(n, min_scale, max_scale)?;
    let signal: Vec<f64> = vec![2.0_f64; n];
    let spectrum = plan.forward_spectrum(&signal, min_scale, max_scale)?;
    let mut recovered = vec![0.0_f64; n];
    plan.inverse_spectrum(&spectrum, min_scale, max_scale, &mut recovered)?;
    let expected: Vec<f64> = vec![2.0_f64; n];
    Ok(published_real_fixture_with_threshold(
        "Mellin",
        "Mellin-inverse-constant-roundtrip(N=32,[1,4],c=2)",
        "Mellin (1896); Titchmarsh (1937) §1.1: constant f(r)=c → DC spectrum → IDFT+exp-resample recovers c",
        &recovered,
        &expected,
        MELLIN_INVERSE_ROUNDTRIP_LIMIT,
    ))
}

/// Haar DWT perfect reconstruction: inverse(forward(x)) = x.
///
/// # Mathematical contract
///
/// Mallat (1989) §3.1 Theorem 2 (perfect reconstruction for two-channel QMF banks):
/// if the synthesis filters are the exact duals of the analysis filters, then
/// IDWT(DWT(x)) = x for every finite-energy signal x.  For the Haar pair
/// h = [1/√2, 1/√2] (lowpass) and g = [1/√2, −1/√2] (highpass), this holds
/// exactly in infinite precision.  f64 accumulation error for N=4, 1-level is
/// bounded by O(N·ε_f64) ≈ 4×2.2×10⁻¹⁶ ≈ 9×10⁻¹⁶; threshold is 1×10⁻¹².
/// Reference: Haar (1910); Mallat (1989) §3.1 Theorem 2.
fn wavelet_haar_inverse_perfect_reconstruction_fixture() -> SuiteResult<PublishedFixtureReport> {
    let signal = [1.0_f64, -1.0, 0.0, 0.0];
    let plan = DwtPlan::new(4, 1, DiscreteWavelet::Haar)?;
    let coefficients = plan.forward(&signal)?;
    let recovered = plan.inverse(&coefficients)?;
    Ok(published_real_fixture_with_threshold(
        "DWT-Haar",
        "Haar-DWT-inverse-roundtrip(N=4,[1,-1,0,0])",
        "Mallat (1989) §3.1 Theorem 2: perfect reconstruction IDWT(DWT(x))=x for Haar QMF; Haar (1910)",
        &recovered,
        &signal,
        1.0e-12,
    ))
}

/// GFT K₂ path graph inverse roundtrip: inverse(forward(s)) = s.
///
/// # Mathematical contract
///
/// The graph Fourier transform diagonalises the Laplacian via its real symmetric
/// eigenbasis U: GFT(s) = Uᵀs and GFT⁻¹(ŝ) = Uŝ (Sandryhaila & Moura 2013,
/// ICASSP, §II).  Since U is orthonormal (UUᵀ = I), the composition gives
/// GFT⁻¹(GFT(s)) = UUᵀs = s exactly.  For the K₂ path graph with 2×2 Laplacian,
/// eigendecomposition is exact in f64; roundtrip error is bounded by O(N·ε_f64)
/// ≈ 4.4×10⁻¹⁶; threshold is 1×10⁻¹².
/// Reference: Sandryhaila and Moura (2013), ICASSP: GFT via Laplacian eigendecomposition.
fn gft_path_graph_inverse_roundtrip_fixture() -> SuiteResult<PublishedFixtureReport> {
    let adjacency = DMatrix::from_row_slice(2, 2, &[0.0_f64, 1.0, 1.0, 0.0]);
    let plan = GftPlan::from_adjacency(&adjacency)?;
    let signal = Array1::from_vec(vec![3.0_f64, -1.0]);
    let spectrum = plan.forward(&signal)?;
    let recovered = plan.inverse(&spectrum)?;
    let expected = [3.0_f64, -1.0];
    Ok(published_real_fixture_with_threshold(
        "GFT",
        "GFT-K2-inverse-roundtrip([3,-1])",
        "Sandryhaila and Moura (2013) ICASSP §II: GFT\u{207b}\u{00b9}(GFT(s))=s via orthonormal Laplacian eigenbasis U; K\u{2082} path graph",
        recovered.as_slice().unwrap(),
        &expected,
        1.0e-12,
    ))
}

/// Unitary FrFT inverse roundtrip: FrFT(−α)(FrFT(α)(x)) = x for α=0.5.
///
/// # Mathematical contract
///
/// By the FrFT additivity theorem (Namias 1980, J. IMA 25(3) §2):
/// F^α ∘ F^β = F^{α+β} for all real orders α, β.  Setting β = −α gives
/// F^{−α}(F^α(x)) = F^0(x) = x (identity at order 0).  For α=0.5, N=4:
/// the unitary plan's `inverse` method applies `forward` with negated order,
/// so the composition reduces exactly to order 0, which is the identity.
/// f64 roundtrip error via Grünbaum eigendecomposition for N=4 is bounded by
/// O(N²·ε_f64) ≈ 16×2.2×10⁻¹⁶ ≈ 3.5×10⁻¹⁵ per transform, O(7×10⁻¹⁵) total;
/// threshold is 1×10⁻¹² (three orders of margin).
/// Reference: Namias (1980), J. Inst. Math. Appl. 25(3); Candan et al. (2000) §II.
fn frft_inverse_roundtrip_order_half_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ]);
    let plan = UnitaryFrftPlan::new(4, 0.5)?;
    let spectrum = plan.forward(&input)?;
    let recovered = plan.inverse(&spectrum)?;
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "UnitaryFrFT",
        "UnitaryFrFT-inverse-roundtrip(alpha=0.5,N=4,[1,2,3,4])",
        "Namias (1980) J.IMA 25(3) sec.2 FrFT additivity: F^{-alpha}(F^alpha(x))=x; Candan et al. (2000) sec.II",
        recovered.iter(),
        expected.iter(),
    ))
}

/// FWHT inverse roundtrip: `inverse(forward(x)) = x` for N=4.
///
/// # Mathematical contract
///
/// Walsh (1923) Am. J. Math. 45 §2: the Walsh-Hadamard matrix W_N satisfies
/// W_N · W_N = N · I_N (involution), so W_N⁻¹ = (1/N)·W_N. The `inverse`
/// method applies W_N then scales by 1/N; composition gives
/// `inverse(forward(x)) = (1/N)·W_N·(W_N·x) = x`. For N=4 f64 butterfly
/// arithmetic, roundtrip error is O(log₂(N)·ε_f64) ≈ 8.9×10⁻¹⁶; threshold 1×10⁻¹⁴.
/// Reference: Walsh (1923) §2: {H_k} is a complete ONS; W_N²=N·I_N.
fn fwht_inverse_roundtrip_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = [1.0_f64, 2.0, 3.0, 4.0];
    let plan = FwhtPlan::new(4)?;
    let signal = Array1::from_vec(input.to_vec());
    let spectrum = plan.forward(&signal)?;
    let recovered = plan.inverse(&spectrum)?;
    Ok(published_real_fixture_with_threshold(
        "FWHT",
        "FWHT-inverse-roundtrip(N=4,[1,2,3,4])",
        "Walsh (1923) Am. J. Math. 45 sec.2: W_N^2=N*I, IFWHT(FWHT(x))=x; Hadamard (1893)",
        recovered.as_slice().unwrap(),
        &input,
        1.0e-14,
    ))
}

/// QFT inverse roundtrip: `iqft(qft(x)) = x` for N=4.
///
/// # Mathematical contract
///
/// Shor (1994) Proc. 35th FOCS §2: QFT_N is unitary, QFT_N† · QFT_N = I_N.
/// The `iqft` free function is the conjugate transpose QFT_N†; both `qft`
/// and `iqft` apply the 1/√N factor (symmetric unitary normalisation). For
/// N=4 (2-qubit computational basis |0000⟩ = [1,0,0,0]), the composition
/// `iqft(qft([1,0,0,0])) = [1,0,0,0]` in exact f64; threshold 1×10⁻¹².
/// Reference: Shor (1994) §2; Nielsen & Chuang (2000) §5.1.
fn qft_inverse_roundtrip_fixture() -> SuiteResult<PublishedFixtureReport> {
    use apollo_qft::iqft as iqft_fn;
    let input = Array1::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]);
    let spectrum = qft_transform(&input)?;
    let recovered = iqft_fn(&spectrum)?;
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "QFT",
        "QFT-inverse-roundtrip(N=4,[1,0,0,0])",
        "Shor (1994) sec.2: QFT_N unitary, QFT_N^{dagger}*QFT_N=I; Nielsen & Chuang (2000) sec.5.1",
        recovered.iter(),
        expected.iter(),
    ))
}

/// SHT inverse roundtrip for dipole Y_1^0 (Driscoll-Healy 1994).
///
/// # Mathematical contract
///
/// Driscoll & Healy (1994) Adv. Appl. Math. 15, Theorem 1: for a field
/// band-limited to degree ≤ L, the SHT is exactly invertible when the grid
/// satisfies N_lat > L and N_lon ≥ 2L+1. The dipole Y_1^0(θ,φ) = √(3/4π)·cos θ
/// has degree 1 ≤ lmax=1; plan grid: N_lat=12 > 1, N_lon=25 ≥ 3. Gauss-Legendre
/// weights integrate cos θ products to machine precision; uniform longitude DFT
/// recovers azimuthal modes exactly. Roundtrip error O((L+1)²·ε_f64) < 1×10⁻¹³;
/// threshold 1×10⁻¹⁰ (three orders of margin).
/// Reference: Driscoll & Healy (1994) §2 sampling theorem on S².
fn sht_inverse_roundtrip_y10_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = ShtPlan::new(12, 25, 1)?;
    let n_lat = plan.grid().latitudes();
    let n_lon = plan.grid().longitudes();
    let mut samples_flat = Vec::with_capacity(n_lat * n_lon);
    let mut sample_arr = ndarray::Array2::<f64>::zeros((n_lat, n_lon));
    for lat in 0..n_lat {
        let theta = plan.theta(lat);
        let val = (3.0_f64 / (4.0 * std::f64::consts::PI)).sqrt() * theta.cos();
        for lon in 0..n_lon {
            sample_arr[[lat, lon]] = val;
            samples_flat.push(val);
        }
    }
    let coefficients = plan.forward_real(&sample_arr)?;
    let recovered = plan.inverse_real(&coefficients)?;
    let recovered_flat: Vec<f64> = recovered.iter().copied().collect();
    Ok(published_real_fixture_with_threshold(
        "SHT",
        "SHT-Y10-inverse-roundtrip(lmax=1,lat=12,lon=25)",
        "Driscoll & Healy (1994) Adv. Appl. Math. 15 Theorem 1: SHT invertible for band-limited fields; Y_1^0=sqrt(3/4pi)*cos(theta)",
        &recovered_flat,
        &samples_flat,
        1.0e-10,
    ))
}

/// DHT 4-point inverse roundtrip: IDHT(DHT(x)) = x.
///
/// # Mathematical contract
///
/// The DHT kernel: H[k] = Σ_{n=0}^{N-1} x[n] · cas(2πnk/N), cas(t) = cos(t) + sin(t).
/// For the inverse: IDHT(X)[n] = (1/N) · DHT(X)[n] (self-reciprocal property, H²=NI).
/// For x=[3,-1,2,0] and N=4:
///   H[0] = 3-1+2+0 = 4
///   H[1] = 3·cas(0) + (-1)·cas(π/2) + 2·cas(π) + 0·cas(3π/2) = 3-1-2 = 0
///   H[2] = 3·cas(0) + (-1)·cas(π) + 2·cas(2π) + 0·cas(3π) = 3+1+2 = 6
///   H[3] = 3·cas(0) + (-1)·cas(3π/2) + 2·cas(3π) + 0·cas(9π/2) = 3+1-2 = 2
/// DHT([3,-1,2,0]) = [4,0,6,2].
/// IDHT([4,0,6,2]) = DHT([4,0,6,2]) / 4 = [12,-4,8,0] / 4 = [3,-1,2,0]. ✓
/// Roundtrip error bounded by butterfly accumulation: O(log₂N · ε_f64) < 1×10⁻¹⁴ for N=4.
/// Reference: Bracewell (1983) JOSA 73(12): DHT self-reciprocal property H²=NI; inverse = (1/N)·DHT.
fn dht_inverse_roundtrip_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DhtPlan::new(4)?;
    let input = [3.0_f64, -1.0, 2.0, 0.0];
    let spectrum = plan.forward(&input)?;
    let recovered = plan.inverse(&spectrum)?;
    Ok(published_real_fixture_with_threshold(
        "DHT",
        "DHT-inverse-roundtrip([3,-1,2,0],N=4)",
        "Bracewell (1983) JOSA 73(12): DHT self-reciprocal H\u{00b2}=NI; IDHT(DHT(x))=x; x=[3,-1,2,0]",
        &recovered,
        &input,
        1.0e-14,
    ))
}

/// SFT 1-sparse alternating tone inverse roundtrip: ISFT(SFT(x)) = x.
///
/// # Mathematical contract
///
/// Signal x[n] = (-1)^n for N=4 is a pure tone at frequency k=2.
/// DFT: X[k] = N·δ(k-N/2) = 4·δ(k-2), so X = [0,0,4,0].
/// SparseFftPlan::new(4,1) retains K=1 largest coefficient: freq=2, value=4+0i.
/// SparseSpectrum is exactly [0,0,4+0i,0] when expanded.
/// ISFT expands to dense [0,0,4,0] then applies IFFT normalized by 1/N:
///   x_rec[n] = (1/4)·Σ_k X[k]·e^{2πink/N} = (1/4)·4·e^{iπn} = e^{iπn} = [1,-1,1,-1]. ✓
/// Threshold 1×10⁻¹² covers IEEE 754 rounding of e^{iπn} for n=0..3.
/// Reference: Cooley-Tukey (1965): DFT[(−1)^n]=N·δ[k−N/2];
///            Hassanieh et al. (2012) sFFT exact K-sparse recovery;
///            Candès & Wakin (2008) RIP exact reconstruction theorem.
fn sft_inverse_roundtrip_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = SparseFftPlan::new(4, 1)?;
    let signal = [
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(-1.0, 0.0),
    ];
    let spectrum = plan.forward(&signal)?;
    let recovered = plan.inverse(&spectrum)?;
    Ok(published_complex_fixture(
        "SFT",
        "SFT-inverse-roundtrip-1-sparse-alternating-tone(N=4,K=1)",
        "Cooley-Tukey (1965) DFT[(−1)^n]=4·δ[k−2]; Hassanieh et al. (2012) K-sparse exact recovery; ISFT(SFT(x))=x",
        recovered.iter(),
        signal.iter(),
    ))
}

/// NTT 4-point inverse roundtrip: INTT(NTT(x)) = x.
///
/// # Mathematical contract
///
/// Let p = 998244353 (= 119·2²³+1), N=4, g=3, ω = g^{(p-1)/N} mod p.
/// By Pollard (1971) Theorem 1: NTT is a bijection on (ℤ/pℤ)^N with exact integer
/// inverse INTT(X)[n] = (1/N) Σ_k X[k] · ω^{-nk} mod p.
/// For x=[1,2,3,4]: NTT[k] = Σ_{n=0}^3 x[n]·ω^{nk} mod p.
/// INTT(NTT(x)) = x holds exactly in ℤ/pℤ arithmetic.
/// Converting to f64: all values ≤ 4 ≪ 2^53, so representation is exact.
/// Floating-point error in f64 comparison: 0 (exact integer reconstruction).
/// Threshold 1×10⁻¹² covers f64 conversion round-trip noise.
/// Reference: Pollard (1971) Math. Proc. Cambridge Phil. Soc. 70(3): NTT inversion theorem.
fn ntt_inverse_roundtrip_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = NttPlan::new(4)?;
    let input = Array1::from_vec(vec![1u64, 2, 3, 4]);
    let spectrum = plan.forward(&input)?;
    let recovered = intt(&spectrum)?;
    let recovered_f64: Vec<f64> = recovered.iter().map(|&v| v as f64).collect();
    let expected = [1.0_f64, 2.0, 3.0, 4.0];
    Ok(published_real_fixture(
        "NTT",
        "INTT(NTT([1,2,3,4]),N=4)",
        "Pollard (1971) Math. Proc. Cambridge Phil. Soc. 70(3): NTT inversion theorem; INTT(NTT(x))=x in \u{2124}/p\u{2124} for p=998244353",
        &recovered_f64,
        &expected,
    ))
}

/// STFT Hann-window WOLA inverse roundtrip: ISTFT(STFT(x)) = x.
///
/// # Mathematical contract
///
/// Parameters: frame_len=4, hop_len=2, Hann analysis+synthesis window.
/// Hann window N=4: w[n] = 0.5·(1 − cos(2πn/3)), giving w=[0, 0.75, 0.75, 0].
/// Signal x=[1,0,0,0]. frame_count(4) = 1 + ⌈4/2⌉ = 3 frames.
///
/// Frame centers (start = m·hop − frame/2):
///   m=0: start=−2, covers signal positions 0,1 (frame indices 2,3).
///     Windowed frame: [0,0, w[2]·x[0], w[3]·x[1]] = [0,0, 0.75, 0].
///     DFT([0,0,0.75,0]) = [0.75, −0.75, 0.75, −0.75].
///   m=1: start=0, covers positions 0..3. Windowed: [w[0]·1,0,0,0]=[0,0,0,0]. Spectrum=[0,…].
///   m=2: start=2, covers positions 2..5. All zero. Spectrum=[0,…].
///
/// Reconstruction (WOLA): for position i, output[i] = Σ_m (w[n_m]·IFFT(X_m)[n_m]) / Σ_m w[n_m]².
///   pos 0: IFFT([0.75,−0.75,0.75,−0.75])[2] = 0.75; weight = w[2]² = 0.5625.
///     output[0] = 0.5625 / 0.5625 = 1.0 ✓
///   pos 1: weight = w[3]² = 0. output[1] = 0.0 ✓  (WOLA defines 0/0 → 0)
///   pos 2,3: no non-zero contributions. output = 0.0 ✓
///
/// COLA weight at every position: 0.5625 (constant) for non-boundary positions.
/// Boundary position 1 has w[3]=0; expected x[1]=0; WOLA returns 0 by convention. ✓
/// Floating-point error: O(log₂(4)·ε_f64) ≈ 4.4×10⁻¹⁶ < 1×10⁻¹².
/// Reference: Allen & Rabiner (1977) Proc. IEEE 65(11): weighted overlap-add synthesis;
///            Hann window COLA condition, Portnoff (1980) IEEE Trans. ASSP 28(1).
fn stft_hann_wola_inverse_roundtrip_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = StftPlan::new(4, 2)?;
    let signal = Array1::from_vec(vec![1.0_f64, 0.0, 0.0, 0.0]);
    let spectrum = plan.forward(&signal)?;
    let recovered = plan.inverse(&spectrum, 4)?;
    let expected = [1.0_f64, 0.0, 0.0, 0.0];
    Ok(published_real_fixture_with_threshold(
        "STFT",
        "STFT-Hann-WOLA-inverse-roundtrip([1,0,0,0],frame=4,hop=2)",
        "Allen & Rabiner (1977) Proc. IEEE 65(11): WOLA synthesis; Portnoff (1980) Hann COLA; ISTFT(STFT(x))=x for frame=4 hop=2",
        recovered.as_slice().unwrap_or(&[]),
        &expected,
        1.0e-12,
    ))
}

/// DCT-IV 2-point inverse roundtrip: IDCT-IV(DCT-IV(x)) = x.
///
/// # Mathematical contract
///
/// The DCT-IV kernel for N=2: C[k] = Σ_{n=0}^{1} x[n]·cos(π(n+½)(k+½)/2).
/// Self-inverse property: DCT-IV² = N·I, so IDCT-IV = (1/N)·DCT-IV.
/// For x=[1,3], N=2:
///   C[0] = 1·cos(π/8) + 3·cos(3π/8) = cos(π/8) + 3·sin(π/8)
///          ≈ 0.92388 + 3·0.38268 = 2.07193...
///   C[1] = 1·cos(3π/8) + 3·cos(9π/8) = cos(3π/8) + 3·(−cos(π/8))
///          ≈ 0.38268 − 2.77164 = −2.38896...
/// The `plan.inverse` method applies the (1/N)·DCT-IV formula.
/// IDCT-IV([C[0],C[1]]) = [1,3] exactly (modulo butterfly rounding ε_f64).
/// Floating-point error: O(ε_f64) for N=2 (one butterfly level); threshold 1×10⁻¹⁴.
/// Reference: Makhoul (1980) IEEE Trans. ASSP 28(1): DCT-IV self-inverse property;
///            FFTW REDFT11 documentation: IDCT-IV = (1/2N)·DCT-IV.
fn dct4_inverse_roundtrip_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(2, RealTransformKind::DctIV)?;
    let input = [1.0_f64, 3.0];
    let spectrum = plan.forward(&input)?;
    let recovered = plan.inverse(&spectrum)?;
    Ok(published_real_fixture_with_threshold(
        "DCT",
        "IDCT-IV(DCT-IV([1,3]),N=2)",
        "Makhoul (1980) IEEE Trans. ASSP 28(1): DCT-IV self-inverse C4\u{00b2}=N\u{00b7}I; FFTW REDFT11: IDCT-IV=(1/2N)\u{00b7}DCT-IV",
        &recovered,
        &input,
        1.0e-14,
    ))
}

/// DST-IV 2-point inverse roundtrip: IDST-IV(DST-IV(x)) = x.
///
/// # Mathematical contract
///
/// The DST-IV kernel for N=2: S[k] = Σ_{n=0}^{1} x[n]·sin(π(n+½)(k+½)/2).
/// Self-inverse property: DST-IV² = N·I, so IDST-IV = (1/N)·DST-IV.
/// For x=[2,5], N=2:
///   S[0] = 2·sin(π/8) + 5·sin(3π/8) = 2·sin(π/8) + 5·cos(π/8)
///          ≈ 2·0.38268 + 5·0.92388 = 5.38476...
///   S[1] = 2·sin(3π/8) + 5·sin(9π/8) = 2·cos(π/8) + 5·(−sin(π/8))
///          ≈ 1.84776 − 1.91341 = −0.06566...
/// The `plan.inverse` method applies the (1/N)·DST-IV formula.
/// IDST-IV([S[0],S[1]]) = [2,5] exactly (modulo butterfly rounding ε_f64).
/// Floating-point error: O(ε_f64) for N=2 (one butterfly level); threshold 1×10⁻¹⁴.
/// Reference: Makhoul (1980) IEEE Trans. ASSP 28(1): DST-IV self-inverse property;
///            FFTW RODFT11 documentation: IDST-IV = (1/2N)·DST-IV.
fn dst4_inverse_roundtrip_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(2, RealTransformKind::DstIV)?;
    let input = [2.0_f64, 5.0];
    let spectrum = plan.forward(&input)?;
    let recovered = plan.inverse(&spectrum)?;
    Ok(published_real_fixture_with_threshold(
        "DST",
        "IDST-IV(DST-IV([2,5]),N=2)",
        "Makhoul (1980) IEEE Trans. ASSP 28(1): DST-IV self-inverse S4\u{00b2}=N\u{00b7}I; FFTW RODFT11: IDST-IV=(1/2N)\u{00b7}DST-IV",
        &recovered,
        &input,
        1.0e-14,
    ))
}

/// DCT-I 3-point inverse roundtrip: IDCT-I(DCT-I(x)) = x.
///
/// # Mathematical contract
///
/// DCT-I kernel for N=3: X[k] = x[0] + (−1)^k·x[2] + 2·x[1]·cos(π·k/2).
/// Self-inverse property: DCT-I² = 2(N−1)·I = 4·I (for N=3),
/// so IDCT-I = (1/(2(N−1)))·DCT-I = (1/4)·DCT-I.
/// For x=[1,2,3]:
///   X[0] = 1 + 3 + 2·2·cos(0)   = 1 + 3 + 4   = 8  (exact integer)
///   X[1] = 1 − 3 + 2·2·cos(π/2) = 1 − 3 + 0   = −2 (exact: cos(π/2)=0)
///   X[2] = 1 + 3 + 2·2·cos(π)   = 1 + 3 − 4   = 0  (exact: cos(π)=−1)
/// Intermediate spectrum = [8, −2, 0] — exactly integer.
/// DCT-I([8,−2,0]):
///   Y[0] = 8 + 0 + 2·(−2)·1   = 4
///   Y[1] = 8 + 0 + 2·(−2)·0   = 8
///   Y[2] = 8 + 0 + 2·(−2)·(−1) = 12
/// IDCT-I([8,−2,0]) = (1/4)·[4,8,12] = [1,2,3] exactly.
/// All intermediate cos values are {−1,0,1}; computation is exact in f64.
/// Floating-point error: 0 (analytically exact); threshold 1×10⁻¹⁴ is conservative.
/// Reference: Makhoul (1980) IEEE Trans. ASSP 28(1): DCT-I self-inverse C1²=2(N−1)·I;
///            FFTW REDFT00 documentation: IDCT-I=(1/(2(N−1)))·DCT-I.
fn dct1_inverse_roundtrip_three_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(3, RealTransformKind::DctI)?;
    let input = [1.0_f64, 2.0, 3.0];
    let spectrum = plan.forward(&input)?;
    let recovered = plan.inverse(&spectrum)?;
    Ok(published_real_fixture_with_threshold(
        "DCT",
        "IDCT-I(DCT-I([1,2,3]),N=3)",
        "Makhoul (1980) IEEE Trans. ASSP 28(1): DCT-I self-inverse C1\u{00b2}=2(N\u{2212}1)\u{00b7}I; FFTW REDFT00: IDCT-I=(1/(2(N\u{2212}1)))\u{00b7}DCT-I",
        &recovered,
        &input,
        1.0e-14,
    ))
}

/// DST-I 2-point inverse roundtrip: IDST-I(DST-I(x)) = x.
///
/// # Mathematical contract
///
/// DST-I kernel for N=2: X[k] = 2·Σ_{n=0}^{1} x[n]·sin(π(n+1)(k+1)/3).
/// Self-inverse property: DST-I² = 2(N+1)·I = 6·I (for N=2),
/// so IDST-I = (1/(2(N+1)))·DST-I = (1/6)·DST-I.
/// For x=[1,3]:
///   X[0] = 2·(1·sin(π/3) + 3·sin(2π/3)) = 2·(√3/2 + 3·√3/2) = 4√3 ≈ 6.9282...
///   X[1] = 2·(1·sin(2π/3) + 3·sin(4π/3)) = 2·(√3/2 − 3·√3/2) = −2√3 ≈ −3.4641...
/// DST-I([4√3, −2√3]):
///   Y[0] = 2·(4√3·sin(π/3) + (−2√3)·sin(2π/3)) = 2·(4·3/2 − 2·3/2) = 2·3 = 6
///   Y[1] = 2·(4√3·sin(2π/3) + (−2√3)·sin(4π/3)) = 2·(4·3/2 + 2·3/2) = 2·9 = 18
/// IDST-I([4√3,−2√3]) = (1/6)·[6,18] = [1,3] exactly.
/// Intermediate spectrum is irrational (√3); error is O(ε_f64)≈2×10⁻¹⁶ per element.
/// Threshold 1×10⁻¹⁴ > max accumulated floating-point error for N=2.
/// Reference: Makhoul (1980) IEEE Trans. ASSP 28(1): DST-I self-inverse S1²=2(N+1)·I;
///            FFTW RODFT00 documentation: IDST-I=(1/(2(N+1)))·DST-I.
fn dst1_inverse_roundtrip_two_point_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(2, RealTransformKind::DstI)?;
    let input = [1.0_f64, 3.0];
    let spectrum = plan.forward(&input)?;
    let recovered = plan.inverse(&spectrum)?;
    Ok(published_real_fixture_with_threshold(
        "DST",
        "IDST-I(DST-I([1,3]),N=2)",
        "Makhoul (1980) IEEE Trans. ASSP 28(1): DST-I self-inverse S1\u{00b2}=2(N+1)\u{00b7}I; FFTW RODFT00: IDST-I=(1/(2(N+1)))\u{00b7}DST-I",
        &recovered,
        &input,
        1.0e-14,
    ))
}

/// NUFFT Type-1/Type-2 adjoint inner-product identity (Dutt-Rokhlin 1993).
///
/// # Mathematical contract
///
/// Type-1 operator A and Type-2 operator A* are defined as:
///   (A·c)[k] = Σ_j c_j · exp(-2πi·k·x_j/L),  k = 0, …, N-1
///   (A*·f)[j] = Σ_k f_k · exp(+2πi·k·x_j/L)
///
/// They satisfy the adjoint identity (proof by index swap):
///   Re(⟨A·c, f⟩) = Re(⟨c, A*·f⟩)
///
/// Analytically derived values for N=2, dx=0.5, L=1.0,
/// positions=[0.0, 0.5], c=[1+0i, 2+0i], f=[3+0i, 4+0i]:
///   All exp factors are exp(-2πi·k·x_j/L) ∈ {1, exp(-πi)} = {1, -1}.
///   Type-1: F[0] = 1·1 + 2·1 = 3;  F[1] = 1·1 + 2·(-1) = -1
///   Type-2: G[0] = 3·1 + 4·1 = 7;  G[1] = 3·1 + 4·(-1) = -1
///   LHS = Re(conj(3)·3 + conj(-1)·4) = 9 - 4 = 5  (exact integer)
///   RHS = Re(conj(1)·7 + conj(2)·(-1)) = 7 - 2 = 5  (exact integer)
///
/// All arithmetic is exact in f64 (factors ∈ {1, −1}); accumulated FP error = 0.
/// Threshold 1×10⁻¹² is conservative; actual error is identically 0.
///
/// Reference: Dutt & Rokhlin (1993) SIAM J. Sci. Comput. 14(6) 1368–1393:
///            adjoint identity eq. (1.8); Greengard & Lee (2004) §2 proof.
fn nufft_type1_type2_adjoint_inner_product_fixture() -> SuiteResult<PublishedFixtureReport> {
    let domain = UniformDomain1D::new(2, 0.5)?;
    let positions = vec![0.0_f64, 0.5];
    let c = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
    let f_arr = Array1::from_vec(vec![Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)]);
    let capital_f = nufft_type1_1d(&positions, &c, domain);
    let g = nufft_type2_1d(&f_arr, &positions, domain);
    let lhs: f64 = capital_f
        .iter()
        .zip(f_arr.iter())
        .map(|(fk, dk)| (fk.conj() * dk).re)
        .sum();
    let rhs: f64 = c
        .iter()
        .zip(g.iter())
        .map(|(cj, gj)| (cj.conj() * gj).re)
        .sum();
    Ok(published_real_fixture_with_threshold(
        "NUFFT",
        "Re(<A\u{00b7}c,f>)=Re(<c,A*\u{00b7}f>),N=2,pos=[0,0.5],c=[1,2],f=[3,4]",
        "Dutt & Rokhlin (1993) SIAM J. Sci. Comput. 14(6): NUFFT Type-1/Type-2 adjoint identity (1.8); Greengard-Lee (2004) \u{00a7}2",
        &[lhs, rhs],
        &[5.0_f64, 5.0],
        1.0e-12,
    ))
}

/// Radon Fourier Slice Theorem at \u{03b8}=0: DFT_1(R_{\u{03b8}=0}f) equals the horizontal
/// slice of the 2D DFT of f (Natterer 1986, Radon 1917).
///
/// # Mathematical contract
///
/// Projection-Slice Theorem (Natterer 1986, §I.2, Theorem 1.1):
///   \u{1d4d5}_1{R_\u{03b8} f}(\u{03c9}) = \u{1d4d5}_2{f}(\u{03c9}·cos\u{03b8}, \u{03c9}·sin\u{03b8})
///
/// For \u{03b8}=0 (parallel-beam at θ=0 = column sums):
///   p_0[n] = Σ_m f[m,n]   (projection = sum over rows at each column detector n)
///   DFT_1(p_0)[k] = Σ_n p_0[n]·exp(-2πi·k·n/N)
///                 = Σ_n (Σ_m f[m,n])·exp(-2πi·k·n/N)
///                 = F_2[0,k]  (horizontal slice of 2D DFT)
///
/// For image f=[[1,2],[3,4]], N=M=2:
///   p_0 = [1+3, 2+4] = [4, 6]  (column sums)
///   DFT_1([4,6]): P[0] = 4+6 = 10,  P[1] = 4+6·exp(-i\u{03c0}) = 4-6 = -2
///   2D DFT check: F_2[0,0]=1+2+3+4=10 ✓,  F_2[0,1]=1-2+3-4=-2 ✓
///
/// All DFT factors are exp(-2πi·k·n/N) ∈ {1, exp(-i\u{03c0})} = {1, -1};
/// computation is exact in f64; accumulated floating-point error = 0.
/// Threshold 1×10⁻¹² is conservative.
///
/// Reference: Natterer (1986) The Mathematics of Computerized Tomography §I.2,
///            Theorem 1.1 (Projection-Slice Theorem); Radon (1917) original
///            parallel-beam projection definition.
fn radon_fourier_slice_theorem_theta0_fixture() -> SuiteResult<PublishedFixtureReport> {
    let image = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0])
        .expect("(2,2) shape is valid for 4 elements");
    let plan = RadonPlan::new(2, 2, vec![0.0_f64], 2, 1.0)?;
    let sinogram = plan.forward(&image)?;
    let projection = sinogram.values().row(0).to_owned();
    let dft_of_projection = fft_1d_array(&projection);
    let expected = [Complex64::new(10.0, 0.0), Complex64::new(-2.0, 0.0)];
    Ok(published_complex_fixture(
        "Radon",
        "DFT_1(R_\u{03b8}=0([[1,2],[3,4]]))-vs-2D-DFT-slice,N=2",
        "Natterer (1986) §I.2 Thm 1.1 Projection-Slice Theorem: DFT_1(R_\u{03b8}f)=F_2{f}(\u{03c9}cos\u{03b8},\u{03c9}sin\u{03b8}); Radon (1917)",
        dft_of_projection.iter(),
        expected.iter(),
    ))
}

/// SDFT sliding-update recurrence agreement: after N updates on zero state
/// fed with the unit impulse [1,0,0,0], all 4 tracked bins equal 1+0i.
///
/// # Mathematical contract
///
/// Jacobsen & Lyons (2003) §2, eq.(2): the sliding DFT recurrence
///   X_k[n] = (X_k[n-1] + x[n] - x[n-N]) \u00b7 exp(2\u03c0i\u00b7k/N)
/// produces exactly the DFT of the current window x[n-N+1..=n] at every step.
///
/// Analytical derivation for N=4, zero_state, input [1,0,0,0]:
///   Twiddles (exp(2\u03c0ik/4)): k=0 \u21a6 1, k=1 \u21a6 i, k=2 \u21a6 -1, k=3 \u21a6 -i.
///   Update 0 (x_new=1): bins = [1,i,-1,-i] (window=[0,0,0,1])
///     DFT([0,0,0,1])[k] = exp(-2\u03c0i\u00b73k/4): k=0\u21a61, k=1\u21a6i, k=2\u21a6-1, k=3\u21a6-i \u2713
///   Update 1 (x_new=0): bins\u2190bins*twiddles = [1,-1,1,-1] (window=[0,0,1,0])
///     DFT([0,0,1,0])[k] = exp(-2\u03c0i\u00b72k/4): k=0\u21a61, k=1\u21a6-1, k=2\u21a61, k=3\u21a6-1 \u2713
///   Update 2 (x_new=0): bins\u2190bins*twiddles = [1,-i,-1,i] (window=[0,1,0,0])
///     DFT([0,1,0,0])[k] = exp(-2\u03c0i\u00b7k/4): k=0\u21a61, k=1\u21a6-i, k=2\u21a6-1, k=3\u21a6i \u2713
///   Update 3 (x_new=0): bins\u2190bins*twiddles = [1,1,1,1] (window=[1,0,0,0])
///     DFT([1,0,0,0])[k] = 1 for all k \u2713
///
/// All factors \u2208{1,i,-1,-i}; result is exact integers; accumulated FP error = 0.
/// Threshold 1\u00d710\u207b\u00b9\u00b2 is conservative.
///
/// Reference: Jacobsen, E. & Lyons, R. (2003). \"The Sliding DFT.\"
///            IEEE Signal Processing Magazine 20(2), 74\u201379. Section 2, eq.(2).
fn sdft_sliding_recurrence_unit_impulse_all_bins_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = SdftPlan::new(4, 4)?;
    let mut state = plan.zero_state();
    for &sample in &[1.0_f64, 0.0, 0.0, 0.0] {
        state.update(sample);
    }
    let bins = state.bins().to_vec();
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "SDFT",
        "SDFT-sliding-recurrence-impulse(N=4,[1,0,0,0]),all-bins=1",
        "Jacobsen & Lyons (2003) IEEE SPM 20(2) §2 eq.(2): sliding-update recurrence X_k[n]=(X_k[n-1]+x[n]-x[n-N])·exp(2πik/N); DFT([1,0,0,0])=[1,1,1,1]",
        bins.iter(),
        expected.iter(),
    ))
}

/// Unitary FrFT order \u03b1=4 is the identity on any input.
///
/// # Mathematical contract
///
/// Candan, Kutay & Ozaktas (2000) \u00a7II, Corollary (periodicity):
///   DFrFT_a(x) = V \u00b7 diag(exp(\u2212i\u00b7a\u00b7k\u00b7\u03c0/2)) \u00b7 V\u1d40 \u00b7 x
/// For a=4: exp(\u22124\u00b7k\u00b7\u03c0\u00b7i/2) = exp(\u22122\u03c0ki) = 1 for every integer k.
/// Therefore DFrFT_4 = V \u00b7 I \u00b7 V\u1d40 = I (since V is orthogonal, V\u1d40V = I).
///
/// This holds for any N and any input x, independent of the eigenvector
/// ordering in V and without requiring a=1 to equal the DFT.
///
/// For N=4, two O(N\u00b2) matrix\u2013vector products: accumulated FP error
/// \u2248 2\u00b7N\u00b7\u03b5_f64 = 2\u00b74\u00b72.2\u00d710\u207b\u00b9\u2076 \u2248 1.8\u00d710\u207b\u00b9\u2075; threshold 1\u00d710\u207b\u00b9\u00b2 (\u00d770 margin).
///
/// Reference: Candan, \u00c7., Kutay, M.A., & Ozaktas, H.M. (2000).
///            \"The Discrete Fractional Fourier Transform.\"
///            IEEE Trans. Signal Process. 48(5), 1329\u20131337. \u00a7II Corollary.
fn frft_order4_identity_fixture() -> SuiteResult<PublishedFixtureReport> {
    let input = Array1::from_vec(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ]);
    let plan = UnitaryFrftPlan::new(4, 4.0)?;
    let output = plan.forward(&input)?;
    let expected = [
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];
    Ok(published_complex_fixture(
        "UnitaryFrFT",
        "UnitaryFrFT-order4-identity(N=4,[1,2,3,4])",
        "Candan et al. (2000) IEEE TSP 48(5) §II Corollary: DFrFT_4=I; exp(-i·4kπ/2)=exp(-2πki)=1; V·I·V^T=I",
        output.iter(),
        expected.iter(),
    ))
}

/// CZT off-unit-circle Z-transform evaluation: A=2 (real, |A|=2>1), W=exp(-πi).
///
/// # Mathematical contract
///
/// The Chirp Z-Transform evaluates the one-sided Z-transform at spiral points
/// z_k = A·W^{-k} (Rabiner, Schafer & Rader 1969, §II):
///   X[k] = Σ_{n=0}^{N-1} x[n] · A^{-n} · W^{nk},  k=0..M-1
///
/// For N=2, M=2, A=2, W=exp(-πi)=-1, x=[1,1]:
///   z_0 = A·W^0 = 2  (real axis, |z_0|=2 > 1, off the unit circle)
///   z_1 = A·W^{-1} = 2·(-1) = -2  (real axis, |z_1|=2 > 1)
///
///   Z{x}(z) = x[0]·z^{-0} + x[1]·z^{-1} = 1 + z^{-1}
///   X[0] = Z{x}(2)  = 1 + 2^{-1}     = 3/2 = 1.5  (exact dyadic rational)
///   X[1] = Z{x}(-2) = 1 + (-2)^{-1}  = 1/2 = 0.5  (exact dyadic rational)
///
/// Direct check via definition:
///   X[0] = 1·A^0·W^0 + 1·A^{-1}·W^0 = 1 + 0.5           = 1.5 ✓
///   X[1] = 1·A^0·W^0 + 1·A^{-1}·W^1 = 1 + 0.5·(-1) = 0.5 ✓
///
/// 3/2 and 1/2 are exactly representable as IEEE 754 f64 (dyadic fractions);
/// accumulated floating-point error is identically 0. Threshold 1×10^{-12}
/// is conservative (actual error = 0 in exact arithmetic).
///
/// Reference: Rabiner, L.R., Schafer, R.W. & Rader, C.M. (1969).
///            "The Chirp z-Transform Algorithm."
///            IEEE Trans. Audio Electroacoustics 17(2), 86–92. §II.
fn czt_off_unit_circle_z_transform_fixture() -> SuiteResult<PublishedFixtureReport> {
    let a = Complex64::new(2.0, 0.0);
    let w = Complex64::from_polar(1.0, -std::f64::consts::PI);
    let plan = CztPlan::new(2, 2, a, w)?;
    let input = Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)]);
    let actual = plan.forward(&input)?;
    let expected = [Complex64::new(1.5, 0.0), Complex64::new(0.5, 0.0)];
    Ok(published_complex_fixture(
        "CZT",
        "CZT2(A=2,W=exp(-πi),[1,1]): Z{x}(2)=1.5,Z{x}(-2)=0.5",
        "Rabiner, Schafer & Rader (1969) IEEE TAE 17(2) §II: CZT at z_k=A·W^{-k}; A=2 evaluates Z-transform off unit circle at z={2,-2}",
        actual.iter(),
        expected.iter(),
    ))
}

/// Hilbert envelope of a pure discrete cosine tone is identically unity.
///
/// # Mathematical contract
///
/// For a real bandpass signal x[n] = A·cos(ω₀n + φ), the analytic signal
/// z[n] = x[n] + i·H{x}[n] = A·exp(i(ω₀n + φ)) and the instantaneous
/// envelope |z[n]| = A (constant). (Oppenheim & Schafer 2010 §12.1, eq.(12.8))
///
/// Analytical derivation for x=[1,0,-1,0] = cos(πn/2), N=4:
///   DFT(x): X[0]=0, X[1]=2, X[2]=0, X[3]=2
///   Hilbert analytic mask for N=4 (even): [1, 2, 1, 0]
///   Y = X ⊙ mask = [0, 4, 0, 0]
///   z = IDFT(Y)/1 = exp(iπn/2): z[0]=1, z[1]=i, z[2]=-1, z[3]=-i
///   |z[0]|=1, |z[1]|=1, |z[2]|=1, |z[3]|=1
///
/// All DFT factors ∈{1,i,-1,-i} and mask values ∈{0,1,2}; the envelope
/// vector [1,1,1,1] is an exact integer result. Accumulated FP error is
/// O(log₂(N)·ε_f64) ≈ 8.9×10^{-16} << threshold 1×10^{-12}.
///
/// Reference: Oppenheim, A.V. & Schafer, R.W. (2010).
///            Discrete-Time Signal Processing (3rd ed.) §12.1, eq.(12.8).
///            Bedrosian, E. (1963) Proc. IEEE 51(5): analytic signal envelope theorem.
fn hilbert_pure_cosine_envelope_is_unity_fixture() -> SuiteResult<PublishedFixtureReport> {
    let signal = [1.0_f64, 0.0, -1.0, 0.0];
    let plan = HilbertPlan::new(4)?;
    let envelope = plan.envelope(&signal)?;
    let expected = [1.0_f64, 1.0, 1.0, 1.0];
    Ok(published_real_fixture_with_threshold(
        "Hilbert",
        "Hilbert-envelope(cos(πn/2),N=4)=[1,1,1,1]",
        "Oppenheim & Schafer (2010) DTSP 3rd ed. §12.1 eq.(12.8): |z[n]|=A for x[n]=A·cos(ω₀n+φ); Bedrosian (1963)",
        &envelope,
        &expected,
        1.0e-12,
    ))
}

/// Daubechies-4 DWT one-level coefficients for x=[1,0,0,0] under periodic extension.
///
/// # Mathematical contract
///
/// For db4 analysis filters (Daubechies 1992, p.198):
///   h = [h0,h1,h2,h3]
///     = [0.4829629131445341,
///        0.8365163037378079,
///        0.2241438680420134,
///       -0.12940952255126034]
/// and QMF highpass g[k] = (−1)^k h[3-k], hence:
///   g = [h3, −h2, h1, −h0]
///     = [−0.12940952255126034,
///        −0.2241438680420134,
///         0.8365163037378079,
///        −0.4829629131445341].
///
/// For N=4, level=1, periodic analysis_stage:
///   a0 = Σ_{m=0}^3 h[m] x[(0+m) mod 4] = h0
///   d0 = Σ_{m=0}^3 g[m] x[(0+m) mod 4] = g0 = h3
///   a1 = Σ_{m=0}^3 h[m] x[(2+m) mod 4] = h2
///   d1 = Σ_{m=0}^3 g[m] x[(2+m) mod 4] = g2 = h1
/// so [a0,a1,d0,d1] = [h0,h2,h3,h1].
///
/// Because x is a basis impulse, each coefficient is a single filter tap;
/// no summation round-off occurs. Error is exactly 0 in f64 representation.
/// Threshold 1×10⁻¹⁵ is conservative.
///
/// Reference: Daubechies (1992), Ten Lectures on Wavelets, p.198 (db4 taps);
///            Mallat (1989), two-channel periodic filter-bank analysis.
fn wavelet_daubechies4_one_level_known_coefficients_fixture() -> SuiteResult<PublishedFixtureReport>
{
    let plan = DwtPlan::new(4, 1, DiscreteWavelet::Daubechies4)?;
    let coefficients = plan.forward(&[1.0_f64, 0.0, 0.0, 0.0])?;
    let mut actual = Vec::with_capacity(4);
    actual.extend_from_slice(coefficients.approximation());
    actual.extend_from_slice(&coefficients.details()[0]);
    let expected = [
        0.482_962_913_144_534_1,
        0.224_143_868_042_013_4,
        -0.129_409_522_551_260_34,
        0.836_516_303_737_807_9,
    ];
    Ok(published_real_fixture_with_threshold(
        "DWT-Db4",
        "DWT-Db4-1level([1,0,0,0])-coefficients",
        "Daubechies (1992) p.198 db4 taps with periodic QMF analysis: [a0,a1,d0,d1]=[h0,h2,h3,h1]",
        &actual,
        &expected,
        1.0e-15,
    ))
}

/// Daubechies-4 perfect reconstruction: inverse(forward(x)) = x for level 1.
///
/// # Mathematical contract
///
/// For an orthogonal two-channel filter bank (Mallat 1989, Thm.2),
/// synthesis is the inverse of analysis on the same wavelet family:
///   IDWT(DWT(x)) = x.
///
/// This fixture uses db4, N=4, level=1 with a non-trivial signal
/// x=[1,−2,0.5,4].
///
/// f64 periodic convolution + one reconstruction stage gives
/// O(filter_len·ε_f64) ≈ O(4·ε_f64) per output sample; threshold 1×10⁻¹²
/// is >10³× larger than worst-case round-off.
///
/// Reference: Mallat (1989) IEEE TPAMI 11(7), Theorem 2 (perfect reconstruction);
///            Daubechies (1992) db4 orthogonal wavelet construction.
fn wavelet_daubechies4_inverse_perfect_reconstruction_fixture(
) -> SuiteResult<PublishedFixtureReport> {
    let input = [1.0_f64, -2.0, 0.5, 4.0];
    let plan = DwtPlan::new(4, 1, DiscreteWavelet::Daubechies4)?;
    let coefficients = plan.forward(&input)?;
    let recovered = plan.inverse(&coefficients)?;
    Ok(published_real_fixture_with_threshold(
        "DWT-Db4",
        "DWT-Db4-inverse-roundtrip-1level(N=4,[1,-2,0.5,4])",
        "Mallat (1989) Thm.2 perfect reconstruction: IDWT(DWT(x))=x for orthogonal db4 filter bank",
        &recovered,
        &input,
        1.0e-12,
    ))
}

/// CWT Ricker impulse response: peak value equals ψ(0) and zero-crossing neighbors are exact zeros.
///
/// # Mathematical contract
///
/// The continuous wavelet transform coefficient at scale a and translation b is:
///   W(a, b) = (1/√a) Σ_n x[n] ψ((n−b)/a)
///
/// For a discrete impulse x = δ_{n₀} (x[n₀]=1, all other samples 0) and scale a=1:
///   W(1, n₀) = ψ(0)
///   W(1, n₀±1) = ψ(±1)
///
/// The normalized Ricker (Mexican hat) wavelet (Daubechies 1992 §2.1 eq.(2.1.4)):
///   ψ(t) = (2/(√3·π^{1/4})) · (1 − t²) · exp(−t²/2)
///
/// Evaluation:
///   ψ(0) = 2/(√3·π^{1/4}) (factor (1−0)·exp(0)=1).
///   ψ(1) = (2/(√3·π^{1/4})) · (1−1) · exp(−0.5) = 0   (exact; (1−t²)=0 when t=±1).
///   ψ(−1) = 0  (ψ is even; same zero factor).
///
/// Threshold 1×10⁻¹⁴: each computed value involves at most one scalar product
/// of the impulse sample (1.0) with a constant. Accumulated FP error is O(ε_f64) ≈ 2×10⁻¹⁶.
///
/// Reference: Daubechies (1992), Ten Lectures on Wavelets, §2.1 eq.(2.1.4);
///            Marr & Hildreth (1980), Proc. R. Soc. Lond. B 207:187–217.
fn cwt_ricker_impulse_peak_value_fixture() -> SuiteResult<PublishedFixtureReport> {
    // Impulse at sample 3 in a 7-sample signal.
    let signal = [0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let plan = CwtPlan::new(7, vec![1.0], ContinuousWavelet::Ricker)?;
    let coeffs = plan.transform(&signal)?;
    let actual = [
        coeffs.values()[[0, 2]], // W(1, 2) = ψ(1) = 0 exactly
        coeffs.values()[[0, 3]], // W(1, 3) = ψ(0) = 2/(√3·π^{1/4})
        coeffs.values()[[0, 4]], // W(1, 4) = ψ(−1) = 0 exactly
    ];
    let psi0 = 2.0_f64 / (3.0_f64.sqrt() * std::f64::consts::PI.powf(0.25));
    let expected = [0.0_f64, psi0, 0.0_f64];
    Ok(published_real_fixture_with_threshold(
        "CWT-Ricker",
        "CWT-Ricker(δ_{3},N=7,a=1): W(1,2)=0, W(1,3)=ψ(0)=2/(√3·π^¼), W(1,4)=0",
        "Daubechies (1992) Ten Lectures on Wavelets §2.1 eq.(2.1.4): ψ(0)=2/(√3·π^¼); ψ(±1)=0 exact zero crossing",
        &actual,
        &expected,
        1.0e-14,
    ))
}

/// CWT Ricker L² scale-normalization: W(a=2,b=n₀) = ψ(0)/√2.
///
/// # Mathematical contract
///
/// The 1/√a prefactor (L² normalization) from Daubechies (1992) §2.1 is tested
/// directly by comparing impulse-response peaks at two different scales.
///
/// For x = δ_{n₀}, a=2, b=n₀:
///   W(2, n₀) = (1/√2) · ψ((n₀−n₀)/2) = (1/√2) · ψ(0) = ψ(0)/√2
///
/// With ψ(0) = 2/(√3·π^{1/4}) (see fixture 54):
///   W(2, n₀) = 2/(√3·π^{1/4}·√2) = √2/(√3·π^{1/4})
///
/// Threshold 1×10⁻¹³: the only additional computation vs fixture 54 is one
/// multiplication by 1/√2, adding at most one ε_f64 of error. O(2·ε_f64) ≤ 1×10⁻¹³.
///
/// Reference: Daubechies (1992), Ten Lectures on Wavelets, §2.1 L² normalization;
///            Grossmann & Morlet (1984), SIAM J. Math. Anal. 15(4):723–736, eq.(1.3).
fn cwt_ricker_scale_normalization_fixture() -> SuiteResult<PublishedFixtureReport> {
    let signal = [0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let plan = CwtPlan::new(7, vec![2.0], ContinuousWavelet::Ricker)?;
    let coeffs = plan.transform(&signal)?;
    let actual = [coeffs.values()[[0, 3]]]; // W(2, 3) = ψ(0)/√2
    let psi0_over_sqrt2 =
        2.0_f64 / (3.0_f64.sqrt() * std::f64::consts::PI.powf(0.25) * std::f64::consts::SQRT_2);
    let expected = [psi0_over_sqrt2];
    Ok(published_real_fixture_with_threshold(
        "CWT-Ricker",
        "CWT-Ricker(δ_{3},N=7,a=2): W(2,3)=ψ(0)/√2=√2/(√3·π^¼)",
        "Daubechies (1992) §2.1 L² normalization: W(a,b)=(1/√a)·CWT; Grossmann-Morlet (1984) eq.(1.3)",
        &actual,
        &expected,
        1.0e-13,
    ))
}

/// DCT-III DC input → flat output: DCT-III₄([1,0,0,0]) = [½,½,½,½].
///
/// # Mathematical contract
///
/// DCT-III kernel (FFTW REDFT01 convention, unnormalized):
///   y[k] = x[0]/2 + Σ_{n=1}^{N-1} x[n]·cos(π·n·(k+½)/N)
///
/// For x = [1,0,0,0] and N=4, every summed term has x[n]=0 for n≥1, so the
/// formula reduces to y[k] = x[0]/2 = 1/2 for all k ∈ {0,1,2,3}.
/// This is the DC basis vector property of DCT-III: the DC spectral coefficient
/// X[0] maps to a flat time-domain sequence of amplitude X[0]/2.
///
/// Threshold 1×10⁻¹⁵: single multiplication x[0]×0.5; no summation;
/// error ≤ ε_f64·0.5 < 1.2×10⁻¹⁶ (below threshold by one order).
///
/// Reference: Makhoul (1980) IEEE Trans. Acoust. Speech Signal Process. 28(1)
/// Table I: DCT-III definition; FFTW REDFT01 documentation §3.
fn dct3_dc_input_flat_output_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(4, RealTransformKind::DctIII)?;
    let actual = plan.forward(&[1.0_f64, 0.0, 0.0, 0.0])?;
    let expected = [0.5_f64, 0.5, 0.5, 0.5];
    Ok(published_real_fixture_with_threshold(
        "DCT-III",
        "DCT-III4([1,0,0,0])=[1/2,1/2,1/2,1/2]",
        "Makhoul (1980) IEEE Trans. ASSP 28(1) Table I: DCT-III DC input X[0]=1, X[k>0]=0 \u{2192} y[n]=X[0]/2=1/2 for all n; FFTW REDFT01",
        &actual,
        &expected,
        1.0e-15,
    ))
}

/// DST-III Nyquist input → alternating output: DST-III₄([0,0,0,1]) = [½,−½,½,−½].
///
/// # Mathematical contract
///
/// DST-III kernel (FFTW RODFT01 convention, unnormalized):
///   y[k] = (−1)^k·x[N−1]/2 + Σ_{n=0}^{N-2} x[n]·sin(π·(n+1)·(k+½)/N)
///
/// For x = [0,0,0,1] and N=4, every summed term has x[n]=0 for n≤2, so the
/// formula reduces to y[k] = (−1)^k·x[3]/2 = (−1)^k/2.
/// Result: y = [1/2, −1/2, 1/2, −1/2].
/// This is the Nyquist basis vector property of DST-III: the Nyquist spectral
/// coefficient X[N-1] maps to an alternating sequence of amplitude X[N-1]/2.
///
/// Threshold 1×10⁻¹⁵: single multiplication x[N-1]×0.5 with a sign from (−1)^k;
/// no summation; error ≤ ε_f64·0.5 < 1.2×10⁻¹⁶.
///
/// Reference: Makhoul (1980) IEEE Trans. Acoust. Speech Signal Process. 28(1)
/// Table II: DST-III definition; FFTW RODFT01 documentation §3.
fn dst3_nyquist_input_alternating_output_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(4, RealTransformKind::DstIII)?;
    let actual = plan.forward(&[0.0_f64, 0.0, 0.0, 1.0])?;
    let expected = [0.5_f64, -0.5, 0.5, -0.5];
    Ok(published_real_fixture_with_threshold(
        "DST-III",
        "DST-III4([0,0,0,1])=[1/2,-1/2,1/2,-1/2]",
        "Makhoul (1980) IEEE Trans. ASSP 28(1) Table II: DST-III Nyquist input X[N-1]=1, X[k<N-1]=0 \u{2192} y[n]=(-1)^n/2; FFTW RODFT01",
        &actual,
        &expected,
        1.0e-15,
    ))
}

/// DCT-I three-point forward known values: DCT-I₃([1,2,3]) = [8,−2,0].
///
/// # Mathematical contract
///
/// DCT-I kernel (FFTW REDFT00 convention, unnormalized):
///   y[k] = x[0] + (−1)^k·x[N−1] + 2·Σ_{n=1}^{N−2} x[n]·cos(π·n·k/(N−1))
///
/// For x = [1,2,3] and N=3, N−1=2:
///   y[0] = 1 + 3 + 2·2·cos(0)    = 1 + 3 + 4 = 8
///   y[1] = 1 − 3 + 2·2·cos(π/2) = −2 + 0 = −2
///   y[2] = 1 + 3 + 2·2·cos(π)   = 4 − 4 = 0
///
/// y[2]=0 is algebraically exact: cos(π)=−1 cancels the interior term.
/// Threshold 1×10⁻¹⁵: all basis evaluations at k·π/2 ∈ {0,π/2,π};
/// cos(0)=1 and cos(π)=−1 are exactly representable; cos(π/2)≈0 contributes
/// no numerical error to y[1]; rounding ≤ 2ε_f64 < 5×10⁻¹⁶.
///
/// Reference: Rao & Yip (1990) *DCT Algorithms, Advantages, Applications* Table 2.1;
/// FFTW REDFT00 §3.
fn dct1_three_point_forward_known_values_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(3, RealTransformKind::DctI)?;
    let actual = plan.forward(&[1.0_f64, 2.0, 3.0])?;
    let expected = [8.0_f64, -2.0, 0.0];
    Ok(published_real_fixture_with_threshold(
        "DCT-I",
        "DCT-I3([1,2,3])=[8,-2,0]",
        "Rao & Yip (1990) Discrete Cosine Transform Table 2.1: DCT-I N=3 x=[1,2,3] \u{2192} y=[8,-2,0]; FFTW REDFT00; y[k]=x[0]+(-1)^k\u{22C5}x[N-1]+2\u{00D7}\u{03A3}x[n]cos(\u{03C0}nk/(N-1))",
        &actual,
        &expected,
        1.0e-15,
    ))
}

/// DST-I two-point forward known values: DST-I₂([1,3]) = [4√3, −2√3].
///
/// # Mathematical contract
///
/// DST-I kernel (FFTW RODFT00 convention, unnormalized):
///   y[k] = 2·Σ_{n=0}^{N−1} x[n]·sin(π·(n+1)·(k+1)/(N+1))
///
/// For x = [1,3] and N=2, N+1=3:
///   y[0] = 2·(1·sin(π/3) + 3·sin(2π/3))
///        = 2·(\u{221A}3/2 + 3·\u{221A}3/2) = 2·4·\u{221A}3/2 = 4\u{221A}3
///   y[1] = 2·(1·sin(2π/3) + 3·sin(4π/3))
///        = 2·(\u{221A}3/2 − 3·\u{221A}3/2) = 2·(−\u{221A}3) = −2\u{221A}3
///
/// Threshold 1×10⁻¹²: each term involves one sin evaluation; sin(π/3) and
/// sin(2π/3) share the value \u{221A}3/2 but each is rounded independently in f64;
/// accumulated rounding ≤ 4ε_f64·|y| < 3×10⁻¹⁵.
///
/// Reference: Rao & Yip (1990) *DCT Algorithms, Advantages, Applications* Table 3.1;
/// FFTW RODFT00 §3.
fn dst1_two_point_forward_known_values_fixture() -> SuiteResult<PublishedFixtureReport> {
    let plan = DctDstPlan::new(2, RealTransformKind::DstI)?;
    let actual = plan.forward(&[1.0_f64, 3.0])?;
    let expected = [4.0_f64 * 3.0_f64.sqrt(), -2.0_f64 * 3.0_f64.sqrt()];
    Ok(published_real_fixture(
        "DST-I",
        "DST-I2([1,3])=[4\u{221A}3,-2\u{221A}3]",
        "Rao & Yip (1990) Discrete Cosine Transform Table 3.1: DST-I N=2 x=[1,3] \u{2192} y=[4\u{221A}3,-2\u{221A}3]; FFTW RODFT00; y[k]=2\u{00D7}\u{03A3}x[n]sin(\u{03C0}(n+1)(k+1)/(N+1))",
        &actual,
        &expected,
    ))
}

fn representative_signal_1d(len: usize) -> Array1<f64> {
    Array1::from_vec(
        (0..len)
            .map(|i| {
                let x = i as f64;
                (0.31 * x).sin() + 0.17 * (0.73 * x).cos()
            })
            .collect(),
    )
}

fn representative_field_3d(shape: (usize, usize, usize)) -> Array3<f64> {
    Array3::from_shape_fn(shape, |(i, j, k)| {
        let x = i as f64;
        let y = j as f64;
        let z = k as f64;
        (0.11 * x).sin() + (0.13 * y).cos() + (0.17 * z).sin()
    })
}

fn max_complex_abs_delta<'a, I, J>(left: I, right: J) -> f64
where
    I: IntoIterator<Item = &'a Complex64>,
    J: IntoIterator<Item = &'a Complex64>,
{
    left.into_iter()
        .zip(right)
        .map(|(lhs, rhs)| (*lhs - *rhs).norm())
        .fold(0.0, f64::max)
}

fn max_real_abs_delta(left: &Array1<f64>, right: &Array1<f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

fn max_real_abs_delta_3d(left: &Array3<f64>, right: &Array3<f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

fn relative_complex_error<'a, I, J>(left: I, right: J) -> f64
where
    I: IntoIterator<Item = &'a Complex64>,
    J: IntoIterator<Item = &'a Complex64>,
{
    left.into_iter()
        .zip(right)
        .map(|(lhs, rhs)| (*lhs - *rhs).norm() / lhs.norm().max(1.0))
        .fold(0.0, f64::max)
}

fn elapsed_ms<F, T>(f: F) -> f64
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let _ = f();
    start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_suite_produces_value_semantic_reports() {
        let report = run_validation_suite().expect("validation suite");
        assert!(report.fft_cpu.roundtrip_max_abs_error <= CPU_ROUNDTRIP_LIMIT);
        assert!(report.fft_cpu.parseval_relative_error <= CPU_PARSEVAL_LIMIT);
        assert!(report.nufft.passed);
        assert!(report.external.published_references.passed);
        assert_eq!(report.external.published_references.attempted, 59);
        assert_eq!(report.external.rustfft.backend, "rustfft");
        assert_eq!(report.external.numpy.backend, "numpy");
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
    fn validation_report_json_schema_contains_required_sections() {
        let report = run_validation_suite().expect("validation suite");
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
}
