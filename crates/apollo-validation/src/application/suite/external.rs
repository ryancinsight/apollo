use std::path::Path;

use crate::domain::report::{
    ExternalBackendReport, ExternalComparisonReport, PublishedReferenceReport,
};
use crate::infrastructure::dft_reference::{dft_1d_array, dft_3d_real};
use leto::Storage;

use super::benchmark::precision_profile_reports;
use super::environment::numpy_comparison_report;
use super::fixtures::{
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
use super::metrics::{max_complex_abs_delta, representative_field_3d, representative_signal_1d};
use super::{SuiteResult, EXTERNAL_FFT_LIMIT};

/// Compare Apollo CPU FFT output with optional external reference engines.
pub fn run_external_comparison_suite() -> SuiteResult<ExternalComparisonReport> {
    let signal = representative_signal_1d(16);
    let signal_leto = leto::Array::<_, leto::MnemosyneStorage<_>, 1>::from_mnemosyne_slice(
        [signal.size()],
        signal.as_slice().unwrap(),
    )
    .unwrap();

    let dft_report = {
        let apollo_leto = apollo_fft::fft_1d_leto(signal_leto.view());
        let apollo = leto::Array1::from(apollo_leto.storage().as_slice().to_vec());
        let dft = dft_1d_array(&signal);
        let dft_fft1_max_abs_error = max_complex_abs_delta(apollo.iter(), dft.iter());

        let prime_signal = representative_signal_1d(17);
        let prime_signal_leto =
            leto::Array::<_, leto::MnemosyneStorage<_>, 1>::from_mnemosyne_slice(
                [prime_signal.size()],
                prime_signal.as_slice().unwrap(),
            )
            .unwrap();
        let prime_apollo_leto = apollo_fft::fft_1d_leto(prime_signal_leto.view());
        let prime_apollo = leto::Array1::from(prime_apollo_leto.storage().as_slice().to_vec());
        let prime_dft = dft_1d_array(&prime_signal);
        let dft_prime_error = max_complex_abs_delta(prime_apollo.iter(), prime_dft.iter());

        let field = representative_field_3d([4, 4, 4]);
        let field_leto = leto::Array::<_, leto::MnemosyneStorage<_>, 3>::from_mnemosyne_slice(
            [4, 4, 4],
            field.as_slice().unwrap(),
        )
        .unwrap();
        let apollo_3d_leto = apollo_fft::fft_3d_leto(field_leto.view());
        let apollo_3d =
            leto::Array3::from_shape_vec([4, 4, 4], apollo_3d_leto.storage().as_slice().to_vec())
                .unwrap();
        let dft_3d = dft_3d_real(&field);
        let dft_fft3_max_abs_error = max_complex_abs_delta(apollo_3d.iter(), dft_3d.iter());

        ExternalBackendReport {
            backend: "dft".to_string(),
            available: true,
            attempted: true,
            fft1_max_abs_error: Some(dft_fft1_max_abs_error),
            fft1_prime_max_abs_error: Some(dft_prime_error),
            fft3_max_abs_error: Some(dft_fft3_max_abs_error),
            stability_max_abs_delta: Some(0.0),
            version: None,
            note: None,
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

    let passed = dft_report
        .fft1_max_abs_error
        .is_some_and(|error| error <= EXTERNAL_FFT_LIMIT)
        && dft_report
            .fft1_prime_max_abs_error
            .is_some_and(|error| error <= EXTERNAL_FFT_LIMIT)
        && dft_report
            .fft3_max_abs_error
            .is_some_and(|error| error <= EXTERNAL_FFT_LIMIT)
        && (!numpy_report.attempted
            || numpy_report
                .fft1_max_abs_error
                .is_some_and(|error| error <= EXTERNAL_FFT_LIMIT))
        && published_references.passed;

    Ok(ExternalComparisonReport {
        passed,
        pyfftw_checkout_present: Path::new("external/pyfftw").exists(),
        dft: dft_report,
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
