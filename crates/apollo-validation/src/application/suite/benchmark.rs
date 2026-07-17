use crate::domain::report::{BenchmarkReport, PrecisionBenchmarkReport, PrecisionRunReport};
use crate::infrastructure::dft_reference::{dft_1d_array, dft_3d_real};
use crate::infrastructure::numpy::benchmark_fft;
use apollo_fft::f16;
use apollo_fft::{fft_1d_array, fft_1d_array_typed, ifft_1d_array, ifft_1d_array_typed};
use apollo_nufft::{
    nufft_type1_1d, nufft_type1_1d_fast, nufft_type1_3d_fast, UniformDomain1D, UniformGrid3D,
    DEFAULT_NUFFT_KERNEL_WIDTH,
};
use eunomia::Complex64;
use leto::Storage;

use super::metrics::{
    elapsed_ms, max_real_abs_delta_3d, representative_field_3d, representative_signal_1d,
};
use super::SuiteResult;

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

    let dft_fft1_ms = elapsed_ms(|| {
        let _ = dft_1d_array(&signal);
    });
    let dft_fft3_ms = elapsed_ms(|| {
        let _ = dft_3d_real(&field);
    });

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
        dft_fft1_ms,
        dft_fft3_ms,
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

pub(super) fn precision_profile_reports() -> Vec<PrecisionRunReport> {
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
