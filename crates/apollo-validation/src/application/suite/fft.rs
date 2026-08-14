use crate::domain::report::{CpuFftReport, GpuFftReport, PrecisionRunReport};
use apollo_fft::{FftBackend, Shape3D};
use hephaestus_core::HephaestusError;
use hephaestus_wgpu::WgpuDevice;
use leto::{self, Array1, Storage};

use super::benchmark::precision_profile_reports;
use super::metrics::{
    max_complex_abs_delta, max_real_abs_delta, max_real_abs_delta_3d, representative_field_3d,
};
use super::{SuiteResult, CPU_PARSEVAL_LIMIT, CPU_ROUNDTRIP_LIMIT, CPU_STABILITY_LIMIT};

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
    let backend = match WgpuDevice::try_default("apollo-validation-fft-wgpu") {
        Err(HephaestusError::AdapterUnavailable { .. }) => {
            return Ok(GpuFftReport {
                attempted: false,
                passed: false,
                forward_max_abs_error: None,
                inverse_max_abs_error: None,
                note: Some("WGPU adapter unavailable on this host".to_string()),
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
        Err(error) => return Err(Box::new(error)),
        Ok(device) => apollo_fft::WgpuBackend::new(device),
    };

    let shape = Shape3D::new(4, 4, 4).expect("valid shape");
    let plan = match FftBackend::plan_3d(&backend, shape) {
        Err(error) => {
            return Ok(GpuFftReport {
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
