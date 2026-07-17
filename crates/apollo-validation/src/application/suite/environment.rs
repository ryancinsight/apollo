use crate::domain::report::{EnvironmentReport, ExternalBackendReport};
use crate::infrastructure::numpy::{compare_fft, PythonEnvironmentProbe};
use eunomia::Complex64;
use leto::{Array1, Storage};

use super::metrics::max_complex_abs_delta;

pub(super) fn environment_report(probe: Option<&PythonEnvironmentProbe>) -> EnvironmentReport {
    EnvironmentReport {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        debug_build: cfg!(debug_assertions),
        python_version: probe.map(|value| value.python_version.clone()),
        numpy_version: probe.and_then(|value| value.numpy_version.clone()),
        pyfftw_version: probe.and_then(|value| value.pyfftw_version.clone()),
    }
}

pub(super) fn numpy_comparison_report(signal: &Array1<f64>) -> ExternalBackendReport {
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
