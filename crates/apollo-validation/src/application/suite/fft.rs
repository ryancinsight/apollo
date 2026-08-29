use crate::domain::report::{CpuFftReport, GpuFftReport, PrecisionRunReport};
use hephaestus_core::{
    ComputeDevice, FftDirection, FftOperands, FftOps, HephaestusError, StridedView,
};
use hephaestus_wgpu::{WgpuDevice, WgpuFftOps};
use leto::{self, Array1, Array3, Layout, Storage};

use super::benchmark::precision_profile_reports;
use super::metrics::{
    max_complex_abs_delta, max_real_abs_delta, max_real_abs_delta_3d, representative_field_3d,
};
use super::{SuiteResult, CPU_PARSEVAL_LIMIT, CPU_ROUNDTRIP_LIMIT, CPU_STABILITY_LIMIT};

const GPU_SHAPE: [usize; 3] = [4, 4, 4];

struct GpuFftErrors {
    forward: f64,
    inverse: f64,
    forward_limit: f64,
    inverse_limit: f64,
}

fn axis_rounding_sites(length: usize) -> u32 {
    if length <= 1 {
        0
    } else {
        5 * length.ilog2()
    }
}

fn relative_error_bound(rounding_sites: u32) -> f64 {
    let unit_roundoff = f64::from(f32::EPSILON) / 2.0;
    let accumulated = f64::from(rounding_sites) * unit_roundoff;
    debug_assert!(
        accumulated < 1.0,
        "invariant: FFT rounding model requires sites times unit roundoff below one"
    );
    accumulated / (1.0 - accumulated)
}

fn gpu_rounding_sites(shape: [usize; 3]) -> (u32, u32) {
    let axis_sites = shape.into_iter().map(axis_rounding_sites).sum::<u32>();
    let active_axes = u32::try_from(shape.into_iter().filter(|&n| n > 1).count())
        .expect("invariant: rank-three active-axis count fits u32");
    (1 + axis_sites, 1 + 2 * axis_sites + active_axes)
}

fn gpu_fft_errors(device: &WgpuDevice) -> SuiteResult<GpuFftErrors> {
    let reference = representative_field_3d(GPU_SHAPE);
    let input = reference
        .iter()
        .map(|&value| value as f32)
        .collect::<Vec<_>>();
    let represented =
        Array3::from_shape_vec(GPU_SHAPE, input.iter().copied().map(f64::from).collect())?;
    let reference_leto = leto::Array::<_, leto::MnemosyneStorage<_>, 3>::from_mnemosyne_slice(
        GPU_SHAPE,
        represented
            .as_slice()
            .expect("invariant: constructed reference field is contiguous"),
    )?;
    let cpu_spectrum = apollo_fft::fft_3d_leto(reference_leto.view());
    let real = device.upload(&input)?;
    let imaginary = device.alloc_zeroed(input.len())?;
    let layout = Layout::c_contiguous(GPU_SHAPE)?;
    let operands = || FftOperands {
        real: StridedView::new(&real, &layout),
        imaginary: StridedView::new(&imaginary, &layout),
    };
    let ops = WgpuFftOps;
    let forward = ops.prepare_fft(device, operands(), FftDirection::Forward)?;
    let inverse = ops.prepare_fft(device, operands(), FftDirection::Inverse)?;

    ops.dispatch_fft(device, &forward)?;
    let gpu_real = device.download_owned(&real)?;
    let gpu_imaginary = device.download_owned(&imaginary)?;
    let forward_error = cpu_spectrum
        .storage()
        .as_slice()
        .iter()
        .zip(gpu_real.iter().zip(&gpu_imaginary))
        .map(|(expected, (&actual_real, &actual_imaginary))| {
            (f64::from(actual_real) - expected.re).hypot(f64::from(actual_imaginary) - expected.im)
        })
        .fold(0.0_f64, f64::max);

    ops.dispatch_fft(device, &inverse)?;
    let recovered = Array3::from_shape_vec(
        GPU_SHAPE,
        device
            .download_owned(&real)?
            .into_iter()
            .map(f64::from)
            .collect(),
    )?;
    let inverse_error = max_real_abs_delta_3d(&represented, &recovered);

    // Hephaestus's radix-2 conformance model counts five rounded operations
    // per stage and component. The forward component bound is gamma_k times
    // the complex-input L1 norm. For a normalized forward/inverse pair, the
    // maximum component error is bounded by the relative L2 bound times the
    // input L2 norm because max-norm <= L2-norm.
    let (forward_sites, roundtrip_sites) = gpu_rounding_sites(GPU_SHAPE);
    let input_l1 = input
        .iter()
        .map(|&value| f64::from(value).abs())
        .sum::<f64>();
    let input_l2 = input
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>()
        .sqrt();
    Ok(GpuFftErrors {
        forward: forward_error,
        inverse: inverse_error,
        forward_limit: relative_error_bound(forward_sites) * input_l1,
        inverse_limit: relative_error_bound(roundtrip_sites) * input_l2,
    })
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
    let recovered = apollo_fft::ifft_1d_leto::<f64>(spectrum.view());
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
    let device = match WgpuDevice::try_default("apollo-validation-fft-wgpu") {
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
        Ok(device) => device,
    };
    let errors = gpu_fft_errors(&device)?;
    let passed = errors.forward <= errors.forward_limit && errors.inverse <= errors.inverse_limit;

    Ok(GpuFftReport {
        attempted: true,
        passed,
        forward_max_abs_error: Some(errors.forward),
        inverse_max_abs_error: Some(errors.inverse),
        note: None,
        precision_profiles: vec![PrecisionRunReport {
            profile: "low_precision".to_string(),
            attempted: true,
            passed,
            forward_max_abs_error: Some(errors.forward),
            inverse_max_abs_error: Some(errors.inverse),
            relative_error: Some(errors.forward.max(errors.inverse)),
            note: Some(format!(
                "Hephaestus f32 FFT; derived forward limit {:.3e}, inverse limit {:.3e}",
                errors.forward_limit, errors.inverse_limit
            )),
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::{gpu_rounding_sites, relative_error_bound, GPU_SHAPE};

    #[test]
    fn gpu_error_model_counts_every_radix_stage() {
        assert_eq!(gpu_rounding_sites(GPU_SHAPE), (31, 64));
        assert!(relative_error_bound(31) < relative_error_bound(64));
    }
}
