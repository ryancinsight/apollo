use crate::infrastructure::transport::gpu::{ShtWgpuBackend, ShtWgpuPlan, WgpuError};
use eunomia::{Complex32, Complex64};
use hephaestus_core::HephaestusError;
use leto::Array2;

pub(super) const CPU_DIFFERENTIAL_TOLERANCE: f64 = 2.0e-5;
pub(super) const REPRESENTED_STORAGE_TOLERANCE: f64 = 1.0e-3;

pub(super) fn backend() -> Option<ShtWgpuBackend> {
    match ShtWgpuBackend::try_default() {
        Ok(backend) => Some(backend),
        Err(WgpuError::Provider(HephaestusError::AdapterUnavailable { .. })) => None,
        Err(error) => panic!("SHT GPU verification requires a working provider: {error}"),
    }
}

pub(super) fn complex_samples(plan: &ShtWgpuPlan) -> Array2<Complex64> {
    Array2::from_shape_fn([plan.latitudes(), plan.longitudes()], |[lat, lon]| {
        Complex64::new(
            0.25 + lat as f64 * 0.5 - lon as f64 * 0.125,
            0.1 * (lat as f64 + 1.0) * (lon as f64 + 1.0),
        )
    })
}

pub(super) fn represented_samples(values: &Array2<Complex64>) -> Array2<Complex32> {
    values.mapv(|value| Complex32::new(value.re as f32, value.im as f32))
}

pub(super) fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert!(
        (actual.re - expected.re).abs() <= CPU_DIFFERENTIAL_TOLERANCE,
        "real mismatch: actual={actual:?}, expected={expected:?}"
    );
    assert!(
        (actual.im - expected.im).abs() <= CPU_DIFFERENTIAL_TOLERANCE,
        "imaginary mismatch: actual={actual:?}, expected={expected:?}"
    );
}
