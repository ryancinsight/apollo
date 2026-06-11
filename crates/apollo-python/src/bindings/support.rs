//! Shared NumPy layout checks and precision-profile helpers used by all
//! binding modules. Conversion-only logic; no domain computation.

use apollo_fft::{PrecisionMode, PrecisionProfile, StoragePrecision};
use numpy::{Element, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn require_contiguous_1d<T: Element>(
    input: &PyReadonlyArray1<'_, T>,
    name: &str,
) -> PyResult<()> {
    if input.as_array().is_standard_layout() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )))
    }
}

pub(crate) fn require_contiguous_2d<T: Element>(
    input: &PyReadonlyArray2<'_, T>,
    name: &str,
) -> PyResult<()> {
    if input.as_array().is_standard_layout() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )))
    }
}

pub(crate) fn require_contiguous_3d<T: Element>(
    input: &PyReadonlyArray3<'_, T>,
    name: &str,
) -> PyResult<()> {
    if input.as_array().is_standard_layout() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )))
    }
}

pub(crate) fn parse_precision(precision: Option<&str>) -> PyResult<PrecisionProfile> {
    match precision.unwrap_or("high_accuracy") {
        "high_accuracy" => Ok(PrecisionProfile::HIGH_ACCURACY_F64),
        "low_precision" => Ok(PrecisionProfile::LOW_PRECISION_F32),
        "mixed_precision" => Ok(PrecisionProfile::MIXED_PRECISION_F16_F32),
        other => Err(PyValueError::new_err(format!(
            "unsupported precision `{other}`; expected `high_accuracy`, `low_precision`, or `mixed_precision`"
        ))),
    }
}

pub(crate) fn precision_name(profile: PrecisionProfile) -> &'static str {
    match profile.mode {
        PrecisionMode::HighAccuracy => "high_accuracy",
        PrecisionMode::LowPrecision => "low_precision",
        PrecisionMode::MixedPrecision => "mixed_precision",
    }
}

pub(crate) fn require_profile_matches_f64(profile: PrecisionProfile, name: &str) -> PyResult<()> {
    if profile.storage == StoragePrecision::F64 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} received float64/complex128 input but precision `{}` expects float32/complex64 storage",
            precision_name(profile)
        )))
    }
}

pub(crate) fn require_profile_matches_f32(profile: PrecisionProfile, name: &str) -> PyResult<()> {
    if profile.storage == StoragePrecision::F32 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} received float32/complex64 input but precision `{}` expects float64/complex128 storage",
            precision_name(profile)
        )))
    }
}
