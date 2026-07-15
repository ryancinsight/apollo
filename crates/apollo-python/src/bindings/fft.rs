//! One-shot real-input FFT functions (`fft1`..`ifft3`, `rfft3`, `irfft3`)
//! wrapping `apollo-fft`.

use apollo_fft::{f16, Complex32, Complex64, StoragePrecision};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::support::{
    leto_array1_into_pyarray, leto_array2_into_pyarray, leto_array3_into_pyarray, parse_precision,
    py_array1_leto_view, py_array1_map_to_leto, py_array2_leto_view, py_array2_map_to_leto,
    py_array3_leto_view, py_array3_map_to_leto, require_contiguous_1d, require_contiguous_2d,
    require_contiguous_3d, require_profile_matches_f32, require_profile_matches_f64,
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};

/// Forward 1D FFT of a real signal.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
pub(crate) fn fft1<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray1<f64>>() {
        require_contiguous_1d(&input64, "fft1 input")?;
        require_profile_matches_f64(profile, "fft1")?;
        let leto_view = py_array1_leto_view(&input64, "fft1 input")?;
        let result = py.detach(|| apollo_fft::fft_1d_leto(leto_view));
        leto_array1_into_pyarray(py, result)
    } else {
        match profile.storage {
            StoragePrecision::F16 => {
                let input32 = input.extract::<PyReadonlyArray1<f32>>()?;
                let owned =
                    py_array1_map_to_leto(&input32, "fft1 input", |value| f16::from_f32(*value))?;
                let result = py.detach(|| apollo_fft::fft_1d_array_typed(&owned));
                leto_array1_into_pyarray(py, result)
            }
            _ => {
                let input32 = input.extract::<PyReadonlyArray1<f32>>()?;
                require_profile_matches_f32(profile, "fft1")?;
                let leto_view = py_array1_leto_view(&input32, "fft1 input")?;
                let result = py.detach(|| apollo_fft::fft_1d_leto_typed::<f32>(leto_view));
                leto_array1_into_pyarray(py, result)
            }
        }
    }
}

/// Inverse 1D FFT of a complex spectrum.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
pub(crate) fn ifft1<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray1<Complex64>>() {
        require_contiguous_1d(&input64, "ifft1 input")?;
        require_profile_matches_f64(profile, "ifft1")?;
        let leto_view = py_array1_leto_view(&input64, "ifft1 input")?;
        let result = py.detach(|| apollo_fft::ifft_1d_leto(leto_view));
        leto_array1_into_pyarray(py, result)
    } else {
        let input32 = input.extract::<PyReadonlyArray1<Complex32>>()?;
        match profile.storage {
            StoragePrecision::F16 => {
                let owned = super::support::py_array1_to_leto(&input32, "ifft1 input")?;
                let result = py.detach(|| {
                    apollo_fft::ifft_1d_array_typed::<f16>(&owned).mapv(|value: f16| value.to_f32())
                });
                leto_array1_into_pyarray(py, result)
            }
            _ => {
                require_profile_matches_f32(profile, "ifft1")?;
                let leto_view = py_array1_leto_view(&input32, "ifft1 input")?;
                let result = py.detach(|| apollo_fft::ifft_1d_leto_typed::<f32>(leto_view));
                leto_array1_into_pyarray(py, result)
            }
        }
    }
}

/// Forward 2D FFT of a real array.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
pub(crate) fn fft2<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray2<f64>>() {
        require_contiguous_2d(&input64, "fft2 input")?;
        require_profile_matches_f64(profile, "fft2")?;
        let leto_view = py_array2_leto_view(&input64, "fft2 input")?;
        let result = py.detach(|| apollo_fft::fft_2d_leto(leto_view));
        leto_array2_into_pyarray(py, result)
    } else {
        match profile.storage {
            StoragePrecision::F16 => {
                let input32 = input.extract::<PyReadonlyArray2<f32>>()?;
                let owned =
                    py_array2_map_to_leto(&input32, "fft2 input", |value| f16::from_f32(*value))?;
                let result = py.detach(|| apollo_fft::fft_2d_array_typed(&owned));
                leto_array2_into_pyarray(py, result)
            }
            _ => {
                let input32 = input.extract::<PyReadonlyArray2<f32>>()?;
                require_profile_matches_f32(profile, "fft2")?;
                let leto_view = py_array2_leto_view(&input32, "fft2 input")?;
                let result = py.detach(|| apollo_fft::fft_2d_leto_typed::<f32>(leto_view));
                leto_array2_into_pyarray(py, result)
            }
        }
    }
}

/// Inverse 2D FFT of a complex spectrum.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
pub(crate) fn ifft2<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray2<Complex64>>() {
        require_contiguous_2d(&input64, "ifft2 input")?;
        require_profile_matches_f64(profile, "ifft2")?;
        let leto_view = py_array2_leto_view(&input64, "ifft2 input")?;
        let result = py.detach(|| apollo_fft::ifft_2d_leto(leto_view));
        leto_array2_into_pyarray(py, result)
    } else {
        let input32 = input.extract::<PyReadonlyArray2<Complex32>>()?;
        match profile.storage {
            StoragePrecision::F16 => {
                let owned = super::support::py_array2_to_leto(&input32, "ifft2 input")?;
                let result = py.detach(|| {
                    apollo_fft::ifft_2d_array_typed::<f16>(&owned).mapv(|value: f16| value.to_f32())
                });
                leto_array2_into_pyarray(py, result)
            }
            _ => {
                require_profile_matches_f32(profile, "ifft2")?;
                let leto_view = py_array2_leto_view(&input32, "ifft2 input")?;
                let result = py.detach(|| apollo_fft::ifft_2d_leto_typed::<f32>(leto_view));
                leto_array2_into_pyarray(py, result)
            }
        }
    }
}

/// Forward 3D FFT of a real array.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
pub(crate) fn fft3<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray3<f64>>() {
        require_contiguous_3d(&input64, "fft3 input")?;
        require_profile_matches_f64(profile, "fft3")?;
        let leto_view = py_array3_leto_view(&input64, "fft3 input")?;
        let result = py.detach(|| apollo_fft::fft_3d_leto(leto_view));
        leto_array3_into_pyarray(py, result)
    } else {
        match profile.storage {
            StoragePrecision::F16 => {
                let input32 = input.extract::<PyReadonlyArray3<f32>>()?;
                let owned =
                    py_array3_map_to_leto(&input32, "fft3 input", |value| f16::from_f32(*value))?;
                let result = py.detach(|| apollo_fft::fft_3d_array_typed(&owned));
                leto_array3_into_pyarray(py, result)
            }
            _ => {
                let input32 = input.extract::<PyReadonlyArray3<f32>>()?;
                require_profile_matches_f32(profile, "fft3")?;
                let leto_view = py_array3_leto_view(&input32, "fft3 input")?;
                let result = py.detach(|| apollo_fft::fft_3d_leto_typed::<f32>(leto_view));
                leto_array3_into_pyarray(py, result)
            }
        }
    }
}

/// Inverse 3D FFT of a complex spectrum.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
pub(crate) fn ifft3<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray3<Complex64>>() {
        require_contiguous_3d(&input64, "ifft3 input")?;
        require_profile_matches_f64(profile, "ifft3")?;
        let leto_view = py_array3_leto_view(&input64, "ifft3 input")?;
        let result = py.detach(|| apollo_fft::ifft_3d_leto(leto_view));
        leto_array3_into_pyarray(py, result)
    } else {
        let input32 = input.extract::<PyReadonlyArray3<Complex32>>()?;
        match profile.storage {
            StoragePrecision::F16 => {
                let owned = super::support::py_array3_to_leto(&input32, "ifft3 input")?;
                let result = py.detach(|| {
                    apollo_fft::ifft_3d_array_typed::<f16>(&owned).mapv(|value: f16| value.to_f32())
                });
                leto_array3_into_pyarray(py, result)
            }
            _ => {
                require_profile_matches_f32(profile, "ifft3")?;
                let leto_view = py_array3_leto_view(&input32, "ifft3 input")?;
                let result = py.detach(|| apollo_fft::ifft_3d_leto_typed::<f32>(leto_view));
                leto_array3_into_pyarray(py, result)
            }
        }
    }
}

/// Forward 3D real-to-complex half-spectrum FFT.
#[pyfunction]
pub(crate) fn rfft3<'py>(py: Python<'py>, input: PyReadonlyArray3<f64>) -> PyResult<Py<PyAny>> {
    require_contiguous_3d(&input, "rfft3 input")?;
    let leto_view = py_array3_leto_view(&input, "rfft3 input")?;
    let result = py.detach(|| apollo_fft::fft_3d_leto(leto_view));
    leto_array3_into_pyarray(py, result)
}

/// Inverse 3D half-spectrum FFT.
#[pyfunction]
pub(crate) fn irfft3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<Complex64>,
    nz: usize,
) -> PyResult<Py<PyAny>> {
    require_contiguous_3d(&input, "irfft3 input")?;
    let nz_c = input.shape()[2];
    if nz_c != nz {
        return Err(PyValueError::new_err(
            "irfft3 input shape and nz are inconsistent",
        ));
    }
    let leto_view = py_array3_leto_view(&input, "irfft3 input")?;
    let result = py.detach(|| apollo_fft::ifft_3d_leto(leto_view));
    leto_array3_into_pyarray(py, result)
}
