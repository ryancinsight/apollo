//! One-shot real-input FFT functions (`fft1`..`ifft3`, `rfft3`, `irfft3`)
//! wrapping `apollo-fft`.

use apollo_fft::{f16, Complex32, Complex64, StoragePrecision};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::support::{
    parse_precision, require_contiguous_1d, require_contiguous_2d, require_contiguous_3d,
    require_profile_matches_f32, require_profile_matches_f64,
};

/// Forward 1D FFT of a real signal.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
pub(crate) fn fft1<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<PyObject> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray1<f64>>() {
        require_contiguous_1d(&input64, "fft1 input")?;
        require_profile_matches_f64(profile, "fft1")?;
        let leto_view = leto::ArrayView1::from(input64.as_array());
        let result = py.allow_threads(|| apollo_fft::fft_1d_leto(leto_view));
        let nd_view = ndarray::ArrayView1::try_from(result.view()).map_err(|e| {
            PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
        })?;
        Ok(PyArray1::from_owned_array(py, nd_view.to_owned())
            .into_any()
            .unbind())
    } else {
        match profile.storage {
            StoragePrecision::F16 => {
                let input32 = input.extract::<PyReadonlyArray1<f32>>()?;
                require_contiguous_1d(&input32, "fft1 input")?;
                let owned = input32.as_array().mapv(f16::from_f32);
                let result = py.allow_threads(|| apollo_fft::fft_1d_array_typed(&leto::Array1::from(owned)));
                Ok(PyArray1::from_owned_array(py, ndarray::Array1::try_from(result).expect("leto result is C-contiguous")).into_any().unbind())
            }
            _ => {
                let input32 = input.extract::<PyReadonlyArray1<f32>>()?;
                require_contiguous_1d(&input32, "fft1 input")?;
                require_profile_matches_f32(profile, "fft1")?;
                let leto_view = leto::ArrayView1::from(input32.as_array());
                let result = py.allow_threads(|| apollo_fft::fft_1d_leto_typed::<f32>(leto_view));
                let nd_view = ndarray::ArrayView1::try_from(result.view()).map_err(|e| {
                    PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
                })?;
                Ok(PyArray1::from_owned_array(py, nd_view.to_owned())
                    .into_any()
                    .unbind())
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
) -> PyResult<PyObject> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray1<Complex64>>() {
        require_contiguous_1d(&input64, "ifft1 input")?;
        require_profile_matches_f64(profile, "ifft1")?;
        let leto_view = leto::ArrayView1::from(input64.as_array());
        let result = py.allow_threads(|| apollo_fft::ifft_1d_leto(leto_view));
        let nd_view = ndarray::ArrayView1::try_from(result.view()).map_err(|e| {
            PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
        })?;
        Ok(PyArray1::from_owned_array(py, nd_view.to_owned())
            .into_any()
            .unbind())
    } else {
        let input32 = input.extract::<PyReadonlyArray1<Complex32>>()?;
        require_contiguous_1d(&input32, "ifft1 input")?;
        match profile.storage {
            StoragePrecision::F16 => {
                let owned = input32.as_array().to_owned();
                let result = py.allow_threads(|| {
                    apollo_fft::ifft_1d_array_typed::<f16>(&leto::Array1::from(owned)).mapv(|value: f16| value.to_f32())
                });
                Ok(PyArray1::from_owned_array(py, ndarray::Array1::try_from(result).expect("leto result is C-contiguous")).into_any().unbind())
            }
            _ => {
                require_profile_matches_f32(profile, "ifft1")?;
                let leto_view = leto::ArrayView1::from(input32.as_array());
                let result = py.allow_threads(|| apollo_fft::ifft_1d_leto_typed::<f32>(leto_view));
                let nd_view = ndarray::ArrayView1::try_from(result.view()).map_err(|e| {
                    PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
                })?;
                Ok(PyArray1::from_owned_array(py, nd_view.to_owned())
                    .into_any()
                    .unbind())
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
) -> PyResult<PyObject> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray2<f64>>() {
        require_contiguous_2d(&input64, "fft2 input")?;
        require_profile_matches_f64(profile, "fft2")?;
        let leto_view = leto::ArrayView2::from(input64.as_array());
        let result = py.allow_threads(|| apollo_fft::fft_2d_leto(leto_view));
        let nd_view = ndarray::ArrayView2::try_from(result.view()).map_err(|e| {
            PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
        })?;
        Ok(PyArray2::from_owned_array(py, nd_view.to_owned())
            .into_any()
            .unbind())
    } else {
        match profile.storage {
            StoragePrecision::F16 => {
                let input32 = input.extract::<PyReadonlyArray2<f32>>()?;
                require_contiguous_2d(&input32, "fft2 input")?;
                let owned = input32.as_array().mapv(f16::from_f32);
                let result = py.allow_threads(|| apollo_fft::fft_2d_array_typed(&leto::Array2::from(owned)));
                Ok(PyArray2::from_owned_array(py, ndarray::Array2::try_from(result).expect("leto result is C-contiguous")).into_any().unbind())
            }
            _ => {
                let input32 = input.extract::<PyReadonlyArray2<f32>>()?;
                require_contiguous_2d(&input32, "fft2 input")?;
                require_profile_matches_f32(profile, "fft2")?;
                let leto_view = leto::ArrayView2::from(input32.as_array());
                let result = py.allow_threads(|| apollo_fft::fft_2d_leto_typed::<f32>(leto_view));
                let nd_view = ndarray::ArrayView2::try_from(result.view()).map_err(|e| {
                    PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
                })?;
                Ok(PyArray2::from_owned_array(py, nd_view.to_owned())
                    .into_any()
                    .unbind())
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
) -> PyResult<PyObject> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray2<Complex64>>() {
        require_contiguous_2d(&input64, "ifft2 input")?;
        require_profile_matches_f64(profile, "ifft2")?;
        let leto_view = leto::ArrayView2::from(input64.as_array());
        let result = py.allow_threads(|| apollo_fft::ifft_2d_leto(leto_view));
        let nd_view = ndarray::ArrayView2::try_from(result.view()).map_err(|e| {
            PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
        })?;
        Ok(PyArray2::from_owned_array(py, nd_view.to_owned())
            .into_any()
            .unbind())
    } else {
        let input32 = input.extract::<PyReadonlyArray2<Complex32>>()?;
        require_contiguous_2d(&input32, "ifft2 input")?;
        match profile.storage {
            StoragePrecision::F16 => {
                let owned = input32.as_array().to_owned();
                let result = py.allow_threads(|| {
                    apollo_fft::ifft_2d_array_typed::<f16>(&leto::Array2::from(owned)).mapv(|value: f16| value.to_f32())
                });
                Ok(PyArray2::from_owned_array(py, ndarray::Array2::try_from(result).expect("leto result is C-contiguous")).into_any().unbind())
            }
            _ => {
                require_profile_matches_f32(profile, "ifft2")?;
                let leto_view = leto::ArrayView2::from(input32.as_array());
                let result = py.allow_threads(|| apollo_fft::ifft_2d_leto_typed::<f32>(leto_view));
                let nd_view = ndarray::ArrayView2::try_from(result.view()).map_err(|e| {
                    PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
                })?;
                Ok(PyArray2::from_owned_array(py, nd_view.to_owned())
                    .into_any()
                    .unbind())
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
) -> PyResult<PyObject> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray3<f64>>() {
        require_contiguous_3d(&input64, "fft3 input")?;
        require_profile_matches_f64(profile, "fft3")?;
        let leto_view = leto::ArrayView3::from(input64.as_array());
        let result = py.allow_threads(|| apollo_fft::fft_3d_leto(leto_view));
        let nd_view = ndarray::ArrayView3::try_from(result.view()).map_err(|e| {
            PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
        })?;
        Ok(PyArray3::from_owned_array(py, nd_view.to_owned())
            .into_any()
            .unbind())
    } else {
        match profile.storage {
            StoragePrecision::F16 => {
                let input32 = input.extract::<PyReadonlyArray3<f32>>()?;
                require_contiguous_3d(&input32, "fft3 input")?;
                let owned = input32.as_array().mapv(f16::from_f32);
                let result = py.allow_threads(|| apollo_fft::fft_3d_array_typed(&leto::Array3::from(owned)));
                Ok(PyArray3::from_owned_array(py, ndarray::Array3::try_from(result).expect("leto result is C-contiguous")).into_any().unbind())
            }
            _ => {
                let input32 = input.extract::<PyReadonlyArray3<f32>>()?;
                require_contiguous_3d(&input32, "fft3 input")?;
                require_profile_matches_f32(profile, "fft3")?;
                let leto_view = leto::ArrayView3::from(input32.as_array());
                let result = py.allow_threads(|| apollo_fft::fft_3d_leto_typed::<f32>(leto_view));
                let nd_view = ndarray::ArrayView3::try_from(result.view()).map_err(|e| {
                    PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
                })?;
                Ok(PyArray3::from_owned_array(py, nd_view.to_owned())
                    .into_any()
                    .unbind())
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
) -> PyResult<PyObject> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray3<Complex64>>() {
        require_contiguous_3d(&input64, "ifft3 input")?;
        require_profile_matches_f64(profile, "ifft3")?;
        let leto_view = leto::ArrayView3::from(input64.as_array());
        let result = py.allow_threads(|| apollo_fft::ifft_3d_leto(leto_view));
        let nd_view = ndarray::ArrayView3::try_from(result.view()).map_err(|e| {
            PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
        })?;
        Ok(PyArray3::from_owned_array(py, nd_view.to_owned())
            .into_any()
            .unbind())
    } else {
        let input32 = input.extract::<PyReadonlyArray3<Complex32>>()?;
        require_contiguous_3d(&input32, "ifft3 input")?;
        let owned = input32.as_array().to_owned();
        match profile.storage {
            StoragePrecision::F16 => {
                let result = py.allow_threads(|| {
                    apollo_fft::ifft_3d_array_typed::<f16>(&leto::Array3::from(owned)).mapv(|value: f16| value.to_f32())
                });
                Ok(PyArray3::from_owned_array(py, ndarray::Array3::try_from(result).expect("leto result is C-contiguous")).into_any().unbind())
            }
            _ => {
                require_profile_matches_f32(profile, "ifft3")?;
                let leto_view = leto::ArrayView3::from(input32.as_array());
                let result = py.allow_threads(|| apollo_fft::ifft_3d_leto_typed::<f32>(leto_view));
                let nd_view = ndarray::ArrayView3::try_from(result.view()).map_err(|e| {
                    PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
                })?;
                Ok(PyArray3::from_owned_array(py, nd_view.to_owned())
                    .into_any()
                    .unbind())
            }
        }
    }
}

/// Forward 3D real-to-complex half-spectrum FFT.
#[pyfunction]
pub(crate) fn rfft3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<f64>,
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    require_contiguous_3d(&input, "rfft3 input")?;
    let leto_view = leto::ArrayView3::from(input.as_array());
    let result = py.allow_threads(|| apollo_fft::fft_3d_leto(leto_view));
    let nd_view = ndarray::ArrayView3::try_from(result.view()).map_err(|e| {
        PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
    })?;
    Ok(PyArray3::from_owned_array(py, nd_view.to_owned()))
}

/// Inverse 3D half-spectrum FFT.
#[pyfunction]
pub(crate) fn irfft3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<Complex64>,
    nz: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    require_contiguous_3d(&input, "irfft3 input")?;
    let nz_c = input.as_array().shape()[2];
    if nz_c != nz {
        return Err(PyValueError::new_err(
            "irfft3 input shape and nz are inconsistent",
        ));
    }
    let leto_view = leto::ArrayView3::from(input.as_array());
    let result = py.allow_threads(|| apollo_fft::ifft_3d_leto(leto_view));
    let nd_view = ndarray::ArrayView3::try_from(result.view()).map_err(|e| {
        PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
    })?;
    Ok(PyArray3::from_owned_array(py, nd_view.to_owned()))
}
