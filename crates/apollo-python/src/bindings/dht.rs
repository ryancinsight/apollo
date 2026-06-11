//! Discrete Hartley Transform functions wrapping `apollo-dht`.

use apollo_dht::{DhtPlan, HartleySpectrum};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::support::{require_contiguous_1d, require_contiguous_2d, require_contiguous_3d};

/// Forward 1D Discrete Hartley Transform.
///
/// Returns the unnormalized DHT spectrum of length `n`. Inverse is `idht1`.
#[pyfunction]
pub(crate) fn dht1<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_1d(&input, "dht1 input")?;
    let arr = input.as_array();
    let n = arr.len();
    let plan = DhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let signal: Vec<f64> = arr.iter().copied().collect();
    let result = py.allow_threads(|| {
        plan.forward(&signal)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    Ok(PyArray1::from_vec(py, result.values().to_vec()))
}

/// Inverse 1D Discrete Hartley Transform. Scales by `1/n`.
#[pyfunction]
pub(crate) fn idht1<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_1d(&input, "idht1 input")?;
    let arr = input.as_array();
    let n = arr.len();
    let plan = DhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let signal: Vec<f64> = arr.iter().copied().collect();
    let spectrum = HartleySpectrum::new(signal);
    let recovered = py.allow_threads(|| {
        plan.inverse(&spectrum)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    Ok(PyArray1::from_vec(py, recovered))
}

/// Forward 2D Discrete Hartley Transform. Input must be square (N×N).
#[pyfunction]
pub(crate) fn dht2<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    require_contiguous_2d(&input, "dht2 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.nrows();
    let plan = DhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.allow_threads(|| {
        plan.forward_2d(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    Ok(PyArray2::from_owned_array(py, result))
}

/// Inverse 2D Discrete Hartley Transform. Input must be square (N×N). Scales by `1/N²`.
#[pyfunction]
pub(crate) fn idht2<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    require_contiguous_2d(&input, "idht2 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.nrows();
    let plan = DhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.allow_threads(|| {
        plan.inverse_2d(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    Ok(PyArray2::from_owned_array(py, result))
}

/// Forward 3D Discrete Hartley Transform. Input must be cubic (N×N×N).
#[pyfunction]
pub(crate) fn dht3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'_, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    require_contiguous_3d(&input, "dht3 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.shape()[0];
    let plan = DhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.allow_threads(|| {
        plan.forward_3d(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    Ok(PyArray3::from_owned_array(py, result))
}

/// Inverse 3D Discrete Hartley Transform. Input must be cubic (N×N×N). Scales by `1/N³`.
#[pyfunction]
pub(crate) fn idht3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'_, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    require_contiguous_3d(&input, "idht3 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.shape()[0];
    let plan = DhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.allow_threads(|| {
        plan.inverse_3d(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    Ok(PyArray3::from_owned_array(py, result))
}
