//! Fast Walsh-Hadamard Transform functions wrapping `apollo-fwht`.

use apollo_fwht::{FwhtPlan, FwhtPlan2D, FwhtPlan3D};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::support::{require_contiguous_1d, require_contiguous_2d, require_contiguous_3d};

/// Forward 1D Fast Walsh-Hadamard Transform. Length must be a power of two.
#[pyfunction]
pub(crate) fn fwht1<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_1d(&input, "fwht1 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.len();
    let plan = FwhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.allow_threads(|| {
        plan.forward(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    Ok(PyArray1::from_owned_array(py, result))
}

/// Inverse 1D Fast Walsh-Hadamard Transform. Scales by `1/n`.
#[pyfunction]
pub(crate) fn ifwht1<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_1d(&input, "ifwht1 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.len();
    let plan = FwhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.allow_threads(|| {
        plan.inverse(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    Ok(PyArray1::from_owned_array(py, result))
}

/// Forward 2D Fast Walsh-Hadamard Transform. Input must be square (N×N), N a power of two.
#[pyfunction]
pub(crate) fn fwht2<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    require_contiguous_2d(&input, "fwht2 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.nrows();
    let plan = FwhtPlan2D::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.allow_threads(|| {
        plan.forward(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    Ok(PyArray2::from_owned_array(py, result))
}

/// Inverse 2D Fast Walsh-Hadamard Transform. Scales by `1/N²`.
#[pyfunction]
pub(crate) fn ifwht2<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    require_contiguous_2d(&input, "ifwht2 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.nrows();
    let plan = FwhtPlan2D::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.allow_threads(|| {
        plan.inverse(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    Ok(PyArray2::from_owned_array(py, result))
}

/// Forward 3D Fast Walsh-Hadamard Transform. Input must be cubic (N×N×N), N a power of two.
#[pyfunction]
pub(crate) fn fwht3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'_, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    require_contiguous_3d(&input, "fwht3 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.shape()[0];
    let plan = FwhtPlan3D::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.allow_threads(|| {
        plan.forward(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    Ok(PyArray3::from_owned_array(py, result))
}

/// Inverse 3D Fast Walsh-Hadamard Transform. Scales by `1/N³`.
#[pyfunction]
pub(crate) fn ifwht3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'_, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    require_contiguous_3d(&input, "ifwht3 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.shape()[0];
    let plan = FwhtPlan3D::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.allow_threads(|| {
        plan.inverse(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    Ok(PyArray3::from_owned_array(py, result))
}
