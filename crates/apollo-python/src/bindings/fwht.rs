//! Fast Walsh-Hadamard Transform functions wrapping `apollo-fwht`.

use apollo_fwht::{FwhtPlan, FwhtPlan2D, FwhtPlan3D};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::support::{
    leto_array1_into_pyarray, leto_array2_into_pyarray, leto_array3_into_pyarray,
    py_array1_to_leto, py_array2_to_leto, py_array3_to_leto, require_contiguous_1d,
    require_contiguous_2d, require_contiguous_3d, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3,
};

/// Forward 1D Fast Walsh-Hadamard Transform. Length must be a power of two.
#[pyfunction]
pub(crate) fn fwht1<'py>(py: Python<'py>, input: PyReadonlyArray1<f64>) -> PyResult<Py<PyAny>> {
    require_contiguous_1d(&input, "fwht1 input")?;
    let arr = py_array1_to_leto(&input, "fwht1 input")?;
    let n = arr.size();
    let plan = FwhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.detach(|| {
        plan.forward(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    leto_array1_into_pyarray(py, result)
}

/// Inverse 1D Fast Walsh-Hadamard Transform. Scales by `1/n`.
#[pyfunction]
pub(crate) fn ifwht1<'py>(py: Python<'py>, input: PyReadonlyArray1<f64>) -> PyResult<Py<PyAny>> {
    require_contiguous_1d(&input, "ifwht1 input")?;
    let arr = py_array1_to_leto(&input, "ifwht1 input")?;
    let n = arr.size();
    let plan = FwhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.detach(|| {
        plan.inverse(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    leto_array1_into_pyarray(py, result)
}

/// Forward 2D Fast Walsh-Hadamard Transform. Input must be square (N×N), N a power of two.
#[pyfunction]
pub(crate) fn fwht2<'py>(py: Python<'py>, input: PyReadonlyArray2<f64>) -> PyResult<Py<PyAny>> {
    require_contiguous_2d(&input, "fwht2 input")?;
    let arr = py_array2_to_leto(&input, "fwht2 input")?;
    let n = arr.shape()[0];
    let plan = FwhtPlan2D::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.detach(|| {
        plan.forward(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    leto_array2_into_pyarray(py, result)
}

/// Inverse 2D Fast Walsh-Hadamard Transform. Scales by `1/N²`.
#[pyfunction]
pub(crate) fn ifwht2<'py>(py: Python<'py>, input: PyReadonlyArray2<f64>) -> PyResult<Py<PyAny>> {
    require_contiguous_2d(&input, "ifwht2 input")?;
    let arr = py_array2_to_leto(&input, "ifwht2 input")?;
    let n = arr.shape()[0];
    let plan = FwhtPlan2D::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.detach(|| {
        plan.inverse(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    leto_array2_into_pyarray(py, result)
}

/// Forward 3D Fast Walsh-Hadamard Transform. Input must be cubic (N×N×N), N a power of two.
#[pyfunction]
pub(crate) fn fwht3<'py>(py: Python<'py>, input: PyReadonlyArray3<f64>) -> PyResult<Py<PyAny>> {
    require_contiguous_3d(&input, "fwht3 input")?;
    let arr = py_array3_to_leto(&input, "fwht3 input")?;
    let n = arr.shape()[0];
    let plan = FwhtPlan3D::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.detach(|| {
        plan.forward(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    leto_array3_into_pyarray(py, result)
}

/// Inverse 3D Fast Walsh-Hadamard Transform. Scales by `1/N³`.
#[pyfunction]
pub(crate) fn ifwht3<'py>(py: Python<'py>, input: PyReadonlyArray3<f64>) -> PyResult<Py<PyAny>> {
    require_contiguous_3d(&input, "ifwht3 input")?;
    let arr = py_array3_to_leto(&input, "ifwht3 input")?;
    let n = arr.shape()[0];
    let plan = FwhtPlan3D::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.detach(|| {
        plan.inverse(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    leto_array3_into_pyarray(py, result)
}
