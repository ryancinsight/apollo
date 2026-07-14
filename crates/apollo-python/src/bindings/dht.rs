//! Discrete Hartley Transform functions wrapping `apollo-dht`.

use apollo_dht::{DhtPlan, HartleySpectrum};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::support::{
    leto_array2_into_pyarray, leto_array3_into_pyarray, py_array1_slice, py_array2_to_leto,
    py_array3_to_leto, require_contiguous_1d, require_contiguous_2d, require_contiguous_3d,
    vec1_into_pyarray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};

/// Forward 1D Discrete Hartley Transform.
///
/// Returns the unnormalized DHT spectrum of length `n`. Inverse is `idht1`.
#[pyfunction]
pub(crate) fn dht1<'py>(py: Python<'py>, input: PyReadonlyArray1<f64>) -> PyResult<Py<PyAny>> {
    require_contiguous_1d(&input, "dht1 input")?;
    let signal = py_array1_slice(&input, "dht1 input")?.to_vec();
    let n = signal.len();
    let plan = DhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.detach(|| {
        plan.forward(&signal)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    vec1_into_pyarray(py, result.values().to_vec())
}

/// Inverse 1D Discrete Hartley Transform. Scales by `1/n`.
#[pyfunction]
pub(crate) fn idht1<'py>(py: Python<'py>, input: PyReadonlyArray1<f64>) -> PyResult<Py<PyAny>> {
    require_contiguous_1d(&input, "idht1 input")?;
    let signal = py_array1_slice(&input, "idht1 input")?.to_vec();
    let n = signal.len();
    let plan = DhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let spectrum = HartleySpectrum::new(signal);
    let recovered = py.detach(|| {
        plan.inverse(&spectrum)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    vec1_into_pyarray(py, recovered)
}

/// Forward 2D Discrete Hartley Transform. Input must be square (N×N).
#[pyfunction]
pub(crate) fn dht2<'py>(py: Python<'py>, input: PyReadonlyArray2<f64>) -> PyResult<Py<PyAny>> {
    require_contiguous_2d(&input, "dht2 input")?;
    let arr = py_array2_to_leto(&input, "dht2 input")?;
    let n = arr.shape()[0];
    let plan = DhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.detach(|| {
        plan.forward_2d(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    leto_array2_into_pyarray(py, result)
}

/// Inverse 2D Discrete Hartley Transform. Input must be square (N×N). Scales by `1/N²`.
#[pyfunction]
pub(crate) fn idht2<'py>(py: Python<'py>, input: PyReadonlyArray2<f64>) -> PyResult<Py<PyAny>> {
    require_contiguous_2d(&input, "idht2 input")?;
    let arr = py_array2_to_leto(&input, "idht2 input")?;
    let n = arr.shape()[0];
    let plan = DhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.detach(|| {
        plan.inverse_2d(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    leto_array2_into_pyarray(py, result)
}

/// Forward 3D Discrete Hartley Transform. Input must be cubic (N×N×N).
#[pyfunction]
pub(crate) fn dht3<'py>(py: Python<'py>, input: PyReadonlyArray3<f64>) -> PyResult<Py<PyAny>> {
    require_contiguous_3d(&input, "dht3 input")?;
    let arr = py_array3_to_leto(&input, "dht3 input")?;
    let n = arr.shape()[0];
    let plan = DhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.detach(|| {
        plan.forward_3d(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    leto_array3_into_pyarray(py, result)
}

/// Inverse 3D Discrete Hartley Transform. Input must be cubic (N×N×N). Scales by `1/N³`.
#[pyfunction]
pub(crate) fn idht3<'py>(py: Python<'py>, input: PyReadonlyArray3<f64>) -> PyResult<Py<PyAny>> {
    require_contiguous_3d(&input, "idht3 input")?;
    let arr = py_array3_to_leto(&input, "idht3 input")?;
    let n = arr.shape()[0];
    let plan = DhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = py.detach(|| {
        plan.inverse_3d(&arr)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    leto_array3_into_pyarray(py, result)
}
