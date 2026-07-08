//! Discrete Cosine/Sine Transform functions wrapping `apollo-dctdst`.

use apollo_dctdst::{dct2, dct3, dst2, dst3};
use pyo3::prelude::*;

use super::support::{py_array1_slice, require_contiguous_1d, vec1_into_pyarray, PyReadonlyArray1};

/// Forward 1D DCT-II (the "the DCT" as used by numpy/scipy).
///
/// Equivalent to `scipy.fft.dct(x, type=2, norm=None)` (unnormalized).
/// Inverse via `idct2_1d`.
#[pyfunction]
pub(crate) fn dct2_1d<'py>(py: Python<'py>, input: PyReadonlyArray1<f64>) -> PyResult<PyObject> {
    require_contiguous_1d(&input, "dct2_1d input")?;
    let signal = py_array1_slice(&input, "dct2_1d input")?.to_vec();
    let n = signal.len();
    let mut output = vec![0.0_f64; n];
    py.allow_threads(|| {
        dct2(&signal, &mut output);
    });
    vec1_into_pyarray(py, output)
}

/// Inverse 1D DCT-II (= DCT-III / N).
///
/// Equivalent to `scipy.fft.idct(x, type=2, norm=None)`.
#[pyfunction]
pub(crate) fn idct2_1d<'py>(py: Python<'py>, input: PyReadonlyArray1<f64>) -> PyResult<PyObject> {
    require_contiguous_1d(&input, "idct2_1d input")?;
    let signal = py_array1_slice(&input, "idct2_1d input")?.to_vec();
    let n = signal.len();
    let mut output = vec![0.0_f64; n];
    // DCT-III is the inverse of DCT-II up to N/2 scaling: DCT-III(DCT-II(x)) = (N/2) * x.
    // Therefore: x = DCT-III(X) * (2 / N).
    py.allow_threads(|| {
        dct3(&signal, &mut output);
        let scale = 2.0 / n as f64;
        output.iter_mut().for_each(|v| *v *= scale);
    });
    vec1_into_pyarray(py, output)
}

/// Forward 1D DST-II.
///
/// Equivalent to `scipy.fft.dst(x, type=2, norm=None)` (unnormalized).
#[pyfunction]
pub(crate) fn dst2_1d<'py>(py: Python<'py>, input: PyReadonlyArray1<f64>) -> PyResult<PyObject> {
    require_contiguous_1d(&input, "dst2_1d input")?;
    let signal = py_array1_slice(&input, "dst2_1d input")?.to_vec();
    let n = signal.len();
    let mut output = vec![0.0_f64; n];
    py.allow_threads(|| {
        dst2(&signal, &mut output);
    });
    vec1_into_pyarray(py, output)
}

/// Inverse 1D DST-II (= DST-III / N).
#[pyfunction]
pub(crate) fn idst2_1d<'py>(py: Python<'py>, input: PyReadonlyArray1<f64>) -> PyResult<PyObject> {
    require_contiguous_1d(&input, "idst2_1d input")?;
    let signal = py_array1_slice(&input, "idst2_1d input")?.to_vec();
    let n = signal.len();
    let mut output = vec![0.0_f64; n];
    // DST-III(DST-II(x)) = (N/2) * x; inverse: x = DST-III(X) * (2 / N).
    py.allow_threads(|| {
        dst3(&signal, &mut output);
        let scale = 2.0 / n as f64;
        output.iter_mut().for_each(|v| *v *= scale);
    });
    vec1_into_pyarray(py, output)
}
