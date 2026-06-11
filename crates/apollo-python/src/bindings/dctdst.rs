//! Discrete Cosine/Sine Transform functions wrapping `apollo-dctdst`.

use apollo_dctdst::{dct2, dct3, dst2, dst3};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::support::require_contiguous_1d;

/// Forward 1D DCT-II (the "the DCT" as used by numpy/scipy).
///
/// Equivalent to `scipy.fft.dct(x, type=2, norm=None)` (unnormalized).
/// Inverse via `idct2_1d`.
#[pyfunction]
pub(crate) fn dct2_1d<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_1d(&input, "dct2_1d input")?;
    let arr = input.as_array();
    let n = arr.len();
    let signal: Vec<f64> = arr.iter().copied().collect();
    let mut output = vec![0.0_f64; n];
    py.allow_threads(|| {
        dct2(&signal, &mut output);
    });
    Ok(PyArray1::from_vec(py, output))
}

/// Inverse 1D DCT-II (= DCT-III / N).
///
/// Equivalent to `scipy.fft.idct(x, type=2, norm=None)`.
#[pyfunction]
pub(crate) fn idct2_1d<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_1d(&input, "idct2_1d input")?;
    let arr = input.as_array();
    let n = arr.len();
    let signal: Vec<f64> = arr.iter().copied().collect();
    let mut output = vec![0.0_f64; n];
    // DCT-III is the inverse of DCT-II up to N/2 scaling: DCT-III(DCT-II(x)) = (N/2) * x.
    // Therefore: x = DCT-III(X) * (2 / N).
    py.allow_threads(|| {
        dct3(&signal, &mut output);
        let scale = 2.0 / n as f64;
        output.iter_mut().for_each(|v| *v *= scale);
    });
    Ok(PyArray1::from_vec(py, output))
}

/// Forward 1D DST-II.
///
/// Equivalent to `scipy.fft.dst(x, type=2, norm=None)` (unnormalized).
#[pyfunction]
pub(crate) fn dst2_1d<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_1d(&input, "dst2_1d input")?;
    let arr = input.as_array();
    let n = arr.len();
    let signal: Vec<f64> = arr.iter().copied().collect();
    let mut output = vec![0.0_f64; n];
    py.allow_threads(|| {
        dst2(&signal, &mut output);
    });
    Ok(PyArray1::from_vec(py, output))
}

/// Inverse 1D DST-II (= DST-III / N).
#[pyfunction]
pub(crate) fn idst2_1d<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_1d(&input, "idst2_1d input")?;
    let arr = input.as_array();
    let n = arr.len();
    let signal: Vec<f64> = arr.iter().copied().collect();
    let mut output = vec![0.0_f64; n];
    // DST-III(DST-II(x)) = (N/2) * x; inverse: x = DST-III(X) * (2 / N).
    py.allow_threads(|| {
        dst3(&signal, &mut output);
        let scale = 2.0 / n as f64;
        output.iter_mut().for_each(|v| *v *= scale);
    });
    Ok(PyArray1::from_vec(py, output))
}
