//! Complex-to-complex FFT functions and FFT frequency/shift utilities
//! wrapping `apollo-fft`.

use apollo_fft::{
    fft_1d_complex_inplace, fft_2d_complex_inplace, fft_3d_complex_inplace, fftfreq, fftshift,
    ifft_1d_complex_inplace, ifft_2d_complex_inplace, ifft_3d_complex_inplace, ifftshift, rfftfreq,
    Complex64,
};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

use super::support::{require_contiguous_1d, require_contiguous_2d, require_contiguous_3d};

// ── FFT utility functions ─────────────────────────────────────────────────────

/// Frequency bin centers for a length-`n` complex DFT with sample spacing `d`.
///
/// Numpy-compatible: `fftfreq(n, d)` returns bins `[0, 1/nd, …, −1/nd, …]`.
/// For `n=0` returns an empty array.
#[pyfunction]
#[pyo3(signature = (n, d=1.0))]
pub(crate) fn fftfreq_py<'py>(py: Python<'py>, n: usize, d: f64) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, fftfreq(n, d))
}

/// Frequency bin centers for a length-`n` real-input FFT with sample spacing `d`.
///
/// Returns `n/2 + 1` non-negative bins. Numpy-compatible `rfftfreq(n, d)`.
#[pyfunction]
#[pyo3(signature = (n, d=1.0))]
pub(crate) fn rfftfreq_py<'py>(py: Python<'py>, n: usize, d: f64) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, rfftfreq(n, d))
}

/// Shift the zero-frequency component to the center of the spectrum.
///
/// Numpy-compatible `fftshift`. Accepts 1-D float64 arrays.
#[pyfunction]
pub(crate) fn fftshift_py<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let owned: Vec<f64> = input.as_array().iter().copied().collect();
    PyArray1::from_vec(py, fftshift(&owned))
}

/// Inverse `fftshift`: move zero-frequency back to bin 0.
///
/// Numpy-compatible `ifftshift`. Accepts 1-D float64 arrays.
#[pyfunction]
pub(crate) fn ifftshift_py<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let owned: Vec<f64> = input.as_array().iter().copied().collect();
    PyArray1::from_vec(py, ifftshift(&owned))
}

// ── Complex-to-complex FFT ────────────────────────────────────────────────────

/// Complex-to-complex forward 1D FFT. Accepts complex128 input, returns complex128.
#[pyfunction]
pub(crate) fn fft_complex1<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, Complex64>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    require_contiguous_1d(&input, "fft_complex1 input")?;
    let mut output = leto::Array1::from(input.as_array().to_owned());
    py.allow_threads(|| {
        fft_1d_complex_inplace(&mut output);
    });
    Ok(PyArray1::from_owned_array(py, ndarray::Array1::try_from(output).expect("leto result is C-contiguous")))
}

/// Complex-to-complex inverse 1D FFT. Accepts complex128, returns complex128.
#[pyfunction]
pub(crate) fn ifft_complex1<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, Complex64>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    require_contiguous_1d(&input, "ifft_complex1 input")?;
    let mut output = leto::Array1::from(input.as_array().to_owned());
    py.allow_threads(|| {
        ifft_1d_complex_inplace(&mut output);
    });
    Ok(PyArray1::from_owned_array(py, ndarray::Array1::try_from(output).expect("leto result is C-contiguous")))
}

/// Complex-to-complex forward 2D FFT.
#[pyfunction]
pub(crate) fn fft_complex2<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'_, Complex64>,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    require_contiguous_2d(&input, "fft_complex2 input")?;
    let mut output = leto::Array2::from(input.as_array().to_owned());
    py.allow_threads(|| {
        fft_2d_complex_inplace(&mut output);
    });
    Ok(PyArray2::from_owned_array(py, ndarray::Array2::try_from(output).expect("leto result is C-contiguous")))
}

/// Complex-to-complex inverse 2D FFT.
#[pyfunction]
pub(crate) fn ifft_complex2<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'_, Complex64>,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    require_contiguous_2d(&input, "ifft_complex2 input")?;
    let mut output = leto::Array2::from(input.as_array().to_owned());
    py.allow_threads(|| {
        ifft_2d_complex_inplace(&mut output);
    });
    Ok(PyArray2::from_owned_array(py, ndarray::Array2::try_from(output).expect("leto result is C-contiguous")))
}

/// Complex-to-complex forward 3D FFT.
#[pyfunction]
pub(crate) fn fft_complex3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'_, Complex64>,
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    require_contiguous_3d(&input, "fft_complex3 input")?;
    let mut output = leto::Array3::from(input.as_array().to_owned());
    py.allow_threads(|| {
        fft_3d_complex_inplace(&mut output);
    });
    Ok(PyArray3::from_owned_array(py, ndarray::Array3::try_from(output).expect("leto result is C-contiguous")))
}

/// Complex-to-complex inverse 3D FFT.
#[pyfunction]
pub(crate) fn ifft_complex3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'_, Complex64>,
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    require_contiguous_3d(&input, "ifft_complex3 input")?;
    let mut output = leto::Array3::from(input.as_array().to_owned());
    py.allow_threads(|| {
        ifft_3d_complex_inplace(&mut output);
    });
    Ok(PyArray3::from_owned_array(py, ndarray::Array3::try_from(output).expect("leto result is C-contiguous")))
}
