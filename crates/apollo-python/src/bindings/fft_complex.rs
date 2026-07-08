//! Complex-to-complex FFT functions and FFT frequency/shift utilities
//! wrapping `apollo-fft`.

use apollo_fft::{
    fft_1d_complex_inplace, fft_2d_complex_inplace, fft_3d_complex_inplace, fftfreq, fftshift,
    ifft_1d_complex_inplace, ifft_2d_complex_inplace, ifft_3d_complex_inplace, ifftshift, rfftfreq,
    Complex64,
};
use pyo3::prelude::*;

use super::support::{
    leto_array1_into_pyarray, leto_array2_into_pyarray, leto_array3_into_pyarray, py_array1_slice,
    py_array1_to_leto, py_array2_to_leto, py_array3_to_leto, require_contiguous_1d,
    require_contiguous_2d, require_contiguous_3d, vec1_into_pyarray, PyReadonlyArray1,
    PyReadonlyArray2, PyReadonlyArray3,
};

// ── FFT utility functions ─────────────────────────────────────────────────────

/// Frequency bin centers for a length-`n` complex DFT with sample spacing `d`.
///
/// Numpy-compatible: `fftfreq(n, d)` returns bins `[0, 1/nd, …, −1/nd, …]`.
/// For `n=0` returns an empty array.
#[pyfunction]
#[pyo3(signature = (n, d=1.0))]
pub(crate) fn fftfreq_py<'py>(py: Python<'py>, n: usize, d: f64) -> PyResult<PyObject> {
    vec1_into_pyarray(py, fftfreq(n, d))
}

/// Frequency bin centers for a length-`n` real-input FFT with sample spacing `d`.
///
/// Returns `n/2 + 1` non-negative bins. Numpy-compatible `rfftfreq(n, d)`.
#[pyfunction]
#[pyo3(signature = (n, d=1.0))]
pub(crate) fn rfftfreq_py<'py>(py: Python<'py>, n: usize, d: f64) -> PyResult<PyObject> {
    vec1_into_pyarray(py, rfftfreq(n, d))
}

/// Shift the zero-frequency component to the center of the spectrum.
///
/// Numpy-compatible `fftshift`. Accepts 1-D float64 arrays.
#[pyfunction]
pub(crate) fn fftshift_py<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let owned = py_array1_slice(&input, "fftshift input")?.to_vec();
    vec1_into_pyarray(py, fftshift(&owned))
}

/// Inverse `fftshift`: move zero-frequency back to bin 0.
///
/// Numpy-compatible `ifftshift`. Accepts 1-D float64 arrays.
#[pyfunction]
pub(crate) fn ifftshift_py<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let owned = py_array1_slice(&input, "ifftshift input")?.to_vec();
    vec1_into_pyarray(py, ifftshift(&owned))
}

// ── Complex-to-complex FFT ────────────────────────────────────────────────────

/// Complex-to-complex forward 1D FFT. Accepts complex128 input, returns complex128.
#[pyfunction]
pub(crate) fn fft_complex1<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<Complex64>,
) -> PyResult<PyObject> {
    require_contiguous_1d(&input, "fft_complex1 input")?;
    let mut output = py_array1_to_leto(&input, "fft_complex1 input")?;
    py.allow_threads(|| {
        fft_1d_complex_inplace(&mut output);
    });
    leto_array1_into_pyarray(py, output)
}

/// Complex-to-complex inverse 1D FFT. Accepts complex128, returns complex128.
#[pyfunction]
pub(crate) fn ifft_complex1<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<Complex64>,
) -> PyResult<PyObject> {
    require_contiguous_1d(&input, "ifft_complex1 input")?;
    let mut output = py_array1_to_leto(&input, "ifft_complex1 input")?;
    py.allow_threads(|| {
        ifft_1d_complex_inplace(&mut output);
    });
    leto_array1_into_pyarray(py, output)
}

/// Complex-to-complex forward 2D FFT.
#[pyfunction]
pub(crate) fn fft_complex2<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<Complex64>,
) -> PyResult<PyObject> {
    require_contiguous_2d(&input, "fft_complex2 input")?;
    let mut output = py_array2_to_leto(&input, "fft_complex2 input")?;
    py.allow_threads(|| {
        fft_2d_complex_inplace(&mut output);
    });
    leto_array2_into_pyarray(py, output)
}

/// Complex-to-complex inverse 2D FFT.
#[pyfunction]
pub(crate) fn ifft_complex2<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<Complex64>,
) -> PyResult<PyObject> {
    require_contiguous_2d(&input, "ifft_complex2 input")?;
    let mut output = py_array2_to_leto(&input, "ifft_complex2 input")?;
    py.allow_threads(|| {
        ifft_2d_complex_inplace(&mut output);
    });
    leto_array2_into_pyarray(py, output)
}

/// Complex-to-complex forward 3D FFT.
#[pyfunction]
pub(crate) fn fft_complex3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<Complex64>,
) -> PyResult<PyObject> {
    require_contiguous_3d(&input, "fft_complex3 input")?;
    let mut output = py_array3_to_leto(&input, "fft_complex3 input")?;
    py.allow_threads(|| {
        fft_3d_complex_inplace(&mut output);
    });
    leto_array3_into_pyarray(py, output)
}

/// Complex-to-complex inverse 3D FFT.
#[pyfunction]
pub(crate) fn ifft_complex3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<Complex64>,
) -> PyResult<PyObject> {
    require_contiguous_3d(&input, "ifft_complex3 input")?;
    let mut output = py_array3_to_leto(&input, "ifft_complex3 input")?;
    py.allow_threads(|| {
        ifft_3d_complex_inplace(&mut output);
    });
    leto_array3_into_pyarray(py, output)
}
