//! Non-uniform FFT functions (type-1/type-2, direct and Kaiser-Bessel fast
//! paths) wrapping `apollo-nufft`.

use apollo_fft::Complex64;
use apollo_nufft::{
    nufft_type1_1d, nufft_type1_1d_fast, nufft_type1_3d, nufft_type1_3d_fast, nufft_type2_1d,
    nufft_type2_1d_fast, UniformDomain1D, UniformGrid3D, DEFAULT_NUFFT_KERNEL_WIDTH,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::support::{
    leto_array1_into_pyarray, leto_array3_into_pyarray, py_array1_slice, py_array1_to_leto,
    py_array2_slice, require_contiguous_1d, require_contiguous_2d, vec1_into_pyarray,
    PyReadonlyArray1, PyReadonlyArray2,
};

/// Exact direct 1D type-1 NUFFT.
#[pyfunction(name = "nufft_type1_1d")]
#[pyo3(signature = (positions, values, dx, n_out=None))]
pub(crate) fn nufft_type1_1d_py<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<Complex64>,
    dx: f64,
    n_out: Option<usize>,
) -> PyResult<Py<PyAny>> {
    require_contiguous_1d(&positions, "nufft_type1_1d positions")?;
    require_contiguous_1d(&values, "nufft_type1_1d values")?;
    let positions = py_array1_slice(&positions, "nufft_type1_1d positions")?;
    let values = py_array1_slice(&values, "nufft_type1_1d values")?;
    let domain = UniformDomain1D::new(n_out.unwrap_or(values.len()), dx)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let result = py.detach(|| nufft_type1_1d(positions, values, domain));
    leto_array1_into_pyarray(py, result)
}

/// Exact direct 1D type-2 NUFFT.
#[pyfunction(name = "nufft_type2_1d")]
pub(crate) fn nufft_type2_1d_py<'py>(
    py: Python<'py>,
    fourier_coeffs: PyReadonlyArray1<Complex64>,
    positions: PyReadonlyArray1<f64>,
    dx: f64,
) -> PyResult<Py<PyAny>> {
    require_contiguous_1d(&fourier_coeffs, "nufft_type2_1d fourier_coeffs")?;
    require_contiguous_1d(&positions, "nufft_type2_1d positions")?;
    let coeffs = py_array1_to_leto(&fourier_coeffs, "nufft_type2_1d fourier_coeffs")?;
    let positions = py_array1_slice(&positions, "nufft_type2_1d positions")?;
    let domain = UniformDomain1D::new(coeffs.size(), dx)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let result = py.detach(|| nufft_type2_1d(&coeffs, positions, domain));
    vec1_into_pyarray(py, result)
}

/// Exact direct 3D type-1 NUFFT.
#[pyfunction(name = "nufft_type1_3d")]
pub(crate) fn nufft_type1_3d_py<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<f64>,
    values: PyReadonlyArray1<Complex64>,
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
) -> PyResult<Py<PyAny>> {
    require_contiguous_2d(&positions, "nufft_type1_3d positions")?;
    require_contiguous_1d(&values, "nufft_type1_3d values")?;
    let shape = positions.shape();
    if shape[1] != 3 {
        return Err(PyValueError::new_err(
            "nufft_type1_3d positions must have shape (n_samples, 3)",
        ));
    }
    let positions_slice = py_array2_slice(&positions, "nufft_type1_3d positions")?;
    let values_slice = py_array1_slice(&values, "nufft_type1_3d values")?;
    if shape[0] != values_slice.len() {
        return Err(PyValueError::new_err(
            "nufft_type1_3d positions/value length mismatch",
        ));
    }
    let tuples: Vec<(f64, f64, f64)> = positions_slice
        .chunks_exact(3)
        .map(|row| (row[0], row[1], row[2]))
        .collect();
    let grid = UniformGrid3D::new(nx, ny, nz, dx, dy, dz)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let result = py.detach(|| nufft_type1_3d(&tuples, values_slice, grid));
    leto_array3_into_pyarray(py, result)
}

/// Fast 1D type-1 NUFFT using Kaiser-Bessel spreading.
#[pyfunction(name = "nufft_type1_1d_fast")]
#[pyo3(signature = (positions, values, dx, n_out=None, kernel_width=DEFAULT_NUFFT_KERNEL_WIDTH))]
pub(crate) fn nufft_type1_1d_fast_py<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<Complex64>,
    dx: f64,
    n_out: Option<usize>,
    kernel_width: usize,
) -> PyResult<Py<PyAny>> {
    require_contiguous_1d(&positions, "nufft_type1_1d_fast positions")?;
    require_contiguous_1d(&values, "nufft_type1_1d_fast values")?;
    let positions = py_array1_slice(&positions, "nufft_type1_1d_fast positions")?;
    let values = py_array1_slice(&values, "nufft_type1_1d_fast values")?;
    let domain = UniformDomain1D::new(n_out.unwrap_or(values.len()), dx)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let result = py.detach(|| nufft_type1_1d_fast(positions, values, domain, kernel_width));
    leto_array1_into_pyarray(py, result)
}

/// Fast 1D type-2 NUFFT using Kaiser-Bessel spreading.
#[pyfunction(name = "nufft_type2_1d_fast")]
#[pyo3(signature = (fourier_coeffs, positions, dx, kernel_width=DEFAULT_NUFFT_KERNEL_WIDTH))]
pub(crate) fn nufft_type2_1d_fast_py<'py>(
    py: Python<'py>,
    fourier_coeffs: PyReadonlyArray1<Complex64>,
    positions: PyReadonlyArray1<f64>,
    dx: f64,
    kernel_width: usize,
) -> PyResult<Py<PyAny>> {
    require_contiguous_1d(&fourier_coeffs, "nufft_type2_1d_fast fourier_coeffs")?;
    require_contiguous_1d(&positions, "nufft_type2_1d_fast positions")?;
    let coeffs = py_array1_to_leto(&fourier_coeffs, "nufft_type2_1d_fast fourier_coeffs")?;
    let positions = py_array1_slice(&positions, "nufft_type2_1d_fast positions")?;
    let domain = UniformDomain1D::new(coeffs.size(), dx)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let result = py.detach(|| nufft_type2_1d_fast(&coeffs, positions, domain, kernel_width));
    vec1_into_pyarray(py, result)
}

/// Fast 3D type-1 NUFFT using Kaiser-Bessel spreading.
#[pyfunction(name = "nufft_type1_3d_fast")]
#[pyo3(signature = (positions, values, nx, ny, nz, dx, dy, dz, kernel_width=DEFAULT_NUFFT_KERNEL_WIDTH))]
pub(crate) fn nufft_type1_3d_fast_py<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<f64>,
    values: PyReadonlyArray1<Complex64>,
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
    kernel_width: usize,
) -> PyResult<Py<PyAny>> {
    require_contiguous_2d(&positions, "nufft_type1_3d_fast positions")?;
    require_contiguous_1d(&values, "nufft_type1_3d_fast values")?;
    let shape = positions.shape();
    if shape[1] != 3 {
        return Err(PyValueError::new_err(
            "nufft_type1_3d_fast positions must have shape (n_samples, 3)",
        ));
    }
    let positions_slice = py_array2_slice(&positions, "nufft_type1_3d_fast positions")?;
    let values_slice = py_array1_slice(&values, "nufft_type1_3d_fast values")?;
    if shape[0] != values_slice.len() {
        return Err(PyValueError::new_err(
            "nufft_type1_3d_fast positions/value length mismatch",
        ));
    }
    let tuples: Vec<(f64, f64, f64)> = positions_slice
        .chunks_exact(3)
        .map(|row| (row[0], row[1], row[2]))
        .collect();
    let grid = UniformGrid3D::new(nx, ny, nz, dx, dy, dz)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let result = py.detach(|| nufft_type1_3d_fast(&tuples, values_slice, grid, kernel_width));
    leto_array3_into_pyarray(py, result)
}
