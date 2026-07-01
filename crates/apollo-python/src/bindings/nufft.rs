//! Non-uniform FFT functions (type-1/type-2, direct and Kaiser-Bessel fast
//! paths) wrapping `apollo-nufft`.

use apollo_fft::Complex64;
use apollo_nufft::{
    nufft_type1_1d, nufft_type1_1d_fast, nufft_type1_3d, nufft_type1_3d_fast, nufft_type2_1d,
    nufft_type2_1d_fast, UniformDomain1D, UniformGrid3D, DEFAULT_NUFFT_KERNEL_WIDTH,
};
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::support::{require_contiguous_1d, require_contiguous_2d};

/// Exact direct 1D type-1 NUFFT.
#[pyfunction(name = "nufft_type1_1d")]
#[pyo3(signature = (positions, values, dx, n_out=None))]
pub(crate) fn nufft_type1_1d_py<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<Complex64>,
    dx: f64,
    n_out: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    require_contiguous_1d(&positions, "nufft_type1_1d positions")?;
    require_contiguous_1d(&values, "nufft_type1_1d values")?;
    let positions = positions.as_array().to_owned();
    let values = values.as_array().to_owned();
    let domain = UniformDomain1D::new(n_out.unwrap_or(values.len()), dx)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let result = py.allow_threads(|| {
        nufft_type1_1d(
            positions
                .as_slice()
                .expect("owned positions are contiguous"),
            values.as_slice().expect("owned values are contiguous"),
            domain,
        )
    });
    Ok(PyArray1::from_owned_array(py, ndarray::Array1::try_from(result).expect("leto result is C-contiguous")))
}

/// Exact direct 1D type-2 NUFFT.
#[pyfunction(name = "nufft_type2_1d")]
pub(crate) fn nufft_type2_1d_py<'py>(
    py: Python<'py>,
    fourier_coeffs: PyReadonlyArray1<Complex64>,
    positions: PyReadonlyArray1<f64>,
    dx: f64,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    require_contiguous_1d(&fourier_coeffs, "nufft_type2_1d fourier_coeffs")?;
    require_contiguous_1d(&positions, "nufft_type2_1d positions")?;
    let coeffs = fourier_coeffs.as_array().to_owned();
    let positions = positions.as_array().to_owned();
    let domain = UniformDomain1D::new(coeffs.len(), dx)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let result = py.allow_threads(|| {
        nufft_type2_1d(
            &leto::Array1::from(coeffs),
            positions
                .as_slice()
                .expect("owned positions are contiguous"),
            domain,
        )
    });
    Ok(result.into_pyarray(py))
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
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    require_contiguous_2d(&positions, "nufft_type1_3d positions")?;
    require_contiguous_1d(&values, "nufft_type1_3d values")?;
    let positions = positions.as_array();
    if positions.ncols() != 3 {
        return Err(PyValueError::new_err(
            "nufft_type1_3d positions must have shape (n_samples, 3)",
        ));
    }
    if positions.nrows() != values.as_array().len() {
        return Err(PyValueError::new_err(
            "nufft_type1_3d positions/value length mismatch",
        ));
    }
    let tuples: Vec<(f64, f64, f64)> = positions
        .rows()
        .into_iter()
        .map(|row| (row[0], row[1], row[2]))
        .collect();
    let owned_values = values.as_array().to_owned();
    let grid = UniformGrid3D::new(nx, ny, nz, dx, dy, dz)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let result = py.allow_threads(|| {
        nufft_type1_3d(
            &tuples,
            owned_values
                .as_slice()
                .expect("owned values are contiguous"),
            grid,
        )
    });
    Ok(PyArray3::from_owned_array(py, ndarray::Array3::try_from(result).expect("leto result is C-contiguous")))
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
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    require_contiguous_1d(&positions, "nufft_type1_1d_fast positions")?;
    require_contiguous_1d(&values, "nufft_type1_1d_fast values")?;
    let positions = positions.as_array().to_owned();
    let values = values.as_array().to_owned();
    let domain = UniformDomain1D::new(n_out.unwrap_or(values.len()), dx)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let result = py.allow_threads(|| {
        nufft_type1_1d_fast(
            positions
                .as_slice()
                .expect("owned positions are contiguous"),
            values.as_slice().expect("owned values are contiguous"),
            domain,
            kernel_width,
        )
    });
    Ok(PyArray1::from_owned_array(py, ndarray::Array1::try_from(result).expect("leto result is C-contiguous")))
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
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    require_contiguous_1d(&fourier_coeffs, "nufft_type2_1d_fast fourier_coeffs")?;
    require_contiguous_1d(&positions, "nufft_type2_1d_fast positions")?;
    let coeffs = fourier_coeffs.as_array().to_owned();
    let positions = positions.as_array().to_owned();
    let domain = UniformDomain1D::new(coeffs.len(), dx)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let result = py.allow_threads(|| {
        nufft_type2_1d_fast(
            &leto::Array1::from(coeffs),
            positions
                .as_slice()
                .expect("owned positions are contiguous"),
            domain,
            kernel_width,
        )
    });
    Ok(result.into_pyarray(py))
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
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    require_contiguous_2d(&positions, "nufft_type1_3d_fast positions")?;
    require_contiguous_1d(&values, "nufft_type1_3d_fast values")?;
    let positions = positions.as_array();
    if positions.ncols() != 3 {
        return Err(PyValueError::new_err(
            "nufft_type1_3d_fast positions must have shape (n_samples, 3)",
        ));
    }
    if positions.nrows() != values.as_array().len() {
        return Err(PyValueError::new_err(
            "nufft_type1_3d_fast positions/value length mismatch",
        ));
    }
    let tuples: Vec<(f64, f64, f64)> = positions
        .rows()
        .into_iter()
        .map(|row| (row[0], row[1], row[2]))
        .collect();
    let owned_values = values.as_array().to_owned();
    let grid = UniformGrid3D::new(nx, ny, nz, dx, dy, dz)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let result = py.allow_threads(|| {
        nufft_type1_3d_fast(
            &tuples,
            owned_values
                .as_slice()
                .expect("owned values are contiguous"),
            grid,
            kernel_width,
        )
    });
    Ok(PyArray3::from_owned_array(py, ndarray::Array3::try_from(result).expect("leto result is C-contiguous")))
}
