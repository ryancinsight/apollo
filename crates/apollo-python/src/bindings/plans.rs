//! Reusable FFT plan classes (`FftPlan1D`, `FftPlan2D`, `FftPlan3D`)
//! wrapping `apollo-fft` plan execution.

use apollo_fft::{
    f16, fft_1d_complex_inplace, fft_2d_complex_inplace, ifft_1d_complex_inplace,
    ifft_2d_complex_inplace, Complex32, Complex64, PrecisionProfile, Shape1D, Shape2D, Shape3D,
    StoragePrecision,
};
use numpy::{
    PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::support::{
    parse_precision, require_contiguous_1d, require_contiguous_2d, require_contiguous_3d,
    require_profile_matches_f32, require_profile_matches_f64,
};

/// Python wrapper for a reusable 1D FFT plan.
#[pyclass(name = "FftPlan1D")]
pub(crate) struct PyFftPlan1D {
    #[allow(dead_code)]
    shape: Shape1D,
    profile: PrecisionProfile,
}

#[pymethods]
impl PyFftPlan1D {
    #[new]
    #[pyo3(signature = (n, precision=None))]
    fn new(n: usize, precision: Option<&str>) -> PyResult<Self> {
        let profile = parse_precision(precision)?;
        let shape = Shape1D::new(n).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { shape, profile })
    }

    fn fft<'py>(&self, py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<PyObject> {
        if let Ok(input64) = input.extract::<PyReadonlyArray1<f64>>() {
            require_contiguous_1d(&input64, "fft input")?;
            require_profile_matches_f64(self.profile, "fft")?;
            let leto_view = leto::ArrayView1::from(input64.as_array());
            let result = py.allow_threads(|| apollo_fft::fft_1d_leto(leto_view));
            let nd_view = ndarray::ArrayView1::try_from(result.view()).map_err(|e| {
                PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
            })?;
            Ok(PyArray1::from_owned_array(py, nd_view.to_owned())
                .into_any()
                .unbind())
        } else {
            match self.profile.storage {
                StoragePrecision::F16 => {
                    let input32 = input.extract::<PyReadonlyArray1<f32>>()?;
                    require_contiguous_1d(&input32, "fft input")?;
                    let owned = input32.as_array().mapv(f16::from_f32);
                    let result = py.allow_threads(|| apollo_fft::fft_1d_array_typed(&owned));
                    Ok(PyArray1::from_owned_array(py, result).into_any().unbind())
                }
                _ => {
                    let input32 = input.extract::<PyReadonlyArray1<f32>>()?;
                    require_contiguous_1d(&input32, "fft input")?;
                    require_profile_matches_f32(self.profile, "fft")?;
                    let leto_view = leto::ArrayView1::from(input32.as_array());
                    let result =
                        py.allow_threads(|| apollo_fft::fft_1d_leto_typed::<f32>(leto_view));
                    let nd_view = ndarray::ArrayView1::try_from(result.view()).map_err(|e| {
                        PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
                    })?;
                    Ok(PyArray1::from_owned_array(py, nd_view.to_owned())
                        .into_any()
                        .unbind())
                }
            }
        }
    }

    fn ifft<'py>(&self, py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<PyObject> {
        if let Ok(input64) = input.extract::<PyReadonlyArray1<Complex64>>() {
            require_contiguous_1d(&input64, "ifft input")?;
            require_profile_matches_f64(self.profile, "ifft")?;
            let leto_view = leto::ArrayView1::from(input64.as_array());
            let result = py.allow_threads(|| apollo_fft::ifft_1d_leto(leto_view));
            let nd_view = ndarray::ArrayView1::try_from(result.view()).map_err(|e| {
                PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
            })?;
            Ok(PyArray1::from_owned_array(py, nd_view.to_owned())
                .into_any()
                .unbind())
        } else {
            let input32 = input.extract::<PyReadonlyArray1<Complex32>>()?;
            require_contiguous_1d(&input32, "ifft input")?;
            match self.profile.storage {
                StoragePrecision::F16 => {
                    let owned = input32.as_array().to_owned();
                    let result = py.allow_threads(|| {
                        apollo_fft::ifft_1d_array_typed::<f16>(&owned)
                            .mapv(|value: f16| value.to_f32())
                    });
                    Ok(PyArray1::from_owned_array(py, result).into_any().unbind())
                }
                _ => {
                    require_profile_matches_f32(self.profile, "ifft")?;
                    let leto_view = leto::ArrayView1::from(input32.as_array());
                    let result =
                        py.allow_threads(|| apollo_fft::ifft_1d_leto_typed::<f32>(leto_view));
                    let nd_view = ndarray::ArrayView1::try_from(result.view()).map_err(|e| {
                        PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
                    })?;
                    Ok(PyArray1::from_owned_array(py, nd_view.to_owned())
                        .into_any()
                        .unbind())
                }
            }
        }
    }

    /// Complex-to-complex forward FFT using the plan's cached twiddle factors.
    fn fft_complex<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray1<'_, Complex64>,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        require_contiguous_1d(&input, "fft_complex input")?;
        let mut output = input.as_array().to_owned();
        py.allow_threads(|| {
            fft_1d_complex_inplace(&mut output);
        });
        Ok(PyArray1::from_owned_array(py, output))
    }

    /// Complex-to-complex inverse FFT using the plan's cached twiddle factors.
    fn ifft_complex<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray1<'_, Complex64>,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        require_contiguous_1d(&input, "ifft_complex input")?;
        let mut output = input.as_array().to_owned();
        py.allow_threads(|| {
            ifft_1d_complex_inplace(&mut output);
        });
        Ok(PyArray1::from_owned_array(py, output))
    }
}

/// Python wrapper for a reusable 2D FFT plan.
#[pyclass(name = "FftPlan2D")]
pub(crate) struct PyFftPlan2D {
    #[allow(dead_code)]
    shape: Shape2D,
    profile: PrecisionProfile,
}

#[pymethods]
impl PyFftPlan2D {
    #[new]
    #[pyo3(signature = (nx, ny, precision=None))]
    fn new(nx: usize, ny: usize, precision: Option<&str>) -> PyResult<Self> {
        let profile = parse_precision(precision)?;
        let shape =
            Shape2D::new(nx, ny).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { shape, profile })
    }

    fn fft<'py>(&self, py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<PyObject> {
        if let Ok(input64) = input.extract::<PyReadonlyArray2<f64>>() {
            require_contiguous_2d(&input64, "fft input")?;
            require_profile_matches_f64(self.profile, "fft")?;
            let leto_view = leto::ArrayView2::from(input64.as_array());
            let result = py.allow_threads(|| apollo_fft::fft_2d_leto(leto_view));
            let nd_view = ndarray::ArrayView2::try_from(result.view()).map_err(|e| {
                PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
            })?;
            Ok(PyArray2::from_owned_array(py, nd_view.to_owned())
                .into_any()
                .unbind())
        } else {
            match self.profile.storage {
                StoragePrecision::F16 => {
                    let input32 = input.extract::<PyReadonlyArray2<f32>>()?;
                    require_contiguous_2d(&input32, "fft input")?;
                    let owned = input32.as_array().mapv(f16::from_f32);
                    let result = py.allow_threads(|| apollo_fft::fft_2d_array_typed(&owned));
                    Ok(PyArray2::from_owned_array(py, result).into_any().unbind())
                }
                _ => {
                    let input32 = input.extract::<PyReadonlyArray2<f32>>()?;
                    require_contiguous_2d(&input32, "fft input")?;
                    require_profile_matches_f32(self.profile, "fft")?;
                    let leto_view = leto::ArrayView2::from(input32.as_array());
                    let result =
                        py.allow_threads(|| apollo_fft::fft_2d_leto_typed::<f32>(leto_view));
                    let nd_view = ndarray::ArrayView2::try_from(result.view()).map_err(|e| {
                        PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
                    })?;
                    Ok(PyArray2::from_owned_array(py, nd_view.to_owned())
                        .into_any()
                        .unbind())
                }
            }
        }
    }

    fn ifft<'py>(&self, py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<PyObject> {
        if let Ok(input64) = input.extract::<PyReadonlyArray2<Complex64>>() {
            require_contiguous_2d(&input64, "ifft input")?;
            require_profile_matches_f64(self.profile, "ifft")?;
            let leto_view = leto::ArrayView2::from(input64.as_array());
            let result = py.allow_threads(|| apollo_fft::ifft_2d_leto(leto_view));
            let nd_view = ndarray::ArrayView2::try_from(result.view()).map_err(|e| {
                PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
            })?;
            Ok(PyArray2::from_owned_array(py, nd_view.to_owned())
                .into_any()
                .unbind())
        } else {
            let input32 = input.extract::<PyReadonlyArray2<Complex32>>()?;
            require_contiguous_2d(&input32, "ifft input")?;
            match self.profile.storage {
                StoragePrecision::F16 => {
                    let owned = input32.as_array().to_owned();
                    let result = py.allow_threads(|| {
                        apollo_fft::ifft_2d_array_typed::<f16>(&owned)
                            .mapv(|value: f16| value.to_f32())
                    });
                    Ok(PyArray2::from_owned_array(py, result).into_any().unbind())
                }
                _ => {
                    require_profile_matches_f32(self.profile, "ifft")?;
                    let leto_view = leto::ArrayView2::from(input32.as_array());
                    let result =
                        py.allow_threads(|| apollo_fft::ifft_2d_leto_typed::<f32>(leto_view));
                    let nd_view = ndarray::ArrayView2::try_from(result.view()).map_err(|e| {
                        PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
                    })?;
                    Ok(PyArray2::from_owned_array(py, nd_view.to_owned())
                        .into_any()
                        .unbind())
                }
            }
        }
    }

    /// Complex-to-complex forward 2D FFT.
    fn fft_complex<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray2<'_, Complex64>,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        require_contiguous_2d(&input, "fft_complex input")?;
        let mut output = input.as_array().to_owned();
        py.allow_threads(|| {
            fft_2d_complex_inplace(&mut output);
        });
        Ok(PyArray2::from_owned_array(py, output))
    }

    /// Complex-to-complex inverse 2D FFT.
    fn ifft_complex<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray2<'_, Complex64>,
    ) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        require_contiguous_2d(&input, "ifft_complex input")?;
        let mut output = input.as_array().to_owned();
        py.allow_threads(|| {
            ifft_2d_complex_inplace(&mut output);
        });
        Ok(PyArray2::from_owned_array(py, output))
    }
}

/// Python wrapper for a reusable 3D FFT plan.
#[pyclass(name = "FftPlan3D")]
pub(crate) struct PyFftPlan3D {
    shape: Shape3D,
    profile: PrecisionProfile,
}

#[pymethods]
impl PyFftPlan3D {
    #[new]
    #[pyo3(signature = (nx, ny, nz, precision=None))]
    fn new(nx: usize, ny: usize, nz: usize, precision: Option<&str>) -> PyResult<Self> {
        let profile = parse_precision(precision)?;
        let shape =
            Shape3D::new(nx, ny, nz).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { shape, profile })
    }

    fn fft<'py>(&self, py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<PyObject> {
        if let Ok(input64) = input.extract::<PyReadonlyArray3<f64>>() {
            require_contiguous_3d(&input64, "fft input")?;
            require_profile_matches_f64(self.profile, "fft")?;
            let leto_view = leto::ArrayView3::from(input64.as_array());
            let result = py.allow_threads(|| apollo_fft::fft_3d_leto(leto_view));
            let nd_view = ndarray::ArrayView3::try_from(result.view()).map_err(|e| {
                PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
            })?;
            Ok(PyArray3::from_owned_array(py, nd_view.to_owned())
                .into_any()
                .unbind())
        } else {
            match self.profile.storage {
                StoragePrecision::F16 => {
                    let input32 = input.extract::<PyReadonlyArray3<f32>>()?;
                    require_contiguous_3d(&input32, "fft input")?;
                    let owned = input32.as_array().mapv(f16::from_f32);
                    let result = py.allow_threads(|| apollo_fft::fft_3d_array_typed(&owned));
                    Ok(PyArray3::from_owned_array(py, result).into_any().unbind())
                }
                _ => {
                    let input32 = input.extract::<PyReadonlyArray3<f32>>()?;
                    require_contiguous_3d(&input32, "fft input")?;
                    require_profile_matches_f32(self.profile, "fft")?;
                    let leto_view = leto::ArrayView3::from(input32.as_array());
                    let result =
                        py.allow_threads(|| apollo_fft::fft_3d_leto_typed::<f32>(leto_view));
                    let nd_view = ndarray::ArrayView3::try_from(result.view()).map_err(|e| {
                        PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
                    })?;
                    Ok(PyArray3::from_owned_array(py, nd_view.to_owned())
                        .into_any()
                        .unbind())
                }
            }
        }
    }

    fn ifft<'py>(&self, py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<PyObject> {
        if let Ok(input64) = input.extract::<PyReadonlyArray3<Complex64>>() {
            require_contiguous_3d(&input64, "ifft input")?;
            require_profile_matches_f64(self.profile, "ifft")?;
            let leto_view = leto::ArrayView3::from(input64.as_array());
            let result = py.allow_threads(|| apollo_fft::ifft_3d_leto(leto_view));
            let nd_view = ndarray::ArrayView3::try_from(result.view()).map_err(|e| {
                PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
            })?;
            Ok(PyArray3::from_owned_array(py, nd_view.to_owned())
                .into_any()
                .unbind())
        } else {
            let input32 = input.extract::<PyReadonlyArray3<Complex32>>()?;
            require_contiguous_3d(&input32, "ifft input")?;
            match self.profile.storage {
                StoragePrecision::F16 => {
                    let owned = input32.as_array().to_owned();
                    let result = py.allow_threads(|| {
                        apollo_fft::ifft_3d_array_typed::<f16>(&owned)
                            .mapv(|value: f16| value.to_f32())
                    });
                    Ok(PyArray3::from_owned_array(py, result).into_any().unbind())
                }
                _ => {
                    require_profile_matches_f32(self.profile, "ifft")?;
                    let leto_view = leto::ArrayView3::from(input32.as_array());
                    let result =
                        py.allow_threads(|| apollo_fft::ifft_3d_leto_typed::<f32>(leto_view));
                    let nd_view = ndarray::ArrayView3::try_from(result.view()).map_err(|e| {
                        PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
                    })?;
                    Ok(PyArray3::from_owned_array(py, nd_view.to_owned())
                        .into_any()
                        .unbind())
                }
            }
        }
    }

    fn rfft<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray3<f64>,
    ) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
        require_contiguous_3d(&input, "rfft input")?;
        let shape = input.shape();
        if shape != [self.shape.nx, self.shape.ny, self.shape.nz] {
            return Err(PyValueError::new_err(
                "input shape does not match plan shape",
            ));
        }
        let leto_view = leto::ArrayView3::from(input.as_array());
        let result = py.allow_threads(|| apollo_fft::fft_3d_leto(leto_view));
        let nd_view = ndarray::ArrayView3::try_from(result.view()).map_err(|e| {
            PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
        })?;
        Ok(PyArray3::from_owned_array(py, nd_view.to_owned()))
    }

    fn irfft<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray3<Complex64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        require_contiguous_3d(&input, "irfft input")?;
        let shape = input.shape();
        if shape != [self.shape.nx, self.shape.ny, self.shape.nz] {
            return Err(PyValueError::new_err(
                "input shape does not match plan shape",
            ));
        }
        let leto_view = leto::ArrayView3::from(input.as_array());
        let result = py.allow_threads(|| apollo_fft::ifft_3d_leto(leto_view));
        let nd_view = ndarray::ArrayView3::try_from(result.view()).map_err(|e| {
            PyValueError::new_err(format!("Leto to ndarray conversion failed: {:?}", e))
        })?;
        Ok(PyArray3::from_owned_array(py, nd_view.to_owned()))
    }
}
