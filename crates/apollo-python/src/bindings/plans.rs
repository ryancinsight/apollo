//! Reusable FFT plan classes (`FftPlan1D`, `FftPlan2D`, `FftPlan3D`)
//! wrapping `apollo-fft` plan execution.

use apollo_fft::{
    f16, fft_1d_complex_inplace, fft_2d_complex_inplace, ifft_1d_complex_inplace,
    ifft_2d_complex_inplace, Complex32, Complex64, PrecisionProfile, Shape1D, Shape2D, Shape3D,
    StoragePrecision,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::support::{
    leto_array1_into_pyarray, leto_array2_into_pyarray, leto_array3_into_pyarray, parse_precision,
    py_array1_leto_view, py_array1_map_to_leto, py_array1_to_leto, py_array2_leto_view,
    py_array2_map_to_leto, py_array2_to_leto, py_array3_leto_view, py_array3_map_to_leto,
    py_array3_to_leto, require_contiguous_1d, require_contiguous_2d, require_contiguous_3d,
    require_profile_matches_f32, require_profile_matches_f64, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3,
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

    fn fft<'py>(&self, py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(input64) = input.extract::<PyReadonlyArray1<f64>>() {
            require_contiguous_1d(&input64, "fft input")?;
            require_profile_matches_f64(self.profile, "fft")?;
            let leto_view = py_array1_leto_view(&input64, "fft input")?;
            let result = py.detach(|| apollo_fft::fft_1d_leto(leto_view));
            leto_array1_into_pyarray(py, result)
        } else {
            match self.profile.storage {
                StoragePrecision::F16 => {
                    let input32 = input.extract::<PyReadonlyArray1<f32>>()?;
                    let owned = py_array1_map_to_leto(&input32, "fft input", |value| {
                        f16::from_f32(*value)
                    })?;
                    let result = py.detach(|| apollo_fft::fft_1d_array_typed(&owned));
                    leto_array1_into_pyarray(py, result)
                }
                _ => {
                    let input32 = input.extract::<PyReadonlyArray1<f32>>()?;
                    require_profile_matches_f32(self.profile, "fft")?;
                    let leto_view = py_array1_leto_view(&input32, "fft input")?;
                    let result = py.detach(|| apollo_fft::fft_1d_leto_typed::<f32>(leto_view));
                    leto_array1_into_pyarray(py, result)
                }
            }
        }
    }

    fn ifft<'py>(&self, py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(input64) = input.extract::<PyReadonlyArray1<Complex64>>() {
            require_contiguous_1d(&input64, "ifft input")?;
            require_profile_matches_f64(self.profile, "ifft")?;
            let leto_view = py_array1_leto_view(&input64, "ifft input")?;
            let result = py.detach(|| apollo_fft::ifft_1d_leto(leto_view));
            leto_array1_into_pyarray(py, result)
        } else {
            let input32 = input.extract::<PyReadonlyArray1<Complex32>>()?;
            match self.profile.storage {
                StoragePrecision::F16 => {
                    let owned = py_array1_to_leto(&input32, "ifft input")?;
                    let result = py.detach(|| {
                        apollo_fft::ifft_1d_array_typed::<f16>(&owned)
                            .mapv(|value: f16| value.to_f32())
                    });
                    leto_array1_into_pyarray(py, result)
                }
                _ => {
                    require_profile_matches_f32(self.profile, "ifft")?;
                    let leto_view = py_array1_leto_view(&input32, "ifft input")?;
                    let result = py.detach(|| apollo_fft::ifft_1d_leto_typed::<f32>(leto_view));
                    leto_array1_into_pyarray(py, result)
                }
            }
        }
    }

    /// Complex-to-complex forward FFT using the plan's cached twiddle factors.
    fn fft_complex<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray1<Complex64>,
    ) -> PyResult<Py<PyAny>> {
        require_contiguous_1d(&input, "fft_complex input")?;
        let mut output = py_array1_to_leto(&input, "fft_complex input")?;
        py.detach(|| {
            fft_1d_complex_inplace(&mut output);
        });
        leto_array1_into_pyarray(py, output)
    }

    /// Complex-to-complex inverse FFT using the plan's cached twiddle factors.
    fn ifft_complex<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray1<Complex64>,
    ) -> PyResult<Py<PyAny>> {
        require_contiguous_1d(&input, "ifft_complex input")?;
        let mut output = py_array1_to_leto(&input, "ifft_complex input")?;
        py.detach(|| {
            ifft_1d_complex_inplace(&mut output);
        });
        leto_array1_into_pyarray(py, output)
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

    fn fft<'py>(&self, py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(input64) = input.extract::<PyReadonlyArray2<f64>>() {
            require_contiguous_2d(&input64, "fft input")?;
            require_profile_matches_f64(self.profile, "fft")?;
            let leto_view = py_array2_leto_view(&input64, "fft input")?;
            let result = py.detach(|| apollo_fft::fft_2d_leto(leto_view));
            leto_array2_into_pyarray(py, result)
        } else {
            match self.profile.storage {
                StoragePrecision::F16 => {
                    let input32 = input.extract::<PyReadonlyArray2<f32>>()?;
                    let owned = py_array2_map_to_leto(&input32, "fft input", |value| {
                        f16::from_f32(*value)
                    })?;
                    let result = py.detach(|| apollo_fft::fft_2d_array_typed(&owned));
                    leto_array2_into_pyarray(py, result)
                }
                _ => {
                    let input32 = input.extract::<PyReadonlyArray2<f32>>()?;
                    require_profile_matches_f32(self.profile, "fft")?;
                    let leto_view = py_array2_leto_view(&input32, "fft input")?;
                    let result = py.detach(|| apollo_fft::fft_2d_leto_typed::<f32>(leto_view));
                    leto_array2_into_pyarray(py, result)
                }
            }
        }
    }

    fn ifft<'py>(&self, py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(input64) = input.extract::<PyReadonlyArray2<Complex64>>() {
            require_contiguous_2d(&input64, "ifft input")?;
            require_profile_matches_f64(self.profile, "ifft")?;
            let leto_view = py_array2_leto_view(&input64, "ifft input")?;
            let result = py.detach(|| apollo_fft::ifft_2d_leto(leto_view));
            leto_array2_into_pyarray(py, result)
        } else {
            let input32 = input.extract::<PyReadonlyArray2<Complex32>>()?;
            match self.profile.storage {
                StoragePrecision::F16 => {
                    let owned = py_array2_to_leto(&input32, "ifft input")?;
                    let result = py.detach(|| {
                        apollo_fft::ifft_2d_array_typed::<f16>(&owned)
                            .mapv(|value: f16| value.to_f32())
                    });
                    leto_array2_into_pyarray(py, result)
                }
                _ => {
                    require_profile_matches_f32(self.profile, "ifft")?;
                    let leto_view = py_array2_leto_view(&input32, "ifft input")?;
                    let result = py.detach(|| apollo_fft::ifft_2d_leto_typed::<f32>(leto_view));
                    leto_array2_into_pyarray(py, result)
                }
            }
        }
    }

    /// Complex-to-complex forward 2D FFT.
    fn fft_complex<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray2<Complex64>,
    ) -> PyResult<Py<PyAny>> {
        require_contiguous_2d(&input, "fft_complex input")?;
        let mut output = py_array2_to_leto(&input, "fft_complex input")?;
        py.detach(|| {
            fft_2d_complex_inplace(&mut output);
        });
        leto_array2_into_pyarray(py, output)
    }

    /// Complex-to-complex inverse 2D FFT.
    fn ifft_complex<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray2<Complex64>,
    ) -> PyResult<Py<PyAny>> {
        require_contiguous_2d(&input, "ifft_complex input")?;
        let mut output = py_array2_to_leto(&input, "ifft_complex input")?;
        py.detach(|| {
            ifft_2d_complex_inplace(&mut output);
        });
        leto_array2_into_pyarray(py, output)
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

    fn fft<'py>(&self, py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(input64) = input.extract::<PyReadonlyArray3<f64>>() {
            require_contiguous_3d(&input64, "fft input")?;
            require_profile_matches_f64(self.profile, "fft")?;
            let leto_view = py_array3_leto_view(&input64, "fft input")?;
            let result = py.detach(|| apollo_fft::fft_3d_leto(leto_view));
            leto_array3_into_pyarray(py, result)
        } else {
            match self.profile.storage {
                StoragePrecision::F16 => {
                    let input32 = input.extract::<PyReadonlyArray3<f32>>()?;
                    let owned = py_array3_map_to_leto(&input32, "fft input", |value| {
                        f16::from_f32(*value)
                    })?;
                    let result = py.detach(|| apollo_fft::fft_3d_array_typed(&owned));
                    leto_array3_into_pyarray(py, result)
                }
                _ => {
                    let input32 = input.extract::<PyReadonlyArray3<f32>>()?;
                    require_profile_matches_f32(self.profile, "fft")?;
                    let leto_view = py_array3_leto_view(&input32, "fft input")?;
                    let result = py.detach(|| apollo_fft::fft_3d_leto_typed::<f32>(leto_view));
                    leto_array3_into_pyarray(py, result)
                }
            }
        }
    }

    fn ifft<'py>(&self, py: Python<'py>, input: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(input64) = input.extract::<PyReadonlyArray3<Complex64>>() {
            require_contiguous_3d(&input64, "ifft input")?;
            require_profile_matches_f64(self.profile, "ifft")?;
            let leto_view = py_array3_leto_view(&input64, "ifft input")?;
            let result = py.detach(|| apollo_fft::ifft_3d_leto(leto_view));
            leto_array3_into_pyarray(py, result)
        } else {
            let input32 = input.extract::<PyReadonlyArray3<Complex32>>()?;
            match self.profile.storage {
                StoragePrecision::F16 => {
                    let owned = py_array3_to_leto(&input32, "ifft input")?;
                    let result = py.detach(|| {
                        apollo_fft::ifft_3d_array_typed::<f16>(&owned)
                            .mapv(|value: f16| value.to_f32())
                    });
                    leto_array3_into_pyarray(py, result)
                }
                _ => {
                    require_profile_matches_f32(self.profile, "ifft")?;
                    let leto_view = py_array3_leto_view(&input32, "ifft input")?;
                    let result = py.detach(|| apollo_fft::ifft_3d_leto_typed::<f32>(leto_view));
                    leto_array3_into_pyarray(py, result)
                }
            }
        }
    }

    fn rfft<'py>(&self, py: Python<'py>, input: PyReadonlyArray3<f64>) -> PyResult<Py<PyAny>> {
        require_contiguous_3d(&input, "rfft input")?;
        let shape = input.shape();
        if shape != [self.shape.nx, self.shape.ny, self.shape.nz] {
            return Err(PyValueError::new_err(
                "input shape does not match plan shape",
            ));
        }
        let leto_view = py_array3_leto_view(&input, "rfft input")?;
        let result = py.detach(|| apollo_fft::fft_3d_leto(leto_view));
        leto_array3_into_pyarray(py, result)
    }

    fn irfft<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray3<Complex64>,
    ) -> PyResult<Py<PyAny>> {
        require_contiguous_3d(&input, "irfft input")?;
        let shape = input.shape();
        if shape != [self.shape.nx, self.shape.ny, self.shape.nz] {
            return Err(PyValueError::new_err(
                "input shape does not match plan shape",
            ));
        }
        let leto_view = py_array3_leto_view(&input, "irfft input")?;
        let result = py.detach(|| apollo_fft::ifft_3d_leto(leto_view));
        leto_array3_into_pyarray(py, result)
    }
}
