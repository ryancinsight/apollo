#![warn(missing_docs)]
//! Python bindings for Apollo FFT and NUFFT.

#![allow(clippy::unused_self)]
#![allow(clippy::elidable_lifetime_names)]

pub mod application;
pub mod domain;
pub mod infrastructure;

use apollo_dctdst::{dct2, dct3, dst2, dst3};
use apollo_dht::{DhtPlan, HartleySpectrum};
use apollo_fft::{
    f16, fft_1d_complex_inplace, fft_2d_complex_inplace, fft_3d_complex_inplace, fftfreq, fftshift,
    ifft_1d_complex_inplace, ifft_2d_complex_inplace, ifft_3d_complex_inplace, ifftshift, rfftfreq,
    Complex32, Complex64, CpuBackend, FftBackend, PrecisionMode, PrecisionProfile, Shape1D,
    Shape2D, Shape3D, StoragePrecision,
};
use apollo_fwht::{FwhtPlan, FwhtPlan2D, FwhtPlan3D};
use apollo_nufft::{
    nufft_type1_1d, nufft_type1_1d_fast, nufft_type1_3d, nufft_type1_3d_fast, nufft_type2_1d,
    nufft_type2_1d_fast, UniformDomain1D, UniformGrid3D, DEFAULT_NUFFT_KERNEL_WIDTH,
};
use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn require_contiguous_1d<T: Element>(input: &PyReadonlyArray1<'_, T>, name: &str) -> PyResult<()> {
    if input.as_array().is_standard_layout() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )))
    }
}

fn require_contiguous_2d<T: Element>(input: &PyReadonlyArray2<'_, T>, name: &str) -> PyResult<()> {
    if input.as_array().is_standard_layout() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )))
    }
}

fn require_contiguous_3d<T: Element>(input: &PyReadonlyArray3<'_, T>, name: &str) -> PyResult<()> {
    if input.as_array().is_standard_layout() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )))
    }
}

fn wgpu_backend_usable() -> bool {
    apollo_fft_wgpu::WgpuBackend::try_default().is_ok()
}

fn parse_precision(precision: Option<&str>) -> PyResult<PrecisionProfile> {
    match precision.unwrap_or("high_accuracy") {
        "high_accuracy" => Ok(PrecisionProfile::HIGH_ACCURACY_F64),
        "low_precision" => Ok(PrecisionProfile::LOW_PRECISION_F32),
        "mixed_precision" => Ok(PrecisionProfile::MIXED_PRECISION_F16_F32),
        other => Err(PyValueError::new_err(format!(
            "unsupported precision `{other}`; expected `high_accuracy`, `low_precision`, or `mixed_precision`"
        ))),
    }
}

fn precision_name(profile: PrecisionProfile) -> &'static str {
    match profile.mode {
        PrecisionMode::HighAccuracy => "high_accuracy",
        PrecisionMode::LowPrecision => "low_precision",
        PrecisionMode::MixedPrecision => "mixed_precision",
    }
}

fn require_profile_matches_f64(profile: PrecisionProfile, name: &str) -> PyResult<()> {
    if profile.storage == StoragePrecision::F64 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} received float64/complex128 input but precision `{}` expects float32/complex64 storage",
            precision_name(profile)
        )))
    }
}

fn require_profile_matches_f32(profile: PrecisionProfile, name: &str) -> PyResult<()> {
    if profile.storage == StoragePrecision::F32 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} received float32/complex64 input but precision `{}` expects float64/complex128 storage",
            precision_name(profile)
        )))
    }
}

/// Python wrapper for a reusable 1D FFT plan.
#[pyclass(name = "FftPlan1D")]
struct PyFftPlan1D {
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
            Ok(PyArray1::from_owned_array(
                py,
                apollo_fft::fft_1d_array(&input64.as_array().to_owned()),
            )
            .into_any()
            .unbind())
        } else {
            match self.profile.storage {
                StoragePrecision::F16 => {
                    let input32 = input.extract::<PyReadonlyArray1<f32>>()?;
                    require_contiguous_1d(&input32, "fft input")?;
                    Ok(PyArray1::from_owned_array(
                        py,
                        apollo_fft::fft_1d_array_typed(&input32.as_array().mapv(f16::from_f32)),
                    )
                    .into_any()
                    .unbind())
                }
                _ => {
                    let input32 = input.extract::<PyReadonlyArray1<f32>>()?;
                    require_contiguous_1d(&input32, "fft input")?;
                    require_profile_matches_f32(self.profile, "fft")?;
                    Ok(PyArray1::from_owned_array(
                        py,
                        apollo_fft::fft_1d_array_typed(&input32.as_array().to_owned()),
                    )
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
            Ok(PyArray1::from_owned_array(
                py,
                apollo_fft::ifft_1d_array(&input64.as_array().to_owned()),
            )
            .into_any()
            .unbind())
        } else {
            let input32 = input.extract::<PyReadonlyArray1<Complex32>>()?;
            require_contiguous_1d(&input32, "ifft input")?;
            match self.profile.storage {
                StoragePrecision::F16 => Ok(PyArray1::from_owned_array(
                    py,
                    apollo_fft::ifft_1d_array_typed::<f16>(&input32.as_array().to_owned())
                        .mapv(|value: f16| value.to_f32()),
                )
                .into_any()
                .unbind()),
                _ => {
                    require_profile_matches_f32(self.profile, "ifft")?;
                    Ok(PyArray1::from_owned_array(
                        py,
                        apollo_fft::ifft_1d_array_typed::<f32>(&input32.as_array().to_owned()),
                    )
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
        fft_1d_complex_inplace(&mut output);
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
        ifft_1d_complex_inplace(&mut output);
        Ok(PyArray1::from_owned_array(py, output))
    }
}

/// Python wrapper for a reusable 2D FFT plan.
#[pyclass(name = "FftPlan2D")]
struct PyFftPlan2D {
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
            Ok(PyArray2::from_owned_array(
                py,
                apollo_fft::fft_2d_array(&input64.as_array().to_owned()),
            )
            .into_any()
            .unbind())
        } else {
            match self.profile.storage {
                StoragePrecision::F16 => {
                    let input32 = input.extract::<PyReadonlyArray2<f32>>()?;
                    require_contiguous_2d(&input32, "fft input")?;
                    Ok(PyArray2::from_owned_array(
                        py,
                        apollo_fft::fft_2d_array_typed(&input32.as_array().mapv(f16::from_f32)),
                    )
                    .into_any()
                    .unbind())
                }
                _ => {
                    let input32 = input.extract::<PyReadonlyArray2<f32>>()?;
                    require_contiguous_2d(&input32, "fft input")?;
                    require_profile_matches_f32(self.profile, "fft")?;
                    Ok(PyArray2::from_owned_array(
                        py,
                        apollo_fft::fft_2d_array_typed(&input32.as_array().to_owned()),
                    )
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
            Ok(PyArray2::from_owned_array(
                py,
                apollo_fft::ifft_2d_array(&input64.as_array().to_owned()),
            )
            .into_any()
            .unbind())
        } else {
            let input32 = input.extract::<PyReadonlyArray2<Complex32>>()?;
            require_contiguous_2d(&input32, "ifft input")?;
            match self.profile.storage {
                StoragePrecision::F16 => Ok(PyArray2::from_owned_array(
                    py,
                    apollo_fft::ifft_2d_array_typed::<f16>(&input32.as_array().to_owned())
                        .mapv(|value: f16| value.to_f32()),
                )
                .into_any()
                .unbind()),
                _ => {
                    require_profile_matches_f32(self.profile, "ifft")?;
                    Ok(PyArray2::from_owned_array(
                        py,
                        apollo_fft::ifft_2d_array_typed::<f32>(&input32.as_array().to_owned()),
                    )
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
        fft_2d_complex_inplace(&mut output);
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
        ifft_2d_complex_inplace(&mut output);
        Ok(PyArray2::from_owned_array(py, output))
    }
}

/// Python wrapper for a reusable 3D FFT plan.
#[pyclass(name = "FftPlan3D")]
struct PyFftPlan3D {
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
            Ok(PyArray3::from_owned_array(
                py,
                apollo_fft::fft_3d_array(&input64.as_array().to_owned()),
            )
            .into_any()
            .unbind())
        } else {
            match self.profile.storage {
                StoragePrecision::F16 => {
                    let input32 = input.extract::<PyReadonlyArray3<f32>>()?;
                    require_contiguous_3d(&input32, "fft input")?;
                    Ok(PyArray3::from_owned_array(
                        py,
                        apollo_fft::fft_3d_array_typed(&input32.as_array().mapv(f16::from_f32)),
                    )
                    .into_any()
                    .unbind())
                }
                _ => {
                    let input32 = input.extract::<PyReadonlyArray3<f32>>()?;
                    require_contiguous_3d(&input32, "fft input")?;
                    require_profile_matches_f32(self.profile, "fft")?;
                    Ok(PyArray3::from_owned_array(
                        py,
                        apollo_fft::fft_3d_array_typed(&input32.as_array().to_owned()),
                    )
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
            Ok(PyArray3::from_owned_array(
                py,
                apollo_fft::ifft_3d_array(&input64.as_array().to_owned()),
            )
            .into_any()
            .unbind())
        } else {
            let input32 = input.extract::<PyReadonlyArray3<Complex32>>()?;
            require_contiguous_3d(&input32, "ifft input")?;
            match self.profile.storage {
                StoragePrecision::F16 => Ok(PyArray3::from_owned_array(
                    py,
                    apollo_fft::ifft_3d_array_typed::<f16>(&input32.as_array().to_owned())
                        .mapv(|value: f16| value.to_f32()),
                )
                .into_any()
                .unbind()),
                _ => {
                    require_profile_matches_f32(self.profile, "ifft")?;
                    Ok(PyArray3::from_owned_array(
                        py,
                        apollo_fft::ifft_3d_array_typed::<f32>(&input32.as_array().to_owned()),
                    )
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
        let input = input.as_array().to_owned();
        let mut output = ndarray::Array3::<Complex64>::zeros(input.dim());
        apollo_fft::fft_3d_array_typed_into(&input, &mut output);
        Ok(PyArray3::from_owned_array(py, output))
    }

    fn irfft<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray3<Complex64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        require_contiguous_3d(&input, "irfft input")?;
        let spectrum = input.as_array().to_owned();
        let (nx, ny, nz) = (self.shape.nx, self.shape.ny, self.shape.nz);
        let mut output = ndarray::Array3::<f64>::zeros((nx, ny, nz));
        let mut scratch = ndarray::Array3::<Complex64>::zeros((nx, ny, nz));
        apollo_fft::ifft_3d_array_typed_into(&spectrum, &mut output, &mut scratch);
        Ok(PyArray3::from_owned_array(py, output))
    }
}

/// Forward 1D FFT of a real signal.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
fn fft1<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<PyObject> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray1<f64>>() {
        require_contiguous_1d(&input64, "fft1 input")?;
        require_profile_matches_f64(profile, "fft1")?;
        Ok(
            PyArray1::from_owned_array(
                py,
                apollo_fft::fft_1d_array(&input64.as_array().to_owned()),
            )
            .into_any()
            .unbind(),
        )
    } else {
        match profile.storage {
            StoragePrecision::F16 => {
                let input32 = input.extract::<PyReadonlyArray1<f32>>()?;
                require_contiguous_1d(&input32, "fft1 input")?;
                Ok(PyArray1::from_owned_array(
                    py,
                    apollo_fft::fft_1d_array_typed(&input32.as_array().mapv(f16::from_f32)),
                )
                .into_any()
                .unbind())
            }
            _ => {
                let input32 = input.extract::<PyReadonlyArray1<f32>>()?;
                require_contiguous_1d(&input32, "fft1 input")?;
                require_profile_matches_f32(profile, "fft1")?;
                Ok(PyArray1::from_owned_array(
                    py,
                    apollo_fft::fft_1d_array_typed(&input32.as_array().to_owned()),
                )
                .into_any()
                .unbind())
            }
        }
    }
}

/// Inverse 1D FFT of a complex spectrum.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
fn ifft1<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<PyObject> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray1<Complex64>>() {
        require_contiguous_1d(&input64, "ifft1 input")?;
        require_profile_matches_f64(profile, "ifft1")?;
        Ok(PyArray1::from_owned_array(
            py,
            apollo_fft::ifft_1d_array(&input64.as_array().to_owned()),
        )
        .into_any()
        .unbind())
    } else {
        let input32 = input.extract::<PyReadonlyArray1<Complex32>>()?;
        require_contiguous_1d(&input32, "ifft1 input")?;
        match profile.storage {
            StoragePrecision::F16 => Ok(PyArray1::from_owned_array(
                py,
                apollo_fft::ifft_1d_array_typed::<f16>(&input32.as_array().to_owned())
                    .mapv(|value: f16| value.to_f32()),
            )
            .into_any()
            .unbind()),
            _ => {
                require_profile_matches_f32(profile, "ifft1")?;
                Ok(PyArray1::from_owned_array(
                    py,
                    apollo_fft::ifft_1d_array_typed::<f32>(&input32.as_array().to_owned()),
                )
                .into_any()
                .unbind())
            }
        }
    }
}

/// Forward 2D FFT of a real array.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
fn fft2<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<PyObject> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray2<f64>>() {
        require_contiguous_2d(&input64, "fft2 input")?;
        require_profile_matches_f64(profile, "fft2")?;
        Ok(
            PyArray2::from_owned_array(
                py,
                apollo_fft::fft_2d_array(&input64.as_array().to_owned()),
            )
            .into_any()
            .unbind(),
        )
    } else {
        match profile.storage {
            StoragePrecision::F16 => {
                let input32 = input.extract::<PyReadonlyArray2<f32>>()?;
                require_contiguous_2d(&input32, "fft2 input")?;
                Ok(PyArray2::from_owned_array(
                    py,
                    apollo_fft::fft_2d_array_typed(&input32.as_array().mapv(f16::from_f32)),
                )
                .into_any()
                .unbind())
            }
            _ => {
                let input32 = input.extract::<PyReadonlyArray2<f32>>()?;
                require_contiguous_2d(&input32, "fft2 input")?;
                require_profile_matches_f32(profile, "fft2")?;
                Ok(PyArray2::from_owned_array(
                    py,
                    apollo_fft::fft_2d_array_typed(&input32.as_array().to_owned()),
                )
                .into_any()
                .unbind())
            }
        }
    }
}

/// Inverse 2D FFT of a complex spectrum.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
fn ifft2<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<PyObject> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray2<Complex64>>() {
        require_contiguous_2d(&input64, "ifft2 input")?;
        require_profile_matches_f64(profile, "ifft2")?;
        Ok(PyArray2::from_owned_array(
            py,
            apollo_fft::ifft_2d_array(&input64.as_array().to_owned()),
        )
        .into_any()
        .unbind())
    } else {
        let input32 = input.extract::<PyReadonlyArray2<Complex32>>()?;
        require_contiguous_2d(&input32, "ifft2 input")?;
        match profile.storage {
            StoragePrecision::F16 => Ok(PyArray2::from_owned_array(
                py,
                apollo_fft::ifft_2d_array_typed::<f16>(&input32.as_array().to_owned())
                    .mapv(|value: f16| value.to_f32()),
            )
            .into_any()
            .unbind()),
            _ => {
                require_profile_matches_f32(profile, "ifft2")?;
                Ok(PyArray2::from_owned_array(
                    py,
                    apollo_fft::ifft_2d_array_typed::<f32>(&input32.as_array().to_owned()),
                )
                .into_any()
                .unbind())
            }
        }
    }
}

/// Forward 3D FFT of a real array.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
fn fft3<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<PyObject> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray3<f64>>() {
        require_contiguous_3d(&input64, "fft3 input")?;
        require_profile_matches_f64(profile, "fft3")?;
        Ok(
            PyArray3::from_owned_array(
                py,
                apollo_fft::fft_3d_array(&input64.as_array().to_owned()),
            )
            .into_any()
            .unbind(),
        )
    } else {
        match profile.storage {
            StoragePrecision::F16 => {
                let input32 = input.extract::<PyReadonlyArray3<f32>>()?;
                require_contiguous_3d(&input32, "fft3 input")?;
                Ok(PyArray3::from_owned_array(
                    py,
                    apollo_fft::fft_3d_array_typed(&input32.as_array().mapv(f16::from_f32)),
                )
                .into_any()
                .unbind())
            }
            _ => {
                let input32 = input.extract::<PyReadonlyArray3<f32>>()?;
                require_contiguous_3d(&input32, "fft3 input")?;
                require_profile_matches_f32(profile, "fft3")?;
                Ok(PyArray3::from_owned_array(
                    py,
                    apollo_fft::fft_3d_array_typed(&input32.as_array().to_owned()),
                )
                .into_any()
                .unbind())
            }
        }
    }
}

/// Inverse 3D FFT of a complex spectrum.
#[pyfunction]
#[pyo3(signature = (input, precision=None))]
fn ifft3<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    precision: Option<&str>,
) -> PyResult<PyObject> {
    let profile = parse_precision(precision)?;
    if let Ok(input64) = input.extract::<PyReadonlyArray3<Complex64>>() {
        require_contiguous_3d(&input64, "ifft3 input")?;
        require_profile_matches_f64(profile, "ifft3")?;
        Ok(PyArray3::from_owned_array(
            py,
            apollo_fft::ifft_3d_array(&input64.as_array().to_owned()),
        )
        .into_any()
        .unbind())
    } else {
        let input32 = input.extract::<PyReadonlyArray3<Complex32>>()?;
        require_contiguous_3d(&input32, "ifft3 input")?;
        match profile.storage {
            StoragePrecision::F16 => Ok(PyArray3::from_owned_array(
                py,
                apollo_fft::ifft_3d_array_typed::<f16>(&input32.as_array().to_owned())
                    .mapv(|value: f16| value.to_f32()),
            )
            .into_any()
            .unbind()),
            _ => {
                require_profile_matches_f32(profile, "ifft3")?;
                Ok(PyArray3::from_owned_array(
                    py,
                    apollo_fft::ifft_3d_array_typed::<f32>(&input32.as_array().to_owned()),
                )
                .into_any()
                .unbind())
            }
        }
    }
}

/// Forward 3D real-to-complex half-spectrum FFT.
#[pyfunction]
fn rfft3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<f64>,
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    require_contiguous_3d(&input, "rfft3 input")?;
    let owned = input.as_array().to_owned();
    let mut output = ndarray::Array3::<Complex64>::zeros(owned.dim());
    apollo_fft::fft_3d_array_typed_into(&owned, &mut output);
    Ok(PyArray3::from_owned_array(py, output))
}

/// Inverse 3D half-spectrum FFT.
#[pyfunction]
fn irfft3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<Complex64>,
    nz: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    require_contiguous_3d(&input, "irfft3 input")?;
    let owned = input.as_array().to_owned();
    let (nx, ny, nz_c) = owned.dim();
    if nz_c != nz {
        return Err(PyValueError::new_err(
            "irfft3 input shape and nz are inconsistent",
        ));
    }
    let mut output = ndarray::Array3::<f64>::zeros((nx, ny, nz));
    let mut scratch = ndarray::Array3::<Complex64>::zeros((nx, ny, nz));
    apollo_fft::ifft_3d_array_typed_into(&owned, &mut output, &mut scratch);
    Ok(PyArray3::from_owned_array(py, output))
}

/// Exact direct 1D type-1 NUFFT.
#[pyfunction(name = "nufft_type1_1d")]
#[pyo3(signature = (positions, values, dx, n_out=None))]
fn nufft_type1_1d_py<'py>(
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
    Ok(PyArray1::from_owned_array(
        py,
        nufft_type1_1d(
            positions
                .as_slice()
                .expect("owned positions are contiguous"),
            values.as_slice().expect("owned values are contiguous"),
            domain,
        ),
    ))
}

/// Exact direct 1D type-2 NUFFT.
#[pyfunction(name = "nufft_type2_1d")]
fn nufft_type2_1d_py<'py>(
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
    Ok(nufft_type2_1d(
        &coeffs,
        positions
            .as_slice()
            .expect("owned positions are contiguous"),
        domain,
    )
    .into_pyarray(py))
}

/// Exact direct 3D type-1 NUFFT.
#[pyfunction(name = "nufft_type1_3d")]
fn nufft_type1_3d_py<'py>(
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
    Ok(PyArray3::from_owned_array(
        py,
        nufft_type1_3d(
            &tuples,
            owned_values
                .as_slice()
                .expect("owned values are contiguous"),
            grid,
        ),
    ))
}

/// Fast 1D type-1 NUFFT using Kaiser-Bessel spreading.
#[pyfunction(name = "nufft_type1_1d_fast")]
#[pyo3(signature = (positions, values, dx, n_out=None, kernel_width=DEFAULT_NUFFT_KERNEL_WIDTH))]
fn nufft_type1_1d_fast_py<'py>(
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
    Ok(PyArray1::from_owned_array(
        py,
        nufft_type1_1d_fast(
            positions
                .as_slice()
                .expect("owned positions are contiguous"),
            values.as_slice().expect("owned values are contiguous"),
            domain,
            kernel_width,
        ),
    ))
}

/// Fast 1D type-2 NUFFT using Kaiser-Bessel spreading.
#[pyfunction(name = "nufft_type2_1d_fast")]
#[pyo3(signature = (fourier_coeffs, positions, dx, kernel_width=DEFAULT_NUFFT_KERNEL_WIDTH))]
fn nufft_type2_1d_fast_py<'py>(
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
    Ok(nufft_type2_1d_fast(
        &coeffs,
        positions
            .as_slice()
            .expect("owned positions are contiguous"),
        domain,
        kernel_width,
    )
    .into_pyarray(py))
}

/// Fast 3D type-1 NUFFT using Kaiser-Bessel spreading.
#[pyfunction(name = "nufft_type1_3d_fast")]
#[pyo3(signature = (positions, values, nx, ny, nz, dx, dy, dz, kernel_width=DEFAULT_NUFFT_KERNEL_WIDTH))]
fn nufft_type1_3d_fast_py<'py>(
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
    Ok(PyArray3::from_owned_array(
        py,
        nufft_type1_3d_fast(
            &tuples,
            owned_values
                .as_slice()
                .expect("owned values are contiguous"),
            grid,
            kernel_width,
        ),
    ))
}

// ── FFT utility functions ─────────────────────────────────────────────────────

/// Frequency bin centers for a length-`n` complex DFT with sample spacing `d`.
///
/// Numpy-compatible: `fftfreq(n, d)` returns bins `[0, 1/nd, …, −1/nd, …]`.
/// For `n=0` returns an empty array.
#[pyfunction]
#[pyo3(signature = (n, d=1.0))]
fn fftfreq_py<'py>(py: Python<'py>, n: usize, d: f64) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, fftfreq(n, d))
}

/// Frequency bin centers for a length-`n` real-input FFT with sample spacing `d`.
///
/// Returns `n/2 + 1` non-negative bins. Numpy-compatible `rfftfreq(n, d)`.
#[pyfunction]
#[pyo3(signature = (n, d=1.0))]
fn rfftfreq_py<'py>(py: Python<'py>, n: usize, d: f64) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, rfftfreq(n, d))
}

/// Shift the zero-frequency component to the center of the spectrum.
///
/// Numpy-compatible `fftshift`. Accepts 1-D float64 arrays.
#[pyfunction]
fn fftshift_py<'py>(
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
fn ifftshift_py<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let owned: Vec<f64> = input.as_array().iter().copied().collect();
    PyArray1::from_vec(py, ifftshift(&owned))
}

// ── Complex-to-complex FFT ────────────────────────────────────────────────────

/// Complex-to-complex forward 1D FFT. Accepts complex128 input, returns complex128.
#[pyfunction]
fn fft_complex1<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, Complex64>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    require_contiguous_1d(&input, "fft_complex1 input")?;
    let mut output = input.as_array().to_owned();
    fft_1d_complex_inplace(&mut output);
    Ok(PyArray1::from_owned_array(py, output))
}

/// Complex-to-complex inverse 1D FFT. Accepts complex128, returns complex128.
#[pyfunction]
fn ifft_complex1<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, Complex64>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    require_contiguous_1d(&input, "ifft_complex1 input")?;
    let mut output = input.as_array().to_owned();
    ifft_1d_complex_inplace(&mut output);
    Ok(PyArray1::from_owned_array(py, output))
}

/// Complex-to-complex forward 2D FFT.
#[pyfunction]
fn fft_complex2<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'_, Complex64>,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    require_contiguous_2d(&input, "fft_complex2 input")?;
    let mut output = input.as_array().to_owned();
    fft_2d_complex_inplace(&mut output);
    Ok(PyArray2::from_owned_array(py, output))
}

/// Complex-to-complex inverse 2D FFT.
#[pyfunction]
fn ifft_complex2<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'_, Complex64>,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    require_contiguous_2d(&input, "ifft_complex2 input")?;
    let mut output = input.as_array().to_owned();
    ifft_2d_complex_inplace(&mut output);
    Ok(PyArray2::from_owned_array(py, output))
}

/// Complex-to-complex forward 3D FFT.
#[pyfunction]
fn fft_complex3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'_, Complex64>,
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    require_contiguous_3d(&input, "fft_complex3 input")?;
    let mut output = input.as_array().to_owned();
    fft_3d_complex_inplace(&mut output);
    Ok(PyArray3::from_owned_array(py, output))
}

/// Complex-to-complex inverse 3D FFT.
#[pyfunction]
fn ifft_complex3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'_, Complex64>,
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    require_contiguous_3d(&input, "ifft_complex3 input")?;
    let mut output = input.as_array().to_owned();
    ifft_3d_complex_inplace(&mut output);
    Ok(PyArray3::from_owned_array(py, output))
}

// ── Discrete Hartley Transform ────────────────────────────────────────────────

/// Forward 1D Discrete Hartley Transform.
///
/// Returns the unnormalized DHT spectrum of length `n`. Inverse is `idht1`.
#[pyfunction]
fn dht1<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_1d(&input, "dht1 input")?;
    let arr = input.as_array();
    let n = arr.len();
    let plan = DhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let signal: Vec<f64> = arr.iter().copied().collect();
    let spectrum = plan
        .forward(&signal)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyArray1::from_vec(py, spectrum.values().to_vec()))
}

/// Inverse 1D Discrete Hartley Transform. Scales by `1/n`.
#[pyfunction]
fn idht1<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_1d(&input, "idht1 input")?;
    let arr = input.as_array();
    let n = arr.len();
    let plan = DhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let signal: Vec<f64> = arr.iter().copied().collect();
    let spectrum = HartleySpectrum::new(signal);
    let recovered = plan
        .inverse(&spectrum)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyArray1::from_vec(py, recovered))
}

/// Forward 2D Discrete Hartley Transform. Input must be square (N×N).
#[pyfunction]
fn dht2<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    require_contiguous_2d(&input, "dht2 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.nrows();
    let plan = DhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = plan
        .forward_2d(&arr)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyArray2::from_owned_array(py, result))
}

/// Inverse 2D Discrete Hartley Transform. Input must be square (N×N). Scales by `1/N²`.
#[pyfunction]
fn idht2<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    require_contiguous_2d(&input, "idht2 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.nrows();
    let plan = DhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = plan
        .inverse_2d(&arr)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyArray2::from_owned_array(py, result))
}

/// Forward 3D Discrete Hartley Transform. Input must be cubic (N×N×N).
#[pyfunction]
fn dht3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'_, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    require_contiguous_3d(&input, "dht3 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.shape()[0];
    let plan = DhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = plan
        .forward_3d(&arr)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyArray3::from_owned_array(py, result))
}

/// Inverse 3D Discrete Hartley Transform. Input must be cubic (N×N×N). Scales by `1/N³`.
#[pyfunction]
fn idht3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'_, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    require_contiguous_3d(&input, "idht3 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.shape()[0];
    let plan = DhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = plan
        .inverse_3d(&arr)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyArray3::from_owned_array(py, result))
}

// ── Fast Walsh-Hadamard Transform ─────────────────────────────────────────────

/// Forward 1D Fast Walsh-Hadamard Transform. Length must be a power of two.
#[pyfunction]
fn fwht1<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_1d(&input, "fwht1 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.len();
    let plan = FwhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = plan
        .forward(&arr)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyArray1::from_owned_array(py, result))
}

/// Inverse 1D Fast Walsh-Hadamard Transform. Scales by `1/n`.
#[pyfunction]
fn ifwht1<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_1d(&input, "ifwht1 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.len();
    let plan = FwhtPlan::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = plan
        .inverse(&arr)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyArray1::from_owned_array(py, result))
}

/// Forward 2D Fast Walsh-Hadamard Transform. Input must be square (N×N), N a power of two.
#[pyfunction]
fn fwht2<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    require_contiguous_2d(&input, "fwht2 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.nrows();
    let plan = FwhtPlan2D::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = plan
        .forward(&arr)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyArray2::from_owned_array(py, result))
}

/// Inverse 2D Fast Walsh-Hadamard Transform. Scales by `1/N²`.
#[pyfunction]
fn ifwht2<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    require_contiguous_2d(&input, "ifwht2 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.nrows();
    let plan = FwhtPlan2D::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = plan
        .inverse(&arr)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyArray2::from_owned_array(py, result))
}

/// Forward 3D Fast Walsh-Hadamard Transform. Input must be cubic (N×N×N), N a power of two.
#[pyfunction]
fn fwht3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'_, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    require_contiguous_3d(&input, "fwht3 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.shape()[0];
    let plan = FwhtPlan3D::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = plan
        .forward(&arr)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyArray3::from_owned_array(py, result))
}

/// Inverse 3D Fast Walsh-Hadamard Transform. Scales by `1/N³`.
#[pyfunction]
fn ifwht3<'py>(
    py: Python<'py>,
    input: PyReadonlyArray3<'_, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    require_contiguous_3d(&input, "ifwht3 input")?;
    let arr = input.as_array().to_owned();
    let n = arr.shape()[0];
    let plan = FwhtPlan3D::new(n).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let result = plan
        .inverse(&arr)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyArray3::from_owned_array(py, result))
}

// ── Discrete Cosine / Sine Transform ─────────────────────────────────────────

/// Forward 1D DCT-II (the "the DCT" as used by numpy/scipy).
///
/// Equivalent to `scipy.fft.dct(x, type=2, norm=None)` (unnormalized).
/// Inverse via `idct2_1d`.
#[pyfunction]
fn dct2_1d<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_1d(&input, "dct2_1d input")?;
    let arr = input.as_array();
    let n = arr.len();
    let signal: Vec<f64> = arr.iter().copied().collect();
    let mut output = vec![0.0_f64; n];
    dct2(&signal, &mut output);
    Ok(PyArray1::from_vec(py, output))
}

/// Inverse 1D DCT-II (= DCT-III / N).
///
/// Equivalent to `scipy.fft.idct(x, type=2, norm=None)`.
#[pyfunction]
fn idct2_1d<'py>(
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
    dct3(&signal, &mut output);
    let scale = 2.0 / n as f64;
    output.iter_mut().for_each(|v| *v *= scale);
    Ok(PyArray1::from_vec(py, output))
}

/// Forward 1D DST-II.
///
/// Equivalent to `scipy.fft.dst(x, type=2, norm=None)` (unnormalized).
#[pyfunction]
fn dst2_1d<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_1d(&input, "dst2_1d input")?;
    let arr = input.as_array();
    let n = arr.len();
    let signal: Vec<f64> = arr.iter().copied().collect();
    let mut output = vec![0.0_f64; n];
    dst2(&signal, &mut output);
    Ok(PyArray1::from_vec(py, output))
}

/// Inverse 1D DST-II (= DST-III / N).
#[pyfunction]
fn idst2_1d<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    require_contiguous_1d(&input, "idst2_1d input")?;
    let arr = input.as_array();
    let n = arr.len();
    let signal: Vec<f64> = arr.iter().copied().collect();
    let mut output = vec![0.0_f64; n];
    // DST-III(DST-II(x)) = (N/2) * x; inverse: x = DST-III(X) * (2 / N).
    dst3(&signal, &mut output);
    let scale = 2.0 / n as f64;
    output.iter_mut().for_each(|v| *v *= scale);
    Ok(PyArray1::from_vec(py, output))
}

/// Return the backend names that are genuinely usable from Python on this host.
#[pyfunction]
fn available_backends() -> Vec<String> {
    let mut backends = vec!["cpu".to_string()];
    if wgpu_backend_usable() {
        backends.push("wgpu".to_string());
    }
    backends
}

/// Return backend capability metadata for Python callers.
#[pyfunction]
fn backend_capabilities(py: Python<'_>) -> PyResult<PyObject> {
    let backends = PyDict::new(py);

    let cpu_caps = CpuBackend.capabilities();
    let cpu = PyDict::new(py);
    cpu.set_item("available", true)?;
    cpu.set_item("supports_1d", cpu_caps.supports_1d)?;
    cpu.set_item("supports_2d", cpu_caps.supports_2d)?;
    cpu.set_item("supports_3d", cpu_caps.supports_3d)?;
    cpu.set_item(
        "supports_real_to_complex",
        cpu_caps.supports_real_to_complex,
    )?;
    cpu.set_item(
        "supports_mixed_precision",
        cpu_caps.supports_mixed_precision,
    )?;
    cpu.set_item(
        "default_precision_profile",
        precision_name(cpu_caps.default_precision_profile),
    )?;
    cpu.set_item(
        "supported_precision_profiles",
        cpu_caps
            .supported_precision_profiles
            .iter()
            .map(|profile| precision_name(*profile))
            .collect::<Vec<_>>(),
    )?;
    backends.set_item("cpu", cpu)?;

    let wgpu = PyDict::new(py);
    if let Ok(backend) = apollo_fft_wgpu::WgpuBackend::try_default() {
        let caps = backend.capabilities();
        wgpu.set_item("available", true)?;
        wgpu.set_item("supports_1d", caps.supports_1d)?;
        wgpu.set_item("supports_2d", caps.supports_2d)?;
        wgpu.set_item("supports_3d", caps.supports_3d)?;
        wgpu.set_item("supports_real_to_complex", caps.supports_real_to_complex)?;
        wgpu.set_item("supports_mixed_precision", caps.supports_mixed_precision)?;
        wgpu.set_item(
            "default_precision_profile",
            precision_name(caps.default_precision_profile),
        )?;
        wgpu.set_item(
            "supported_precision_profiles",
            caps.supported_precision_profiles
                .iter()
                .map(|profile| precision_name(*profile))
                .collect::<Vec<_>>(),
        )?;
    } else {
        wgpu.set_item("available", false)?;
        wgpu.set_item("supports_1d", false)?;
        wgpu.set_item("supports_2d", false)?;
        wgpu.set_item("supports_3d", false)?;
        wgpu.set_item("supports_real_to_complex", false)?;
        wgpu.set_item("supports_mixed_precision", false)?;
        wgpu.set_item("default_precision_profile", "low_precision")?;
        wgpu.set_item("supported_precision_profiles", vec!["low_precision"])?;
    }
    backends.set_item("wgpu", wgpu)?;

    Ok(backends.into_any().unbind())
}

/// Python module entry point.
#[pymodule]
fn _pyapollofft(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFftPlan1D>()?;
    m.add_class::<PyFftPlan2D>()?;
    m.add_class::<PyFftPlan3D>()?;
    // Real-to-complex FFT
    m.add_function(wrap_pyfunction!(fft1, m)?)?;
    m.add_function(wrap_pyfunction!(ifft1, m)?)?;
    m.add_function(wrap_pyfunction!(fft2, m)?)?;
    m.add_function(wrap_pyfunction!(ifft2, m)?)?;
    m.add_function(wrap_pyfunction!(fft3, m)?)?;
    m.add_function(wrap_pyfunction!(ifft3, m)?)?;
    m.add_function(wrap_pyfunction!(rfft3, m)?)?;
    m.add_function(wrap_pyfunction!(irfft3, m)?)?;
    // Complex-to-complex FFT
    m.add_function(wrap_pyfunction!(fft_complex1, m)?)?;
    m.add_function(wrap_pyfunction!(ifft_complex1, m)?)?;
    m.add_function(wrap_pyfunction!(fft_complex2, m)?)?;
    m.add_function(wrap_pyfunction!(ifft_complex2, m)?)?;
    m.add_function(wrap_pyfunction!(fft_complex3, m)?)?;
    m.add_function(wrap_pyfunction!(ifft_complex3, m)?)?;
    // FFT frequency and shift utilities
    m.add_function(wrap_pyfunction!(fftfreq_py, m)?)?;
    m.add_function(wrap_pyfunction!(rfftfreq_py, m)?)?;
    m.add_function(wrap_pyfunction!(fftshift_py, m)?)?;
    m.add_function(wrap_pyfunction!(ifftshift_py, m)?)?;
    // NUFFT
    m.add_function(wrap_pyfunction!(nufft_type1_1d_py, m)?)?;
    m.add_function(wrap_pyfunction!(nufft_type2_1d_py, m)?)?;
    m.add_function(wrap_pyfunction!(nufft_type1_3d_py, m)?)?;
    m.add_function(wrap_pyfunction!(nufft_type1_1d_fast_py, m)?)?;
    m.add_function(wrap_pyfunction!(nufft_type2_1d_fast_py, m)?)?;
    m.add_function(wrap_pyfunction!(nufft_type1_3d_fast_py, m)?)?;
    // Discrete Hartley Transform
    m.add_function(wrap_pyfunction!(dht1, m)?)?;
    m.add_function(wrap_pyfunction!(idht1, m)?)?;
    m.add_function(wrap_pyfunction!(dht2, m)?)?;
    m.add_function(wrap_pyfunction!(idht2, m)?)?;
    m.add_function(wrap_pyfunction!(dht3, m)?)?;
    m.add_function(wrap_pyfunction!(idht3, m)?)?;
    // Fast Walsh-Hadamard Transform
    m.add_function(wrap_pyfunction!(fwht1, m)?)?;
    m.add_function(wrap_pyfunction!(ifwht1, m)?)?;
    m.add_function(wrap_pyfunction!(fwht2, m)?)?;
    m.add_function(wrap_pyfunction!(ifwht2, m)?)?;
    m.add_function(wrap_pyfunction!(fwht3, m)?)?;
    m.add_function(wrap_pyfunction!(ifwht3, m)?)?;
    // DCT/DST
    m.add_function(wrap_pyfunction!(dct2_1d, m)?)?;
    m.add_function(wrap_pyfunction!(idct2_1d, m)?)?;
    m.add_function(wrap_pyfunction!(dst2_1d, m)?)?;
    m.add_function(wrap_pyfunction!(idst2_1d, m)?)?;
    // Backend introspection
    m.add_function(wrap_pyfunction!(available_backends, m)?)?;
    m.add_function(wrap_pyfunction!(backend_capabilities, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
