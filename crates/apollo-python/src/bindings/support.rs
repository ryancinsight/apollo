//! Shared Python array layout checks and precision-profile helpers used by all
//! binding modules. Conversion-only logic; no domain computation.

use apollo_fft::{Complex32, Complex64, PrecisionMode, PrecisionProfile, StoragePrecision};
use leto::{Array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Layout, Storage};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub(crate) trait PyArrayElement: bytemuck::Pod {
    const DTYPE: &'static str;
}

impl PyArrayElement for f64 {
    const DTYPE: &'static str = "float64";
}

impl PyArrayElement for f32 {
    const DTYPE: &'static str = "float32";
}

impl PyArrayElement for Complex64 {
    const DTYPE: &'static str = "complex128";
}

impl PyArrayElement for Complex32 {
    const DTYPE: &'static str = "complex64";
}

pub(crate) struct PyReadonlyArray<T, const N: usize> {
    shape: [usize; N],
    values: Vec<T>,
    is_c_contiguous: bool,
}

pub(crate) type PyReadonlyArray1<T> = PyReadonlyArray<T, 1>;
pub(crate) type PyReadonlyArray2<T> = PyReadonlyArray<T, 2>;
pub(crate) type PyReadonlyArray3<T> = PyReadonlyArray<T, 3>;

impl<T, const N: usize> PyReadonlyArray<T, N> {
    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(crate) fn is_c_contiguous(&self) -> bool {
        self.is_c_contiguous
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        &self.values
    }
}

impl<T: PyArrayElement, const N: usize> PyReadonlyArray<T, N> {
    fn from_py_array(input: &Bound<'_, PyAny>) -> PyResult<Self> {
        let ndim = input
            .getattr("ndim")
            .and_then(|value| value.extract::<usize>())
            .map_err(|_| PyValueError::new_err("input must be a NumPy array"))?;
        if ndim != N {
            return Err(PyValueError::new_err(format!(
                "input must be {N}D, got {ndim}D"
            )));
        }

        let dtype_name = input
            .getattr("dtype")?
            .getattr("name")?
            .extract::<String>()?;
        if dtype_name != T::DTYPE {
            return Err(PyValueError::new_err(format!(
                "input must have dtype {}, got {dtype_name}",
                T::DTYPE
            )));
        }

        let shape_vec = input.getattr("shape")?.extract::<Vec<usize>>()?;
        let shape: [usize; N] = shape_vec
            .try_into()
            .map_err(|_| PyValueError::new_err(format!("input must be {N}D")))?;
        let is_c_contiguous = input
            .getattr("flags")?
            .getattr("c_contiguous")?
            .extract::<bool>()?;
        let bytes_any = input.call_method0("tobytes")?;
        let bytes = bytes_any.cast::<PyBytes>()?;
        let values = bytemuck::try_cast_slice::<u8, T>(bytes.as_bytes())
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .to_vec();
        let expected_len = shape
            .iter()
            .try_fold(1_usize, |acc, extent| acc.checked_mul(*extent))
            .ok_or_else(|| PyValueError::new_err("input shape product overflows usize"))?;
        if values.len() != expected_len {
            return Err(PyValueError::new_err(format!(
                "input shape product {expected_len} does not match buffer length {}",
                values.len()
            )));
        }

        Ok(Self {
            shape,
            values,
            is_c_contiguous,
        })
    }
}

impl<'py, T: PyArrayElement, const N: usize> FromPyObject<'_, 'py> for PyReadonlyArray<T, N> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        Self::from_py_array(&obj)
    }
}

fn row_major_layout<const N: usize>(shape: [usize; N]) -> Layout<N> {
    let mut strides = [0_isize; N];
    let mut stride = 1_isize;
    for axis in (0..N).rev() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(shape[axis] as isize)
            .expect("invariant: NumPy shape product fits isize on this platform");
    }
    Layout::new(shape, strides, 0)
}

fn shape2(shape: &[usize]) -> [usize; 2] {
    [shape[0], shape[1]]
}

fn shape3(shape: &[usize]) -> [usize; 3] {
    [shape[0], shape[1], shape[2]]
}

fn leto_shape_error(error: leto::LetoError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) fn require_contiguous_1d<T>(input: &PyReadonlyArray1<T>, name: &str) -> PyResult<()> {
    if input.is_c_contiguous() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )))
    }
}

pub(crate) fn require_contiguous_2d<T>(input: &PyReadonlyArray2<T>, name: &str) -> PyResult<()> {
    if input.is_c_contiguous() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )))
    }
}

pub(crate) fn require_contiguous_3d<T>(input: &PyReadonlyArray3<T>, name: &str) -> PyResult<()> {
    if input.is_c_contiguous() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )))
    }
}

pub(crate) fn py_array1_slice<'a, T>(
    input: &'a PyReadonlyArray1<T>,
    name: &str,
) -> PyResult<&'a [T]> {
    require_contiguous_1d(input, name)?;
    Ok(input.as_slice())
}

pub(crate) fn py_array2_slice<'a, T>(
    input: &'a PyReadonlyArray2<T>,
    name: &str,
) -> PyResult<&'a [T]> {
    require_contiguous_2d(input, name)?;
    Ok(input.as_slice())
}

pub(crate) fn py_array3_slice<'a, T>(
    input: &'a PyReadonlyArray3<T>,
    name: &str,
) -> PyResult<&'a [T]> {
    require_contiguous_3d(input, name)?;
    Ok(input.as_slice())
}

pub(crate) fn py_array1_leto_view<'a, T>(
    input: &'a PyReadonlyArray1<T>,
    name: &str,
) -> PyResult<ArrayView1<'a, T>> {
    let slice = py_array1_slice(input, name)?;
    Ok(ArrayView1::new(row_major_layout([slice.len()]), slice))
}

pub(crate) fn py_array2_leto_view<'a, T>(
    input: &'a PyReadonlyArray2<T>,
    name: &str,
) -> PyResult<ArrayView2<'a, T>> {
    let slice = py_array2_slice(input, name)?;
    let shape = shape2(input.shape());
    Ok(ArrayView2::new(row_major_layout(shape), slice))
}

pub(crate) fn py_array3_leto_view<'a, T>(
    input: &'a PyReadonlyArray3<T>,
    name: &str,
) -> PyResult<ArrayView3<'a, T>> {
    let slice = py_array3_slice(input, name)?;
    let shape = shape3(input.shape());
    Ok(ArrayView3::new(row_major_layout(shape), slice))
}

pub(crate) fn py_array1_to_leto<T: Clone>(
    input: &PyReadonlyArray1<T>,
    name: &str,
) -> PyResult<Array1<T>> {
    let slice = py_array1_slice(input, name)?;
    Array1::from_shape_vec([slice.len()], slice.to_vec()).map_err(leto_shape_error)
}

pub(crate) fn py_array2_to_leto<T: Clone>(
    input: &PyReadonlyArray2<T>,
    name: &str,
) -> PyResult<Array2<T>> {
    let slice = py_array2_slice(input, name)?;
    Array2::from_shape_vec(shape2(input.shape()), slice.to_vec()).map_err(leto_shape_error)
}

pub(crate) fn py_array3_to_leto<T: Clone>(
    input: &PyReadonlyArray3<T>,
    name: &str,
) -> PyResult<Array3<T>> {
    let slice = py_array3_slice(input, name)?;
    Array3::from_shape_vec(shape3(input.shape()), slice.to_vec()).map_err(leto_shape_error)
}

pub(crate) fn py_array1_map_to_leto<T, U, F>(
    input: &PyReadonlyArray1<T>,
    name: &str,
    f: F,
) -> PyResult<Array1<U>>
where
    F: FnMut(&T) -> U,
{
    let slice = py_array1_slice(input, name)?;
    Array1::from_shape_vec([slice.len()], slice.iter().map(f).collect()).map_err(leto_shape_error)
}

pub(crate) fn py_array2_map_to_leto<T, U, F>(
    input: &PyReadonlyArray2<T>,
    name: &str,
    f: F,
) -> PyResult<Array2<U>>
where
    F: FnMut(&T) -> U,
{
    let slice = py_array2_slice(input, name)?;
    Array2::from_shape_vec(shape2(input.shape()), slice.iter().map(f).collect())
        .map_err(leto_shape_error)
}

pub(crate) fn py_array3_map_to_leto<T, U, F>(
    input: &PyReadonlyArray3<T>,
    name: &str,
    f: F,
) -> PyResult<Array3<U>>
where
    F: FnMut(&T) -> U,
{
    let slice = py_array3_slice(input, name)?;
    Array3::from_shape_vec(shape3(input.shape()), slice.iter().map(f).collect())
        .map_err(leto_shape_error)
}

pub(crate) fn parse_precision(precision: Option<&str>) -> PyResult<PrecisionProfile> {
    match precision.unwrap_or("high_accuracy") {
        "high_accuracy" => Ok(PrecisionProfile::HIGH_ACCURACY_F64),
        "low_precision" => Ok(PrecisionProfile::LOW_PRECISION_F32),
        "mixed_precision" => Ok(PrecisionProfile::MIXED_PRECISION_F16_F32),
        other => Err(PyValueError::new_err(format!(
            "unsupported precision `{other}`; expected `high_accuracy`, `low_precision`, or `mixed_precision`"
        ))),
    }
}

pub(crate) fn precision_name(profile: PrecisionProfile) -> &'static str {
    match profile.mode {
        PrecisionMode::HighAccuracy => "high_accuracy",
        PrecisionMode::LowPrecision => "low_precision",
        PrecisionMode::MixedPrecision => "mixed_precision",
    }
}

pub(crate) fn require_profile_matches_f64(profile: PrecisionProfile, name: &str) -> PyResult<()> {
    if profile.storage == StoragePrecision::F64 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} received float64/complex128 input but precision `{}` expects float32/complex64 storage",
            precision_name(profile)
        )))
    }
}

pub(crate) fn require_profile_matches_f32(profile: PrecisionProfile, name: &str) -> PyResult<()> {
    if profile.storage == StoragePrecision::F32 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} received float32/complex64 input but precision `{}` expects float64/complex128 storage",
            precision_name(profile)
        )))
    }
}

pub(crate) fn leto_array1_into_pyarray<'py, T>(
    py: Python<'py>,
    array: Array<T, impl Storage<T>, 1>,
) -> PyResult<Py<PyAny>>
where
    T: PyArrayElement,
{
    let values = array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Leto array must be C-contiguous"))?;
    values_into_pyarray(py, values, [values.len()])
}

pub(crate) fn leto_array2_into_pyarray<'py, T>(
    py: Python<'py>,
    array: Array<T, impl Storage<T>, 2>,
) -> PyResult<Py<PyAny>>
where
    T: PyArrayElement,
{
    let shape = array.shape();
    let values = array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Leto array must be C-contiguous"))?;
    values_into_pyarray(py, values, shape)
}

pub(crate) fn leto_array3_into_pyarray<'py, T>(
    py: Python<'py>,
    array: Array<T, impl Storage<T>, 3>,
) -> PyResult<Py<PyAny>>
where
    T: PyArrayElement,
{
    let shape = array.shape();
    let values = array
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Leto array must be C-contiguous"))?;
    values_into_pyarray(py, values, shape)
}

pub(crate) fn vec1_into_pyarray<'py, T: PyArrayElement>(
    py: Python<'py>,
    values: Vec<T>,
) -> PyResult<Py<PyAny>> {
    values_into_pyarray(py, &values, [values.len()])
}

pub(crate) fn values_into_pyarray<'py, T: PyArrayElement, const N: usize>(
    py: Python<'py>,
    values: &[T],
    shape: [usize; N],
) -> PyResult<Py<PyAny>> {
    let numpy = py.import("numpy")?;
    let bytes = PyBytes::new(py, bytemuck::cast_slice(values));
    let array = numpy.getattr("frombuffer")?.call1((bytes, T::DTYPE))?;
    let copied = array.call_method0("copy")?;
    let reshaped = copied.call_method1("reshape", (shape.to_vec(),))?;
    Ok(reshaped.into_any().unbind())
}

#[cfg(test)]
mod tests {
    use super::{py_array2_to_leto, values_into_pyarray, PyReadonlyArray2};
    use pyo3::types::PyAnyMethods;
    use pyo3::Python;

    #[test]
    fn numpy_boundary_preserves_shape_and_values() {
        Python::initialize();
        Python::attach(|py| {
            let numpy = py.import("numpy").expect("numpy import");
            let input = numpy
                .getattr("array")
                .expect("numpy.array")
                .call1((vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]],))
                .expect("construct array");

            let typed = input.extract::<PyReadonlyArray2<f64>>().expect("extract");
            assert_eq!(typed.shape(), [2, 2]);
            assert_eq!(typed.as_slice(), &[1.0, 2.0, 3.0, 4.0]);

            let leto = py_array2_to_leto(&typed, "input").expect("leto array");
            assert_eq!(leto.shape(), [2, 2]);
            assert_eq!(leto.as_slice(), Some(&[1.0, 2.0, 3.0, 4.0][..]));

            let output =
                values_into_pyarray(py, &[5.0_f64, 6.0, 7.0, 8.0], [2, 2]).expect("numpy output");
            let output = output.bind(py);
            let shape = output
                .getattr("shape")
                .expect("shape")
                .extract::<Vec<usize>>()
                .expect("shape vec");
            assert_eq!(shape, vec![2, 2]);
            let values = output
                .call_method0("tolist")
                .expect("tolist")
                .extract::<Vec<Vec<f64>>>()
                .expect("nested values");
            assert_eq!(values, vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        });
    }
}
