#![warn(missing_docs)]
//! Python bindings for Apollo FFT and NUFFT.

#![allow(clippy::unused_self)]
#![allow(clippy::elidable_lifetime_names)]

pub mod application;
mod bindings;
pub mod domain;
pub mod infrastructure;

use pyo3::prelude::*;

use bindings::backend::{available_backends, backend_capabilities};
use bindings::dctdst::{dct2_1d, dst2_1d, idct2_1d, idst2_1d};
use bindings::dht::{dht1, dht2, dht3, idht1, idht2, idht3};
use bindings::fft::{fft1, fft2, fft3, ifft1, ifft2, ifft3, irfft3, rfft3};
use bindings::fft_complex::{
    fft_complex1, fft_complex2, fft_complex3, fftfreq_py, fftshift_py, ifft_complex1,
    ifft_complex2, ifft_complex3, ifftshift_py, rfftfreq_py,
};
use bindings::fwht::{fwht1, fwht2, fwht3, ifwht1, ifwht2, ifwht3};
use bindings::nufft::{
    nufft_type1_1d_fast_py, nufft_type1_1d_py, nufft_type1_3d_fast_py, nufft_type1_3d_py,
    nufft_type2_1d_fast_py, nufft_type2_1d_py,
};
use bindings::plans::{PyFftPlan1D, PyFftPlan2D, PyFftPlan3D};

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
