//! Backend availability and capability introspection for Python callers.

use apollo_fft::{CpuBackend, FftBackend};
use hephaestus_core::HephaestusError;
use hephaestus_wgpu::WgpuDevice;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::support::precision_name;

fn probe_wgpu_device() -> PyResult<Option<WgpuDevice>> {
    match WgpuDevice::try_default("apollo-python-wgpu") {
        Ok(device) => Ok(Some(device)),
        Err(HephaestusError::AdapterUnavailable { .. }) => Ok(None),
        Err(error) => Err(PyRuntimeError::new_err(format!(
            "Apollo Python WGPU provider acquisition failed: {error}"
        ))),
    }
}

/// Return the backend names that are genuinely usable from Python on this host.
#[pyfunction]
pub(crate) fn available_backends() -> PyResult<Vec<String>> {
    let mut backends = vec!["cpu".to_string()];
    if probe_wgpu_device()?.is_some() {
        backends.push("wgpu".to_string());
    }
    Ok(backends)
}

/// Return backend capability metadata for Python callers.
#[pyfunction]
pub(crate) fn backend_capabilities(py: Python<'_>) -> PyResult<Py<PyAny>> {
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
    if let Some(device) = probe_wgpu_device()? {
        let backend = apollo_fft::WgpuBackend::new(device);
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
