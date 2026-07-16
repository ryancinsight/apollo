//! Shared device acquisition, fixtures, and analytical bounds for CZT tests.

use eunomia::Complex32;

use crate::infrastructure::transport::gpu::CztWgpuBackend;

// Each direct term performs two polar reconstructions and complex products.
// Budgeting 512 eps per term covers those elementary operations; the four-term
// differential therefore carries 4 * 2 * 512 eps.
pub(super) const DIRECT_DIFFERENTIAL_BOUND: f64 = 4096.0 * f32::EPSILON as f64;
// The eight-point DFT roundtrip applies two direct transforms.
pub(super) const DFT_ROUNDTRIP_BOUND: f32 = 8192.0 * f32::EPSILON;

pub(super) fn backend() -> Option<CztWgpuBackend> {
    match hephaestus_wgpu::WgpuDevice::try_default("apollo-czt-wgpu") {
        Ok(device) => Some(CztWgpuBackend::new(device)),
        Err(hephaestus_core::HephaestusError::AdapterUnavailable { .. }) => None,
        Err(error) => panic!("CZT GPU verification requires a working provider: {error}"),
    }
}

pub(super) fn reference_parameters() -> (Complex32, Complex32) {
    (
        Complex32::new(0.95, 0.1),
        Complex32::from_polar(1.0, -std::f32::consts::TAU / 9.0),
    )
}

pub(super) fn reference_input() -> [Complex32; 4] {
    [
        Complex32::new(1.0, 0.0),
        Complex32::new(-0.5, 1.0),
        Complex32::new(0.25, -0.75),
        Complex32::new(1.25, 0.5),
    ]
}

pub(super) fn dft_parameters(len: usize) -> (Complex32, Complex32) {
    (
        Complex32::new(1.0, 0.0),
        Complex32::from_polar(1.0, -std::f32::consts::TAU / len as f32),
    )
}

pub(super) fn dft_input(len: usize) -> Vec<Complex32> {
    (0..len)
        .map(|index| Complex32::new((index as f32 * 0.7).sin(), (index as f32 * 0.31).cos()))
        .collect()
}
