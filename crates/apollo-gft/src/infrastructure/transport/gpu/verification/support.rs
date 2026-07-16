//! Shared device acquisition, path-graph fixture, and derived GFT test bound.

use crate::infrastructure::transport::gpu::GftWgpuBackend;

/// Four products plus three additions have first-order error `gamma_7`.
/// `64 * epsilon_f32 = 2^-17` conservatively covers that bound, f64-to-f32
/// basis quantization, and the second kernel launch in the path-four roundtrip
/// without masking a transform-scale error.
pub(super) const PATH4_F32_DOT_ABS_TOLERANCE: f64 = 1.0 / 131_072.0;

pub(super) fn backend() -> Option<GftWgpuBackend> {
    match hephaestus_wgpu::WgpuDevice::try_default("apollo-gft-wgpu") {
        Ok(device) => Some(GftWgpuBackend::new(device)),
        Err(hephaestus_core::HephaestusError::AdapterUnavailable { .. }) => None,
        Err(error) => panic!("GFT GPU verification requires a working provider: {error}"),
    }
}

/// Builds the path-four CPU plan and extracts its basis and signal as `f32`.
pub(super) fn path4_plan_and_basis() -> (crate::GftPlan, Vec<f32>, Vec<f32>) {
    let adjacency = leto::Array2::from_shape_vec(
        [4, 4],
        vec![
            0.0_f64, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ],
    )
    .expect("path-four adjacency shape");
    let plan = crate::GftPlan::from_adjacency(adjacency.view()).expect("path-four GFT plan");
    let basis = plan.basis().iter().map(|&value| value as f32).collect();
    let signal = vec![1.0_f32, -0.5, 2.0, 0.5];
    (plan, basis, signal)
}
