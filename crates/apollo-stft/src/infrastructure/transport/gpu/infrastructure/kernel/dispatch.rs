//! STFT geometry and dispatch-grid validation.

use apollo_fft::{WgpuError, WgpuResult};
use hephaestus_core::DispatchGrid;

use super::{FRAME_WORKGROUP, OLA_WORKGROUP};

/// Build a dispatch grid covering frame-plane elements.
pub(crate) fn fft_grid(elements: usize) -> WgpuResult<DispatchGrid> {
    DispatchGrid::covering_domain([elements, 1, 1], [FRAME_WORKGROUP, 1, 1]).map_err(Into::into)
}

/// Build a dispatch grid covering overlap-add output samples.
pub(crate) fn ola_grid(elements: usize) -> WgpuResult<DispatchGrid> {
    DispatchGrid::covering_domain([elements, 1, 1], [OLA_WORKGROUP, 1, 1]).map_err(Into::into)
}

/// Validate that a host dimension has an accelerator representation.
pub(crate) fn dimension(value: usize, name: &'static str) -> WgpuResult<u32> {
    u32::try_from(value).map_err(|_| WgpuError::InvalidPlan {
        message: format!("{name} exceeds accelerator u32 range: {value}"),
    })
}
