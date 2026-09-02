//! 1D Short-Time Fourier Transform plan.

pub mod plan;
pub mod windowing;

#[cfg(test)]
mod tests;

pub use plan::StftPlan;
pub use windowing::is_valid_length;

#[cfg(test)]
pub(crate) use windowing::{
    inverse_real_lane_workspace_capacity, inverse_wola_workspace_capacities,
    typed_workspace_capacities, window_complex_real_frame_into, window_signal_frame_into,
    HERMES_WINDOW_FRAME_THRESHOLD,
};
