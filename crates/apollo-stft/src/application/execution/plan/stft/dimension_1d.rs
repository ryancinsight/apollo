//! 1D Short-Time Fourier Transform plan.

pub mod helpers;
pub mod plan;

#[cfg(test)]
mod tests;

pub use helpers::is_valid_length;
pub use plan::StftPlan;

#[cfg(test)]
pub(crate) use helpers::{
    forward_window_workspace_capacity, inverse_wola_workspace_capacities,
    typed_workspace_capacities, window_complex_real_frame_into, window_signal_frame_into,
    HERMES_WINDOW_FRAME_THRESHOLD,
};
