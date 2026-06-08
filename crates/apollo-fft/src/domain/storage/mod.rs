//! Domain-owned FFT storage layout contracts.
//!
//! This module is the single source of truth for complex sample layout. The
//! application layer depends on this trait for orchestration, and
//! infrastructure kernels implement concrete loads and stores through it.

mod contract;
mod views;

pub use contract::{FftSample, FftStorage};
pub use views::{FftInterleavedCow, FftInterleavedMut, FftPlanarMut};
