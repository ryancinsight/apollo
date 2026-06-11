//! 3D FFT plan.
//!
//! Apollo-owned 3D FFT implementation based on separable FFT passes.

pub(crate) mod helpers;
pub(crate) mod static_impl;
pub(crate) mod dynamic_impl;

#[cfg(test)]
pub(crate) mod tests;

pub use static_impl::StaticFftPlan3D;
pub use dynamic_impl::FftPlan3D;

/// Use Moirai parallel iteration when total elements exceed this threshold.
pub(crate) const MOIRAI_PARALLEL_THRESHOLD: usize = 32768;

/// Tile size for cache-blocked gather/scatter in axis-1 and axis-0 passes.
pub(crate) const GATHER_TILE: usize = 32;
