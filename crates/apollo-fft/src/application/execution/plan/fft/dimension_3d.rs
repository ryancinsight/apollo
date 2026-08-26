//! 3D FFT plan.
//!
//! Apollo-owned 3D FFT implementation based on separable FFT passes.

pub(crate) mod dynamic_impl;
pub(crate) mod static_impl;
pub(crate) mod twiddles;

#[cfg(test)]
pub(crate) mod tests;

pub use dynamic_impl::FftPlan3D;
pub use static_impl::StaticFftPlan3D;

/// Use Moirai parallel iteration when total elements exceed this threshold.
pub(crate) const MOIRAI_PARALLEL_THRESHOLD: usize = 32768;
