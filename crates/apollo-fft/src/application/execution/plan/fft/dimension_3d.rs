//! 3D FFT plan.
//!
//! Apollo-owned 3D FFT implementation based on separable FFT passes.

pub(crate) mod dynamic_impl;
mod passes;
mod rotated;
pub(crate) mod static_impl;

#[cfg(test)]
pub(crate) mod tests;
// Windows-gated like `base128::pinned_probe`: reads processor classes.
#[cfg(all(test, windows, target_arch = "x86_64"))]
mod pass_attribution;

pub use dynamic_impl::FftPlan3D;
pub use rotated::RotatedSpectrum;
pub use static_impl::StaticFftPlan3D;
