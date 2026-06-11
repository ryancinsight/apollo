//! 1D FFT plan.
//!
//! Apollo-owned 1D FFT implementation.

pub(crate) mod executors;
pub(crate) mod helpers;
pub(crate) mod static_impl;
pub(crate) mod dynamic_impl;

#[cfg(test)]
pub(crate) mod plan_tests;

pub use static_impl::StaticFftPlan1D;
pub use dynamic_impl::FftPlan1D;
