//! Power-of-two CPU SIMD FFT kernels.

pub(crate) use crate::application::execution::kernel::radix_shape;

/// Radix-2 Cooley-Tukey kernels.
pub mod radix2;
