//! CUDA C source for the shared typed radix-two FFT descriptors.

use std::borrow::Cow;

use hephaestus_core::{CudaC, KernelSource};

use crate::infrastructure::transport::fft::kernel::{FftEntry, FftKernel};

/// CUDA C kernel source for the shared typed radix-two FFT descriptors.
///
/// The bound names [`FftEntry`] directly: every entry point the
/// dialect-independent descriptors define is represented by `shaders/fft.cu`.
impl<E: FftEntry> KernelSource<CudaC> for FftKernel<f32, E> {
    const ENTRY: &'static str = E::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(include_str!("shaders/fft.cu"))
    }
}
