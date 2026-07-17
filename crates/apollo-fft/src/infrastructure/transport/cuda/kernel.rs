//! CUDA C source for the shared typed radix-two FFT descriptors.

use std::borrow::Cow;

use hephaestus_core::{CudaC, KernelSource};

use crate::infrastructure::transport::fft::kernel::{
    BitReverse, Butterfly, FftEntry, FftKernel, Scale,
};

/// Entries represented by the CUDA radix-two source.
pub(crate) trait CudaFftEntry: FftEntry {}

impl CudaFftEntry for BitReverse {}
impl CudaFftEntry for Butterfly {}
impl CudaFftEntry for Scale {}

impl<E: CudaFftEntry> KernelSource<CudaC> for FftKernel<f32, E> {
    const ENTRY: &'static str = E::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(include_str!("shaders/fft.cu"))
    }
}
