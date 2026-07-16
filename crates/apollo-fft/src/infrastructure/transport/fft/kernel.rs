//! Typed dense-FFT kernel interface shared by accelerator dialects.

use core::marker::PhantomData;

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{BindingDecl, KernelInterface};

/// Threads launched in one accelerator workgroup.
pub(crate) const WORKGROUP_SIZE: u32 = 256;

/// Per-dispatch radix FFT values shared by all kernel dialects.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct FftParams {
    pub(crate) n: u32,
    pub(crate) stage: u32,
    pub(crate) inverse: u32,
    pub(crate) batch_count: u32,
}

const _: () = assert!(core::mem::size_of::<FftParams>() == 16);

/// Dialect-independent identity of one FFT entry point.
pub(crate) trait FftEntry {
    const LABEL: &'static str;
    const ENTRY: &'static str;
}

/// Zero-sized typed descriptor for one split-complex FFT operation.
pub(crate) struct FftKernel<T, E>(PhantomData<(T, E)>);

impl<T, E> FftKernel<T, E> {
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Pod, E: FftEntry> KernelInterface for FftKernel<T, E> {
    type Params = FftParams;
    const LABEL: &'static str = E::LABEL;
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_write::<T>(),
        BindingDecl::read_write::<T>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE, 1, 1];
}

pub(crate) struct BitReverse;
pub(crate) struct RadixFourBitReverse;
pub(crate) struct Butterfly;
pub(crate) struct RadixFourButterfly;
pub(crate) struct Scale;

impl FftEntry for BitReverse {
    const LABEL: &'static str = "apollo-fft-bit-reverse";
    const ENTRY: &'static str = "fft_bitrev";
}
impl FftEntry for RadixFourBitReverse {
    const LABEL: &'static str = "apollo-fft-radix-four-bit-reverse";
    const ENTRY: &'static str = "fft_bitrev_radix4";
}
impl FftEntry for Butterfly {
    const LABEL: &'static str = "apollo-fft-butterfly";
    const ENTRY: &'static str = "fft_forward";
}
impl FftEntry for RadixFourButterfly {
    const LABEL: &'static str = "apollo-fft-radix-four-butterfly";
    const ENTRY: &'static str = "fft_forward_radix4";
}
impl FftEntry for Scale {
    const LABEL: &'static str = "apollo-fft-scale";
    const ENTRY: &'static str = "fft_scale";
}
