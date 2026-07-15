//! Typed Hephaestus kernel descriptors for dense FFT storage.
//!
//! Each descriptor is zero-sized.  Apollo owns the FFT equations and WGSL;
//! Hephaestus owns pipeline construction, binding validation, command encoding,
//! submission, and transfers.

use std::{borrow::Cow, marker::PhantomData};

use crate::f16 as HalfF16;
use bytemuck::{Pod, Zeroable};
use hephaestus_core::{BindingDecl, KernelInterface, KernelSource, Wgsl};

pub(crate) const WORKGROUP_SIZE: u32 = 256;

mod storage_sealed {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for u16 {}
}

/// Physical storage and WGSL source contract for one dense-FFT precision.
///
/// `u16` stores native half-precision bit patterns while its WGSL source
/// declares `array<f16>`; the typed binding still validates the two-byte
/// physical layout at the provider boundary.
pub trait FftStorage: Pod + storage_sealed::Sealed {
    /// WGSL source containing radix and scaling entries for this storage.
    const FFT_SOURCE: &'static str;
    /// WGSL source containing axis pack and unpack entries for this storage.
    const PACK_SOURCE: &'static str;
    /// WGSL source containing Bluestein chirp entries for this storage.
    const CHIRP_SOURCE: &'static str;

    /// Whether this source provides the radix-four entry points.
    ///
    /// The planner selects radix two when this is `false`, preserving the
    /// source contract without a separate dispatch implementation.
    const SUPPORTS_RADIX_FOUR: bool;

    /// Encode an f32 coefficient at this storage precision.
    fn encode_coefficient(value: f32) -> Self;
}

impl FftStorage for f32 {
    const FFT_SOURCE: &'static str = include_str!("../shaders/fft.wgsl");
    const PACK_SOURCE: &'static str = include_str!("../shaders/pack.wgsl");
    const CHIRP_SOURCE: &'static str = include_str!("../shaders/chirp.wgsl");
    const SUPPORTS_RADIX_FOUR: bool = true;

    fn encode_coefficient(value: f32) -> Self {
        value
    }
}

impl FftStorage for u16 {
    const FFT_SOURCE: &'static str = include_str!("../shaders/fft_native_f16.wgsl");
    const PACK_SOURCE: &'static str = include_str!("../shaders/pack_native_f16.wgsl");
    const CHIRP_SOURCE: &'static str = include_str!("../shaders/chirp_native_f16.wgsl");
    const SUPPORTS_RADIX_FOUR: bool = false;

    fn encode_coefficient(value: f32) -> Self {
        HalfF16::from_f32(value).to_bits()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct FftParams {
    pub(crate) n: u32,
    pub(crate) stage: u32,
    pub(crate) inverse: u32,
    pub(crate) batch_count: u32,
}

const _: () = assert!(core::mem::size_of::<FftParams>() == 16);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct PackParams {
    pub(crate) n: u32,
    pub(crate) stage: u32,
    pub(crate) inverse: u32,
    pub(crate) batch_count: u32,
    pub(crate) nx: u32,
    pub(crate) ny: u32,
    pub(crate) nz: u32,
    pub(crate) axis: u32,
    pub(crate) fft_len: u32,
    pub(crate) padding: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<PackParams>() == 48);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct ChirpParams {
    pub(crate) n: u32,
    pub(crate) m: u32,
    pub(crate) batch_count: u32,
    pub(crate) padding: u32,
}

const _: () = assert!(core::mem::size_of::<ChirpParams>() == 16);

pub(crate) trait FftEntry {
    const LABEL: &'static str;
    const ENTRY: &'static str;
}

pub(crate) struct FftKernel<T, E>(PhantomData<(T, E)>);

impl<T, E> FftKernel<T, E> {
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: FftStorage, E: FftEntry> KernelInterface for FftKernel<T, E> {
    type Params = FftParams;
    const LABEL: &'static str = E::LABEL;
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_write::<T>(),
        BindingDecl::read_write::<T>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE, 1, 1];
}

impl<T: FftStorage, E: FftEntry> KernelSource<Wgsl> for FftKernel<T, E> {
    const ENTRY: &'static str = E::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(T::FFT_SOURCE)
    }
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

pub(crate) trait PackEntry {
    const LABEL: &'static str;
    const ENTRY: &'static str;
}

pub(crate) struct PackKernel<T, E>(PhantomData<(T, E)>);

impl<T, E> PackKernel<T, E> {
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: FftStorage, E: PackEntry> KernelInterface for PackKernel<T, E> {
    type Params = PackParams;
    const LABEL: &'static str = E::LABEL;
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_write::<T>(),
        BindingDecl::read_write::<T>(),
        BindingDecl::read_write::<T>(),
        BindingDecl::read_write::<T>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE, 1, 1];
}

impl<T: FftStorage, E: PackEntry> KernelSource<Wgsl> for PackKernel<T, E> {
    const ENTRY: &'static str = E::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(T::PACK_SOURCE)
    }
}

pub(crate) struct Pack;
pub(crate) struct Unpack;

impl PackEntry for Pack {
    const LABEL: &'static str = "apollo-fft-pack";
    const ENTRY: &'static str = "fft_pack_axis";
}
impl PackEntry for Unpack {
    const LABEL: &'static str = "apollo-fft-unpack";
    const ENTRY: &'static str = "fft_unpack_axis";
}

pub(crate) trait ChirpEntry {
    const LABEL: &'static str;
    const ENTRY: &'static str;
}

pub(crate) struct ChirpKernel<T, E>(PhantomData<(T, E)>);

impl<T, E> ChirpKernel<T, E> {
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: FftStorage, E: ChirpEntry> KernelInterface for ChirpKernel<T, E> {
    type Params = ChirpParams;
    const LABEL: &'static str = E::LABEL;
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_write::<T>(),
        BindingDecl::read_write::<T>(),
        BindingDecl::read_only::<T>(),
        BindingDecl::read_only::<T>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE, 1, 1];
}

impl<T: FftStorage, E: ChirpEntry> KernelSource<Wgsl> for ChirpKernel<T, E> {
    const ENTRY: &'static str = E::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(T::CHIRP_SOURCE)
    }
}

pub(crate) struct ChirpPremultiply;
pub(crate) struct ChirpPointMultiply;
pub(crate) struct ChirpScale;
pub(crate) struct ChirpPostmultiply;
pub(crate) struct ChirpNegateImaginary;

impl ChirpEntry for ChirpPremultiply {
    const LABEL: &'static str = "apollo-fft-chirp-premultiply";
    const ENTRY: &'static str = "chirp_premul";
}
impl ChirpEntry for ChirpPointMultiply {
    const LABEL: &'static str = "apollo-fft-chirp-point-multiply";
    const ENTRY: &'static str = "chirp_pointmul";
}
impl ChirpEntry for ChirpScale {
    const LABEL: &'static str = "apollo-fft-chirp-scale";
    const ENTRY: &'static str = "chirp_scale";
}
impl ChirpEntry for ChirpPostmultiply {
    const LABEL: &'static str = "apollo-fft-chirp-postmultiply";
    const ENTRY: &'static str = "chirp_postmul";
}
impl ChirpEntry for ChirpNegateImaginary {
    const LABEL: &'static str = "apollo-fft-chirp-negate-imaginary";
    const ENTRY: &'static str = "chirp_negate_im";
}
