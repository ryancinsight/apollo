//! Typed Hephaestus kernels for the short-time Fourier transform.
//!
//! The WGSL sources remain the numerical implementation.  This module owns
//! their binding declarations, parameter ABI, and dispatch tiles once, while
//! the forward, inverse, and Bluestein leaves own operation sequencing.

use core::marker::PhantomData;
use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{
    BindingDecl, GroupedBindingDecl, GroupedKernelInterface, GroupedKernelSource, KernelInterface,
    KernelSource, Wgsl,
};

/// Shared geometry validation and typed dispatch operations.
mod dispatch;
/// Execution implementations for power-of-two forward frames.
pub mod forward;
/// Execution implementations for non-power-of-two forward frames.
pub mod forward_chirp;
/// Execution implementations for power-of-two inverse frames.
pub mod inverse;
/// Execution implementations for non-power-of-two inverse frames.
pub mod inverse_chirp;

pub(crate) use dispatch::{
    chirp_frequency_kernel, chirp_padded_len, dimension, dispatch_chirp_radix, dispatch_grouped,
    fft_grid, ola_grid,
};

pub(crate) const OLA_WORKGROUP: usize = 64;
pub(crate) const FFT_WORKGROUP: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct ComplexPod {
    pub(crate) re: f32,
    pub(crate) im: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct StftParams {
    pub(crate) signal_len: u32,
    pub(crate) frame_len: u32,
    pub(crate) hop_len: u32,
    pub(crate) frame_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct FftStageParams {
    pub(crate) frame_count: u32,
    pub(crate) frame_len: u32,
    pub(crate) stage: u32,
    pub(crate) padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct FwdFftStageParams {
    pub(crate) frame_count: u32,
    pub(crate) frame_len: u32,
    pub(crate) hop_len: u32,
    pub(crate) stage: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct StftChirpParams {
    pub(crate) frame_count: u32,
    pub(crate) frame_len: u32,
    pub(crate) chirp_len: u32,
    pub(crate) hop_len: u32,
    pub(crate) signal_len: u32,
    pub(crate) padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct ChirpFftParams {
    pub(crate) fft_len: u32,
    pub(crate) stage: u32,
    pub(crate) inverse_flag: u32,
    pub(crate) batch_count: u32,
}

const _: () = assert!(core::mem::size_of::<ComplexPod>() == 8);
const _: () = assert!(core::mem::size_of::<StftParams>() == 16);
const _: () = assert!(core::mem::size_of::<FftStageParams>() == 16);
const _: () = assert!(core::mem::size_of::<FwdFftStageParams>() == 16);
const _: () = assert!(core::mem::size_of::<StftChirpParams>() == 32);
const _: () = assert!(core::mem::size_of::<ChirpFftParams>() == 16);

const FORWARD_BINDINGS: [GroupedBindingDecl; 4] = [
    GroupedBindingDecl::read_only::<f32>(0, 0),
    GroupedBindingDecl::read_write::<f32>(0, 1),
    GroupedBindingDecl::read_write::<f32>(0, 2),
    GroupedBindingDecl::read_write::<ComplexPod>(0, 3),
];
const INVERSE_BINDINGS: [GroupedBindingDecl; 4] = [
    GroupedBindingDecl::read_only::<f32>(0, 0),
    GroupedBindingDecl::read_write::<f32>(0, 1),
    GroupedBindingDecl::read_write::<f32>(0, 2),
    GroupedBindingDecl::read_write::<f32>(0, 3),
];
const CHIRP_FORWARD_BINDINGS: [GroupedBindingDecl; 6] = [
    GroupedBindingDecl::read_write::<f32>(0, 0),
    GroupedBindingDecl::read_write::<f32>(0, 1),
    GroupedBindingDecl::read_only::<f32>(0, 2),
    GroupedBindingDecl::read_only::<f32>(0, 3),
    GroupedBindingDecl::read_only::<f32>(2, 0),
    GroupedBindingDecl::read_write::<ComplexPod>(2, 1),
];
const CHIRP_INVERSE_BINDINGS: [GroupedBindingDecl; 6] = [
    GroupedBindingDecl::read_write::<f32>(0, 0),
    GroupedBindingDecl::read_write::<f32>(0, 1),
    GroupedBindingDecl::read_only::<f32>(0, 2),
    GroupedBindingDecl::read_only::<f32>(0, 3),
    GroupedBindingDecl::read_only::<f32>(2, 0),
    GroupedBindingDecl::read_write::<f32>(2, 1),
];
const CHIRP_FFT_BINDINGS: [GroupedBindingDecl; 4] = [
    GroupedBindingDecl::read_write::<f32>(0, 0),
    GroupedBindingDecl::read_write::<f32>(0, 1),
    GroupedBindingDecl::read_only::<f32>(0, 2),
    GroupedBindingDecl::read_only::<f32>(0, 3),
];
const OLA_BINDINGS: [BindingDecl; 2] = [
    BindingDecl::read_only::<f32>(),
    BindingDecl::read_write::<f32>(),
];

trait GroupedSpec {
    type Params: Pod;
    const LABEL: &'static str;
    const ENTRY: &'static str;
    const SOURCE: &'static str;
    const BINDINGS: &'static [GroupedBindingDecl];
    const WORKGROUP: [u32; 3];
}

struct GroupedKernel<S>(PhantomData<S>);

impl<S> GroupedKernel<S> {
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<S: GroupedSpec> GroupedKernelInterface for GroupedKernel<S> {
    type Params = S::Params;
    const LABEL: &'static str = S::LABEL;
    const BINDINGS: &'static [GroupedBindingDecl] = S::BINDINGS;
    const PARAM_GROUP: u32 = 1;
    const PARAM_BINDING: u32 = 0;
    const WORKGROUP: [u32; 3] = S::WORKGROUP;
}

impl<S: GroupedSpec> GroupedKernelSource<Wgsl> for GroupedKernel<S> {
    const ENTRY: &'static str = S::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(S::SOURCE)
    }
}

trait FlatSpec {
    type Params: Pod;
    const LABEL: &'static str;
    const ENTRY: &'static str;
    const SOURCE: &'static str;
    const BINDINGS: &'static [BindingDecl];
    const WORKGROUP: [u32; 3];
}

struct FlatKernel<S>(PhantomData<S>);

impl<S> FlatKernel<S> {
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<S: FlatSpec> KernelInterface for FlatKernel<S> {
    type Params = S::Params;
    const LABEL: &'static str = S::LABEL;
    const BINDINGS: &'static [BindingDecl] = S::BINDINGS;
    const WORKGROUP: [u32; 3] = S::WORKGROUP;
}

impl<S: FlatSpec> KernelSource<Wgsl> for FlatKernel<S> {
    const ENTRY: &'static str = S::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(S::SOURCE)
    }
}

macro_rules! grouped_specs {
    ($(($marker:ident, $params:ty, $label:literal, $entry:literal, $source:expr, $bindings:expr)),+ $(,)?) => {
        $(
            struct $marker;
            impl GroupedSpec for $marker {
                type Params = $params;
                const LABEL: &'static str = $label;
                const ENTRY: &'static str = $entry;
                const SOURCE: &'static str = $source;
                const BINDINGS: &'static [GroupedBindingDecl] = $bindings;
                const WORKGROUP: [u32; 3] = [FFT_WORKGROUP as u32, 1, 1];
            }
        )+
    };
}

grouped_specs!(
    (
        ForwardPack,
        FwdFftStageParams,
        "apollo-stft-forward-pack",
        "stft_fwd_pack_window",
        include_str!("shaders/stft_forward_fft.wgsl"),
        &FORWARD_BINDINGS
    ),
    (
        ForwardBitReverse,
        FwdFftStageParams,
        "apollo-stft-forward-bit-reverse",
        "stft_fwd_bitrev",
        include_str!("shaders/stft_forward_fft.wgsl"),
        &FORWARD_BINDINGS
    ),
    (
        ForwardButterfly,
        FwdFftStageParams,
        "apollo-stft-forward-butterfly",
        "stft_fwd_butterfly",
        include_str!("shaders/stft_forward_fft.wgsl"),
        &FORWARD_BINDINGS
    ),
    (
        ForwardInterleave,
        FwdFftStageParams,
        "apollo-stft-forward-interleave",
        "stft_fwd_interleave",
        include_str!("shaders/stft_forward_fft.wgsl"),
        &FORWARD_BINDINGS
    ),
    (
        InverseDeinterleave,
        FftStageParams,
        "apollo-stft-inverse-deinterleave",
        "stft_deinterleave",
        include_str!("shaders/stft_inverse_fft.wgsl"),
        &INVERSE_BINDINGS
    ),
    (
        InverseBitReverse,
        FftStageParams,
        "apollo-stft-inverse-bit-reverse",
        "stft_bitrev",
        include_str!("shaders/stft_inverse_fft.wgsl"),
        &INVERSE_BINDINGS
    ),
    (
        InverseButterfly,
        FftStageParams,
        "apollo-stft-inverse-butterfly",
        "stft_butterfly",
        include_str!("shaders/stft_inverse_fft.wgsl"),
        &INVERSE_BINDINGS
    ),
    (
        InverseScaleWindow,
        FftStageParams,
        "apollo-stft-inverse-scale-window",
        "stft_scale_and_window",
        include_str!("shaders/stft_inverse_fft.wgsl"),
        &INVERSE_BINDINGS
    ),
    (
        ChirpForwardPremultiply,
        StftChirpParams,
        "apollo-stft-chirp-forward-premultiply",
        "stft_chirp_premul_fwd",
        include_str!("shaders/stft_chirp.wgsl"),
        &CHIRP_FORWARD_BINDINGS
    ),
    (
        ChirpForwardPointMultiply,
        StftChirpParams,
        "apollo-stft-chirp-forward-point-multiply",
        "stft_chirp_pointmul_fwd",
        include_str!("shaders/stft_chirp.wgsl"),
        &CHIRP_FORWARD_BINDINGS
    ),
    (
        ChirpForwardPostmultiply,
        StftChirpParams,
        "apollo-stft-chirp-forward-postmultiply",
        "stft_chirp_postmul_fwd",
        include_str!("shaders/stft_chirp.wgsl"),
        &CHIRP_FORWARD_BINDINGS
    ),
    (
        ChirpInversePremultiply,
        StftChirpParams,
        "apollo-stft-chirp-inverse-premultiply",
        "stft_chirp_premul_inv",
        include_str!("shaders/stft_chirp.wgsl"),
        &CHIRP_INVERSE_BINDINGS
    ),
    (
        ChirpInversePointMultiply,
        StftChirpParams,
        "apollo-stft-chirp-inverse-point-multiply",
        "stft_chirp_pointmul",
        include_str!("shaders/stft_chirp.wgsl"),
        &CHIRP_INVERSE_BINDINGS
    ),
    (
        ChirpInversePostmultiply,
        StftChirpParams,
        "apollo-stft-chirp-inverse-postmultiply",
        "stft_chirp_postmul_inv",
        include_str!("shaders/stft_chirp.wgsl"),
        &CHIRP_INVERSE_BINDINGS
    ),
    (
        ChirpBitReverse,
        ChirpFftParams,
        "apollo-stft-chirp-bit-reverse",
        "chirp_fft_bitrev",
        include_str!("shaders/stft_chirp_fft.wgsl"),
        &CHIRP_FFT_BINDINGS
    ),
    (
        ChirpForwardButterfly,
        ChirpFftParams,
        "apollo-stft-chirp-forward-butterfly",
        "chirp_fft_butterfly_fwd",
        include_str!("shaders/stft_chirp_fft.wgsl"),
        &CHIRP_FFT_BINDINGS
    ),
    (
        ChirpInverseButterfly,
        ChirpFftParams,
        "apollo-stft-chirp-inverse-butterfly",
        "chirp_fft_butterfly_inv",
        include_str!("shaders/stft_chirp_fft.wgsl"),
        &CHIRP_FFT_BINDINGS
    ),
    (
        ChirpScale,
        ChirpFftParams,
        "apollo-stft-chirp-scale",
        "chirp_fft_scale",
        include_str!("shaders/stft_chirp_fft.wgsl"),
        &CHIRP_FFT_BINDINGS
    ),
);

struct OverlapAdd;

impl FlatSpec for OverlapAdd {
    type Params = StftParams;
    const LABEL: &'static str = "apollo-stft-overlap-add";
    const ENTRY: &'static str = "stft_inverse_ola";
    const SOURCE: &'static str = include_str!("shaders/stft_inverse.wgsl");
    const BINDINGS: &'static [BindingDecl] = &OLA_BINDINGS;
    const WORKGROUP: [u32; 3] = [OLA_WORKGROUP as u32, 1, 1];
}

type ForwardPackKernel = GroupedKernel<ForwardPack>;
type ForwardBitReverseKernel = GroupedKernel<ForwardBitReverse>;
type ForwardButterflyKernel = GroupedKernel<ForwardButterfly>;
type ForwardInterleaveKernel = GroupedKernel<ForwardInterleave>;
type InverseDeinterleaveKernel = GroupedKernel<InverseDeinterleave>;
type InverseBitReverseKernel = GroupedKernel<InverseBitReverse>;
type InverseButterflyKernel = GroupedKernel<InverseButterfly>;
type InverseScaleWindowKernel = GroupedKernel<InverseScaleWindow>;
type ChirpForwardPremultiplyKernel = GroupedKernel<ChirpForwardPremultiply>;
type ChirpForwardPointMultiplyKernel = GroupedKernel<ChirpForwardPointMultiply>;
type ChirpForwardPostmultiplyKernel = GroupedKernel<ChirpForwardPostmultiply>;
type ChirpInversePremultiplyKernel = GroupedKernel<ChirpInversePremultiply>;
type ChirpInversePointMultiplyKernel = GroupedKernel<ChirpInversePointMultiply>;
type ChirpInversePostmultiplyKernel = GroupedKernel<ChirpInversePostmultiply>;
type ChirpBitReverseKernel = GroupedKernel<ChirpBitReverse>;
type ChirpForwardButterflyKernel = GroupedKernel<ChirpForwardButterfly>;
type ChirpInverseButterflyKernel = GroupedKernel<ChirpInverseButterfly>;
type ChirpScaleKernel = GroupedKernel<ChirpScale>;
type OverlapAddKernel = FlatKernel<OverlapAdd>;

/// Zero-sized typed STFT GPU orchestration.
#[derive(Clone, Copy, Debug, Default)]
pub struct StftGpuKernel;
