//! Typed Hephaestus kernels for STFT-domain framing and reconstruction.
//!
//! Dense Fourier arithmetic is prepared and executed by Hephaestus. This
//! module owns only Hann framing, split/interleaved conversion, synthesis
//! windowing, and weighted overlap-add.

use core::marker::PhantomData;
use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{GroupedBindingDecl, GroupedKernelInterface, GroupedKernelSource, Wgsl};

/// Shared STFT geometry and dispatch-grid validation.
mod dispatch;
/// Forward framing and output conversion.
pub mod forward;
/// Inverse input conversion and reconstruction.
pub mod inverse;

pub(crate) use dispatch::{dimension, fft_grid, ola_grid};

pub(crate) const FRAME_WORKGROUP: usize = 256;
pub(crate) const OLA_WORKGROUP: usize = 64;

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

const _: () = assert!(core::mem::size_of::<ComplexPod>() == 8);
const _: () = assert!(core::mem::size_of::<StftParams>() == 16);

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
const OLA_BINDINGS: [GroupedBindingDecl; 2] = [
    GroupedBindingDecl::read_only::<f32>(0, 0),
    GroupedBindingDecl::read_write::<f32>(0, 1),
];

pub(crate) trait GroupedSpec {
    const LABEL: &'static str;
    const ENTRY: &'static str;
    const SOURCE: &'static str;
    const BINDINGS: &'static [GroupedBindingDecl];
    const WORKGROUP: [u32; 3];
}

pub(crate) struct GroupedKernel<S>(PhantomData<S>);

impl<S> GroupedKernel<S> {
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<S: GroupedSpec> GroupedKernelInterface for GroupedKernel<S> {
    type Params = StftParams;
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

macro_rules! grouped_specs {
    ($(($marker:ident, $label:literal, $entry:literal, $source:expr, $bindings:expr, $workgroup:expr)),+ $(,)?) => {
        $(
            pub(crate) struct $marker;
            impl GroupedSpec for $marker {
                const LABEL: &'static str = $label;
                const ENTRY: &'static str = $entry;
                const SOURCE: &'static str = $source;
                const BINDINGS: &'static [GroupedBindingDecl] = $bindings;
                const WORKGROUP: [u32; 3] = [$workgroup as u32, 1, 1];
            }
        )+
    };
}

grouped_specs!(
    (
        ForwardPack,
        "apollo-stft-forward-pack",
        "stft_pack_window",
        include_str!("shaders/stft_forward.wgsl"),
        &FORWARD_BINDINGS,
        FRAME_WORKGROUP
    ),
    (
        ForwardInterleave,
        "apollo-stft-forward-interleave",
        "stft_interleave",
        include_str!("shaders/stft_forward.wgsl"),
        &FORWARD_BINDINGS,
        FRAME_WORKGROUP
    ),
    (
        InverseDeinterleave,
        "apollo-stft-inverse-deinterleave",
        "stft_deinterleave",
        include_str!("shaders/stft_inverse_frame.wgsl"),
        &INVERSE_BINDINGS,
        FRAME_WORKGROUP
    ),
    (
        InverseWindow,
        "apollo-stft-inverse-window",
        "stft_synthesis_window",
        include_str!("shaders/stft_inverse_frame.wgsl"),
        &INVERSE_BINDINGS,
        FRAME_WORKGROUP
    ),
    (
        OverlapAdd,
        "apollo-stft-overlap-add",
        "stft_inverse_ola",
        include_str!("shaders/stft_inverse.wgsl"),
        &OLA_BINDINGS,
        OLA_WORKGROUP
    ),
);

pub(crate) type ForwardPackKernel = GroupedKernel<ForwardPack>;
pub(crate) type ForwardInterleaveKernel = GroupedKernel<ForwardInterleave>;
pub(crate) type InverseDeinterleaveKernel = GroupedKernel<InverseDeinterleave>;
pub(crate) type InverseWindowKernel = GroupedKernel<InverseWindow>;
pub(crate) type OverlapAddKernel = GroupedKernel<OverlapAdd>;

/// Zero-sized typed STFT GPU orchestration marker.
#[derive(Clone, Copy, Debug, Default)]
pub struct StftGpuKernel;

impl apollo_fft::GpuTransformPlanner for StftGpuKernel {
    type Plan = crate::infrastructure::transport::gpu::FramePlan;

    fn input_len(plan: &crate::infrastructure::transport::gpu::FramePlan) -> usize {
        plan.frame_len()
    }

    fn validate(
        plan: &crate::infrastructure::transport::gpu::FramePlan,
    ) -> apollo_fft::WgpuResult<()> {
        plan.validate_geometry()
    }
}
