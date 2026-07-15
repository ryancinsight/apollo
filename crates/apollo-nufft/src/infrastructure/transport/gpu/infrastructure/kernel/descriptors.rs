//! Typed kernel descriptors and parameter layouts for NUFFT dispatch.

use core::{marker::PhantomData, mem};
use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use eunomia::Complex32;
use hephaestus_core::{BindingDecl, KernelInterface, KernelSource, Wgsl};

pub(crate) const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct Position3Pod {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) z: f32,
    pub(crate) padding: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct NufftParams {
    pub(crate) n0: u32,
    pub(crate) n1: u32,
    pub(crate) n2: u32,
    pub(crate) sample_count: u32,
    pub(crate) l0: f32,
    pub(crate) l1: f32,
    pub(crate) l2: f32,
    pub(crate) padding: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct FastNufftParams {
    pub(crate) n: u32,
    pub(crate) m: u32,
    pub(crate) sample_count: u32,
    pub(crate) kernel_width: u32,
    pub(crate) length: f32,
    pub(crate) beta: f32,
    pub(crate) i0_beta: f32,
    pub(crate) padding: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct FastNufftParams3D {
    pub(crate) nx: u32,
    pub(crate) ny: u32,
    pub(crate) nz: u32,
    pub(crate) mx: u32,
    pub(crate) my: u32,
    pub(crate) mz: u32,
    pub(crate) sample_count: u32,
    pub(crate) kernel_width: u32,
    pub(crate) lx: f32,
    pub(crate) ly: f32,
    pub(crate) lz: f32,
    pub(crate) beta: f32,
    pub(crate) i0_beta: f32,
    pub(crate) padding: [f32; 3],
}

const _: () = assert!(mem::size_of::<Position3Pod>() == 16);
const _: () = assert!(mem::size_of::<NufftParams>() == 32);
const _: () = assert!(mem::size_of::<FastNufftParams>() == 32);
const _: () = assert!(mem::size_of::<FastNufftParams3D>() == 64);

pub(crate) trait DirectOperation {
    const LABEL: &'static str;
    const ENTRY: &'static str;
}

pub(crate) struct Type1One;
pub(crate) struct Type2One;
pub(crate) struct Type1Three;
pub(crate) struct Type2Three;

impl DirectOperation for Type1One {
    const LABEL: &'static str = "apollo-nufft-direct-type1-1d";
    const ENTRY: &'static str = "nufft_type1_1d";
}
impl DirectOperation for Type2One {
    const LABEL: &'static str = "apollo-nufft-direct-type2-1d";
    const ENTRY: &'static str = "nufft_type2_1d";
}
impl DirectOperation for Type1Three {
    const LABEL: &'static str = "apollo-nufft-direct-type1-3d";
    const ENTRY: &'static str = "nufft_type1_3d";
}
impl DirectOperation for Type2Three {
    const LABEL: &'static str = "apollo-nufft-direct-type2-3d";
    const ENTRY: &'static str = "nufft_type2_3d";
}

pub(crate) struct DirectKernel<O>(PhantomData<O>);

impl<O> DirectKernel<O> {
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<O: DirectOperation> KernelInterface for DirectKernel<O> {
    type Params = NufftParams;
    const LABEL: &'static str = O::LABEL;
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<Position3Pod>(),
        BindingDecl::read_only::<Complex32>(),
        BindingDecl::read_write::<Complex32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE, 1, 1];
}

impl<O: DirectOperation> KernelSource<Wgsl> for DirectKernel<O> {
    const ENTRY: &'static str = O::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(include_str!("../shaders/nufft.wgsl"))
    }
}

pub(crate) trait FastOperation {
    const LABEL: &'static str;
    const ENTRY: &'static str;
}

pub(crate) struct SpreadOne;
pub(crate) struct ExtractOne;
pub(crate) struct LoadOne;
pub(crate) struct InterpolateOne;
pub(crate) struct SpreadThree;
pub(crate) struct ExtractThree;
pub(crate) struct LoadThree;
pub(crate) struct InterpolateThree;

impl FastOperation for SpreadOne {
    const LABEL: &'static str = "apollo-nufft-fast-type1-spread-1d";
    const ENTRY: &'static str = "fast_type1_spread_1d";
}
impl FastOperation for ExtractOne {
    const LABEL: &'static str = "apollo-nufft-fast-type1-extract-1d";
    const ENTRY: &'static str = "fast_type1_extract_1d";
}
impl FastOperation for LoadOne {
    const LABEL: &'static str = "apollo-nufft-fast-type2-load-1d";
    const ENTRY: &'static str = "fast_type2_load_1d";
}
impl FastOperation for InterpolateOne {
    const LABEL: &'static str = "apollo-nufft-fast-type2-interpolate-1d";
    const ENTRY: &'static str = "fast_type2_interpolate_1d";
}
impl FastOperation for SpreadThree {
    const LABEL: &'static str = "apollo-nufft-fast-type1-spread-3d";
    const ENTRY: &'static str = "fast_type1_spread_3d";
}
impl FastOperation for ExtractThree {
    const LABEL: &'static str = "apollo-nufft-fast-type1-extract-3d";
    const ENTRY: &'static str = "fast_type1_extract_3d";
}
impl FastOperation for LoadThree {
    const LABEL: &'static str = "apollo-nufft-fast-type2-load-3d";
    const ENTRY: &'static str = "fast_type2_load_3d";
}
impl FastOperation for InterpolateThree {
    const LABEL: &'static str = "apollo-nufft-fast-type2-interpolate-3d";
    const ENTRY: &'static str = "fast_type2_interpolate_3d";
}

pub(crate) struct FastOneKernel<O>(PhantomData<O>);

impl<O> FastOneKernel<O> {
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<O: FastOperation> KernelInterface for FastOneKernel<O> {
    type Params = FastNufftParams;
    const LABEL: &'static str = O::LABEL;
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<Complex32>(),
        BindingDecl::read_only::<Complex32>(),
        BindingDecl::read_write::<f32>(),
        BindingDecl::read_write::<f32>(),
        BindingDecl::read_only::<Complex32>(),
        BindingDecl::read_write::<Complex32>(),
        BindingDecl::read_only::<Complex32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE, 1, 1];
}

impl<O: FastOperation> KernelSource<Wgsl> for FastOneKernel<O> {
    const ENTRY: &'static str = O::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(include_str!("../shaders/nufft_fast_1d.wgsl"))
    }
}

pub(crate) struct FastThreeKernel<O>(PhantomData<O>);

impl<O> FastThreeKernel<O> {
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<O: FastOperation> KernelInterface for FastThreeKernel<O> {
    type Params = FastNufftParams3D;
    const LABEL: &'static str = O::LABEL;
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<Position3Pod>(),
        BindingDecl::read_only::<Complex32>(),
        BindingDecl::read_write::<f32>(),
        BindingDecl::read_write::<f32>(),
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<Complex32>(),
        BindingDecl::read_only::<Complex32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE, 1, 1];
}

impl<O: FastOperation> KernelSource<Wgsl> for FastThreeKernel<O> {
    const ENTRY: &'static str = O::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(include_str!("../shaders/nufft_fast_3d.wgsl"))
    }
}
