//! Hephaestus execution for real-to-real DCT and DST transforms.
//!
//! The DCT-II/DCT-III and DST-II/DST-III pairs compose to `(N / 2) I` under
//! Apollo's unnormalized convention. DCT-I, DCT-IV, DST-I, and DST-IV are
//! self-inverse up to their documented normalization factors. The host derives
//! those factors once and encodes the concrete `f32` kernel sequence below.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{
    Binding, BindingDecl, CommandStream, DispatchGrid, KernelDevice, KernelInterface, KernelSource,
    Wgsl,
};

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};

const WORKGROUP_SIZE: usize = 64;
const DCT_SOURCE: &str = include_str!("shaders/dct.wgsl");

/// Implemented real-to-real transform modes.
#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum DctMode {
    /// Type-II discrete cosine transform.
    Dct2 = 0,
    /// Type-III discrete cosine transform.
    Dct3 = 1,
    /// Type-II discrete sine transform.
    Dst2 = 2,
    /// Type-III discrete sine transform.
    Dst3 = 3,
    /// Type-I discrete cosine transform.
    Dct1 = 4,
    /// Type-IV discrete cosine transform.
    Dct4 = 5,
    /// Type-I discrete sine transform.
    Dst1 = 6,
    /// Type-IV discrete sine transform.
    Dst4 = 7,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct DctParams {
    len: u32,
    mode: u32,
    scale_bits: u32,
    elem_stride: u32,
    num_fibers: u32,
    fiber_stride_a: u32,
    fiber_stride_b: u32,
    fiber_dim_b: u32,
}

const _: () = assert!(core::mem::size_of::<DctParams>() == 32);

impl DctParams {
    fn transform(mode: DctMode, layout: FiberLayout) -> WgpuResult<Self> {
        let encode = |value: usize, field: &str| {
            u32::try_from(value).map_err(|_| WgpuError::InvalidPlan {
                message: format!("{field} {value} exceeds the provider parameter range"),
            })
        };
        Ok(Self {
            len: encode(layout.len, "fiber length")?,
            mode: mode as u32,
            scale_bits: 1.0_f32.to_bits(),
            elem_stride: encode(layout.elem_stride, "element stride")?,
            num_fibers: encode(layout.num_fibers, "fiber count")?,
            fiber_stride_a: encode(layout.fiber_stride_a, "outer fiber stride")?,
            fiber_stride_b: encode(layout.fiber_stride_b, "inner fiber stride")?,
            fiber_dim_b: encode(layout.fiber_dim_b, "inner fiber dimension")?,
        })
    }

    fn scale(len: u32, scale: f32) -> Self {
        Self {
            len,
            mode: DctMode::Dct2 as u32,
            scale_bits: scale.to_bits(),
            elem_stride: 1,
            num_fibers: 1,
            fiber_stride_a: 0,
            fiber_stride_b: 0,
            fiber_dim_b: 1,
        }
    }
}

/// Addressing of the one-dimensional fibers in a separable transform pass.
///
/// Element `n` of fiber `f` is at `base(f) + n * elem_stride`; `axis` builds
/// the row-major layout for an axis of a rank-one, rank-two, or rank-three
/// cubic field. This descriptor is host-only and resolves before dispatch.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FiberLayout {
    len: usize,
    elem_stride: usize,
    num_fibers: usize,
    fiber_stride_a: usize,
    fiber_stride_b: usize,
    fiber_dim_b: usize,
}

impl FiberLayout {
    /// Construct the fiber layout for `axis` of a row-major `n^rank` cube.
    pub(crate) fn axis(n: usize, rank: usize, axis: usize) -> WgpuResult<Self> {
        if n == 0 || !(1..=3).contains(&rank) || axis >= rank {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid fiber layout: length={n}, rank={rank}, axis={axis}"),
            });
        }
        let power = |exponent: u32| {
            n.checked_pow(exponent)
                .ok_or_else(|| WgpuError::InvalidPlan {
                    message: format!("fiber layout overflows usize for length {n} and rank {rank}"),
                })
        };
        let stride = |dimension: usize| power((rank - 1 - dimension) as u32);
        let elem_stride = stride(axis)?;
        let volume = power(rank as u32)?;
        let num_fibers = volume / n;
        let others = (0..rank)
            .filter(|&dimension| dimension != axis)
            .collect::<Vec<_>>();
        Ok(match others.as_slice() {
            [] => Self {
                len: n,
                elem_stride,
                num_fibers: 1,
                fiber_stride_a: 0,
                fiber_stride_b: 0,
                fiber_dim_b: 1,
            },
            [other] => Self {
                len: n,
                elem_stride,
                num_fibers,
                fiber_stride_a: stride(*other)?,
                fiber_stride_b: 0,
                fiber_dim_b: 1,
            },
            [outer, inner] => Self {
                len: n,
                elem_stride,
                num_fibers,
                fiber_stride_a: stride(*outer)?,
                fiber_stride_b: stride(*inner)?,
                fiber_dim_b: n,
            },
            _ => unreachable!("rank is constrained to one through three"),
        })
    }
}

pub(crate) struct TransformKernel;

impl KernelInterface for TransformKernel {
    type Params = DctParams;

    const LABEL: &'static str = "apollo-dctdst-transform";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl KernelSource<Wgsl> for TransformKernel {
    const ENTRY: &'static str = "dct_transform";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(DCT_SOURCE)
    }
}

pub(crate) struct ScaleKernel;

impl KernelInterface for ScaleKernel {
    type Params = DctParams;

    const LABEL: &'static str = "apollo-dctdst-scale";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl KernelSource<Wgsl> for ScaleKernel {
    const ENTRY: &'static str = "dct_scale";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(DCT_SOURCE)
    }
}

/// Zero-sized DCT/DST kernel orchestration over a Hephaestus device.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DctGpuKernel;

impl DctGpuKernel {
    /// Execute one transform pass and optional normalization into caller storage.
    pub(crate) fn execute_into<D>(
        device: &D,
        input: &[f32],
        output: &mut [f32],
        mode: DctMode,
        scale: f32,
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        TransformKernel: KernelSource<D::Dialect>,
        ScaleKernel: KernelSource<D::Dialect>,
    {
        Self::execute_separable_into(
            device,
            input,
            output,
            &[(mode, FiberLayout::axis(input.len(), 1, 0)?)],
            scale,
        )
    }

    /// Execute ordered separable passes without intermediate host transfers.
    pub(crate) fn execute_separable_into<D>(
        device: &D,
        input: &[f32],
        output: &mut [f32],
        passes: &[(DctMode, FiberLayout)],
        final_scale: f32,
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        TransformKernel: KernelSource<D::Dialect>,
        ScaleKernel: KernelSource<D::Dialect>,
    {
        let total = u32::try_from(input.len()).map_err(|_| WgpuError::InvalidPlan {
            message: format!(
                "length {} exceeds the provider parameter range",
                input.len()
            ),
        })?;
        let buffers = [
            device.upload(input)?,
            device.alloc_zeroed::<f32>(output.len())?,
        ];
        let transform = device.prepare(&TransformKernel)?;
        let scale = (final_scale != 1.0)
            .then(|| device.prepare(&ScaleKernel))
            .transpose()?;
        let mut source = 0usize;
        let mut destination = 1usize;
        let mut stream = device.stream()?;

        for &(mode, layout) in passes {
            let params = DctParams::transform(mode, layout)?;
            let grid = DispatchGrid::covering_domain(
                [layout.len, layout.num_fibers, 1],
                [WORKGROUP_SIZE, 1, 1],
            )?;
            let bindings = [
                Binding::read(&buffers[source]),
                Binding::read_write(&buffers[destination]),
            ];
            stream.encode(&transform, &bindings, &params, grid)?;
            core::mem::swap(&mut source, &mut destination);
        }

        if let Some(scale) = scale {
            let params = DctParams::scale(total, final_scale);
            let grid = DispatchGrid::covering_domain([input.len(), 1, 1], [WORKGROUP_SIZE, 1, 1])?;
            let bindings = [
                Binding::read(&buffers[destination]),
                Binding::read_write(&buffers[source]),
            ];
            stream.encode(&scale, &bindings, &params, grid)?;
        }

        stream.submit()?;
        device.download(&buffers[source], output)?;
        Ok(())
    }
}
