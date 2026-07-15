//! Typed Hephaestus kernels for direct spherical harmonic execution.
//!
//! The forward pass evaluates the quadrature projection
//! `a_l^m = sum_j f_j conj(Y_l^m(j)) w_j`, while the inverse pass evaluates
//! `f_j = sum_lm a_l^m Y_l^m(j)`.  A typed command stream materializes the
//! respective basis before consuming it in the matrix reduction, so the
//! provider owns the required device-side write-before-read synchronization.

use std::{borrow::Cow, marker::PhantomData};

use bytemuck::{Pod, Zeroable};
use eunomia::Complex32;
use hephaestus_core::{
    Binding, BindingDecl, CommandStream, DispatchGrid, KernelDevice, KernelInterface, KernelSource,
    Wgsl,
};

use crate::infrastructure::transport::gpu::{
    application::plan::ShtWgpuPlan,
    domain::error::{WgpuError, WgpuResult},
};

const WORKGROUP_SIZE: usize = 64;
const SHT_COMMON_SOURCE: &str = include_str!("shaders/common.wgsl");
const SHT_BASIS_SOURCE: &str = include_str!("shaders/basis.wgsl");
const SHT_MATRIX_SOURCE: &str = include_str!("shaders/matrix.wgsl");

/// Uniform parameters for the matrix reduction.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct ShtParams {
    output_count: u32,
    reduction_count: u32,
    padding: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<ShtParams>() == 16);

impl ShtParams {
    fn new(output_count: usize, reduction_count: usize) -> WgpuResult<Self> {
        Ok(Self {
            output_count: u32::try_from(output_count).map_err(|_| WgpuError::InvalidPlan {
                message: format!(
                    "matrix output count {output_count} exceeds the accelerator parameter range"
                ),
            })?,
            reduction_count: u32::try_from(reduction_count).map_err(|_| WgpuError::InvalidPlan {
                message: format!(
                    "matrix reduction count {reduction_count} exceeds the accelerator parameter range"
                ),
            })?,
            padding: [0; 2],
        })
    }
}

/// One sampled spherical-grid point and its quadrature weight.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GridPod {
    pub(crate) cos_theta: f32,
    pub(crate) phi: f32,
    pub(crate) weight: f32,
    pub(crate) padding: f32,
}

const _: () = assert!(core::mem::size_of::<GridPod>() == 16);

/// Uniform parameters for basis materialization.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct BasisParams {
    mode_count: u32,
    sample_count: u32,
    max_degree: u32,
    weighted: u32,
    conjugate: u32,
    padding: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<BasisParams>() == 32);

impl BasisParams {
    fn new<P: ShtMatrixPass>(plan: &ShtWgpuPlan) -> WgpuResult<Self> {
        Ok(Self {
            mode_count: u32::try_from(plan.mode_count()).map_err(|_| WgpuError::InvalidPlan {
                message: format!(
                    "mode count {} exceeds the accelerator parameter range",
                    plan.mode_count()
                ),
            })?,
            sample_count: u32::try_from(plan.sample_count()).map_err(|_| {
                WgpuError::InvalidPlan {
                    message: format!(
                        "sample count {} exceeds the accelerator parameter range",
                        plan.sample_count()
                    ),
                }
            })?,
            max_degree: u32::try_from(plan.max_degree()).map_err(|_| WgpuError::InvalidPlan {
                message: format!(
                    "maximum degree {} exceeds the accelerator parameter range",
                    plan.max_degree()
                ),
            })?,
            weighted: u32::from(P::WEIGHTED),
            conjugate: u32::from(P::CONJUGATE),
            padding: [0; 3],
        })
    }
}

/// Compile-time direction selection for the SHT matrix pass.
trait ShtMatrixPass {
    /// Provider diagnostic label for the basis pipeline.
    const BASIS_LABEL: &'static str;
    /// Provider diagnostic label for the reduction pipeline.
    const MATRIX_LABEL: &'static str;
    /// Matrix entry point.
    const ENTRY: &'static str;
    /// Whether basis values include quadrature weights.
    const WEIGHTED: bool;
    /// Whether basis values use complex conjugation.
    const CONJUGATE: bool;
}

/// Marker selecting forward quadrature projection.
pub(crate) struct Forward;

impl ShtMatrixPass for Forward {
    const BASIS_LABEL: &'static str = "apollo-sht-forward-basis";
    const MATRIX_LABEL: &'static str = "apollo-sht-forward-matrix";
    const ENTRY: &'static str = "sht_forward";
    const WEIGHTED: bool = true;
    const CONJUGATE: bool = true;
}

/// Marker selecting inverse harmonic synthesis.
pub(crate) struct Inverse;

impl ShtMatrixPass for Inverse {
    const BASIS_LABEL: &'static str = "apollo-sht-inverse-basis";
    const MATRIX_LABEL: &'static str = "apollo-sht-inverse-matrix";
    const ENTRY: &'static str = "sht_inverse";
    const WEIGHTED: bool = false;
    const CONJUGATE: bool = false;
}

/// Typed basis-generation interface for one transform direction.
pub(crate) struct ShtBasisKernel<P>(PhantomData<P>);

impl<P: ShtMatrixPass> KernelInterface for ShtBasisKernel<P> {
    type Params = BasisParams;

    const LABEL: &'static str = P::BASIS_LABEL;
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<GridPod>(),
        BindingDecl::read_write::<Complex32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl<P: ShtMatrixPass> KernelSource<Wgsl> for ShtBasisKernel<P> {
    const ENTRY: &'static str = "sht_basis";

    fn source(&self) -> Cow<'static, str> {
        Cow::Owned([SHT_COMMON_SOURCE, SHT_BASIS_SOURCE].concat())
    }
}

/// Typed matrix-reduction interface for one transform direction.
pub(crate) struct ShtMatrixKernel<P>(PhantomData<P>);

impl<P: ShtMatrixPass> KernelInterface for ShtMatrixKernel<P> {
    type Params = ShtParams;

    const LABEL: &'static str = P::MATRIX_LABEL;
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<Complex32>(),
        BindingDecl::read_only::<Complex32>(),
        BindingDecl::read_write::<Complex32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
}

impl<P: ShtMatrixPass> KernelSource<Wgsl> for ShtMatrixKernel<P> {
    const ENTRY: &'static str = P::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        Cow::Owned([SHT_COMMON_SOURCE, SHT_MATRIX_SOURCE].concat())
    }
}

/// Zero-sized SHT orchestration over a Hephaestus device.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ShtGpuKernel;

impl ShtGpuKernel {
    /// Execute forward quadrature into caller-owned mode storage.
    pub(crate) fn execute_forward_into<D>(
        device: &D,
        plan: &ShtWgpuPlan,
        samples: &[Complex32],
        grid: &[GridPod],
        coefficients: &mut [Complex32],
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        ShtBasisKernel<Forward>: KernelSource<D::Dialect> + KernelInterface<Params = BasisParams>,
        ShtMatrixKernel<Forward>: KernelSource<D::Dialect> + KernelInterface<Params = ShtParams>,
    {
        Self::execute_into::<D, Forward>(device, plan, samples, grid, coefficients)
    }

    /// Execute inverse synthesis into caller-owned sample storage.
    pub(crate) fn execute_inverse_into<D>(
        device: &D,
        plan: &ShtWgpuPlan,
        coefficients: &[Complex32],
        grid: &[GridPod],
        samples: &mut [Complex32],
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        ShtBasisKernel<Inverse>: KernelSource<D::Dialect> + KernelInterface<Params = BasisParams>,
        ShtMatrixKernel<Inverse>: KernelSource<D::Dialect> + KernelInterface<Params = ShtParams>,
    {
        Self::execute_into::<D, Inverse>(device, plan, coefficients, grid, samples)
    }

    fn execute_into<D, P>(
        device: &D,
        plan: &ShtWgpuPlan,
        input: &[Complex32],
        grid: &[GridPod],
        output: &mut [Complex32],
    ) -> WgpuResult<()>
    where
        D: KernelDevice,
        P: ShtMatrixPass,
        ShtBasisKernel<P>: KernelSource<D::Dialect> + KernelInterface<Params = BasisParams>,
        ShtMatrixKernel<P>: KernelSource<D::Dialect> + KernelInterface<Params = ShtParams>,
    {
        let (expected_input, expected_output) = if P::WEIGHTED {
            (plan.sample_count(), plan.mode_count())
        } else {
            (plan.mode_count(), plan.sample_count())
        };
        if input.len() != expected_input || output.len() != expected_output {
            return Err(WgpuError::ShapeMismatch {
                message: format!(
                    "SHT {} expects input length {expected_input} and output length {expected_output}, got {} and {}",
                    P::MATRIX_LABEL,
                    input.len(),
                    output.len()
                ),
            });
        }
        if grid.len() != plan.sample_count() {
            return Err(WgpuError::ShapeMismatch {
                message: format!(
                    "SHT grid expects {} samples, got {}",
                    plan.sample_count(),
                    grid.len()
                ),
            });
        }

        let input_buffer = device.upload(input)?;
        let grid_buffer = device.upload(grid)?;
        let basis_length = plan
            .mode_count()
            .checked_mul(plan.sample_count())
            .ok_or_else(|| WgpuError::InvalidPlan {
                message: "basis storage length overflows usize".to_owned(),
            })?;
        let basis_buffer = device.alloc_zeroed::<Complex32>(basis_length)?;
        let output_buffer = device.alloc_zeroed::<Complex32>(output.len())?;
        let basis = device.prepare(&ShtBasisKernel::<P>(PhantomData))?;
        let matrix = device.prepare(&ShtMatrixKernel::<P>(PhantomData))?;
        let basis_bindings = [
            Binding::read(&grid_buffer),
            Binding::read_write(&basis_buffer),
        ];
        let matrix_bindings = [
            Binding::read(&input_buffer),
            Binding::read(&basis_buffer),
            Binding::read_write(&output_buffer),
        ];
        let basis_grid =
            DispatchGrid::covering_domain([basis_length, 1, 1], [WORKGROUP_SIZE, 1, 1])?;
        let matrix_grid =
            DispatchGrid::covering_domain([output.len(), 1, 1], [WORKGROUP_SIZE, 1, 1])?;
        let basis_params = BasisParams::new::<P>(plan)?;
        let matrix_params = ShtParams::new(output.len(), expected_input)?;
        let mut stream = device.stream()?;
        stream.encode(&basis, &basis_bindings, &basis_params, basis_grid)?;
        stream.encode(&matrix, &matrix_bindings, &matrix_params, matrix_grid)?;
        stream.submit()?;
        device.download(&output_buffer, output)?;
        Ok(())
    }
}
