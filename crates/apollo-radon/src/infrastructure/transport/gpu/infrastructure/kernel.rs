//! Typed Hephaestus kernels for Radon projection and reconstruction.
//!
//! The forward kernel is the discrete parallel-beam projection, the second
//! kernel is its interpolation-based adjoint, and filtered backprojection
//! records the Ram-Lak convolution before that adjoint in one ordered stream.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use hephaestus_core::{
    Binding, BindingDecl, CommandStream, ComputeDevice, DeviceBuffer, DispatchGrid, KernelDevice,
    KernelInterface, KernelSource, Wgsl,
};
use hephaestus_wgpu::WgpuDevice;
use leto::Array2;

use crate::{
    infrastructure::transport::gpu::{
        application::plan::RadonWgpuPlan,
        domain::error::{WgpuError, WgpuResult},
    },
    ramp_filter_projection,
};

const WORKGROUP_SIZE: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RadonParams {
    rows: u32,
    cols: u32,
    angle_count: u32,
    detector_count: u32,
    detector_spacing: f32,
    padding: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<RadonParams>() == 32);

impl RadonParams {
    fn new(plan: RadonWgpuPlan) -> WgpuResult<Self> {
        Ok(Self {
            rows: u32::try_from(plan.rows()).map_err(|_| WgpuError::InvalidPlan {
                message: "row count exceeds u32".to_owned(),
            })?,
            cols: u32::try_from(plan.cols()).map_err(|_| WgpuError::InvalidPlan {
                message: "column count exceeds u32".to_owned(),
            })?,
            angle_count: u32::try_from(plan.angle_count()).map_err(|_| WgpuError::InvalidPlan {
                message: "angle count exceeds u32".to_owned(),
            })?,
            detector_count: u32::try_from(plan.detector_count()).map_err(|_| {
                WgpuError::InvalidPlan {
                    message: "detector count exceeds u32".to_owned(),
                }
            })?,
            detector_spacing: plan.detector_spacing() as f32,
            padding: [0; 3],
        })
    }
}

macro_rules! radon_kernel {
    ($name:ident, $label:literal, $entry:literal, $source:expr) => {
        struct $name;
        impl KernelInterface for $name {
            type Params = RadonParams;
            const LABEL: &'static str = $label;
            const BINDINGS: &'static [BindingDecl] = &[
                BindingDecl::read_only::<f32>(),
                BindingDecl::read_only::<f32>(),
                BindingDecl::read_write::<f32>(),
            ];
            const WORKGROUP: [u32; 3] = [WORKGROUP_SIZE as u32, 1, 1];
        }
        impl KernelSource<Wgsl> for $name {
            const ENTRY: &'static str = $entry;
            fn source(&self) -> Cow<'static, str> {
                Cow::Borrowed($source)
            }
        }
    };
}

radon_kernel!(
    ForwardKernel,
    "apollo-radon-forward",
    "radon_forward",
    include_str!("shaders/radon.wgsl")
);
radon_kernel!(
    BackprojectKernel,
    "apollo-radon-backproject",
    "radon_backproject",
    include_str!("shaders/radon_backproject.wgsl")
);
radon_kernel!(
    FilterKernel,
    "apollo-radon-filter",
    "radon_fbp_filter",
    include_str!("shaders/radon_fbp_filter.wgsl")
);

/// Zero-sized Radon orchestration over a typed Hephaestus device.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RadonGpuKernel;

impl RadonGpuKernel {
    pub(crate) fn execute_forward(
        device: &WgpuDevice,
        plan: RadonWgpuPlan,
        image: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>> {
        let image = device.upload(&image.iter().copied().collect::<Vec<_>>())?;
        let angles = device.upload(angles)?;
        let output = device.alloc_zeroed::<f32>(plan.angle_count() * plan.detector_count())?;
        Self::run(
            device,
            &ForwardKernel,
            &image,
            &angles,
            &output,
            plan,
            output.len(),
        )?;
        Self::download_matrix(
            device,
            &output,
            [plan.angle_count(), plan.detector_count()],
            "sinogram",
        )
    }

    pub(crate) fn execute_backproject(
        device: &WgpuDevice,
        plan: RadonWgpuPlan,
        sinogram: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>> {
        let sinogram = device.upload(&sinogram.iter().copied().collect::<Vec<_>>())?;
        let angles = device.upload(angles)?;
        let output = device.alloc_zeroed::<f32>(plan.rows() * plan.cols())?;
        Self::run(
            device,
            &BackprojectKernel,
            &sinogram,
            &angles,
            &output,
            plan,
            output.len(),
        )?;
        Self::download_matrix(
            device,
            &output,
            [plan.rows(), plan.cols()],
            "backprojection",
        )
    }

    pub(crate) fn execute_filtered_backproject(
        device: &WgpuDevice,
        plan: RadonWgpuPlan,
        sinogram: &Array2<f32>,
        angles: &[f32],
    ) -> WgpuResult<Array2<f32>> {
        let sinogram = device.upload(&sinogram.iter().copied().collect::<Vec<_>>())?;
        let filter = device.upload(&ramp_kernel(plan.detector_count(), plan.detector_spacing()))?;
        let angles = device.upload(angles)?;
        let filtered = device.alloc_zeroed::<f32>(plan.angle_count() * plan.detector_count())?;
        let image = device.alloc_zeroed::<f32>(plan.rows() * plan.cols())?;
        let params = RadonParams::new(plan)?;
        let filter_kernel = device.prepare(&FilterKernel)?;
        let backproject_kernel = device.prepare(&BackprojectKernel)?;
        let filter_bindings = [
            Binding::read(&sinogram),
            Binding::read(&filter),
            Binding::read_write(&filtered),
        ];
        let backproject_bindings = [
            Binding::read(&filtered),
            Binding::read(&angles),
            Binding::read_write(&image),
        ];
        let filter_grid = grid(filtered.len())?;
        let image_grid = grid(image.len())?;
        let mut stream = device.stream()?;
        stream.encode(&filter_kernel, &filter_bindings, &params, filter_grid)?;
        stream.encode(
            &backproject_kernel,
            &backproject_bindings,
            &params,
            image_grid,
        )?;
        stream.submit()?;
        let mut output = Self::download_matrix(
            device,
            &image,
            [plan.rows(), plan.cols()],
            "filtered backprojection",
        )?;
        let scale = core::f32::consts::PI / plan.angle_count() as f32;
        output.iter_mut().for_each(|value| *value *= scale);
        Ok(output)
    }

    fn run<K>(
        device: &WgpuDevice,
        kernel: &K,
        left: &hephaestus_wgpu::WgpuBuffer<f32>,
        right: &hephaestus_wgpu::WgpuBuffer<f32>,
        output: &hephaestus_wgpu::WgpuBuffer<f32>,
        plan: RadonWgpuPlan,
        output_len: usize,
    ) -> WgpuResult<()>
    where
        K: KernelInterface<Params = RadonParams> + KernelSource<Wgsl>,
    {
        let prepared = device.prepare(kernel)?;
        let bindings = [
            Binding::read(left),
            Binding::read(right),
            Binding::read_write(output),
        ];
        let mut stream = device.stream()?;
        stream.encode(
            &prepared,
            &bindings,
            &RadonParams::new(plan)?,
            grid(output_len)?,
        )?;
        stream.submit()?;
        Ok(())
    }

    fn download_matrix(
        device: &WgpuDevice,
        buffer: &hephaestus_wgpu::WgpuBuffer<f32>,
        shape: [usize; 2],
        label: &str,
    ) -> WgpuResult<Array2<f32>> {
        let mut values = vec![0.0; buffer.len()];
        device.download(buffer, &mut values)?;
        Array2::from_shape_vec(shape, values).map_err(|_| WgpuError::InvalidPlan {
            message: format!("failed to reshape {label} readback"),
        })
    }
}

fn grid(len: usize) -> WgpuResult<DispatchGrid> {
    Ok(DispatchGrid::covering_domain(
        [len, 1, 1],
        [WORKGROUP_SIZE, 1, 1],
    )?)
}

fn ramp_kernel(detector_count: usize, spacing: f64) -> Vec<f32> {
    let mut impulse = vec![0.0; detector_count];
    if let Some(first) = impulse.first_mut() {
        *first = 1.0;
    }
    ramp_filter_projection(&impulse, spacing)
        .into_iter()
        .map(|value| value as f32)
        .collect()
}
