//! Retained Hephaestus plans and grouped NUFFT-domain dispatch state.

use bytemuck::Pod;
use eunomia::Complex32;
use hephaestus_core::{
    DispatchGrid, FftDirection, FftOperands, FftOps, GroupedBinding, GroupedKernelDevice,
    GroupedKernelSource, StridedView, Wgsl,
};
use hephaestus_wgpu::{
    WgpuBoundGroupedDispatch, WgpuBuffer, WgpuDevice, WgpuFftOps, WgpuGroupedSequence,
    WgpuPreparedFft,
};
use leto::Layout;

use super::super::descriptors::{
    ExtractOne, ExtractThree, FastNufftParams, FastNufftParams3D, FastOneKernel, FastThreeKernel,
    InterpolateOne, InterpolateThree, LoadOne, LoadThree, SpreadOne, SpreadThree,
};
use crate::infrastructure::transport::gpu::domain::error::{NufftWgpuError, NufftWgpuResult};

pub(super) struct PreparedFftPair<const R: usize> {
    forward: WgpuPreparedFft<R>,
    inverse: WgpuPreparedFft<R>,
}

impl<const R: usize> core::fmt::Debug for PreparedFftPair<R> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let Self {
            forward: _,
            inverse: _,
        } = self;
        formatter
            .debug_struct("PreparedFftPair")
            .field("forward", &"prepared")
            .field("inverse", &"prepared")
            .finish()
    }
}

impl<const R: usize> PreparedFftPair<R> {
    pub(super) fn new(
        device: &WgpuDevice,
        real: &WgpuBuffer<f32>,
        imaginary: &WgpuBuffer<f32>,
        shape: [usize; R],
    ) -> NufftWgpuResult<Self> {
        let layout = Layout::c_contiguous(shape).map_err(|_| NufftWgpuError::InvalidPlan {
            message: "oversampled FFT shape cannot form a dense layout",
        })?;
        let operands = || FftOperands {
            real: StridedView::new(real, &layout),
            imaginary: StridedView::new(imaginary, &layout),
        };
        let ops = WgpuFftOps;
        Ok(Self {
            forward: ops.prepare_fft(device, operands(), FftDirection::Forward)?,
            inverse: ops.prepare_fft(device, operands(), FftDirection::Inverse)?,
        })
    }

    pub(super) fn encode_forward(
        &self,
        sequence: &mut WgpuGroupedSequence<'_>,
    ) -> hephaestus_core::Result<()> {
        self.forward.encode_in_sequence(sequence)
    }

    pub(super) fn encode_inverse(
        &self,
        sequence: &mut WgpuGroupedSequence<'_>,
    ) -> hephaestus_core::Result<()> {
        self.inverse.encode_in_sequence(sequence)
    }
}

pub(super) struct NufftKernelOperands<'a, P, D> {
    pub(super) positions: &'a WgpuBuffer<P>,
    pub(super) values: &'a WgpuBuffer<Complex32>,
    pub(super) real: &'a WgpuBuffer<f32>,
    pub(super) imaginary: &'a WgpuBuffer<f32>,
    pub(super) deconvolution: &'a WgpuBuffer<D>,
    pub(super) output: &'a WgpuBuffer<Complex32>,
}

impl<'a, P: Pod, D: Pod> NufftKernelOperands<'a, P, D> {
    fn bindings(
        &self,
        coefficients: &'a WgpuBuffer<Complex32>,
    ) -> [GroupedBinding<'a, WgpuDevice>; 7] {
        [
            GroupedBinding::read(0, 0, self.positions),
            GroupedBinding::read(0, 1, self.values),
            GroupedBinding::read_write(0, 2, self.real),
            GroupedBinding::read_write(0, 3, self.imaginary),
            GroupedBinding::read(0, 4, self.deconvolution),
            GroupedBinding::read_write(0, 5, self.output),
            GroupedBinding::read(0, 6, coefficients),
        ]
    }
}

#[derive(Clone, Copy)]
pub(super) struct NufftKernelGrids {
    pub(super) oversampled: DispatchGrid,
    pub(super) modes: DispatchGrid,
    pub(super) samples: DispatchGrid,
}

fn bind<K: GroupedKernelSource<Wgsl>>(
    device: &WgpuDevice,
    kernel: &K,
    bindings: &[GroupedBinding<'_, WgpuDevice>],
    params: &K::Params,
    grid: DispatchGrid,
) -> NufftWgpuResult<WgpuBoundGroupedDispatch<K>> {
    let prepared = device.prepare_grouped(kernel)?;
    Ok(device.bind_grouped_dispatch(&prepared, bindings, params, grid)?)
}

#[derive(Debug)]
pub(super) struct PreparedNufftKernels1D {
    pub(super) spread: WgpuBoundGroupedDispatch<FastOneKernel<SpreadOne>>,
    pub(super) extract: WgpuBoundGroupedDispatch<FastOneKernel<ExtractOne>>,
    pub(super) load: WgpuBoundGroupedDispatch<FastOneKernel<LoadOne>>,
    pub(super) interpolate: WgpuBoundGroupedDispatch<FastOneKernel<InterpolateOne>>,
}

impl PreparedNufftKernels1D {
    pub(super) fn new(
        device: &WgpuDevice,
        operands: &NufftKernelOperands<'_, Complex32, Complex32>,
        type_one_coefficients: &WgpuBuffer<Complex32>,
        type_two_coefficients: &WgpuBuffer<Complex32>,
        params: &FastNufftParams,
        grids: NufftKernelGrids,
    ) -> NufftWgpuResult<Self> {
        let type_one = operands.bindings(type_one_coefficients);
        let type_two = operands.bindings(type_two_coefficients);
        Ok(Self {
            spread: bind(
                device,
                &FastOneKernel::<SpreadOne>::new(),
                &type_one,
                params,
                grids.oversampled,
            )?,
            extract: bind(
                device,
                &FastOneKernel::<ExtractOne>::new(),
                &type_one,
                params,
                grids.modes,
            )?,
            load: bind(
                device,
                &FastOneKernel::<LoadOne>::new(),
                &type_two,
                params,
                grids.oversampled,
            )?,
            interpolate: bind(
                device,
                &FastOneKernel::<InterpolateOne>::new(),
                &type_two,
                params,
                grids.samples,
            )?,
        })
    }

    pub(super) fn update_type_one(
        &mut self,
        device: &WgpuDevice,
        params: &FastNufftParams,
    ) -> NufftWgpuResult<()> {
        self.spread.update_params(device, params)?;
        self.extract.update_params(device, params)?;
        Ok(())
    }

    pub(super) fn update_type_two(
        &mut self,
        device: &WgpuDevice,
        params: &FastNufftParams,
    ) -> NufftWgpuResult<()> {
        self.load.update_params(device, params)?;
        self.interpolate.update_params(device, params)?;
        Ok(())
    }
}

#[derive(Debug)]
pub(super) struct PreparedNufftKernels3D {
    pub(super) spread: WgpuBoundGroupedDispatch<FastThreeKernel<SpreadThree>>,
    pub(super) extract: WgpuBoundGroupedDispatch<FastThreeKernel<ExtractThree>>,
    pub(super) load: WgpuBoundGroupedDispatch<FastThreeKernel<LoadThree>>,
    pub(super) interpolate: WgpuBoundGroupedDispatch<FastThreeKernel<InterpolateThree>>,
}

impl PreparedNufftKernels3D {
    pub(super) fn new(
        device: &WgpuDevice,
        operands: &NufftKernelOperands<'_, super::super::descriptors::Position3Pod, f32>,
        type_one_coefficients: &WgpuBuffer<Complex32>,
        type_two_coefficients: &WgpuBuffer<Complex32>,
        params: &FastNufftParams3D,
        grids: NufftKernelGrids,
    ) -> NufftWgpuResult<Self> {
        let type_one = operands.bindings(type_one_coefficients);
        let type_two = operands.bindings(type_two_coefficients);
        Ok(Self {
            spread: bind(
                device,
                &FastThreeKernel::<SpreadThree>::new(),
                &type_one,
                params,
                grids.oversampled,
            )?,
            extract: bind(
                device,
                &FastThreeKernel::<ExtractThree>::new(),
                &type_one,
                params,
                grids.modes,
            )?,
            load: bind(
                device,
                &FastThreeKernel::<LoadThree>::new(),
                &type_two,
                params,
                grids.oversampled,
            )?,
            interpolate: bind(
                device,
                &FastThreeKernel::<InterpolateThree>::new(),
                &type_two,
                params,
                grids.samples,
            )?,
        })
    }

    pub(super) fn update_type_one(
        &mut self,
        device: &WgpuDevice,
        params: &FastNufftParams3D,
    ) -> NufftWgpuResult<()> {
        self.spread.update_params(device, params)?;
        self.extract.update_params(device, params)?;
        Ok(())
    }

    pub(super) fn update_type_two(
        &mut self,
        device: &WgpuDevice,
        params: &FastNufftParams3D,
    ) -> NufftWgpuResult<()> {
        self.load.update_params(device, params)?;
        self.interpolate.update_params(device, params)?;
        Ok(())
    }
}
