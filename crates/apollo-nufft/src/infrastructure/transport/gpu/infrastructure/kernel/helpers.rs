use super::{FastNufftParams3D, NufftGpuKernel, NufftParams, Position3Pod, WORKGROUP_SIZE};
use crate::infrastructure::transport::gpu::domain::error::{NufftWgpuError, NufftWgpuResult};
use bytemuck::Pod;
use eunomia::Complex32;
use std::sync::mpsc;
use wgpu::util::DeviceExt;

impl NufftGpuKernel {
    pub(crate) fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &[Position3Pod],
        values: &[Complex32],
        output_len: usize,
        params: NufftParams,
        pipeline: &wgpu::ComputePipeline,
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let position_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-nufft-wgpu positions"),
            contents: bytemuck::cast_slice(positions),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let value_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-nufft-wgpu values"),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu output"),
            size: (output_len * std::mem::size_of::<Complex32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu staging"),
            size: (output_len * std::mem::size_of::<Complex32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apollo-nufft-wgpu params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: position_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: value_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu direct type1 pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count(output_len as u32), 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging,
            0,
            (output_len * std::mem::size_of::<Complex32>()) as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        match receiver.recv() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                return Err(NufftWgpuError::BufferMapFailed {
                    message: error.to_string(),
                });
            }
            Err(error) => {
                return Err(NufftWgpuError::BufferMapFailed {
                    message: error.to_string(),
                });
            }
        }

        let output = {
            let mapped =
                slice
                    .get_mapped_range()
                    .map_err(|error| NufftWgpuError::BufferMapFailed {
                        message: error.to_string(),
                    })?;
            bytemuck::cast_slice::<u8, Complex32>(&mapped).to_vec()
        };
        staging.unmap();
        Ok(output)
    }
}

pub(crate) fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn uniform_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(
                std::num::NonZeroU64::new(std::mem::size_of::<NufftParams>() as u64)
                    .expect("nonzero uniform size"),
            ),
        },
        count: None,
    }
}

pub(crate) fn uniform_layout_entry_3d(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(
                std::num::NonZeroU64::new(std::mem::size_of::<FastNufftParams3D>() as u64)
                    .expect("nonzero 3d uniform size"),
            ),
        },
        count: None,
    }
}

pub(crate) fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub(crate) fn positions_to_complex_1d(positions: &[f32]) -> Vec<Complex32> {
    positions.iter().map(|x| Complex32::new(*x, 0.0)).collect()
}

pub(crate) fn real_to_complex(values: &[f32]) -> Vec<Complex32> {
    values
        .iter()
        .map(|value| Complex32::new(*value, 0.0))
        .collect()
}

pub(crate) fn real_to_complex_scaled(values: &[f32], scale: f32) -> Vec<Complex32> {
    values
        .iter()
        .map(|value| Complex32::new(*value * scale, 0.0))
        .collect()
}

pub(crate) fn storage_buffer<T: Pod>(
    device: &wgpu::Device,
    label: &'static str,
    data: &[T],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

pub(crate) fn split_grid_buffers(
    device: &wgpu::Device,
    len: usize,
) -> (wgpu::Buffer, wgpu::Buffer) {
    let size = (len * std::mem::size_of::<f32>()) as u64;
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let re = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("apollo-nufft-wgpu fast grid re"),
        size,
        usage,
        mapped_at_creation: false,
    });
    let im = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("apollo-nufft-wgpu fast grid im"),
        size,
        usage,
        mapped_at_creation: false,
    });
    (re, im)
}

pub(crate) fn read_complex_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    len: usize,
) -> NufftWgpuResult<Vec<Complex32>> {
    let size = (len * std::mem::size_of::<Complex32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("apollo-nufft-wgpu fast output staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("apollo-nufft-wgpu fast readback encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    match receiver.recv() {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            return Err(NufftWgpuError::BufferMapFailed {
                message: error.to_string(),
            });
        }
        Err(error) => {
            return Err(NufftWgpuError::BufferMapFailed {
                message: error.to_string(),
            });
        }
    }
    let output = {
        let mapped = slice
            .get_mapped_range()
            .map_err(|error| NufftWgpuError::BufferMapFailed {
                message: error.to_string(),
            })?;
        bytemuck::cast_slice::<u8, Complex32>(&mapped).to_vec()
    };
    staging.unmap();
    Ok(output)
}

pub(crate) fn read_complex_buffer_with_staging(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    staging: &wgpu::Buffer,
    len: usize,
) -> NufftWgpuResult<Vec<Complex32>> {
    let size = (len * std::mem::size_of::<Complex32>()) as u64;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("apollo-nufft-wgpu fast readback encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..size);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    match receiver.recv() {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            return Err(NufftWgpuError::BufferMapFailed {
                message: error.to_string(),
            });
        }
        Err(error) => {
            return Err(NufftWgpuError::BufferMapFailed {
                message: error.to_string(),
            });
        }
    }
    let output = {
        let mapped = slice
            .get_mapped_range()
            .map_err(|error| NufftWgpuError::BufferMapFailed {
                message: error.to_string(),
            })?;
        bytemuck::cast_slice::<u8, Complex32>(&mapped).to_vec()
    };
    staging.unmap();
    Ok(output)
}

#[cfg(any(test, feature = "diagnostics"))]
pub(crate) fn read_real_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    len: usize,
) -> NufftWgpuResult<Vec<f32>> {
    let size = (len * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("apollo-nufft-wgpu diagnostic grid staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("apollo-nufft-wgpu diagnostic grid readback encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..size);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    match receiver.recv() {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            return Err(NufftWgpuError::BufferMapFailed {
                message: error.to_string(),
            });
        }
        Err(error) => {
            return Err(NufftWgpuError::BufferMapFailed {
                message: error.to_string(),
            });
        }
    }
    let output = {
        let mapped = slice
            .get_mapped_range()
            .map_err(|error| NufftWgpuError::BufferMapFailed {
                message: error.to_string(),
            })?;
        bytemuck::cast_slice(&mapped).to_vec()
    };
    staging.unmap();
    Ok(output)
}

#[cfg(any(test, feature = "diagnostics"))]
pub(crate) fn read_split_grid_snapshot(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    re_buffer: &wgpu::Buffer,
    im_buffer: &wgpu::Buffer,
    len: usize,
) -> NufftWgpuResult<super::buffers::NufftGridSnapshot> {
    Ok(super::buffers::NufftGridSnapshot {
        re: read_real_buffer(device, queue, re_buffer, len)?,
        im: read_real_buffer(device, queue, im_buffer, len)?,
    })
}

pub(crate) fn dispatch_count(items: u32) -> u32 {
    items.div_ceil(WORKGROUP_SIZE)
}

/// Read back split real/imaginary GPU buffers into interleaved `Complex32` values.
#[cfg(feature = "debug-readbacks")]
pub(crate) fn read_split_grid(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    re_buffer: &wgpu::Buffer,
    im_buffer: &wgpu::Buffer,
    len: usize,
) -> NufftWgpuResult<Vec<Complex32>> {
    let size = (len * std::mem::size_of::<f32>()) as u64;
    let staging_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
    let re_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("apollo-nufft-wgpu debug re staging"),
        size,
        usage: staging_usage,
        mapped_at_creation: false,
    });
    let im_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("apollo-nufft-wgpu debug im staging"),
        size,
        usage: staging_usage,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("apollo-nufft-wgpu debug readback encoder"),
    });
    encoder.copy_buffer_to_buffer(re_buffer, 0, &re_staging, 0, size);
    encoder.copy_buffer_to_buffer(im_buffer, 0, &im_staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));
    let re_slice = re_staging.slice(..);
    let im_slice = im_staging.slice(..);
    let (re_tx, re_rx) = mpsc::channel();
    let (im_tx, im_rx) = mpsc::channel();
    re_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = re_tx.send(result);
    });
    im_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = im_tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    match (re_rx.recv(), im_rx.recv()) {
        (Ok(Ok(())), Ok(Ok(()))) => {}
        (Ok(Err(e)), _) | (_, Ok(Err(e))) => {
            return Err(NufftWgpuError::BufferMapFailed {
                message: e.to_string(),
            });
        }
        (Err(e), _) | (_, Err(e)) => {
            return Err(NufftWgpuError::BufferMapFailed {
                message: e.to_string(),
            });
        }
    }
    let output = {
        let re_mapped =
            re_slice
                .get_mapped_range()
                .map_err(|error| NufftWgpuError::BufferMapFailed {
                    message: error.to_string(),
                })?;
        let im_mapped =
            im_slice
                .get_mapped_range()
                .map_err(|error| NufftWgpuError::BufferMapFailed {
                    message: error.to_string(),
                })?;
        let re_data: &[f32] = bytemuck::cast_slice(&re_mapped);
        let im_data: &[f32] = bytemuck::cast_slice(&im_mapped);
        re_data
            .iter()
            .zip(im_data.iter())
            .map(|(&re, &im)| Complex32::new(re, im))
            .collect()
    };
    re_staging.unmap();
    im_staging.unmap();
    Ok(output)
}
