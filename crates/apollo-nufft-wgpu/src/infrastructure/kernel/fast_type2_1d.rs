use std::sync::Arc;
use num_complex::Complex32;
use crate::domain::error::{NufftWgpuError, NufftWgpuResult};
use super::{
    binding, complex_to_pods, dispatch_count, positions_to_complex_pods_1d,
    read_complex_buffer, real_to_complex_pods_scaled, split_grid_buffers,
    storage_buffer, ComplexPod, FastNufftParams, NufftGpuKernel,
};
use crate::infrastructure::kernel::NufftGpuBuffers1D;
use crate::infrastructure::kernel::buffers::ensure_sample_capacity;
use apollo_fft_wgpu::GpuFft3d;

#[cfg(any(test, feature = "diagnostics"))]
use super::{read_split_grid_snapshot, NufftType2GridDiagnostics};

impl NufftGpuKernel {
    /// Execute fast gridded Type-2 1D NUFFT with GPU deconvolution, FFT, and interpolation.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type2_1d(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        n: usize,
        oversampled_len: usize,
        kernel_width: usize,
        length: f32,
        beta: f32,
        i0_beta: f32,
        deconv: &[f32],
        coefficients: &[Complex32],
        positions: &[f32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let coefficient_data = complex_to_pods(coefficients);
        let position_data = positions_to_complex_pods_1d(positions);
        let deconv_data = real_to_complex_pods_scaled(deconv, oversampled_len as f32);
        let coefficients_buffer = storage_buffer(
            device,
            "apollo-nufft-wgpu fast coefficients",
            &coefficient_data,
        );
        let position_buffer =
            storage_buffer(device, "apollo-nufft-wgpu fast positions", &position_data);
        let deconv_buffer = storage_buffer(device, "apollo-nufft-wgpu fast deconv", &deconv_data);
        let (re_buffer, im_buffer) = split_grid_buffers(device, oversampled_len);
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu fast type2 output"),
            size: (positions.len() * std::mem::size_of::<ComplexPod>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = FastNufftParams {
            n: n as u32,
            m: oversampled_len as u32,
            sample_count: positions.len() as u32,
            kernel_width: kernel_width as u32,
            length,
            beta,
            i0_beta,
            _pad: 0.0,
        };
        queue.write_buffer(&self.fast_params_buffer, 0, bytemuck::bytes_of(&params));

        let load_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast type2 load bg"),
            layout: &self.fast_extract_layout,
            entries: &[
                binding(0, &position_buffer),
                binding(1, &coefficients_buffer),
                binding(2, &re_buffer),
                binding(3, &im_buffer),
                binding(4, &deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coefficients_buffer),
                binding(7, &self.fast_params_buffer),
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu fast type2 load encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast type2 load pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_load_1d_pipeline);
            pass.set_bind_group(0, &load_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(oversampled_len as u32), 1, 1);
        }

        let fft = GpuFft3d::new(Arc::clone(device), Arc::clone(queue), oversampled_len, 1, 1)
            .map_err(|_| NufftWgpuError::InvalidPlan {
                message: "oversampled FFT plan is invalid for WGPU execution",
            })?;
        fft.encode_inverse_split(&mut encoder, &re_buffer, &im_buffer);

        let interp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast type2 interpolate bg"),
            layout: &self.fast_spread_layout,
            entries: &[
                binding(0, &position_buffer),
                binding(1, &coefficients_buffer),
                binding(2, &re_buffer),
                binding(3, &im_buffer),
                binding(4, &deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coefficients_buffer),
                binding(7, &self.fast_params_buffer),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast type2 interpolate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_interpolate_1d_pipeline);
            pass.set_bind_group(0, &interp_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(positions.len() as u32), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        read_complex_buffer(device, queue, &output_buffer, positions.len())
    }

    /// Execute fast gridded Type-2 1D NUFFT with pre-allocated buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type2_1d_with_buffers(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        buffers: &NufftGpuBuffers1D,
        kernel_width: usize,
        length: f32,
        beta: f32,
        i0_beta: f32,
        deconv: &[f32],
        coefficients: &[Complex32],
        positions: &[f32],
    ) -> NufftWgpuResult<Vec<Complex32>> {
        let oversampled_len = buffers.m;
        ensure_sample_capacity(buffers.max_samples, positions.len())?;

        let coefficient_data = complex_to_pods(coefficients);
        let position_data = positions_to_complex_pods_1d(positions);
        let deconv_data = real_to_complex_pods_scaled(deconv, oversampled_len as f32);

        let coefficients_buffer = storage_buffer(
            device,
            "apollo-nufft-wgpu fast coefficients",
            &coefficient_data,
        );

        queue.write_buffer(
            &buffers.position_buffer,
            0,
            bytemuck::cast_slice(&position_data),
        );
        queue.write_buffer(
            &buffers.deconv_buffer,
            0,
            bytemuck::cast_slice(&deconv_data),
        );

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu fast type2 output"),
            size: (positions.len() * std::mem::size_of::<ComplexPod>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = FastNufftParams {
            n: buffers.n as u32,
            m: oversampled_len as u32,
            sample_count: positions.len() as u32,
            kernel_width: kernel_width as u32,
            length,
            beta,
            i0_beta,
            _pad: 0.0,
        };
        queue.write_buffer(&self.fast_params_buffer, 0, bytemuck::bytes_of(&params));

        let load_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast type2 load bg"),
            layout: &self.fast_extract_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &coefficients_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coefficients_buffer),
                binding(7, &self.fast_params_buffer),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu fast type2 load encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast type2 load pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_load_1d_pipeline);
            pass.set_bind_group(0, &load_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(oversampled_len as u32), 1, 1);
        }

        let fft = GpuFft3d::new(Arc::clone(device), Arc::clone(queue), oversampled_len, 1, 1)
            .map_err(|_| NufftWgpuError::InvalidPlan {
                message: "oversampled FFT plan is invalid for WGPU execution",
            })?;
        fft.encode_inverse_split(&mut encoder, &buffers.re_buffer, &buffers.im_buffer);

        let interp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu fast type2 interpolate bg"),
            layout: &self.fast_spread_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &buffers.value_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coefficients_buffer),
                binding(7, &self.fast_params_buffer),
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu fast type2 interpolate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_interpolate_1d_pipeline);
            pass.set_bind_group(0, &interp_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(positions.len() as u32), 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        read_complex_buffer(device, queue, &output_buffer, positions.len())
    }

    /// Execute fast gridded Type-2 1D NUFFT and return diagnostic grid snapshots.
    #[cfg(any(test, feature = "diagnostics"))]
    #[allow(clippy::too_many_arguments)]
    pub fn execute_fast_type2_1d_with_diagnostics(
        &self,
        device: &Arc<wgpu::Device>,
        queue: &Arc<wgpu::Queue>,
        buffers: &NufftGpuBuffers1D,
        kernel_width: usize,
        length: f32,
        beta: f32,
        i0_beta: f32,
        deconv: &[f32],
        coefficients: &[Complex32],
        positions: &[f32],
    ) -> NufftWgpuResult<(Vec<Complex32>, NufftType2GridDiagnostics)> {
        let oversampled_len = buffers.m;
        ensure_sample_capacity(buffers.max_samples, positions.len())?;
        let coefficient_data = complex_to_pods(coefficients);
        let position_data = positions_to_complex_pods_1d(positions);
        let deconv_data = real_to_complex_pods_scaled(deconv, oversampled_len as f32);
        let coefficients_buffer = storage_buffer(
            device,
            "apollo-nufft-wgpu diagnostic fast coefficients",
            &coefficient_data,
        );
        queue.write_buffer(
            &buffers.position_buffer,
            0,
            bytemuck::cast_slice(&position_data),
        );
        queue.write_buffer(
            &buffers.deconv_buffer,
            0,
            bytemuck::cast_slice(&deconv_data),
        );

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast type2 output"),
            size: (positions.len() * std::mem::size_of::<ComplexPod>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = FastNufftParams {
            n: buffers.n as u32,
            m: oversampled_len as u32,
            sample_count: positions.len() as u32,
            kernel_width: kernel_width as u32,
            length,
            beta,
            i0_beta,
            _pad: 0.0,
        };
        queue.write_buffer(&self.fast_params_buffer, 0, bytemuck::bytes_of(&params));

        let load_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast type2 load bg"),
            layout: &self.fast_extract_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &coefficients_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coefficients_buffer),
                binding(7, &self.fast_params_buffer),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast type2 load encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu diagnostic fast type2 load pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_load_1d_pipeline);
            pass.set_bind_group(0, &load_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(oversampled_len as u32), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        let after_load = read_split_grid_snapshot(
            device,
            queue,
            &buffers.re_buffer,
            &buffers.im_buffer,
            oversampled_len,
        )?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast type2 ifft/interp encoder"),
        });
        let fft = GpuFft3d::new(Arc::clone(device), Arc::clone(queue), oversampled_len, 1, 1)
            .map_err(|_| NufftWgpuError::InvalidPlan {
                message: "oversampled FFT plan is invalid for WGPU execution",
            })?;
        fft.encode_inverse_split(&mut encoder, &buffers.re_buffer, &buffers.im_buffer);
        queue.submit(std::iter::once(encoder.finish()));
        let after_ifft = read_split_grid_snapshot(
            device,
            queue,
            &buffers.re_buffer,
            &buffers.im_buffer,
            oversampled_len,
        )?;

        let interp_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast type2 interpolate bg"),
            layout: &self.fast_spread_layout,
            entries: &[
                binding(0, &buffers.position_buffer),
                binding(1, &coefficients_buffer),
                binding(2, &buffers.re_buffer),
                binding(3, &buffers.im_buffer),
                binding(4, &buffers.deconv_buffer),
                binding(5, &output_buffer),
                binding(6, &coefficients_buffer),
                binding(7, &self.fast_params_buffer),
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-nufft-wgpu diagnostic fast type2 interpolate encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-nufft-wgpu diagnostic fast type2 interpolate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fast_type2_interpolate_1d_pipeline);
            pass.set_bind_group(0, &interp_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(positions.len() as u32), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        let output = read_complex_buffer(device, queue, &output_buffer, positions.len())?;
        Ok((
            output,
            NufftType2GridDiagnostics {
                after_load,
                after_ifft,
            },
        ))
    }
}
