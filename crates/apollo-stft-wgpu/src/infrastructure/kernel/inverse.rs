use num_complex::Complex32;
use std::sync::mpsc;
use wgpu::util::DeviceExt;

use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::{
    dispatch_count, fft_dispatch_count, FftStageParams, StftGpuKernel, StftParams,
};
use apollo_wgpu_helpers::WgpuDevice;

impl StftGpuKernel {
    /// Execute the inverse STFT via FFT-accelerated WOLA reconstruction.
    ///
    /// ## Invariants
    /// - `frame_len` must be a power of two (Radix-2 requirement).
    /// - `spectrum.len() == frame_count * frame_len`
    pub fn execute_inverse(
        &self,
        device: &WgpuDevice,
        spectrum: &[Complex32],
        frame_len: usize,
        hop_len: usize,
        frame_count: usize,
        signal_len: usize,
    ) -> WgpuResult<Vec<f32>> {
        // Non-power-of-two frame_len: delegate to Bluestein/Chirp-Z path.
        if !frame_len.is_power_of_two() {
            return self.execute_inverse_chirp(
                device,
                spectrum,
                frame_len,
                hop_len,
                frame_count,
                signal_len,
            );
        }
        let log2_n = frame_len.trailing_zeros();

        // ── Step 1: Build flat interleaved spectrum for GPU upload ────────────
        let spectrum_flat: Vec<f32> = spectrum.iter().flat_map(|c| [c.re, c.im]).collect();

        // ── Step 2: Allocate GPU buffers ──────────────────────────────────────
        let spectrum_buf = device
            .inner()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-stft-wgpu inverse spectrum"),
                contents: bytemuck::cast_slice(&spectrum_flat),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let scratch_size = (frame_count * frame_len * std::mem::size_of::<f32>()) as u64;
        let re_scratch_buf = device.inner().create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu re scratch"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let im_scratch_buf = device.inner().create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu im scratch"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let frame_data_buf = device.inner().create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu frame data"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let signal_size = (signal_len * std::mem::size_of::<f32>()) as u64;
        let signal_buf = device.inner().create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu inverse signal"),
            size: signal_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.get_staging_buffer(signal_size);

        // ── Step 3: Write OLA uniform params (StftParams) ─────────────────────
        device.queue().write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&StftParams {
                signal_len: signal_len as u32,
                frame_len: frame_len as u32,
                hop_len: hop_len as u32,
                frame_count: frame_count as u32,
            }),
        );

        // ── Step 4: Build the shared FFT data bind group (group 0) ────────────
        let fft_data_bg = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-stft-wgpu FFT data bind group"),
                layout: &self.fft_data_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: spectrum_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: re_scratch_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: im_scratch_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: frame_data_buf.as_entire_binding(),
                    },
                ],
            });

        // ── Step 5: Pre-allocate per-stage params buffers and bind groups ─────
        let base_params = FftStageParams {
            frame_count: frame_count as u32,
            frame_len: frame_len as u32,
            stage: 0,
            _pad: 0,
        };
        let base_params_buf =
            device
                .inner()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("apollo-stft-wgpu base FFT params"),
                    contents: bytemuck::bytes_of(&base_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
        let base_params_bg = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-stft-wgpu base FFT params BG"),
                layout: &self.fft_params_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: base_params_buf.as_entire_binding(),
                }],
            });

        let mut butterfly_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(log2_n as usize);
        let mut butterfly_bgs: Vec<wgpu::BindGroup> = Vec::with_capacity(log2_n as usize);
        for s in 0..log2_n {
            let stage_params = FftStageParams {
                frame_count: frame_count as u32,
                frame_len: frame_len as u32,
                stage: s,
                _pad: 0,
            };
            let buf = device
                .inner()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("apollo-stft-wgpu butterfly stage params"),
                    contents: bytemuck::bytes_of(&stage_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bg = device
                .inner()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("apollo-stft-wgpu butterfly stage params BG"),
                    layout: &self.fft_params_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
            butterfly_bufs.push(buf);
            butterfly_bgs.push(bg);
        }

        // ── Step 6: Build the OLA bind group (3-binding layout) ──────────────
        let ola_bg = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-stft-wgpu ola bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: frame_data_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: signal_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                ],
            });

        // ── Step 7: Encode all passes in one CommandEncoder ───────────────────
        let mut enc = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-stft-wgpu inverse encoder"),
            });

        // Pass 1: deinterleave.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu deinterleave pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.deinterleave_pipeline);
            pass.set_bind_group(0, &fft_data_bg, &[]);
            pass.set_bind_group(1, &base_params_bg, &[]);
            pass.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        // Pass 2: bitrev.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu bitrev pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bitrev_pipeline);
            pass.set_bind_group(0, &fft_data_bg, &[]);
            pass.set_bind_group(1, &base_params_bg, &[]);
            pass.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        // Pass 3: butterfly stages.
        for s in 0..log2_n as usize {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu butterfly pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.butterfly_pipeline);
            pass.set_bind_group(0, &fft_data_bg, &[]);
            pass.set_bind_group(1, &butterfly_bgs[s], &[]);
            pass.dispatch_workgroups(
                fft_dispatch_count((frame_count * frame_len / 2) as u32),
                1,
                1,
            );
        }

        // Pass 4: scale + window.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu scale-window pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scale_window_pipeline);
            pass.set_bind_group(0, &fft_data_bg, &[]);
            pass.set_bind_group(1, &base_params_bg, &[]);
            pass.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        // Pass 5: OLA.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu inverse ola pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inverse_ola_pipeline);
            pass.set_bind_group(0, &ola_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(signal_len as u32), 1, 1);
        }

        enc.copy_buffer_to_buffer(&signal_buf, 0, &staging, 0, signal_size);

        // ── Step 8: Submit, poll, map, and collect output ─────────────────────
        device.queue().submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..signal_size);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.inner().poll(wgpu::PollType::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                staging.unmap();
                device.recycle_staging_buffer(staging);
                return Err(WgpuError::BufferMapFailed {
                    message: e.to_string(),
                });
            }
            Err(e) => {
                staging.unmap();
                device.recycle_staging_buffer(staging);
                return Err(WgpuError::BufferMapFailed {
                    message: e.to_string(),
                });
            }
        }
        let output = {
            let m = slice.get_mapped_range();
            bytemuck::cast_slice::<_, f32>(&m).to_vec()
        };
        staging.unmap();
        device.recycle_staging_buffer(staging);
        Ok(output)
    }

    /// Execute the inverse STFT using pre-allocated GPU resources.
    pub fn execute_inverse_with_buffers(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spectrum: &[Complex32],
        signal_len: usize,
        buffers: &mut crate::infrastructure::buffers::StftGpuBuffers,
    ) -> WgpuResult<()> {
        let expected_spectrum = buffers.frame_count() * buffers.frame_len();
        if spectrum.len() != expected_spectrum {
            return Err(WgpuError::LengthMismatch {
                expected: expected_spectrum,
                actual: spectrum.len(),
            });
        }
        if signal_len != buffers.signal_len() {
            return Err(WgpuError::LengthMismatch {
                expected: buffers.signal_len(),
                actual: signal_len,
            });
        }

        let frame_count = buffers.frame_count();
        let frame_len = buffers.frame_len();
        let hop_len = buffers.hop_len();
        let log2_n = buffers.log2_n;
        let inv_signal_size = (signal_len * std::mem::size_of::<f32>()) as u64;

        let spectrum_flat: Vec<f32> = spectrum.iter().flat_map(|c| [c.re, c.im]).collect();
        queue.write_buffer(
            &buffers.spectrum_buf,
            0,
            bytemuck::cast_slice(&spectrum_flat),
        );

        queue.write_buffer(
            &buffers.inv_ola_params_buf,
            0,
            bytemuck::bytes_of(&StftParams {
                signal_len: signal_len as u32,
                frame_len: frame_len as u32,
                hop_len: hop_len as u32,
                frame_count: frame_count as u32,
            }),
        );

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-stft-wgpu inv reuse encoder"),
        });

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu deinterleave pass (reuse)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.deinterleave_pipeline);
            pass.set_bind_group(0, &buffers.inv_data_bg, &[]);
            pass.set_bind_group(1, &buffers.inv_base_params_bg, &[]);
            pass.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu bitrev pass (reuse)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bitrev_pipeline);
            pass.set_bind_group(0, &buffers.inv_data_bg, &[]);
            pass.set_bind_group(1, &buffers.inv_base_params_bg, &[]);
            pass.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        for s in 0..log2_n as usize {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu butterfly pass (reuse)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.butterfly_pipeline);
            pass.set_bind_group(0, &buffers.inv_data_bg, &[]);
            pass.set_bind_group(1, &buffers.inv_butterfly_bgs[s], &[]);
            pass.dispatch_workgroups(
                fft_dispatch_count((frame_count * frame_len / 2) as u32),
                1,
                1,
            );
        }

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu scale-window pass (reuse)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scale_window_pipeline);
            pass.set_bind_group(0, &buffers.inv_data_bg, &[]);
            pass.set_bind_group(1, &buffers.inv_base_params_bg, &[]);
            pass.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-stft-wgpu inverse ola pass (reuse)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inverse_ola_pipeline);
            pass.set_bind_group(0, &buffers.ola_bg, &[]);
            pass.dispatch_workgroups(dispatch_count(signal_len as u32), 1, 1);
        }

        enc.copy_buffer_to_buffer(
            &buffers.inv_signal_buf,
            0,
            &buffers.inv_staging_buf,
            0,
            inv_signal_size,
        );

        queue.submit(std::iter::once(enc.finish()));

        let slice = buffers.inv_staging_buf.slice(..inv_signal_size);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::Wait);

        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return Err(WgpuError::BufferMapFailed {
                    message: e.to_string(),
                })
            }
            Err(e) => {
                return Err(WgpuError::BufferMapFailed {
                    message: e.to_string(),
                })
            }
        }

        {
            let m = slice.get_mapped_range();
            buffers
                .inv_output_host
                .copy_from_slice(bytemuck::cast_slice::<u8, f32>(&m));
        }

        buffers.inv_staging_buf.unmap();
        Ok(())
    }
}
