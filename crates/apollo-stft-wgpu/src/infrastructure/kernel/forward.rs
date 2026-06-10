use num_complex::Complex32;
use std::sync::mpsc;
use wgpu::util::DeviceExt;

use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::{
    fft_dispatch_count, ComplexPod, FwdFftStageParams, StftGpuKernel,
};
use apollo_wgpu_helpers::WgpuDevice;

impl StftGpuKernel {
    /// Execute the forward STFT via FFT-accelerated batch DFT (O(N log N) per frame).
    ///
    /// ## Invariants
    /// - `frame_len` must be a power of two (Radix-2 requirement).
    pub fn execute_forward_fft(
        &self,
        device: &WgpuDevice,
        signal: &[f32],
        frame_len: usize,
        hop_len: usize,
        frame_count: usize,
    ) -> WgpuResult<Vec<Complex32>> {
        // Non-power-of-two frame_len: delegate to Bluestein/Chirp-Z path.
        if !frame_len.is_power_of_two() {
            return Self::execute_forward_fft_chirp(
                device,
                signal,
                frame_len,
                hop_len,
                frame_count,
            );
        }
        let log2_n = frame_len.trailing_zeros();

        let signal_buf = device
            .inner()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-stft-wgpu fwd signal"),
                contents: bytemuck::cast_slice(signal),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let scratch_size = (frame_count * frame_len * std::mem::size_of::<f32>()) as u64;
        let re_scratch_buf = device.inner().create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu fwd re scratch"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let im_scratch_buf = device.inner().create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu fwd im scratch"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let out_size = (frame_count * frame_len * std::mem::size_of::<ComplexPod>()) as u64;
        let output_buf = device.inner().create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu fwd output"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.get_staging_buffer(out_size);

        // Bind group 0: reuse fft_data_bgl (binding types are identical: ro, rw, rw, rw).
        let fft_fwd_data_bg = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-stft-wgpu fwd FFT data BG"),
                layout: &self.fft_data_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: signal_buf.as_entire_binding(),
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
                        resource: output_buf.as_entire_binding(),
                    },
                ],
            });

        // Base params bind group: stage=0, hop_len filled.
        // Used for pack_window, bitrev, and interleave passes (stage field unused for these).
        let base_params = FwdFftStageParams {
            frame_count: frame_count as u32,
            frame_len: frame_len as u32,
            hop_len: hop_len as u32,
            stage: 0,
        };
        let base_params_buf =
            device
                .inner()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("apollo-stft-wgpu fwd base params"),
                    contents: bytemuck::bytes_of(&base_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
        let base_params_bg = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-stft-wgpu fwd base params BG"),
                layout: &self.fft_params_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: base_params_buf.as_entire_binding(),
                }],
            });

        // Per-butterfly-stage params bind groups (one per stage, stage index varies).
        let mut butterfly_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(log2_n as usize);
        let mut butterfly_bgs: Vec<wgpu::BindGroup> = Vec::with_capacity(log2_n as usize);
        for s in 0..log2_n {
            let stage_params = FwdFftStageParams {
                frame_count: frame_count as u32,
                frame_len: frame_len as u32,
                hop_len: hop_len as u32,
                stage: s,
            };
            let buf = device
                .inner()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("apollo-stft-wgpu fwd butterfly stage params"),
                    contents: bytemuck::bytes_of(&stage_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bg = device
                .inner()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("apollo-stft-wgpu fwd butterfly stage params BG"),
                    layout: &self.fft_params_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
            butterfly_bufs.push(buf);
            butterfly_bgs.push(bg);
        }

        let mut enc = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-stft-wgpu fwd encoder"),
            });

        // Pass 1: pack + Hann analysis window → split re/im scratch.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_fwd_pack_window"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.fwd_pack_window_pipeline);
            p.set_bind_group(0, &fft_fwd_data_bg, &[]);
            p.set_bind_group(1, &base_params_bg, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }
        // Pass 2: bit-reversal permutation.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_fwd_bitrev"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.fwd_bitrev_pipeline);
            p.set_bind_group(0, &fft_fwd_data_bg, &[]);
            p.set_bind_group(1, &base_params_bg, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }
        // Pass 3: Radix-2 DIT butterfly × log₂(frame_len).
        for s in 0..log2_n as usize {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_fwd_butterfly"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.fwd_butterfly_pipeline);
            p.set_bind_group(0, &fft_fwd_data_bg, &[]);
            p.set_bind_group(1, &butterfly_bgs[s], &[]);
            p.dispatch_workgroups(
                fft_dispatch_count((frame_count * frame_len / 2) as u32),
                1,
                1,
            );
        }
        // Pass 4: interleave split re/im → output ComplexValue array.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_fwd_interleave"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.fwd_interleave_pipeline);
            p.set_bind_group(0, &fft_fwd_data_bg, &[]);
            p.set_bind_group(1, &base_params_bg, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        enc.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, out_size);
        device.queue().submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..out_size);
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
            bytemuck::cast_slice::<_, ComplexPod>(&m)
                .iter()
                .map(|p| Complex32::new(p.re, p.im))
                .collect()
        };
        staging.unmap();
        device.recycle_staging_buffer(staging);
        Ok(output)
    }

    /// Execute the forward STFT using pre-allocated GPU resources.
    pub fn execute_forward_fft_with_buffers(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        signal: &[f32],
        buffers: &mut crate::infrastructure::buffers::StftGpuBuffers,
    ) -> WgpuResult<()> {
        if signal.len() != buffers.signal_len() {
            return Err(WgpuError::LengthMismatch {
                expected: buffers.signal_len(),
                actual: signal.len(),
            });
        }

        let frame_count = buffers.frame_count();
        let frame_len = buffers.frame_len();
        let log2_n = buffers.log2_n;
        let out_size = (frame_count * frame_len * std::mem::size_of::<ComplexPod>()) as u64;

        queue.write_buffer(&buffers.signal_buf, 0, bytemuck::cast_slice(signal));

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apollo-stft-wgpu fwd reuse encoder"),
        });

        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_fwd_pack_window (reuse)"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.fwd_pack_window_pipeline);
            p.set_bind_group(0, &buffers.fwd_data_bg, &[]);
            p.set_bind_group(1, &buffers.fwd_base_params_bg, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_fwd_bitrev (reuse)"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.fwd_bitrev_pipeline);
            p.set_bind_group(0, &buffers.fwd_data_bg, &[]);
            p.set_bind_group(1, &buffers.fwd_base_params_bg, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        for s in 0..log2_n as usize {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_fwd_butterfly (reuse)"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.fwd_butterfly_pipeline);
            p.set_bind_group(0, &buffers.fwd_data_bg, &[]);
            p.set_bind_group(1, &buffers.fwd_butterfly_bgs[s], &[]);
            p.dispatch_workgroups(
                fft_dispatch_count((frame_count * frame_len / 2) as u32),
                1,
                1,
            );
        }

        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_fwd_interleave (reuse)"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.fwd_interleave_pipeline);
            p.set_bind_group(0, &buffers.fwd_data_bg, &[]);
            p.set_bind_group(1, &buffers.fwd_base_params_bg, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        enc.copy_buffer_to_buffer(
            &buffers.fwd_output_buf,
            0,
            &buffers.fwd_staging_buf,
            0,
            out_size,
        );

        queue.submit(std::iter::once(enc.finish()));

        let slice = buffers.fwd_staging_buf.slice(..out_size);
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
            let pods = bytemuck::cast_slice::<u8, ComplexPod>(&m);
            for (slot, p) in buffers.fwd_output_host.iter_mut().zip(pods.iter()) {
                *slot = Complex32::new(p.re, p.im);
            }
        }

        buffers.fwd_staging_buf.unmap();
        Ok(())
    }
}
