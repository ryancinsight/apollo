use num_complex::Complex32;
use std::sync::mpsc;
use wgpu::util::DeviceExt;

use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::chirp::{chirp_padded_len, StftChirpData};
use crate::infrastructure::kernel::{fft_dispatch_count, ComplexPod, StftGpuKernel};
use apollo_wgpu_helpers::WgpuDevice;

impl StftGpuKernel {
    /// Execute the forward STFT for non-power-of-two `frame_len` via Bluestein's identity.
    pub(crate) fn execute_forward_fft_chirp(
        device: &WgpuDevice,
        signal: &[f32],
        frame_len: usize,
        hop_len: usize,
        frame_count: usize,
    ) -> WgpuResult<Vec<Complex32>> {
        let signal_len = signal.len();
        let m = chirp_padded_len(frame_len);
        let log2_m = m.trailing_zeros();

        // Build StftChirpData (precomputes chirp kernel and all pipeline objects).
        let chirp = StftChirpData::new(
            device.inner(),
            device.queue().as_ref(),
            frame_len,
            frame_count,
            hop_len,
            signal_len,
        );

        // Upload signal to GPU.
        let signal_buf = device
            .inner()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-stft-wgpu chirp fwd signal"),
                contents: bytemuck::cast_slice(signal),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Output buffer: frame_count × frame_len × ComplexPod.
        let out_size = (frame_count * frame_len * std::mem::size_of::<ComplexPod>()) as u64;
        let output_buf = device.inner().create_buffer(&wgpu::BufferDescriptor {
            label: Some("apollo-stft-wgpu chirp fwd output"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.get_staging_buffer(out_size);

        // IO bind group for premul_fwd (binding 0 = signal, binding 1 = output_data).
        let io_bg_fwd = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-stft-wgpu chirp fwd IO BG"),
                layout: &chirp.io_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: signal_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_buf.as_entire_binding(),
                    },
                ],
            });

        let mut enc = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-stft-wgpu chirp fwd encoder"),
            });

        // Pass A: premul_fwd — Hann window + Bluestein exp(+πi·n²/N) premultiply.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_chirp_premul_fwd"),
                timestamp_writes: None,
            });
            p.set_pipeline(&chirp.premul_fwd_pipeline);
            p.set_bind_group(0, &chirp.chirp_data_bg, &[]);
            p.set_bind_group(1, &chirp.chirp_params_bg, &[]);
            p.set_bind_group(2, &io_bg_fwd, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * m) as u32), 1, 1);
        }

        // Pass B: Radix-2 forward sub-FFT over M on chirp working buffers.
        Self::dispatch_chirp_radix2(&mut enc, &chirp, frame_count, m, log2_m, false);

        // Pass C: pointmul — pointwise multiply by precomputed H in DFT domain.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_chirp_pointmul_fwd"),
                timestamp_writes: None,
            });
            p.set_pipeline(&chirp.pointmul_fwd_pipeline);
            p.set_bind_group(0, &chirp.chirp_data_bg, &[]);
            p.set_bind_group(1, &chirp.chirp_params_bg, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * m) as u32), 1, 1);
        }

        // Pass D: Radix-2 inverse sub-FFT over M (+ 1/M scale).
        Self::dispatch_chirp_radix2(&mut enc, &chirp, frame_count, m, log2_m, true);

        // Pass E: postmul_fwd — Bluestein exp(+πi·k²/N) postmultiply + write output.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_chirp_postmul_fwd"),
                timestamp_writes: None,
            });
            p.set_pipeline(&chirp.postmul_fwd_pipeline);
            p.set_bind_group(0, &chirp.chirp_data_bg, &[]);
            p.set_bind_group(1, &chirp.chirp_params_bg, &[]);
            p.set_bind_group(2, &io_bg_fwd, &[]);
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
            let mapped = slice.get_mapped_range();
            bytemuck::cast_slice::<_, ComplexPod>(&mapped)
                .iter()
                .map(|p| Complex32::new(p.re, p.im))
                .collect()
        };
        staging.unmap();
        device.recycle_staging_buffer(staging);
        Ok(output)
    }
}
