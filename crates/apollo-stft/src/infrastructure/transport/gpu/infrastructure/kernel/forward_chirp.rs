use num_complex::Complex32;
use wgpu::util::DeviceExt;

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::chirp::{chirp_padded_len, StftChirpData};
use crate::infrastructure::transport::gpu::infrastructure::kernel::{fft_dispatch_count, ComplexPod, StftGpuKernel};
use apollo_wgpu_helpers::hephaestus_wgpu::ComputeDevice;
use apollo_wgpu_helpers::WgpuDevice;

impl StftGpuKernel {
    /// Execute the forward STFT for non-power-of-two `frame_len` via Bluestein's identity.
    pub(crate) fn execute_forward_fft_chirp(
        &self,
        device: &WgpuDevice,
        signal: &[f32],
        frame_len: usize,
        hop_len: usize,
        frame_count: usize,
    ) -> WgpuResult<Vec<Complex32>> {
        let signal_len = signal.len();
        let m = chirp_padded_len(frame_len);
        let log2_m = m.trailing_zeros();
        let hep_device = device.hephaestus();

        // Build StftChirpData (precomputes chirp kernel and all pipeline objects).
        let chirp = StftChirpData::new(
            self,
            device.inner(),
            device.queue().as_ref(),
            frame_len,
            frame_count,
            hop_len,
            signal_len,
        );

        // Upload signal to GPU.
        let signal_buf = hep_device.upload(signal).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        // Output buffer: frame_count × frame_len × ComplexPod.
        let output_buf = hep_device.alloc_zeroed::<ComplexPod>(frame_count * frame_len).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

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

        device.queue().submit(std::iter::once(enc.finish()));

        let mut pods = vec![ComplexPod { re: 0.0, im: 0.0 }; frame_count * frame_len];
        hep_device.download(&output_buf, &mut pods).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        Ok(pods.iter().map(|p| Complex32::new(p.re, p.im)).collect())
    }
}
