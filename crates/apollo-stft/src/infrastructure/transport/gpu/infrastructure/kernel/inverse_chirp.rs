use num_complex::Complex32;
use wgpu::util::DeviceExt;

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::chirp::{chirp_padded_len, StftChirpData};
use crate::infrastructure::transport::gpu::infrastructure::kernel::{
    dispatch_count, fft_dispatch_count, ComplexPod, StftGpuKernel, StftParams,
};
use apollo_wgpu_helpers::hephaestus_wgpu::ComputeDevice;
use apollo_wgpu_helpers::WgpuDevice;

impl StftGpuKernel {
    /// Execute the inverse STFT for non-power-of-two `frame_len` via Bluestein's identity.
    pub(crate) fn execute_inverse_chirp(
        &self,
        device: &WgpuDevice,
        spectrum: &[Complex32],
        frame_len: usize,
        hop_len: usize,
        frame_count: usize,
        signal_len: usize,
    ) -> WgpuResult<Vec<f32>> {
        let m = chirp_padded_len(frame_len);
        let log2_m = m.trailing_zeros();
        let hep_device = device.hephaestus();

        let chirp = StftChirpData::new(
            self,
            device.inner(),
            device.queue().as_ref(),
            frame_len,
            frame_count,
            hop_len,
            signal_len,
        );

        // Upload spectrum to GPU.
        let spectrum_pods: Vec<ComplexPod> = spectrum
            .iter()
            .map(|c| ComplexPod { re: c.re, im: c.im })
            .collect();
        let spectrum_buf = hep_device.upload(&spectrum_pods).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        // frame_data buffer: written by postmul_inv, read by OLA pass.
        let frame_data_buf = hep_device.alloc_zeroed::<f32>(frame_count * frame_len).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        // OLA signal output.
        let signal_buf = hep_device.alloc_zeroed::<f32>(signal_len).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        // IO bind group for premul_inv (binding 0 = interleaved spectrum, binding 1 = frame_data).
        let io_bg_inv = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-stft-wgpu chirp inv IO BG"),
                layout: &chirp.io_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: spectrum_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: frame_data_buf.as_entire_binding(),
                    },
                ],
            });

        // OLA bind group (3-binding layout: frame_data ro, signal rw, params uniform).
        let params_buffer = device
            .inner()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-stft-wgpu chirp inv params"),
                contents: bytemuck::bytes_of(&StftParams {
                    signal_len: signal_len as u32,
                    frame_len: frame_len as u32,
                    hop_len: hop_len as u32,
                    frame_count: frame_count as u32,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let ola_bg = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-stft-wgpu chirp inv OLA BG"),
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
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut enc = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-stft-wgpu chirp inv encoder"),
            });

        // Pass A: premul_inv.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_chirp_premul_inv"),
                timestamp_writes: None,
            });
            p.set_pipeline(&chirp.premul_inv_pipeline);
            p.set_bind_group(0, &chirp.chirp_data_bg, &[]);
            p.set_bind_group(1, &chirp.chirp_params_bg, &[]);
            p.set_bind_group(2, &io_bg_inv, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * m) as u32), 1, 1);
        }

        // Pass B: Radix-2 forward sub-FFT over M.
        Self::dispatch_chirp_radix2(&mut enc, &chirp, frame_count, m, log2_m, false);

        // Pass C: pointmul.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_chirp_pointmul"),
                timestamp_writes: None,
            });
            p.set_pipeline(&chirp.pointmul_pipeline);
            p.set_bind_group(0, &chirp.chirp_data_bg, &[]);
            p.set_bind_group(1, &chirp.chirp_params_bg, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * m) as u32), 1, 1);
        }

        // Pass D: Radix-2 inverse sub-FFT over M (+ 1/M scale).
        Self::dispatch_chirp_radix2(&mut enc, &chirp, frame_count, m, log2_m, true);

        // Pass E: postmul_inv — conjugate postmul + 1/N scale + Hann window → frame_data.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_chirp_postmul_inv"),
                timestamp_writes: None,
            });
            p.set_pipeline(&chirp.postmul_inv_pipeline);
            p.set_bind_group(0, &chirp.chirp_data_bg, &[]);
            p.set_bind_group(1, &chirp.chirp_params_bg, &[]);
            p.set_bind_group(2, &io_bg_inv, &[]);
            p.dispatch_workgroups(fft_dispatch_count((frame_count * frame_len) as u32), 1, 1);
        }

        // Pass F: OLA reconstruction.
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stft_inverse_ola (chirp)"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.inverse_ola_pipeline);
            p.set_bind_group(0, &ola_bg, &[]);
            p.dispatch_workgroups(dispatch_count(signal_len as u32), 1, 1);
        }

        device.queue().submit(std::iter::once(enc.finish()));

        let mut output = vec![0.0f32; signal_len];
        hep_device.download(&signal_buf, &mut output).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        Ok(output)
    }

    /// Dispatch the Radix-2 sub-FFT passes of the Chirp-Z path over the chirp working buffers.
    pub(crate) fn dispatch_chirp_radix2(
        enc: &mut wgpu::CommandEncoder,
        chirp: &StftChirpData,
        frame_count: usize,
        m: usize,
        log2_m: u32,
        inverse: bool,
    ) {
        let bgs = if inverse {
            &chirp.radix2_inv_bgs
        } else {
            &chirp.radix2_fwd_bgs
        };
        let bitrev_total = (frame_count * m) as u32;
        let butterfly_total = (frame_count * m / 2) as u32;

        // Bitrev pass (bgs[0]).
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chirp_fft_bitrev"),
                timestamp_writes: None,
            });
            let pipeline = &chirp.chirp_bitrev_pipeline;
            p.set_pipeline(pipeline);
            p.set_bind_group(0, &chirp.chirp_data_bg, &[]);
            p.set_bind_group(1, &bgs[0], &[]);
            p.dispatch_workgroups(fft_dispatch_count(bitrev_total), 1, 1);
        }

        // Butterfly passes (bgs[1..=log2_m]).
        for s in 0..log2_m as usize {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chirp_fft_butterfly"),
                timestamp_writes: None,
            });
            let pipeline = if inverse {
                &chirp.chirp_inv_butterfly_pipeline
            } else {
                &chirp.chirp_fwd_butterfly_pipeline
            };
            p.set_pipeline(pipeline);
            p.set_bind_group(0, &chirp.chirp_data_bg, &[]);
            p.set_bind_group(1, &bgs[1 + s], &[]);
            p.dispatch_workgroups(fft_dispatch_count(butterfly_total), 1, 1);
        }

        // Scale pass for inverse (bgs[log2_m + 1]).
        if inverse {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chirp_fft_scale"),
                timestamp_writes: None,
            });
            p.set_pipeline(&chirp.chirp_scale_pipeline);
            p.set_bind_group(0, &chirp.chirp_data_bg, &[]);
            p.set_bind_group(1, bgs.last().unwrap(), &[]);
            p.dispatch_workgroups(fft_dispatch_count(bitrev_total), 1, 1);
        }
    }
}
