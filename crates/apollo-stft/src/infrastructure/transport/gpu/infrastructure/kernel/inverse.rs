use eunomia::Complex32;
use wgpu::util::DeviceExt;

use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::kernel::{
    dispatch_count, fft_dispatch_count, ComplexPod, FftStageParams, StftGpuKernel, StftParams,
};
use apollo_wgpu_helpers::hephaestus_wgpu::ComputeDevice;
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
        let hep_device = device.hephaestus();

        // ── Step 1: Build flat interleaved spectrum for GPU upload ────────────
        let spectrum_pods: Vec<ComplexPod> = spectrum.iter().map(|c| ComplexPod { re: c.re, im: c.im }).collect();

        // ── Step 2: Allocate GPU buffers ──────────────────────────────────────
        let spectrum_buf = hep_device.upload(&spectrum_pods).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        let re_scratch_buf = hep_device.alloc_zeroed::<f32>(frame_count * frame_len).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;
        let im_scratch_buf = hep_device.alloc_zeroed::<f32>(frame_count * frame_len).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;
        let frame_data_buf = hep_device.alloc_zeroed::<f32>(frame_count * frame_len).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        let signal_buf = hep_device.alloc_zeroed::<f32>(signal_len).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        // ── Step 3: Write OLA uniform params (StftParams) ─────────────────────
        let params_buffer = device
            .inner()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-stft-wgpu params"),
                contents: bytemuck::bytes_of(&StftParams {
                    signal_len: signal_len as u32,
                    frame_len: frame_len as u32,
                    hop_len: hop_len as u32,
                    frame_count: frame_count as u32,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // ── Step 4: Write FFT uniform base params ─────────────────────────────
        let base_params = FftStageParams {
            frame_count: frame_count as u32,
            frame_len: frame_len as u32,
            stage: 0,
            _pad: 0,
        };
        let base_params_buf = device
            .inner()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("apollo-stft-wgpu base params"),
                contents: bytemuck::bytes_of(&base_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // ── Step 5: Write butterfly uniform params (one buffer per stage) ─────
        let mut butterfly_bufs = Vec::with_capacity(log2_n as usize);
        let mut butterfly_bgs = Vec::with_capacity(log2_n as usize);
        for s in 0..log2_n {
            let buf = device
                .inner()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("apollo-stft-wgpu butterfly params"),
                    contents: bytemuck::bytes_of(&FftStageParams {
                        frame_count: frame_count as u32,
                        frame_len: frame_len as u32,
                        stage: s,
                        _pad: 0,
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bg = device
                .inner()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("apollo-stft-wgpu butterfly params BG"),
                    layout: &self.fft_params_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
            butterfly_bufs.push(buf);
            butterfly_bgs.push(bg);
        }

        // ── Step 6: Create bind groups ────────────────────────────────────────
        let fft_data_bg = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-stft-wgpu FFT data BG"),
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
        let base_params_bg = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-stft-wgpu base params BG"),
                layout: &self.fft_params_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: base_params_buf.as_entire_binding(),
                }],
            });
        let ola_bg = device
            .inner()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apollo-stft-wgpu OLA BG"),
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

        // ── Step 7: Record compute passes ─────────────────────────────────────
        let mut enc = device
            .inner()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apollo-stft-wgpu inverse encoder"),
            });

        // Pass 1: deinterleave flat spectrum into re/im scratch.
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

        // Pass 2: bit-reversal sorting.
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

        // Pass 3: Radix-2 butterfly merges.
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

        // Pass 4: scale by 1/N and apply synthesis window.
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
        device.queue().submit(std::iter::once(enc.finish()));

        let mut output = vec![0.0f32; signal_len];
        hep_device.download(&signal_buf, &mut output).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;
        Ok(output)
    }

    /// Execute the inverse STFT using pre-allocated GPU resources.
    pub fn execute_inverse_with_buffers(
        &self,
        device: &WgpuDevice,
        spectrum: &[Complex32],
        signal_len: usize,
        buffers: &mut crate::infrastructure::transport::gpu::infrastructure::buffers::StftGpuBuffers,
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
        let hep_device = device.hephaestus();

        let spectrum_pods: Vec<ComplexPod> = spectrum.iter().map(|c| ComplexPod { re: c.re, im: c.im }).collect();
        hep_device.write_buffer(&buffers.spectrum_buf, &spectrum_pods).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        device.queue().write_buffer(
            &buffers.inv_ola_params_buf,
            0,
            bytemuck::bytes_of(&StftParams {
                signal_len: signal_len as u32,
                frame_len: frame_len as u32,
                hop_len: hop_len as u32,
                frame_count: frame_count as u32,
            }),
        );

        let mut enc = device.inner().create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

        device.queue().submit(std::iter::once(enc.finish()));

        hep_device.download(&buffers.inv_signal_buf, &mut buffers.inv_output_host).map_err(|e| WgpuError::BufferMapFailed {
            message: e.to_string(),
        })?;

        Ok(())
    }
}
