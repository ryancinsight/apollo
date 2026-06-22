//! WGPU compute dispatch logic for the FFT execution.

use crate::infrastructure::transport::gpu::infrastructure::gpu_fft::pipeline::GpuFft3d;
use crate::infrastructure::transport::gpu::infrastructure::gpu_fft::strategy::{Axis, ChirpData, RadixStages};

impl GpuFft3d {
    pub(crate) fn dispatch_pack(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        axis: Axis,
        axis_len: u32,
        batch_count: u32,
    ) {
        let stage = match axis {
            Axis::X => &self.pack_bg_x,
            Axis::Y => &self.pack_bg_y,
            Axis::Z => &self.pack_bg_z,
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("apollo-fft-wgpu pack pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pack_pipeline);
        pass.set_bind_group(0, &self.data_bg, &[]);
        pass.set_bind_group(1, &stage.fft_params_bg, &[]);
        pass.set_bind_group(2, &stage.bg, &[]);
        let total = axis_len * batch_count;
        pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
    }

    pub(crate) fn dispatch_unpack(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        axis: Axis,
        axis_len: u32,
        batch_count: u32,
    ) {
        let stage = match axis {
            Axis::X => &self.pack_bg_x,
            Axis::Y => &self.pack_bg_y,
            Axis::Z => &self.pack_bg_z,
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("apollo-fft-wgpu unpack pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.unpack_pipeline);
        pass.set_bind_group(0, &self.data_bg, &[]);
        pass.set_bind_group(1, &stage.fft_params_bg, &[]);
        pass.set_bind_group(2, &stage.bg, &[]);
        let total = axis_len * batch_count;
        pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
    }

    pub(crate) fn dispatch_radix2(&self, encoder: &mut wgpu::CommandEncoder, stages: &RadixStages) {
        if stages.fft_m == 0 {
            return;
        }

        if stages.radix4 {
            let bitrev_total = stages.batch_count * stages.fft_m;
            let butterfly_total = stages.batch_count * (stages.fft_m / 4);

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("apollo-fft-wgpu radix4 bitrev pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.bitrev_radix4_pipeline);
                pass.set_bind_group(0, &self.data_bg, &[]);
                pass.set_bind_group(1, &stages.bgs[0], &[]);
                pass.dispatch_workgroups(bitrev_total.div_ceil(256), 1, 1);
            }

            for stage_idx in 0..(stages.fft_m.trailing_zeros() as usize / 2) {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("apollo-fft-wgpu radix4 butterfly pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.forward_radix4_pipeline);
                pass.set_bind_group(0, &self.data_bg, &[]);
                pass.set_bind_group(1, &stages.bgs[1 + stage_idx], &[]);
                pass.dispatch_workgroups(butterfly_total.div_ceil(256), 1, 1);
            }

            if stages.bgs.len() > 1 + (stages.fft_m.trailing_zeros() as usize / 2) {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("apollo-fft-wgpu radix4 scale pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scale_pipeline);
                pass.set_bind_group(0, &self.data_bg, &[]);
                pass.set_bind_group(1, stages.bgs.last().unwrap(), &[]);
                pass.dispatch_workgroups(bitrev_total.div_ceil(256), 1, 1);
            }
            return;
        }

        let bitrev_total = stages.batch_count * stages.fft_m;
        let butterfly_total = stages.batch_count * (stages.fft_m / 2);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-fft-wgpu radix2 bitrev pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bitrev_pipeline);
            pass.set_bind_group(0, &self.data_bg, &[]);
            pass.set_bind_group(1, &stages.bgs[0], &[]);
            pass.dispatch_workgroups(bitrev_total.div_ceil(256), 1, 1);
        }

        for stage_idx in 0..stages.fft_m.trailing_zeros() as usize {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-fft-wgpu radix2 butterfly pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.forward_pipeline);
            pass.set_bind_group(0, &self.data_bg, &[]);
            pass.set_bind_group(1, &stages.bgs[1 + stage_idx], &[]);
            pass.dispatch_workgroups(butterfly_total.div_ceil(256), 1, 1);
        }

        if stages.bgs.len() > 1 + stages.fft_m.trailing_zeros() as usize {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-fft-wgpu radix2 scale pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scale_pipeline);
            pass.set_bind_group(0, &self.data_bg, &[]);
            pass.set_bind_group(1, stages.bgs.last().unwrap(), &[]);
            pass.dispatch_workgroups(bitrev_total.div_ceil(256), 1, 1);
        }
    }

    pub(crate) fn dispatch_chirp(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        chirp: &ChirpData,
        inverse: bool,
    ) {
        if chirp.n == 0 || chirp.m == 0 {
            return;
        }

        let padded_total = chirp.m * chirp.batch_count;
        let output_total = chirp.n * chirp.batch_count;

        if inverse {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-fft-wgpu chirp pass (pre-conjugate)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&chirp.negate_im_pipeline);
            pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
            pass.set_bind_group(1, &chirp.params_bg, &[]);
            pass.dispatch_workgroups(output_total.div_ceil(256), 1, 1);
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("apollo-fft-wgpu chirp pass (premul)"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&chirp.premul_pipeline);
        pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
        pass.set_bind_group(1, &chirp.params_bg, &[]);
        pass.dispatch_workgroups(padded_total.div_ceil(256), 1, 1);
        drop(pass);

        self.dispatch_radix2(encoder, &chirp.radix2_fwd);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("apollo-fft-wgpu chirp pass (pointmul)"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&chirp.pointmul_pipeline);
        pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
        pass.set_bind_group(1, &chirp.params_bg, &[]);
        pass.dispatch_workgroups(padded_total.div_ceil(256), 1, 1);
        drop(pass);

        self.dispatch_radix2(encoder, &chirp.radix2_inv);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("apollo-fft-wgpu chirp pass (post)"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&chirp.postmul_pipeline);
        pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
        pass.set_bind_group(1, &chirp.params_bg, &[]);
        pass.dispatch_workgroups(output_total.div_ceil(256), 1, 1);
        drop(pass);

        if inverse {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-fft-wgpu chirp pass (post-conjugate)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&chirp.negate_im_pipeline);
            pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
            pass.set_bind_group(1, &chirp.params_bg, &[]);
            pass.dispatch_workgroups(output_total.div_ceil(256), 1, 1);
            drop(pass);

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apollo-fft-wgpu chirp pass (scale)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&chirp.scale_pipeline);
            pass.set_bind_group(0, &chirp.data_chirp_bg, &[]);
            pass.set_bind_group(1, &chirp.params_bg, &[]);
            pass.dispatch_workgroups(output_total.div_ceil(256), 1, 1);
        }
    }
}
