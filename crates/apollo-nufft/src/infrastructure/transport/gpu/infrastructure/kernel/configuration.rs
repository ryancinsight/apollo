//! Plan-derived fast NUFFT configuration retained across dispatches.

use crate::infrastructure::kernel::kaiser_bessel::{fft_signed_index, i0, kb_kernel_ft};
use crate::infrastructure::transport::gpu::application::plan::{NufftWgpuPlan1D, NufftWgpuPlan3D};
use crate::infrastructure::transport::gpu::domain::error::{NufftWgpuError, NufftWgpuResult};

#[derive(Debug)]
pub(super) struct FastConfiguration1D {
    pub(super) n: usize,
    pub(super) oversampled_len: usize,
    pub(super) kernel_width: usize,
    pub(super) length: f32,
    pub(super) beta: f32,
    pub(super) i0_beta: f32,
    pub(super) deconvolution: Vec<f32>,
}

impl FastConfiguration1D {
    pub(super) fn new(plan: &NufftWgpuPlan1D) -> NufftWgpuResult<Self> {
        let n = plan.domain().n;
        let oversampling = plan.oversampling();
        let kernel_width = plan.kernel_width();
        if oversampling < 2 {
            return Err(NufftWgpuError::InvalidPlan {
                message: "fast 1D NUFFT oversampling factor must be >= 2",
            });
        }
        if kernel_width < 2 {
            return Err(NufftWgpuError::InvalidPlan {
                message: "fast 1D NUFFT kernel width must be >= 2",
            });
        }
        dimension(kernel_width, "kernel width")?;
        dimension(n, "mode count")?;
        let oversampled_len = n
            .checked_mul(oversampling)
            .ok_or(NufftWgpuError::InvalidPlan {
                message: "fast 1D NUFFT oversampled length overflow",
            })?;
        dimension(oversampled_len, "oversampled grid length")?;
        let beta = core::f64::consts::PI
            * (1.0 - 1.0 / (2.0 * oversampling as f64))
            * (2.0 * kernel_width as f64);
        let i0_beta = i0(beta);
        let deconvolution = (0..n)
            .map(|index| {
                let frequency = fft_signed_index(index, n) as f64 / oversampled_len as f64;
                (1.0 / kb_kernel_ft(frequency, kernel_width, beta, i0_beta)) as f32
            })
            .collect();
        Ok(Self {
            n,
            oversampled_len,
            kernel_width,
            length: plan.domain().length() as f32,
            beta: beta as f32,
            i0_beta: i0_beta as f32,
            deconvolution,
        })
    }
}

#[derive(Debug)]
pub(super) struct FastConfiguration3D {
    pub(super) shape: (usize, usize, usize),
    pub(super) oversampled: (usize, usize, usize),
    pub(super) kernel_width: usize,
    pub(super) lengths: (f32, f32, f32),
    pub(super) beta: f32,
    pub(super) i0_beta: f32,
    pub(super) deconvolution: Vec<f32>,
}

impl FastConfiguration3D {
    pub(super) fn new(plan: &NufftWgpuPlan3D) -> NufftWgpuResult<Self> {
        let grid = plan.grid();
        let oversampling = plan.oversampling();
        let kernel_width = plan.kernel_width();
        if oversampling < 2 {
            return Err(NufftWgpuError::InvalidPlan {
                message: "fast 3D NUFFT oversampling factor must be >= 2",
            });
        }
        if kernel_width < 2 {
            return Err(NufftWgpuError::InvalidPlan {
                message: "fast 3D NUFFT kernel width must be >= 2",
            });
        }
        dimension(kernel_width, "kernel width")?;
        let mx = oversampled_dimension(grid.nx, oversampling, kernel_width, "x")?;
        let my = oversampled_dimension(grid.ny, oversampling, kernel_width, "y")?;
        let mz = oversampled_dimension(grid.nz, oversampling, kernel_width, "z")?;
        dimension(
            mx.checked_mul(my)
                .and_then(|value| value.checked_mul(mz))
                .ok_or(NufftWgpuError::InvalidPlan {
                    message: "fast 3D NUFFT oversampled grid length overflow",
                })?,
            "oversampled grid length",
        )?;
        let beta = core::f64::consts::PI
            * (1.0 - 1.0 / (2.0 * oversampling as f64))
            * (2.0 * kernel_width as f64);
        let i0_beta = i0(beta);
        let mut deconvolution = Vec::with_capacity(
            grid.nx
                .checked_add(grid.ny)
                .and_then(|value| value.checked_add(grid.nz))
                .ok_or(NufftWgpuError::InvalidPlan {
                    message: "3D deconvolution length overflows usize",
                })?,
        );
        extend_deconvolution(&mut deconvolution, grid.nx, mx, kernel_width, beta, i0_beta);
        extend_deconvolution(&mut deconvolution, grid.ny, my, kernel_width, beta, i0_beta);
        extend_deconvolution(&mut deconvolution, grid.nz, mz, kernel_width, beta, i0_beta);
        let (lx, ly, lz) = grid.lengths();
        Ok(Self {
            shape: (grid.nx, grid.ny, grid.nz),
            oversampled: (mx, my, mz),
            kernel_width,
            lengths: (lx as f32, ly as f32, lz as f32),
            beta: beta as f32,
            i0_beta: i0_beta as f32,
            deconvolution,
        })
    }
}

fn oversampled_dimension(
    length: usize,
    oversampling: usize,
    kernel_width: usize,
    axis: &'static str,
) -> NufftWgpuResult<usize> {
    let minimum_length = kernel_width
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT kernel support length overflow",
        })?;
    let raw = length
        .checked_mul(oversampling)
        .ok_or(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT oversampled dimension overflow",
        })?
        .max(minimum_length);
    let value = raw
        .checked_next_power_of_two()
        .ok_or(NufftWgpuError::InvalidPlan {
            message: "fast 3D NUFFT radix-2 length overflow",
        })?;
    dimension(value, axis)?;
    Ok(value)
}

fn extend_deconvolution(
    output: &mut Vec<f32>,
    length: usize,
    oversampled_length: usize,
    kernel_width: usize,
    beta: f64,
    i0_beta: f64,
) {
    output.extend((0..length).map(|index| {
        let frequency = fft_signed_index(index, length) as f64 / oversampled_length as f64;
        (1.0 / kb_kernel_ft(frequency, kernel_width, beta, i0_beta)) as f32
    }));
}

fn dimension(value: usize, message: &'static str) -> NufftWgpuResult<u32> {
    u32::try_from(value).map_err(|_| NufftWgpuError::InvalidPlan { message })
}

#[cfg(test)]
mod tests {
    use super::{FastConfiguration1D, FastConfiguration3D};
    use crate::{
        infrastructure::transport::gpu::{NufftWgpuError, NufftWgpuPlan1D, NufftWgpuPlan3D},
        UniformDomain1D, UniformGrid3D,
    };

    fn assert_kernel_width_error(error: NufftWgpuError) {
        match error {
            NufftWgpuError::InvalidPlan { message } => assert_eq!(message, "kernel width"),
            other => panic!("expected invalid plan, received {other:?}"),
        }
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn oversized_kernel_width_is_rejected_before_derived_arithmetic() {
        let kernel_width = u32::MAX as usize + 1;
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        let one_dimensional = NufftWgpuPlan1D::new(domain, 2, kernel_width);
        assert_kernel_width_error(
            FastConfiguration1D::new(&one_dimensional)
                .expect_err("oversized 1D kernel width must fail"),
        );

        let grid = UniformGrid3D::new(2, 2, 2, 0.5, 0.5, 0.5).expect("grid");
        let three_dimensional = NufftWgpuPlan3D::new(grid, 2, kernel_width);
        assert_kernel_width_error(
            FastConfiguration3D::new(&three_dimensional)
                .expect_err("oversized 3D kernel width must fail"),
        );
    }
}
