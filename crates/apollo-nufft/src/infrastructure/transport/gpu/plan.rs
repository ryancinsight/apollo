//! NUFFT plan payload for the shared WGPU transform transport.
//!
//! One payload carries both uniform geometries the non-uniform transform
//! binds: a 1D Fourier domain and a 3D mode grid. The rank is payload state
//! because the non-uniform sample count is a per-call operand, not plan
//! metadata; execution surfaces reject a rank mismatch with a typed error.

use crate::{UniformDomain1D, UniformGrid3D};
use apollo_fft::{WgpuError, WgpuResult};

/// Uniform geometry and gridding parameters bound by one NUFFT plan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NufftPlan {
    /// Uniform 1D Fourier domain producing `n` modes.
    OneDimension {
        /// Validated uniform domain descriptor.
        domain: UniformDomain1D,
        /// Fast-path oversampling factor.
        oversampling: usize,
        /// Fast-path Kaiser-Bessel kernel width.
        kernel_width: usize,
    },
    /// Uniform 3D mode grid.
    ThreeDimension {
        /// Validated uniform grid descriptor.
        grid: UniformGrid3D,
        /// Fast-path oversampling factor.
        oversampling: usize,
        /// Fast-path Kaiser-Bessel kernel width.
        kernel_width: usize,
    },
}

impl NufftPlan {
    /// Create a 1D NUFFT plan payload from validated metadata.
    #[must_use]
    pub const fn one_dimension(
        domain: UniformDomain1D,
        oversampling: usize,
        kernel_width: usize,
    ) -> Self {
        Self::OneDimension {
            domain,
            oversampling,
            kernel_width,
        }
    }

    /// Create a 3D NUFFT plan payload from validated metadata.
    #[must_use]
    pub const fn three_dimension(
        grid: UniformGrid3D,
        oversampling: usize,
        kernel_width: usize,
    ) -> Self {
        Self::ThreeDimension {
            grid,
            oversampling,
            kernel_width,
        }
    }

    /// Return the uniform 1D domain.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan rejection when the payload is 3D.
    pub fn domain(&self) -> WgpuResult<UniformDomain1D> {
        match self {
            Self::OneDimension { domain, .. } => Ok(*domain),
            Self::ThreeDimension { .. } => Err(WgpuError::InvalidPlan {
                message: "NUFFT plan is not one-dimensional".to_owned(),
            }),
        }
    }

    /// Return the uniform 3D grid.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan rejection when the payload is 1D.
    pub fn grid(&self) -> WgpuResult<UniformGrid3D> {
        match self {
            Self::ThreeDimension { grid, .. } => Ok(*grid),
            Self::OneDimension { .. } => Err(WgpuError::InvalidPlan {
                message: "NUFFT plan is not three-dimensional".to_owned(),
            }),
        }
    }

    /// Return the fast-path oversampling factor.
    #[must_use]
    pub const fn oversampling(&self) -> usize {
        match self {
            Self::OneDimension { oversampling, .. } | Self::ThreeDimension { oversampling, .. } => {
                *oversampling
            }
        }
    }

    /// Return the fast-path Kaiser-Bessel kernel width.
    #[must_use]
    pub const fn kernel_width(&self) -> usize {
        match self {
            Self::OneDimension { kernel_width, .. } | Self::ThreeDimension { kernel_width, .. } => {
                *kernel_width
            }
        }
    }

    /// Total uniform-grid element count demanded by the payload.
    ///
    /// Zero reports an unusable grid the shared validation rejects; the
    /// count saturates rather than wrapping when the 3D product overflows.
    #[must_use]
    pub fn grid_len(&self) -> usize {
        match self {
            Self::OneDimension { domain, .. } => domain.n,
            Self::ThreeDimension { grid, .. } => grid
                .nx
                .checked_mul(grid.ny)
                .and_then(|value| value.checked_mul(grid.nz))
                .unwrap_or(0),
        }
    }
}
