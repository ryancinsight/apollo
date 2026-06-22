//! NUFFT WGPU plan descriptors.

use crate::{UniformDomain1D, UniformGrid3D};

/// WGPU Type-1/Type-2 1D NUFFT plan descriptor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NufftWgpuPlan1D {
    domain: UniformDomain1D,
    oversampling: usize,
    kernel_width: usize,
}

impl NufftWgpuPlan1D {
    /// Create a 1D NUFFT WGPU plan descriptor from validated NUFFT metadata.
    #[must_use]
    pub const fn new(domain: UniformDomain1D, oversampling: usize, kernel_width: usize) -> Self {
        Self {
            domain,
            oversampling,
            kernel_width,
        }
    }

    /// Return the uniform domain descriptor.
    #[must_use]
    pub const fn domain(self) -> UniformDomain1D {
        self.domain
    }

    /// Return the oversampling factor.
    #[must_use]
    pub const fn oversampling(self) -> usize {
        self.oversampling
    }

    /// Return the Kaiser-Bessel kernel width.
    #[must_use]
    pub const fn kernel_width(self) -> usize {
        self.kernel_width
    }
}

/// WGPU Type-1/Type-2 3D NUFFT plan descriptor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NufftWgpuPlan3D {
    grid: UniformGrid3D,
    oversampling: usize,
    kernel_width: usize,
}

impl NufftWgpuPlan3D {
    /// Create a 3D NUFFT WGPU plan descriptor from validated NUFFT metadata.
    #[must_use]
    pub const fn new(grid: UniformGrid3D, oversampling: usize, kernel_width: usize) -> Self {
        Self {
            grid,
            oversampling,
            kernel_width,
        }
    }

    /// Return the uniform grid descriptor.
    #[must_use]
    pub const fn grid(self) -> UniformGrid3D {
        self.grid
    }

    /// Return the oversampling factor.
    #[must_use]
    pub const fn oversampling(self) -> usize {
        self.oversampling
    }

    /// Return the Kaiser-Bessel kernel width.
    #[must_use]
    pub const fn kernel_width(self) -> usize {
        self.kernel_width
    }
}
