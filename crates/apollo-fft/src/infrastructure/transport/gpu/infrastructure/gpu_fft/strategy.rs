//! Dense-FFT execution strategies and provider-neutral stage metadata.

use hephaestus_wgpu::WgpuBuffer;

use crate::infrastructure::transport::fft::stages::RadixStages;

use super::kernel::ChirpParams;

/// Execution strategy chosen for a single FFT axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisStrategy {
    /// Native radix decomposition for power-of-two lengths.
    Radix2,
    /// Bluestein / Chirp-Z reduction for arbitrary lengths.
    ChirpZ {
        /// Original transform length.
        n: usize,
        /// Padded radix decomposition length.
        m: usize,
    },
}

/// Cartesian axis of a 3D FFT sweep.
#[derive(Debug, Clone, Copy)]
pub enum Axis {
    /// X dimension.
    X,
    /// Y dimension.
    Y,
    /// Z dimension.
    Z,
}

impl Axis {
    /// Return the transform length for this axis.
    pub fn len(self, nx: usize, ny: usize, nz: usize) -> usize {
        match self {
            Self::X => nx,
            Self::Y => ny,
            Self::Z => nz,
        }
    }

    /// Return the number of batched transforms induced by this axis.
    pub fn batch_count(self, nx: usize, ny: usize, nz: usize) -> usize {
        match self {
            Self::X => ny * nz,
            Self::Y => nx * nz,
            Self::Z => nx * ny,
        }
    }
}

/// Typed provider buffers and recurrence values for one Bluestein axis.
pub(crate) struct ChirpData<T> {
    pub(crate) real_kernel: WgpuBuffer<T>,
    pub(crate) imaginary_kernel: WgpuBuffer<T>,
    pub(crate) params: ChirpParams,
    pub(crate) forward_radix: RadixStages,
    pub(crate) inverse_radix: RadixStages,
}
