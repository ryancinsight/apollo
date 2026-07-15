//! Dense-FFT execution strategies and provider-neutral stage metadata.

use hephaestus_wgpu::WgpuBuffer;

use super::kernel::{ChirpParams, FftParams};

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

/// Value parameters for one typed radix execution plan.
///
/// Hephaestus owns the per-dispatch uniform allocation and binding.  Apollo
/// retains only the transform-stage values required by the FFT recurrence.
pub(crate) struct RadixStages {
    pub(crate) bit_reverse: FftParams,
    pub(crate) butterflies: Box<[FftParams]>,
    pub(crate) inverse_scale: Option<FftParams>,
    pub(crate) fft_len: u32,
    pub(crate) batch_count: u32,
    pub(crate) radix_four: bool,
}

impl RadixStages {
    pub(crate) fn empty() -> Self {
        Self {
            bit_reverse: FftParams {
                n: 0,
                stage: 0,
                inverse: 0,
                batch_count: 0,
            },
            butterflies: Box::default(),
            inverse_scale: None,
            fft_len: 0,
            batch_count: 0,
            radix_four: false,
        }
    }

    pub(crate) fn radix_two(fft_len: u32, batch_count: u32, inverse: bool) -> Self {
        let inverse_flag = u32::from(inverse);
        let butterflies = (0..fft_len.trailing_zeros())
            .map(|stage| FftParams {
                n: fft_len,
                stage,
                inverse: inverse_flag,
                batch_count,
            })
            .collect();
        Self {
            bit_reverse: FftParams {
                n: fft_len,
                stage: 0,
                inverse: inverse_flag,
                batch_count,
            },
            butterflies,
            inverse_scale: inverse.then_some(FftParams {
                n: fft_len,
                stage: 0,
                inverse: 1,
                batch_count,
            }),
            fft_len,
            batch_count,
            radix_four: false,
        }
    }

    pub(crate) fn radix_four(fft_len: u32, batch_count: u32, inverse: bool) -> Self {
        let inverse_flag = u32::from(inverse);
        let butterflies = (0..(fft_len.trailing_zeros() / 2))
            .map(|stage| FftParams {
                n: fft_len,
                stage,
                inverse: inverse_flag,
                batch_count,
            })
            .collect();
        Self {
            bit_reverse: FftParams {
                n: fft_len,
                stage: 0,
                inverse: inverse_flag,
                batch_count,
            },
            butterflies,
            inverse_scale: inverse.then_some(FftParams {
                n: fft_len,
                stage: 0,
                inverse: 1,
                batch_count,
            }),
            fft_len,
            batch_count,
            radix_four: true,
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
