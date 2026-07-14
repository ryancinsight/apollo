//! Reusable discrete wavelet transform plan.

use crate::domain::contracts::error::{WaveletError, WaveletResult};
use crate::domain::metadata::wavelet::DiscreteWavelet;
use apollo_fft::f16;

/// Forward transform implementations.
pub mod forward;
/// Internal conversion and validation helpers.
pub mod helpers;
/// Inverse transform implementations.
pub mod inverse;
/// Typed storage transform implementations.
pub mod typed;

#[cfg(test)]
mod tests;

pub use typed::WaveletStorage;

mod gpu_storage_sealed {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for apollo_fft::f16 {}
}

/// Storage admitted by the concrete `f32` Haar GPU contract.
///
/// The accelerator executes `f32` arithmetic. Native `f32` storage borrows
/// directly; `f16` quantizes only at the caller-owned boundary. `f64` is not
/// admitted, so the GPU path cannot silently narrow a high-accuracy request.
///
/// ```compile_fail
/// use apollo_wavelet::WaveletGpuStorage;
///
/// fn require_gpu_storage<T: WaveletGpuStorage>() {}
///
/// require_gpu_storage::<f64>();
/// ```
pub trait WaveletGpuStorage: WaveletStorage + gpu_storage_sealed::Sealed {
    /// Convert one storage value to the GPU arithmetic representation.
    fn to_gpu(self) -> f32;

    /// Convert one GPU arithmetic result to caller-owned storage.
    fn from_gpu(value: f32) -> Self;

    /// Borrow native GPU values when their storage layout is exact.
    fn as_f32_slice(_values: &[Self]) -> Option<&[f32]> {
        None
    }

    /// Borrow mutable native GPU values when their storage layout is exact.
    fn as_f32_slice_mut(_values: &mut [Self]) -> Option<&mut [f32]> {
        None
    }
}

impl WaveletGpuStorage for f32 {
    fn to_gpu(self) -> f32 {
        self
    }

    fn from_gpu(value: f32) -> Self {
        value
    }

    fn as_f32_slice(values: &[Self]) -> Option<&[f32]> {
        Some(values)
    }

    fn as_f32_slice_mut(values: &mut [Self]) -> Option<&mut [f32]> {
        Some(values)
    }
}

impl WaveletGpuStorage for f16 {
    fn to_gpu(self) -> f32 {
        self.to_f32()
    }

    fn from_gpu(value: f32) -> Self {
        f16::from_f32(value)
    }
}

/// Reusable 1D orthogonal DWT plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DwtPlan {
    len: usize,
    levels: usize,
    wavelet: DiscreteWavelet,
}

/// Multilevel DWT coefficient storage backed by Leto arrays.
pub struct DwtLetoCoefficients<T> {
    len: usize,
    levels: usize,
    approximation: leto::Array<T, leto::MnemosyneStorage<T>, 1>,
    details: Vec<leto::Array<T, leto::MnemosyneStorage<T>, 1>>,
}

impl<T> DwtLetoCoefficients<T> {
    /// Create DWT coefficient storage backed by Leto arrays.
    #[must_use]
    pub fn new(
        len: usize,
        levels: usize,
        approximation: leto::Array<T, leto::MnemosyneStorage<T>, 1>,
        details: Vec<leto::Array<T, leto::MnemosyneStorage<T>, 1>>,
    ) -> Self {
        Self {
            len,
            levels,
            approximation,
            details,
        }
    }

    /// Return original signal length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Return true when original signal length is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return decomposition levels.
    #[must_use]
    pub const fn levels(&self) -> usize {
        self.levels
    }

    /// Return the coarsest approximation coefficients.
    #[must_use]
    pub const fn approximation(&self) -> &leto::Array<T, leto::MnemosyneStorage<T>, 1> {
        &self.approximation
    }

    /// Return detail coefficients from finest to coarsest.
    #[must_use]
    pub fn details(&self) -> &[leto::Array<T, leto::MnemosyneStorage<T>, 1>] {
        &self.details
    }
}

impl DwtPlan {
    /// Create a DWT plan for a power-of-two signal length.
    pub fn new(len: usize, levels: usize, wavelet: DiscreteWavelet) -> WaveletResult<Self> {
        if len == 0 {
            return Err(WaveletError::EmptySignal);
        }
        if !len.is_power_of_two() {
            return Err(WaveletError::NonPowerOfTwoLength);
        }
        if levels == 0 {
            return Err(WaveletError::EmptyLevelCount);
        }
        if levels > len.trailing_zeros() as usize {
            return Err(WaveletError::LevelExceedsLength);
        }
        Ok(Self {
            len,
            levels,
            wavelet,
        })
    }

    /// Return signal length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Return true when signal length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Return decomposition level count.
    #[must_use]
    pub const fn levels(self) -> usize {
        self.levels
    }

    /// Return wavelet family.
    #[must_use]
    pub const fn wavelet(self) -> DiscreteWavelet {
        self.wavelet
    }

    pub(crate) fn coefficient_shapes(&self) -> impl Iterator<Item = usize> {
        let len = self.len;
        let levels = self.levels;
        (0..levels).map(move |level| len >> (level + 1))
    }
}
