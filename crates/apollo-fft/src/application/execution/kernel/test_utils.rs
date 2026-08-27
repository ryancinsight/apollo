//! Shared test utilities for FFT kernel testing.
//!
//! Centralizes common assertion and comparison functions used across
//! radix and Rader kernel tests to reduce DRY violations and
//! maintain consistent error tolerances.

use eunomia::{Complex32, Complex64};

/// Compute the maximum absolute error between two f64 FFT outputs.
///
/// Returns the largest element-wise norm distance: `max_k |output_k - expected_k|`.
/// Used for validating forward/inverse transforms against direct DFT reference.
#[cfg(test)]
#[inline]
pub(crate) fn max_abs_err_64(got: &[Complex64], expected: &[Complex64]) -> f64 {
    got.iter()
        .zip(expected.iter())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0, f64::max)
}

/// Compute the maximum absolute error between two f32 FFT outputs.
///
/// Returns the largest element-wise norm distance: `max_k |output_k - expected_k|`.
/// Used for validating forward/inverse transforms against direct DFT reference.
#[cfg(test)]
#[inline]
pub(crate) fn max_abs_err_32(got: &[Complex32], expected: &[Complex32]) -> f32 {
    got.iter()
        .zip(expected.iter())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0, f32::max)
}

/// Thread pinning for the measurement probes, shared so the Win32 shim exists
/// once. Unpinned bench threads receive EcoQoS from the hybrid scheduler and
/// report route costs that say more about scheduling than about routes.
#[cfg(windows)]
mod pinning {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetThreadAffinityMask(thread: isize, mask: usize) -> usize;
        fn GetCurrentThread() -> isize;
        fn GetCurrentProcessorNumber() -> u32;
    }

    /// Pins the current thread to logical `cpu` and returns where it landed.
    pub(crate) fn pin(cpu: u32) -> u32 {
        // SAFETY: pins the current thread; both calls are documented Win32.
        unsafe {
            SetThreadAffinityMask(GetCurrentThread(), 1usize << cpu);
        }
        std::thread::yield_now();
        // SAFETY: no arguments, no state.
        unsafe { GetCurrentProcessorNumber() }
    }
}
#[cfg(windows)]
pub(crate) use pinning::pin;
