//! FFT utility functions: frequency bin computation and spectrum shift.
//!
//! These utilities correspond to `numpy.fft.fftfreq`, `numpy.fft.rfftfreq`,
//! `numpy.fft.fftshift`, and `numpy.fft.ifftshift`.

/// Frequency bin utilities: `fftfreq` and `rfftfreq`.
pub mod freq;
/// Canonical Leto ↔ slice/ndarray interop helpers shared workspace-wide.
pub mod leto_interop;
/// Spectrum shift utilities: `fftshift` and `ifftshift`.
pub mod shift;
