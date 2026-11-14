//! FFT utility functions: frequency bin computation and spectrum shift.
//!
//! These utilities correspond to `numpy.fft.fftfreq`, `numpy.fft.rfftfreq`,
//! `numpy.fft.fftshift`, and `numpy.fft.ifftshift`.

/// Frequency bin utilities: `fftfreq` and `rfftfreq`.
pub mod freq;
/// Spectrum shift utilities: `fftshift` and `ifftshift`.
pub mod shift;
/// Axis-set spectrum shifts over Leto arrays: `fftshift_axes` and `ifftshift_axes`.
pub mod shift_axes;
