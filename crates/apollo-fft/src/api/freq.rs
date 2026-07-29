//! Frequency-grid constructors for FFT output axes.
//!
//! Provides `fftfreq` and `rfftfreq` to build the discrete-frequency sample
//! vectors corresponding to FFT output arrays, following the NumPy/SciPy
//! convention.

pub use crate::application::utilities::freq::{fftfreq, rfftfreq};
