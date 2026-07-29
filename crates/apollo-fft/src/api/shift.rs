//! Frequency-axis shift operations (fftshift / ifftshift).
//!
//! Provides in-place and allocating variants to move the zero-frequency
//! component to the centre of an FFT output array, following the NumPy
//! convention.

pub use crate::application::utilities::shift::{
    fftshift, fftshift_inplace, ifftshift, ifftshift_inplace,
};
