//! Inverse real FFT API functions.

mod dimension_1d;
mod dimension_2d;
mod dimension_3d;

pub use dimension_1d::{
    ifft_1d_array, ifft_1d_array_into, ifft_1d_array_into_spectrum_scratch,
    ifft_1d_array_static_into, ifft_1d_leto, ifft_1d_slice, ifft_1d_slice_half_into,
};
pub use dimension_2d::{
    ifft_2d_array, ifft_2d_array_half_into, ifft_2d_array_into,
    ifft_2d_array_into_spectrum_scratch, ifft_2d_array_static_into, ifft_2d_leto,
};
pub use dimension_3d::{
    ifft_3d_array, ifft_3d_array_half_into, ifft_3d_array_into,
    ifft_3d_array_into_spectrum_scratch, ifft_3d_array_static_into, ifft_3d_leto,
};
