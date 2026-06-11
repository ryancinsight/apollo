//! API wrapper functions for Apollo FFT.

pub mod cfft;
pub mod icfft;
pub mod irfft;
pub mod rfft;
pub mod utils;

pub use rfft::{
    fft_1d_array, fft_1d_array_into, fft_1d_array_static_into, fft_1d_array_static_typed_into,
    fft_1d_array_typed, fft_1d_array_typed_into, fft_1d_leto, fft_1d_leto_typed,
    fft_1d_slice_typed, fft_2d_array, fft_2d_array_into, fft_2d_array_static_into,
    fft_2d_array_static_typed_into, fft_2d_array_typed, fft_2d_array_typed_into, fft_2d_leto,
    fft_2d_leto_typed, fft_3d_array, fft_3d_array_into, fft_3d_array_static_into,
    fft_3d_array_static_typed_into, fft_3d_array_typed, fft_3d_array_typed_into, fft_3d_leto,
    fft_3d_leto_typed,
};

pub use irfft::{
    ifft_1d_array, ifft_1d_array_into, ifft_1d_array_into_spectrum_scratch,
    ifft_1d_array_static_into, ifft_1d_array_static_typed_into, ifft_1d_array_typed,
    ifft_1d_array_typed_into, ifft_1d_array_typed_into_spectrum_scratch, ifft_1d_leto,
    ifft_1d_leto_typed, ifft_1d_slice_typed, ifft_2d_array, ifft_2d_array_into,
    ifft_2d_array_into_spectrum_scratch, ifft_2d_array_static_into,
    ifft_2d_array_static_typed_into, ifft_2d_array_typed, ifft_2d_array_typed_into,
    ifft_2d_array_typed_into_spectrum_scratch, ifft_2d_leto, ifft_2d_leto_typed, ifft_3d_array,
    ifft_3d_array_into, ifft_3d_array_into_scratch, ifft_3d_array_static_into,
    ifft_3d_array_static_typed_into, ifft_3d_array_typed, ifft_3d_array_typed_into,
    ifft_3d_array_typed_into_spectrum_scratch, ifft_3d_leto, ifft_3d_leto_typed,
};

pub use cfft::{
    fft_1d_complex, fft_1d_complex_inplace, fft_1d_complex_into, fft_1d_complex_owned,
    fft_1d_complex_static, fft_1d_complex_static_inplace, fft_1d_complex_static_into,
    fft_1d_complex_static_typed, fft_1d_complex_static_typed_inplace,
    fft_1d_complex_static_typed_into, fft_1d_complex_typed, fft_1d_complex_typed_inplace,
    fft_1d_complex_typed_into, fft_1d_complex_typed_owned, fft_2d_complex, fft_2d_complex_inplace,
    fft_2d_complex_into, fft_2d_complex_owned, fft_2d_complex_static,
    fft_2d_complex_static_inplace, fft_2d_complex_static_into, fft_2d_complex_static_typed,
    fft_2d_complex_static_typed_inplace, fft_2d_complex_static_typed_into, fft_2d_complex_typed,
    fft_2d_complex_typed_inplace, fft_2d_complex_typed_into, fft_2d_complex_typed_owned,
    fft_3d_complex, fft_3d_complex_inplace, fft_3d_complex_into, fft_3d_complex_owned,
    fft_3d_complex_static, fft_3d_complex_static_inplace, fft_3d_complex_static_into,
    fft_3d_complex_static_typed, fft_3d_complex_static_typed_inplace,
    fft_3d_complex_static_typed_into, fft_3d_complex_typed, fft_3d_complex_typed_inplace,
    fft_3d_complex_typed_into, fft_3d_complex_typed_owned,
};

pub use icfft::{
    ifft_1d_complex, ifft_1d_complex_inplace, ifft_1d_complex_into, ifft_1d_complex_owned,
    ifft_1d_complex_static, ifft_1d_complex_static_inplace, ifft_1d_complex_static_into,
    ifft_1d_complex_static_typed, ifft_1d_complex_static_typed_inplace,
    ifft_1d_complex_static_typed_into, ifft_1d_complex_typed, ifft_1d_complex_typed_inplace,
    ifft_1d_complex_typed_into, ifft_1d_complex_typed_owned, ifft_2d_complex,
    ifft_2d_complex_inplace, ifft_2d_complex_into, ifft_2d_complex_owned, ifft_2d_complex_static,
    ifft_2d_complex_static_inplace, ifft_2d_complex_static_into, ifft_2d_complex_static_typed,
    ifft_2d_complex_static_typed_inplace, ifft_2d_complex_static_typed_into, ifft_2d_complex_typed,
    ifft_2d_complex_typed_inplace, ifft_2d_complex_typed_into, ifft_2d_complex_typed_owned,
    ifft_3d_complex, ifft_3d_complex_inplace, ifft_3d_complex_into, ifft_3d_complex_owned,
    ifft_3d_complex_static, ifft_3d_complex_static_inplace, ifft_3d_complex_static_into,
    ifft_3d_complex_static_typed, ifft_3d_complex_static_typed_inplace,
    ifft_3d_complex_static_typed_into, ifft_3d_complex_typed, ifft_3d_complex_typed_inplace,
    ifft_3d_complex_typed_into, ifft_3d_complex_typed_owned,
};

pub use utils::{fftfreq, fftshift, fftshift_inplace, ifftshift, ifftshift_inplace, rfftfreq};
