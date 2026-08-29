//! Public API functions for Apollo FFT.
//!
//! The multidimensional real-FFT Leto boundaries materialize the kernel output
//! directly through `leto::Array::from_mnemosyne_slice`. For a contiguous
//! output `O` with shape `s` and logical sequence `o_i`, the provider contract
//! gives the returned array `M` the same shape and `M[i] = o_i` for every valid
//! index. This is a provider-contract proof sketch; the 2D/3D forward and
//! inverse parity tests in `tests/slice_api.rs` are the executable differential
//! evidence (see ADR 0032).

pub mod cfft;
pub mod freq;
pub mod icfft;
pub mod irfft;
pub mod rfft;
pub mod routing;
pub mod shift;

pub use rfft::{
    fft_1d_array, fft_1d_array_into, fft_1d_array_static_into, fft_1d_leto, fft_1d_slice,
    fft_1d_slice_half, fft_1d_slice_half_into, fft_2d_array, fft_2d_array_into,
    fft_2d_array_static_into, fft_2d_leto, fft_3d_array, fft_3d_array_into,
    fft_3d_array_static_into, fft_3d_leto,
};

pub use irfft::{
    ifft_1d_array, ifft_1d_array_into, ifft_1d_array_into_spectrum_scratch,
    ifft_1d_array_static_into, ifft_1d_leto, ifft_1d_slice, ifft_2d_array, ifft_2d_array_into,
    ifft_2d_array_into_spectrum_scratch, ifft_2d_array_static_into, ifft_2d_leto, ifft_3d_array,
    ifft_3d_array_into, ifft_3d_array_into_spectrum_scratch, ifft_3d_array_static_into,
    ifft_3d_leto,
};

pub use cfft::{
    fft_1d_complex, fft_1d_complex_inplace, fft_1d_complex_into, fft_1d_complex_owned,
    fft_1d_complex_static, fft_1d_complex_static_inplace, fft_1d_complex_static_into,
    fft_2d_complex, fft_2d_complex_inplace, fft_2d_complex_into, fft_2d_complex_owned,
    fft_2d_complex_static, fft_2d_complex_static_inplace, fft_2d_complex_static_into,
    fft_3d_complex, fft_3d_complex_inplace, fft_3d_complex_into, fft_3d_complex_owned,
    fft_3d_complex_static, fft_3d_complex_static_inplace, fft_3d_complex_static_into,
};

pub use icfft::{
    ifft_1d_complex, ifft_1d_complex_inplace, ifft_1d_complex_into, ifft_1d_complex_owned,
    ifft_1d_complex_static, ifft_1d_complex_static_inplace, ifft_1d_complex_static_into,
    ifft_2d_complex, ifft_2d_complex_inplace, ifft_2d_complex_into, ifft_2d_complex_owned,
    ifft_2d_complex_static, ifft_2d_complex_static_inplace, ifft_2d_complex_static_into,
    ifft_3d_complex, ifft_3d_complex_inplace, ifft_3d_complex_into, ifft_3d_complex_owned,
    ifft_3d_complex_static, ifft_3d_complex_static_inplace, ifft_3d_complex_static_into,
};

pub use freq::{fftfreq, rfftfreq};
pub use shift::{fftshift, fftshift_inplace, ifftshift, ifftshift_inplace};

pub use routing::supports_length;
