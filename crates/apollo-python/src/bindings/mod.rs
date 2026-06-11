//! PyO3 binding modules grouped by transform family.
//!
//! Each leaf module wraps one apollo crate (FFT, NUFFT, DHT, FWHT, DCT/DST)
//! or one binding concern (backend introspection). Shared NumPy conversion
//! and precision-profile helpers live in [`support`].

pub(crate) mod backend;
pub(crate) mod dctdst;
pub(crate) mod dht;
pub(crate) mod fft;
pub(crate) mod fft_complex;
pub(crate) mod fwht;
pub(crate) mod nufft;
pub(crate) mod plans;
pub(crate) mod support;
