//! Mixed-radix 8 x 128 experiment — `ATLAS-APOLLO-BASE-BUTTERFLY-128`.
//!
//! The RustFFT-class construction for N = 1024: gather the eight stride-8
//! subsequences into contiguous scratch rows, run eight inner 128-point
//! transforms, then one twiddled column pass of lane-wise 8-point FFTs whose
//! stores land in natural output order. Two-and-a-half passes over the data
//! where the batched four-step pays six.
//!
//! The construction is measurement-gated: it beats the batched route only if
//! the inner 128-point transform runs at RustFFT's class (~600 TSC). The
//! pinned probe beside this module measures that gate first.

pub(crate) mod butterfly;

#[cfg(test)]
mod tests;
// Windows-gated: pins threads through Win32 to control the hybrid scheduler.
#[cfg(all(test, windows))]
mod pinned_probe;
