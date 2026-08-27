//! Mixed-radix 8 x 128 experiment — `ATLAS-APOLLO-BASE-BUTTERFLY-128`.
//!
//! The RustFFT-class construction for N = 1024: gather the eight stride-8
//! subsequences into contiguous scratch rows, run eight inner 128-point
//! transforms, then one twiddled column pass of lane-wise 8-point FFTs whose
//! stores land in natural output order. Two-and-a-half passes over the data
//! where the batched four-step pays six.
//!
//! The current register map requires four native scalar lanes: f64 on AVX2 and
//! f32 on NEON. A different widest native width declines without mutation.
//! The distribution-free median interval clears the production N = 128 route
//! on both measured core types. The construction remains test-only until an
//! exact-width capability is available on every routed host and the immutable
//! plan moves into production plan ownership. The pinned probe times the
//! zero-instrumentation specialization; phase attribution runs as a separate
//! const-specialized pass.

pub(crate) mod butterfly;

#[cfg(test)]
mod tests;
// Windows-gated: pins threads through Win32 to control the hybrid scheduler.
#[cfg(all(test, windows))]
mod pinned_probe;
