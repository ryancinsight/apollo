//! Value-semantic GFT GPU verification manifest.
//!
//! ADR 0020 records the orthonormal graph-Fourier theorem whose finite-precision
//! evidence remains in the forward and inverse contract leaves.

#![cfg(test)]

mod forward;
mod inverse;
mod leto;
mod metadata;
mod precision;
mod support;
mod typed;
