//! 2D FFT plan.
//!
//! Apollo-owned 2D FFT implementation.
//!
//! The 2D DFT is separable, so this plan applies the in-repo auto-selected 1D
//! FFT kernel across rows and columns. The inverse path is normalized on each
//! inverse axis pass, which gives the standard `1 / (nx * ny)` inverse
//! normalization.
//!
//! # Mathematical contract
//!
//! For a complex input field `x in C^(nx x ny)`, the forward transform is
//!
//! `X[k,l] = sum_i sum_j x[i,j] exp(-2*pi*i*(k*i/nx + l*j/ny))`.
//!
//! The inverse transform is
//!
//! `x[i,j] = (1/(nx*ny)) sum_k sum_l X[k,l] exp(2*pi*i*(k*i/nx + l*j/ny))`.
//!
//! The implementation is linear and separable. Floating-point error follows
//! from the selected scalar precision and the selected 1D FFT kernel.
//!
//! # Complexity
//!
//! Let `C(n)` be the selected 1D FFT cost. The plan costs
//! `O(ny * C(nx) + nx * C(ny))`, with `C(n) = O(n log n)` for radix-2,
//! mixed-radix, and Rader plan paths. Contiguous innermost-axis passes transform
//! lanes in place. Other axes transpose into reusable full-volume scratch and
//! transpose back after lane execution.

mod dynamic_impl;
mod static_impl;

pub use dynamic_impl::FftPlan2D;
pub use static_impl::StaticFftPlan2D;

#[cfg(test)]
mod tests;
