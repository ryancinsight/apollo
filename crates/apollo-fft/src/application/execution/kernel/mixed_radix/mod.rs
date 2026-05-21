//! Mixed-radix strategy facade.
//!
//! Routes in-place FFTs to the best available kernel for the given length:
//!
//! ## Dispatch hierarchy (f64 / f32)
//!
//! | Input length       | Kernel selected |
//! |--------------------|-----------------|
//! | Power of two < 64  | **Winograd/Good-Thomas codelets** — monomorphized stack-resident DFTs while they remain faster than the generic power route. |
//! | Power of two >= 64 | **Stockham autosort / four-step** — out-of-place ping-pong FFT using a thread-local scratch buffer, with square four-step retained for large even-exponent powers. |
//! | Short non-power composite | **Winograd/Good-Thomas codelets** — monomorphized stack-resident DFTs. |
//! | 2/3/5/7-smooth     | **Composite mixed-radix** — Cooley-Tukey DIT with digit-reversal. |
//! | Coprime composite  | **Good-Thomas PFA** — CRT remapping without inter-stage twiddles. |
//! | Prime              | **Rader convolution** — N-1 cyclic convolution with cached spectrum/permutation. |
//!
//! ## Dispatch hierarchy (f16)
//!
//! `Complex<f16>` is storage-only. All PoT sizes promote to f32, run through
//! the Stockham f32 kernel, and demote back to f16 via `run_via_complex32`.
//!
//! ## Monomorphization
//!
//! All f64/f32 dispatch is driven by a single generic body in `dispatch.rs`
//! parameterized by `scalar::MixedRadixScalar`. The compiler emits fully
//! inlined, optimal machine code per precision through monomorphization.
//! `const INVERSE` and `const NORMALIZE` booleans eliminate dead branches at
//! compile time.
//!
//! ## SSOT principle
//!
//! One dispatch body serves all precision × direction × normalization
//! combinations. No algorithm body is duplicated across precision variants.

#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::uninit_vec)]

pub(crate) mod caches;
pub(crate) mod dispatch;
pub(crate) mod scalar;
#[cfg(test)]
mod tests;
pub(crate) mod traits;

pub(crate) use dispatch::*;
pub(crate) use scalar::MixedRadixScalar;
