//! Mixed-radix strategy facade.
//!
//! Routes in-place FFTs to the best available kernel for the given length:
//!
//! ## Dispatch hierarchy (f64 / f32)
//!
//! | Input length       | Kernel selected |
//! |--------------------|-----------------|
//! | Power of two >= 2  | **Stockham autosort** — out-of-place ping-pong FFT using a thread-local scratch buffer; no bit-reversal. AVX/FMA SIMD codelets for N=4,8,64,4096 (f32); N=64 (f64). Falls back to generic `transform<P>` for other PoT sizes. |
//! | 2/3/5/7-smooth     | **Composite mixed-radix** — Cooley-Tukey DIT with digit-reversal. |
//! | Other non-PoT      | **Bluestein chirp-Z** — pads to next PoT and runs Stockham internally. |
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

pub(crate) use caches::*;
pub(crate) use dispatch::*;
pub(crate) use scalar::MixedRadixScalar;
