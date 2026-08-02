//! Apollo FFT kernel module.
//!
//! ## Kernel implementations
//!
//! | Module            | Role |
//! |-------------------|------|
//! | `direct`          | O(N²) reference DFT; used only for testing. |
//! | `radix2`          | Twiddle-table builders used by Stockham, Rader, and tests. |
//! | `winograd`        | Short-DFT codelets (DFT-3/5/7/8/N) used by the composite kernel. |
//! | `radix_composite` | Mixed-radix Stockham autosort FFT for 2/3/5/7-smooth composite lengths. |
//! | `stockham`        | Radix-2 Stockham autosort FFT for all power-of-two lengths. |
//! | `mixed_radix`     | Dispatch facade: Stockham for PoT, composite/PFA for smooth, Rader for primes. |

// ── Module declarations ──────────────────────────────────────────────────────

pub(crate) mod auto_dispatch;
pub(crate) mod components;
pub mod direct;
pub mod mixed_radix;
pub(crate) mod pot;
pub(crate) mod precision_bridge;
pub(crate) mod radix_shape;
pub(crate) mod radix_stage;
pub mod real_fft;
pub(crate) mod tuning;
pub(crate) mod twiddle_table;

#[cfg(any(test, debug_assertions, feature = "kernel-strategy-bench"))]
#[doc(hidden)]
pub mod benchmark_kernels;

#[cfg(test)]
pub(crate) mod test_utils;

// ── Re-exports (public API) ──────────────────────────────────────────────────

pub use auto_dispatch::{fft_forward, fft_inverse, fft_inverse_unnorm, FftPrecision};
pub use direct::{dft_forward, dft_inverse, KernelScalar};
