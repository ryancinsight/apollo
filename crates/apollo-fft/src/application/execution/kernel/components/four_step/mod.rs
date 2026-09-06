//! Cache-optimal Four-Step FFT for large power-of-two transforms.
//!
//! Implements Bailey's 4-step algorithm: N = N1 × N2 decomposes the transform
//! into N1 transforms of length N2 and N2 transforms of length N1, interleaved
//! by a twiddle-multiply step using a cached W_N^{j·k} matrix.
//!
//! ## Twiddle caching
//!
//! The W_N^{j·k} matrix (N entries) is evaluated directly once per length and
//! direction, then reused across transforms. Direct evaluation avoids the
//! O(sqrt(N) * u) error growth of the superseded recurrence without putting
//! trigonometric work in the execution path.
//!
//! ## Parallelism
//!
//! Steps 2 and 4 (N1 independent row-FFTs of length N2 and N2 independent
//! row-FFTs of length N1 respectively) are embarrassingly parallel and are
//! executed via Moirai above a configurable threshold.

mod execution;
mod selection;
mod transpose;
mod workspace;

pub(crate) use execution::four_step_fft;
pub(crate) use selection::try_four_step;
pub(crate) use workspace::{scratch_len, PARALLEL_ROW_THRESHOLD};

#[cfg(test)]
mod tests;
