//! Twiddle-table construction and real-FFT half-complex split routines.
//!
//! ## Current role
//!
//! This module no longer contains a DIT execution kernel. The radix-2 iterative
//! DIT butterfly engine was retired in favour of the Stockham autosort kernel
//! (`mixed_radix.rs`) which requires no bit-reversal permutation pass and
//! delivers better throughput via cache-friendly ping-pong buffering.
//!
//! The functions remaining here are twiddle-table builders
//! (`build_forward_twiddle_table_{32,64}`, `build_inverse_twiddle_table_{32,64}`).
//! They construct contiguous per-stage twiddle tables used by the Stockham
//! kernel and by the 2-D / 3-D plan axes. All four delegate to the SSOT in
//! `twiddle_table.rs`.
//!
//! ## Twiddle-table mathematical contract
//!
//! Theorem (Unified Twiddle Table): A single (N-1)-entry contiguous table
//! with per-stage layout suffices for all log2(N) Stockham stages.
//!
//! Layout invariant: for stage s with sub-transform length L = 2^s,
//! table[base..base+L/2] holds W_L^j = exp(-2*pi*i*j/L) for j = 0..L/2-1,
//! where base = L/2 - 1 (sum of all shorter stage lengths). This lets
//! the Stockham kernel read twiddles sequentially with no stride. QED.
//!
//! ## Failure modes
//!
//! - Empty slice: returns immediately (N=0).
//! - N=1: returns immediately (trivial transform).
//! - N not a power of 2: triggers `debug_assert!` in debug builds.

/// Kernel-level twiddle-table trait consumed by the active twiddle caches.
pub(crate) trait RealFft:
    crate::application::execution::kernel::mixed_radix::MixedRadixScalar
{
    fn build_forward_twiddle_table(n: usize) -> Vec<Self::Complex>;
    fn build_inverse_twiddle_table(n: usize) -> Vec<Self::Complex>;
}

impl RealFft for f64 {
    #[inline]
    fn build_forward_twiddle_table(n: usize) -> Vec<eunomia::Complex64> {
        super::twiddle_table::build_twiddle_table(n, -1.0)
    }

    #[inline]
    fn build_inverse_twiddle_table(n: usize) -> Vec<eunomia::Complex64> {
        super::twiddle_table::build_twiddle_table(n, 1.0)
    }
}

impl RealFft for f32 {
    #[inline]
    fn build_forward_twiddle_table(n: usize) -> Vec<eunomia::Complex32> {
        super::twiddle_table::build_twiddle_table(n, -1.0)
    }

    #[inline]
    fn build_inverse_twiddle_table(n: usize) -> Vec<eunomia::Complex32> {
        super::twiddle_table::build_twiddle_table(n, 1.0)
    }
}
