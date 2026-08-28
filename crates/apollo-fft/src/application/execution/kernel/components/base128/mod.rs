//! Mixed-radix 8 x 16 base — `ATLAS-APOLLO-BASE-BUTTERFLY-128`.
//!
//! The RustFFT-class construction for N = 1024: gather the eight stride-8
//! subsequences into contiguous scratch rows, run eight inner 128-point
//! transforms, then one twiddled column pass of lane-wise 8-point FFTs whose
//! stores land in natural output order. Two-and-a-half passes over the data
//! where the batched four-step pays six.
//!
//! The current register map requests exactly four scalar lanes: f64 selects
//! AVX2 even on AVX-512 hosts, while f32 selects NEON or Hermes' portable
//! packed backend. A host without that width declines without mutation.
//! The distribution-free median interval clears the production N = 128 route
//! on both measured core types. [`crate::FftPlan1D`] owns the immutable forward
//! plan and initializes inverse state on first use; plan clones share both
//! directions. The pinned probe times the zero-instrumentation specialization;
//! phase attribution runs as a separate const-specialized pass.

pub(crate) mod butterfly;

#[cfg(test)]
mod tests;
// Windows-gated: pins threads through Win32 to control the hybrid scheduler.
#[cfg(all(test, windows))]
mod pinned_probe;

/// Powers of two that decompose down to the 128-point base: 128 itself, and
/// 256 and 512 by repeated radix-2 decimation.
pub(crate) const BASE_SPLIT_LENGTHS: [usize; 3] = [128, 256, 512];

/// Transforms `data` by decimating down to the 128-point base.
///
/// The four-step route pays six passes over the array regardless of size,
/// which at these lengths costs more than the transform: n = 256 measured
/// 2.96x the cost of n = 128 where the arithmetic asks for about 2.3x. One
/// radix-2 decimation per level instead leaves halves that reach the base
/// kernel directly, and the combine reads `W_N^j` from the final stage of
/// the twiddle table the Stockham route already caches
/// (gap_audit.md#small-size-splitting).
///
/// Reports whether the dispatched width ran it, matching
/// [`butterfly::transform_128`].
pub(crate) fn transform_via_base_128<F, const INVERSE: bool>(
    data: &mut [F::Complex],
    plan: &butterfly::Plan128<F>,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
            Complex = eunomia::Complex<F>,
        >,
    eunomia::Complex<F>: bytemuck::Pod,
{
    let n = data.len();
    debug_assert!(BASE_SPLIT_LENGTHS.contains(&n));
    if n == 128 {
        return butterfly::transform_128::<F, INVERSE>(data, plan);
    }

    let half = n / 2;
    let twiddles = if INVERSE {
        F::cached_twiddle_inv(n)
    } else {
        F::cached_twiddle_fwd(n)
    };
    // The stage-major table ends with the length-`n` stage, whose `n / 2`
    // entries are `W_N^j` in order; earlier stages occupy `n / 2 - 1` slots.
    let combine = &twiddles[half - 1..n - 1];

    <F as crate::application::execution::kernel::mixed_radix::MixedRadixScalar>::with_scratch(
        n,
        |scratch| {
            for (j, pair) in data.chunks_exact(2).enumerate() {
                scratch[j] = pair[0];
                scratch[half + j] = pair[1];
            }
            let (even, odd) = scratch.split_at_mut(half);
            if !transform_via_base_128::<F, INVERSE>(even, plan)
                || !transform_via_base_128::<F, INVERSE>(odd, plan)
            {
                return false;
            }
            let (low, high) = data.split_at_mut(half);
            // The combining loop stays scalar: a hand-vectorized sibling
            // measured 728.9 ns against 725.2 at n = 256, so the compiler is
            // already doing what it would have done.
            for j in 0..half {
                let rotated = odd[j] * combine[j];
                low[j] = even[j] + rotated;
                high[j] = even[j] - rotated;
            }
            true
        },
    )
}
