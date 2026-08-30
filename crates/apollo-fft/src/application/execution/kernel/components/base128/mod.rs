//! Mixed-radix 8 x 16 base — `ATLAS-APOLLO-BASE-BUTTERFLY-128`.
//!
//! The RustFFT-class construction for N = 1024: gather the eight stride-8
//! subsequences into contiguous scratch rows, run eight inner 128-point
//! transforms, then one twiddled column pass of lane-wise 8-point FFTs whose
//! stores land in natural output order. Two-and-a-half passes over the data
//! where the batched four-step pays six.
//!
//! The register map selects a native width once: f64 retains the four-lane
//! AVX2 layout, f32 uses the eight-lane AVX2 layout or the four-lane NEON
//! layout, and a host without either native capability declines without
//! mutation. Hermes' scalar fallback is not reported as a base capability.
//! The distribution-free median interval clears the production N = 128 route
//! on both measured core types. [`crate::FftPlan1D`] owns the immutable forward
//! plan and initializes inverse state on first use; plan clones share both
//! directions. The pinned probe times the zero-instrumentation specialization;
//! phase attribution runs as a separate const-specialized pass.

pub(crate) mod cmul;

/// The base kernels: registers hold two FFT instances rather than two
/// samples, which turns every row twiddle into a broadcast and cuts the row
/// multiplies from 64 to 16 (gap_audit.md#across-instance-outlining). One
/// generic construction serves both the 128-point (eight rows) and 64-point
/// (four rows) routes; the sample-major sibling it replaced is deleted
/// (gap_audit.md#instance-major-64).
pub(crate) mod instance_major;

#[cfg(test)]
mod tests;
// x86-64 Windows-gated: pins threads and reads TSC phase counters.
#[cfg(all(test, windows, target_arch = "x86_64"))]
mod pinned_probe;

/// Powers of two that decompose down to the 128-point base: 128 itself, and
/// 256 and 512 by repeated radix-2 decimation.
pub(crate) const BASE_SPLIT_LENGTHS: [usize; 3] = [128, 256, 512];

/// Length of the base transform every split bottoms out in.
const BASE: usize = 128;

/// Transforms `data` by decimating down to the 128-point base.
///
/// The four-step route pays six passes over the array regardless of size,
/// which at these lengths costs more than the transform: n = 256 measured
/// 2.96x the cost of n = 128 where the arithmetic asks for about 2.3x.
/// Radix-2 decimation instead leaves subsequences that reach the base kernel
/// directly, and the combines read `W_N^j` from the final stage of the
/// twiddle tables the Stockham route already caches
/// (gap_audit.md#small-size-splitting).
///
/// The decimation is flat rather than recursive. Halving at every level
/// gathers at every level: n = 512 paid three gathers and two nested scratch
/// acquisitions where one gather suffices, because `2^d` subsequences at
/// stride `2^d` are exactly what `d` levels of halving produce. Subsequence
/// `b` starts at offset `rev(b)` over `d` bits, bit reversal being what
/// repeated even/odd splitting does to the block index
/// (gap_audit.md#flat-base-split).
///
/// Reports whether the dispatched width ran it, matching
/// [`instance_major::transform_128`].
pub(crate) fn transform_via_base_128<F, const INVERSE: bool>(
    data: &mut [F::Complex],
    plan: &instance_major::Plan128<F>,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: bytemuck::Pod,
{
    let n = data.len();
    debug_assert!(BASE_SPLIT_LENGTHS.contains(&n));
    if n == BASE {
        return instance_major::transform_128::<F, INVERSE>(data, plan);
    }

    let blocks = n / BASE;
    let bits = blocks.trailing_zeros();
    <F as crate::application::execution::kernel::mixed_radix::MixedRadixScalar>::with_scratch(
        n,
        |scratch| {
            // One gather covering every level.
            for (b, block) in scratch.chunks_exact_mut(BASE).enumerate().take(blocks) {
                let offset = b.reverse_bits() >> (usize::BITS - bits);
                for (j, slot) in block.iter_mut().enumerate() {
                    *slot = data[j * blocks + offset];
                }
            }
            for block in scratch.chunks_exact_mut(BASE).take(blocks) {
                if !instance_major::transform_128::<F, INVERSE>(block, plan) {
                    return false;
                }
            }

            // Combining stages, each doubling the transform length. All but
            // the last run in place in scratch; the last writes `data`, so no
            // pass exists only to copy the result back.
            let mut len = BASE;
            while len * 2 < n {
                combine_stage::<F, INVERSE>(scratch, len);
                len *= 2;
            }
            combine_final::<F, INVERSE>(data, scratch, len);
            true
        },
    )
}

/// `W_(2 * len)^j` for `j < len`, from the final stage of the cached
/// stage-major table: that stage holds `len` entries in order, and the earlier
/// stages occupy the `len - 1` slots before it.
fn combine_twiddles<F, const INVERSE: bool>(len: usize) -> std::sync::Arc<[F::Complex]>
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
{
    if INVERSE {
        F::cached_twiddle_inv(2 * len)
    } else {
        F::cached_twiddle_fwd(2 * len)
    }
}

/// One in-place combining stage, pairing adjacent length-`len` transforms into
/// length-`2 * len` ones.
fn combine_stage<F, const INVERSE: bool>(data: &mut [F::Complex], len: usize)
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
{
    let twiddles = combine_twiddles::<F, INVERSE>(len);
    let combine = &twiddles[len - 1..2 * len - 1];
    for pair in data.chunks_exact_mut(2 * len) {
        let (low, high) = pair.split_at_mut(len);
        for j in 0..len {
            let rotated = high[j] * combine[j];
            let even = low[j];
            low[j] = even + rotated;
            high[j] = even - rotated;
        }
    }
}

/// The last combining stage, reading `scratch` and writing `out`.
///
/// The combining loop stays scalar: a hand-vectorized sibling measured
/// 728.9 ns against 725.2 at n = 256, so the compiler is already doing what it
/// would have done.
fn combine_final<F, const INVERSE: bool>(out: &mut [F::Complex], scratch: &[F::Complex], len: usize)
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
{
    let twiddles = combine_twiddles::<F, INVERSE>(len);
    let combine = &twiddles[len - 1..2 * len - 1];
    let (even, odd) = scratch.split_at(len);
    let (low, high) = out.split_at_mut(len);
    for j in 0..len {
        let rotated = odd[j] * combine[j];
        low[j] = even[j] + rotated;
        high[j] = even[j] - rotated;
    }
}
