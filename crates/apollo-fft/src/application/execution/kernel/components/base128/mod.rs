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
pub(crate) mod split_boundary;

/// The base kernels: registers hold two FFT instances rather than two
/// samples, which turns every row twiddle into a broadcast and cuts the row
/// multiplies from 64 to 16 (gap_audit.md#across-instance-outlining). One
/// generic construction serves both the 128-point (eight rows) and 64-point
/// (four rows) routes; the sample-major sibling it replaced is deleted
/// (gap_audit.md#instance-major-64).
pub(crate) mod instance_major;

#[cfg(test)]
mod tests;
// x86-64 Windows-gated: uses Hermes exact processor binding and reads TSC phase counters.
#[cfg(all(test, windows, target_arch = "x86_64"))]
mod pinned_probe;

/// Powers of two that decompose down to the 128-point base: 128 itself, and
/// 256, 512 and 1024 by repeated radix-2 decimation.
pub(crate) const BASE_SPLIT_LENGTHS: [usize; 4] = [128, 256, 512, 1024];

/// Length of the base transform every split bottoms out in.
const BASE: usize = 128;

fn base_lanes<T>(data: &[eunomia::Complex<T>]) -> &[T; 2 * BASE]
where
    T: eunomia::layout::Pod,
    eunomia::Complex<T>: eunomia::layout::Pod,
{
    eunomia::layout::cast_slice(data)
        .try_into()
        .expect("invariant: one base block is exactly 256 scalar lanes")
}

fn base_lanes_mut<T>(data: &mut [eunomia::Complex<T>]) -> &mut [T; 2 * BASE]
where
    T: eunomia::layout::Pod,
    eunomia::Complex<T>: eunomia::layout::Pod,
{
    eunomia::layout::cast_slice_mut(data)
        .try_into()
        .expect("invariant: one base block is exactly 256 scalar lanes")
}

/// Transforms `data` by decimating down to the 128-point base.
///
/// The four-step route pays six passes over the array regardless of size,
/// which at these lengths costs more than the transform: n = 256 measured
/// 2.96x the cost of n = 128 where the arithmetic asks for about 2.3x.
/// Radix-2 decimation instead leaves subsequences that reach the base kernel
/// directly. The plan passes its complete, immutable stage-major table by
/// borrow, so the combines select `W_N^j` without another cache lookup or
/// temporary shared-owner acquisition (gap_audit.md#base-split-twiddle-reuse).
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
    twiddles: &[F::Complex],
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let n = data.len();
    debug_assert!(BASE_SPLIT_LENGTHS.contains(&n));
    if n == BASE {
        return instance_major::transform_128::<F, INVERSE>(data, plan);
    }
    debug_assert_eq!(twiddles.len(), n - 1);

    let blocks = n / BASE;
    let bits = blocks.trailing_zeros();
    <F as crate::application::execution::kernel::mixed_radix::MixedRadixScalar>::with_scratch(
        n,
        |scratch| {
            // One gather covering every level — the pair-deinterleave
            // network at the plan's native width, the scalar strided read
            // otherwise (gap_audit.md#split-boundary). Dispatching at the
            // base kernel's width keeps a four-byte scalar out of the
            // scalar-emulated four-lane frame it previously gathered in.
            let gathered = {
                let src = eunomia::layout::cast_slice(&*data);
                let dst = eunomia::layout::cast_slice_mut(&mut scratch[..n]);
                match (blocks, plan.native_eight_lanes()) {
                    (2, false) => {
                        hermes_simd::vectorize_lanes::<4, F, _>(split_boundary::GatherBlocks::<
                            F,
                            2,
                        > {
                            src,
                            dst,
                        })
                    }
                    (2, true) => {
                        hermes_simd::vectorize_lanes::<8, F, _>(split_boundary::GatherBlocks::<
                            F,
                            2,
                        > {
                            src,
                            dst,
                        })
                    }
                    (4, false) => {
                        hermes_simd::vectorize_lanes::<4, F, _>(split_boundary::GatherBlocks::<
                            F,
                            4,
                        > {
                            src,
                            dst,
                        })
                    }
                    (4, true) => {
                        hermes_simd::vectorize_lanes::<8, F, _>(split_boundary::GatherBlocks::<
                            F,
                            4,
                        > {
                            src,
                            dst,
                        })
                    }
                    (_, false) => {
                        hermes_simd::vectorize_lanes::<4, F, _>(split_boundary::GatherBlocks::<
                            F,
                            8,
                        > {
                            src,
                            dst,
                        })
                    }
                    (_, true) => {
                        hermes_simd::vectorize_lanes::<8, F, _>(split_boundary::GatherBlocks::<
                            F,
                            8,
                        > {
                            src,
                            dst,
                        })
                    }
                }
                .unwrap_or(false)
            };
            if !gathered {
                for (b, block) in scratch.chunks_exact_mut(BASE).enumerate().take(blocks) {
                    let offset = b.reverse_bits() >> (usize::BITS - bits);
                    for (j, slot) in block.iter_mut().enumerate() {
                        *slot = data[j * blocks + offset];
                    }
                }
            }
            // Two blocks: the even block transforms into scratch and the
            // odd block combines on the way out of its own column pass,
            // writing both halves of `data` directly — no separate combine
            // pass and no store-then-reload of the odd spectrum
            // (gap_audit.md#combine-sink). A width that does not carry the
            // sink falls back to the two-pass form.
            if blocks == 2 {
                let (even, odd) = scratch.split_at_mut(BASE);
                if !instance_major::transform_128::<F, INVERSE>(even, plan) {
                    return false;
                }
                let combine = &twiddles[BASE - 1..2 * BASE - 1];
                {
                    let (low, high) = data.split_at_mut(BASE);
                    if instance_major::transform_128_combining::<F, INVERSE>(
                        odd,
                        plan,
                        instance_major::CombineSink {
                            peer: base_lanes(even),
                            tw: base_lanes(combine),
                            low: base_lanes_mut(low),
                            high: base_lanes_mut(high),
                        },
                    ) {
                        return true;
                    }
                }
                if !instance_major::transform_128::<F, INVERSE>(odd, plan) {
                    return false;
                }
                combine_final::<F>(data, scratch, twiddles, BASE);
                return true;
            }
            // Eight blocks: two fused four-block chains and one in-place
            // level.
            //
            // Walking the three levels separately loses for the reason the
            // four-step route was rejected at the top of this file — passes.
            // `combine_final4` fuses two levels, so each half reaches its
            // 512-point spectrum in one pass; the last level then reads and
            // writes the same array, since output `j` and `j + 512` depend
            // only on inputs `j` and `j + 512`
            // (`backlog.md#atlas-apollo-eight-block-split`).
            if blocks == 8 {
                let half = n / 2;
                let (scratch_low, scratch_high) = scratch[..n].split_at_mut(half);
                let combined = {
                    let (data_low, data_high) = data.split_at_mut(half);
                    combine_four_blocks::<F, INVERSE>(data_low, scratch_low, plan, twiddles)
                        && combine_four_blocks::<F, INVERSE>(
                            data_high,
                            scratch_high,
                            plan,
                            twiddles,
                        )
                };
                if !combined {
                    return false;
                }
                combine_level_in_place::<F>(data, twiddles, half);
                return true;
            }
            // Four blocks keep only the even pair's two halves as an
            // intermediate; the shared column pass carries the sinks at
            // every native width. Block one writes those halves into the
            // first two output quarters; block three applies its pair
            // butterfly and the outer level as its registers leave the base
            // kernel, replacing the intermediates and filling the last two
            // quarters. The detached scalar final pass disappears.
            combine_four_blocks::<F, INVERSE>(data, &mut scratch[..n], plan, twiddles)
        },
    )
}

#[cfg(all(test, windows, target_arch = "x86_64"))]
fn transform_via_base_128_incumbent<F, const INVERSE: bool>(
    data: &mut [F::Complex],
    plan: &instance_major::Plan128<F>,
    twiddles: &[F::Complex],
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let n = data.len();
    assert_eq!(n, 4 * BASE, "the incumbent probe covers only N=512");
    <F as crate::application::execution::kernel::mixed_radix::MixedRadixScalar>::with_scratch(
        n,
        |scratch| {
            let gathered =
                hermes_simd::vectorize_lanes::<4, F, _>(split_boundary::GatherBlocks::<F, 4> {
                    src: eunomia::layout::cast_slice(&*data),
                    dst: eunomia::layout::cast_slice_mut(&mut scratch[..n]),
                })
                .unwrap_or(false);
            if !gathered {
                for (block_index, block) in scratch.chunks_exact_mut(BASE).enumerate().take(4) {
                    let offset = block_index.reverse_bits() >> (usize::BITS - 2);
                    for (index, slot) in block.iter_mut().enumerate() {
                        *slot = data[4 * index + offset];
                    }
                }
            }
            for block in scratch.chunks_exact_mut(BASE).take(4) {
                if !instance_major::transform_128::<F, INVERSE>(block, plan) {
                    return false;
                }
            }
            combine_final4::<F>(data, scratch, twiddles, BASE);
            true
        },
    )
}

/// Four transformed 128-blocks into one `4 * BASE` spectrum, both combine
/// levels fused into the base kernel's register exit.
///
/// The four-block split is this over the whole array; each half of the
/// eight-block split is this over its half, which is why it is a function
/// rather than an inline block. The twiddle slices are the same either way:
/// a half of the eight-block split is a 512-point sub-problem whose levels
/// have half-lengths `BASE` and `2 * BASE`, exactly the four-block case.
fn combine_four_blocks<F, const INVERSE: bool>(
    data: &mut [F::Complex],
    scratch: &mut [F::Complex],
    plan: &instance_major::Plan128<F>,
    twiddles: &[F::Complex],
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
            Complex = eunomia::Complex<F>,
        > + eunomia::layout::Pod,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let inner = &twiddles[BASE - 1..2 * BASE - 1];
    let outer = &twiddles[2 * BASE - 1..4 * BASE - 1];
    let (outer_low, outer_high) = outer.split_at(BASE);

    let (b01, b23) = scratch.split_at_mut(2 * BASE);
    let (b0, b1) = b01.split_at_mut(BASE);
    let (b2, b3) = b23.split_at_mut(BASE);
    // Establish that both ordinary base calls run before the first output
    // mutation. The plan's one selected native width makes the following two
    // sink calls the same capability decision.
    if !instance_major::transform_128::<F, INVERSE>(b0, plan)
        || !instance_major::transform_128::<F, INVERSE>(b2, plan)
    {
        return false;
    }
    {
        let (even, _) = data.split_at_mut(2 * BASE);
        let (even_low, even_high) = even.split_at_mut(BASE);
        if !instance_major::transform_128_combining::<F, INVERSE>(
            b1,
            plan,
            instance_major::CombineSink {
                peer: base_lanes(b0),
                tw: base_lanes(inner),
                low: base_lanes_mut(even_low),
                high: base_lanes_mut(even_high),
            },
        ) {
            return false;
        }
    }
    let (low, high) = data.split_at_mut(2 * BASE);
    let (even_low, even_high) = low.split_at_mut(BASE);
    let (high_low, high_high) = high.split_at_mut(BASE);
    instance_major::transform_128_combining_final::<F, INVERSE>(
        b3,
        plan,
        instance_major::FinalCombineSink {
            peer: base_lanes(b2),
            inner_tw: base_lanes(inner),
            even_low: base_lanes_mut(even_low),
            even_high: base_lanes_mut(even_high),
            outer_low_tw: base_lanes(outer_low),
            outer_high_tw: base_lanes(outer_high),
            high_low: base_lanes_mut(high_low),
            high_high: base_lanes_mut(high_high),
        },
    )
}

/// The final radix-2 combine level, in place.
///
/// Output `j` and `j + len` depend only on inputs `j` and `j + len`, so the
/// halves are read and written where they already sit — no second buffer, no
/// copy back, one pass. The twiddle table is stage-major: the level whose
/// half-length is `len` reads `len` entries starting at `len - 1`, the same
/// slice `combine_final` takes for its own level.
fn combine_level_in_place<F>(data: &mut [F::Complex], twiddles: &[F::Complex], len: usize)
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
{
    let combine = &twiddles[len - 1..2 * len - 1];
    let (low, high) = data.split_at_mut(len);
    for j in 0..len {
        let rotated = high[j] * combine[j];
        let even = low[j];
        low[j] = even + rotated;
        high[j] = even - rotated;
    }
}

/// The last combining stage, reading `scratch` and writing `out`.
///
/// The combining loop stays scalar: a hand-vectorized sibling measured
/// 728.9 ns against 725.2 at n = 256, so the compiler is already doing what it
/// would have done.
fn combine_final<F>(
    out: &mut [F::Complex],
    scratch: &[F::Complex],
    twiddles: &[F::Complex],
    len: usize,
) where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
{
    let combine = &twiddles[len - 1..2 * len - 1];
    let (even, odd) = scratch.split_at(len);
    let (low, high) = out.split_at_mut(len);
    for j in 0..len {
        let rotated = odd[j] * combine[j];
        low[j] = even[j] + rotated;
        high[j] = even[j] - rotated;
    }
}

/// Both combine levels of a four-block split in one pass.
///
/// The chained form runs `combine_stage` and then `combine_final` — two
/// full reads and writes of the array. Fusing them applies both butterfly
/// levels per index while every operand is in registers: block values
/// `b0..b3` at index `j` produce the four outputs `j`, `j + len`,
/// `j + 2 * len`, `j + 3 * len` directly, and the array is read and
/// written once (gap_audit.md#split-boundary).
///
/// Level one pairs `(b0, b1)` and `(b2, b3)` with `W_{2 * len}`; level two
/// combines those with `W_{4 * len}` at `j` and `j + len` — the block
/// order is the gather's bit-reversed one, which is exactly what makes the
/// adjacent-pair pairing correct.
#[cfg(all(test, windows, target_arch = "x86_64"))]
fn combine_final4<F>(
    out: &mut [F::Complex],
    scratch: &[F::Complex],
    twiddles: &[F::Complex],
    len: usize,
) where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
{
    let inner = &twiddles[len - 1..2 * len - 1];
    let outer = &twiddles[2 * len - 1..4 * len - 1];
    let (b01, b23) = scratch.split_at(2 * len);
    let (b0, b1) = b01.split_at(len);
    let (b2, b3) = b23.split_at(len);
    let (lo, hi) = out.split_at_mut(2 * len);
    let (out0, out1) = lo.split_at_mut(len);
    let (out2, out3) = hi.split_at_mut(len);
    for j in 0..len {
        let r = b1[j] * inner[j];
        let (e_lo, e_hi) = (b0[j] + r, b0[j] - r);
        let r = b3[j] * inner[j];
        let (o_lo, o_hi) = (b2[j] + r, b2[j] - r);
        let r = o_lo * outer[j];
        out0[j] = e_lo + r;
        out2[j] = e_lo - r;
        let r = o_hi * outer[j + len];
        out1[j] = e_hi + r;
        out3[j] = e_hi - r;
    }
}
