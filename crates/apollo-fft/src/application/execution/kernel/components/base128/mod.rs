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

/// Length of the base transform every split bottoms out in.
/// The lanes of one base block of `LANES / 2` samples.
fn base_lanes<T, const LANES: usize>(data: &[eunomia::Complex<T>]) -> &[T; LANES]
where
    T: eunomia::layout::Pod,
    eunomia::Complex<T>: eunomia::layout::Pod,
{
    eunomia::layout::cast_slice(data)
        .try_into()
        .expect("invariant: one base block is exactly LANES scalar lanes")
}

fn base_lanes_mut<T, const LANES: usize>(data: &mut [eunomia::Complex<T>]) -> &mut [T; LANES]
where
    T: eunomia::layout::Pod,
    eunomia::Complex<T>: eunomia::layout::Pod,
{
    eunomia::layout::cast_slice_mut(data)
        .try_into()
        .expect("invariant: one base block is exactly LANES scalar lanes")
}

/// The split route over the 256-point base (ADR 0061): [`transform_via_base`]
/// at 32-sample rows, so 512 and 1024 are two and four blocks under one
/// radix step.
pub(crate) fn transform_via_base_256<F, const INVERSE: bool, const MEASURE: bool>(
    data: &mut [F::Complex],
    plan: &instance_major::Plan256<F>,
    twiddles: &[F::Complex],
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    transform_via_base::<F, INVERSE, MEASURE, 32, 256, 512, { instance_major::table_lanes(8, 32) }>(
        data, plan, twiddles,
    )
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
fn transform_via_base<
    F,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROW_LEN: usize,
    const BASE: usize,
    const BLOCK_LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [F::Complex],
    plan: &instance_major::BasePlan<F, 8, ROW_LEN, TABLE_LANES>,
    twiddles: &[F::Complex],
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let n = data.len();
    debug_assert!(BASE == 8 * ROW_LEN && BLOCK_LANES == 2 * BASE);
    debug_assert!(n % BASE == 0 && (n / BASE).is_power_of_two() && n / BASE <= 4);
    if n == BASE {
        return instance_major::transform_block::<
            F,
            INVERSE,
            MEASURE,
            ROW_LEN,
            BLOCK_LANES,
            TABLE_LANES,
        >(data, plan);
    }
    #[cfg(all(test, windows, target_arch = "x86_64"))]
    let t0 = if MEASURE {
        instance_major::phase_meter::stamp()
    } else {
        0
    };
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
                            BLOCK_LANES,
                        > {
                            src,
                            dst,
                        })
                    }
                    (2, true) => {
                        hermes_simd::vectorize_lanes::<8, F, _>(split_boundary::GatherBlocks::<
                            F,
                            2,
                            BLOCK_LANES,
                        > {
                            src,
                            dst,
                        })
                    }
                    (4, false) => {
                        hermes_simd::vectorize_lanes::<4, F, _>(split_boundary::GatherBlocks::<
                            F,
                            4,
                            BLOCK_LANES,
                        > {
                            src,
                            dst,
                        })
                    }
                    (4, true) => {
                        hermes_simd::vectorize_lanes::<8, F, _>(split_boundary::GatherBlocks::<
                            F,
                            4,
                            BLOCK_LANES,
                        > {
                            src,
                            dst,
                        })
                    }
                    // The assertion above bounds the split at four blocks.
                    _ => Some(false),
                }
                .unwrap_or(false)
            };
            #[cfg(all(test, windows, target_arch = "x86_64"))]
            let t1 = if MEASURE {
                let t = instance_major::phase_meter::stamp();
                instance_major::phase_meter::add_outer(0, t - t0);
                t
            } else {
                0
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
            // (gap_audit.md#combine-sink).
            if blocks == 2 {
                let (even, odd) = scratch.split_at_mut(BASE);
                if !instance_major::transform_block::<
                    F,
                    INVERSE,
                    MEASURE,
                    ROW_LEN,
                    BLOCK_LANES,
                    TABLE_LANES,
                >(even, plan)
                {
                    return false;
                }
                let combine = &twiddles[BASE - 1..2 * BASE - 1];
                {
                    let (low, high) = data.split_at_mut(BASE);
                    if instance_major::transform_block_combining::<
                        F,
                        INVERSE,
                        MEASURE,
                        ROW_LEN,
                        BLOCK_LANES,
                        TABLE_LANES,
                    >(
                        odd,
                        plan,
                        instance_major::CombineSink {
                            peer: base_lanes::<F, BLOCK_LANES>(even),
                            tw: base_lanes::<F, BLOCK_LANES>(combine),
                            low: base_lanes_mut::<F, BLOCK_LANES>(low),
                            high: base_lanes_mut::<F, BLOCK_LANES>(high),
                        },
                    ) {
                        #[cfg(all(test, windows, target_arch = "x86_64"))]
                        if MEASURE {
                            let t = instance_major::phase_meter::stamp();
                            instance_major::phase_meter::add_outer(1, t - t1);
                            instance_major::phase_meter::OUTER_CALLS
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        return true;
                    }
                }
                return false;
            }
            // Four blocks keep only the even pair's two halves as an
            // intermediate; the shared column pass carries the sinks at
            // every native width. Block one writes those halves into the
            // first two output quarters; block three applies its pair
            // butterfly and the outer level as its registers leave the base
            // kernel, replacing the intermediates and filling the last two
            // quarters. The detached scalar final pass disappears.
            let combined = combine_four_blocks::<
                F,
                INVERSE,
                MEASURE,
                ROW_LEN,
                BASE,
                BLOCK_LANES,
                TABLE_LANES,
            >(data, &mut scratch[..n], plan, twiddles);
            #[cfg(all(test, windows, target_arch = "x86_64"))]
            if MEASURE {
                let t = instance_major::phase_meter::stamp();
                instance_major::phase_meter::add_outer(1, t - t1);
                instance_major::phase_meter::OUTER_CALLS
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            combined
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
fn combine_four_blocks<
    F,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROW_LEN: usize,
    const BASE: usize,
    const BLOCK_LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [F::Complex],
    scratch: &mut [F::Complex],
    plan: &instance_major::BasePlan<F, 8, ROW_LEN, TABLE_LANES>,
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
    if !instance_major::transform_block::<F, INVERSE, MEASURE, ROW_LEN, BLOCK_LANES, TABLE_LANES>(
        b0, plan,
    ) || !instance_major::transform_block::<F, INVERSE, MEASURE, ROW_LEN, BLOCK_LANES, TABLE_LANES>(
        b2, plan,
    ) {
        return false;
    }
    {
        let (even, _) = data.split_at_mut(2 * BASE);
        let (even_low, even_high) = even.split_at_mut(BASE);
        if !instance_major::transform_block_combining::<
            F,
            INVERSE,
            MEASURE,
            ROW_LEN,
            BLOCK_LANES,
            TABLE_LANES,
        >(
            b1,
            plan,
            instance_major::CombineSink {
                peer: base_lanes::<F, BLOCK_LANES>(b0),
                tw: base_lanes::<F, BLOCK_LANES>(inner),
                low: base_lanes_mut::<F, BLOCK_LANES>(even_low),
                high: base_lanes_mut::<F, BLOCK_LANES>(even_high),
            },
        ) {
            return false;
        }
    }
    let (low, high) = data.split_at_mut(2 * BASE);
    let (even_low, even_high) = low.split_at_mut(BASE);
    let (high_low, high_high) = high.split_at_mut(BASE);
    instance_major::transform_block_combining_final::<
        F,
        INVERSE,
        MEASURE,
        ROW_LEN,
        BLOCK_LANES,
        TABLE_LANES,
    >(
        b3,
        plan,
        instance_major::FinalCombineSink {
            peer: base_lanes::<F, BLOCK_LANES>(b2),
            inner_tw: base_lanes::<F, BLOCK_LANES>(inner),
            even_low: base_lanes_mut::<F, BLOCK_LANES>(even_low),
            even_high: base_lanes_mut::<F, BLOCK_LANES>(even_high),
            outer_low_tw: base_lanes::<F, BLOCK_LANES>(outer_low),
            outer_high_tw: base_lanes::<F, BLOCK_LANES>(outer_high),
            high_low: base_lanes_mut::<F, BLOCK_LANES>(high_low),
            high_high: base_lanes_mut::<F, BLOCK_LANES>(high_high),
        },
    )
}
