//! The register-resident bases and the radix step over them —
//! `ATLAS-APOLLO-BASE-BUTTERFLY-128`, ADR 0061.
//!
//! One instance-major kernel serves 64, 128, and 256 points; 512 and 1024
//! are two and four 256-blocks whose combining butterflies ride the last
//! blocks' column passes. At four lanes every block loads its samples
//! straight out of the parent, so the route is the blocks' own passes and
//! nothing else; at eight lanes the blocks are gathered first, the
//! measured better of the two (`transform_via_base`), and 512 is instead
//! one sixteen-row block (`ROWS = 16` over 32-sample rows, the column
//! pass a sixteen-point DIF), the gather and the combining sink gone.
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

/// A sink table as the fixed-size array its sink indexes.
fn lane_array<T, const N: usize>(lanes: &[T]) -> &[T; N] {
    lanes
        .try_into()
        .expect("invariant: a sink table spans exactly its lane count")
}

/// The lanes of a whole split parent.
fn lanes<T>(data: &[eunomia::Complex<T>]) -> &[T]
where
    T: eunomia::layout::Pod,
    eunomia::Complex<T>: eunomia::layout::Pod,
{
    eunomia::layout::cast_slice(data)
}

/// The 256 base and one radix step over it for 256, 512, and 1024
/// samples, from the plan state that owns its tables.
pub(crate) fn transform_via_base_256<F, const INVERSE: bool, const MEASURE: bool>(
    data: &mut [F::Complex],
    state: &instance_major::State256<F>,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let (plan, sinks) = if INVERSE {
        (state.inverse(), state.inverse_sinks())
    } else {
        (state.forward(), state.sinks())
    };
    transform_via_base::<
        F,
        INVERSE,
        MEASURE,
        32,
        256,
        512,
        1024,
        { instance_major::table_lanes(8, 32) },
    >(data, plan, sinks)
}

/// The base and one radix step over it: `n = BASE` runs the kernel in
/// place, `2 BASE` pairs the even and odd subsequences, `4 BASE` runs the
/// radix-4 step over its four subsequences' spectra. The combining
/// butterflies ride the last block's column pass
/// ([`instance_major::CombineSink`], [`instance_major::FinalRadix4Sink`]);
/// the twiddle table is stage-major, the level of half-length `len`
/// reading `len` entries from `len - 1`.
///
/// Where a block reads its samples is the plan width's measured choice
/// (ADR 0061). At four lanes every block loads its stride-`blocks`
/// subsequence out of the parent directly ([`instance_major::BlockSource`]):
/// two loads and one shuffle a register, the gather pass gone, and the
/// parent read by the last block's rows before its sink writes it — `f64`
/// 1024 from 1.33 to 1.15 of RustFFT. At eight lanes the same loads cost
/// two or four windows and up to three shuffles a register, more than the
/// gather they replace (`f32` 512 and 1024 measured 2 to 4% slower), so
/// that width gathers the subsequences into scratch first
/// ([`split_boundary::GatherBlocks`]) and every block reads contiguously.
/// Both forms run the first three blocks of four into scratch and the
/// last through the radix-4 sink, so no intermediate pair is written.
///
/// Reports whether the dispatched width ran it.
fn transform_via_base<
    F,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROW_LEN: usize,
    const BASE: usize,
    const BLOCK_LANES: usize,
    const SINK_LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [F::Complex],
    plan: &instance_major::BasePlan<F, 8, ROW_LEN, TABLE_LANES>,
    sinks: &instance_major::SplitSinks<F>,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let n = data.len();
    debug_assert!(BASE == 8 * ROW_LEN && BLOCK_LANES == 2 * BASE && SINK_LANES == 2 * BLOCK_LANES);
    debug_assert!(n % BASE == 0 && (n / BASE).is_power_of_two() && n / BASE <= 4);
    if n == BASE {
        return instance_major::transform_block::<
            F,
            INVERSE,
            MEASURE,
            8,
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
    debug_assert_eq!(sinks.inner().len(), SINK_LANES);
    let blocks = n / BASE;
    let gathered_width = plan.native_eight_lanes();
    let scratch_len = n;
    <F as crate::application::execution::kernel::mixed_radix::MixedRadixScalar>::with_scratch(
        scratch_len,
        |scratch| {
            let scratch = &mut scratch[..scratch_len];
            if gathered_width {
                let gathered = match blocks {
                    2 => hermes_simd::vectorize_lanes::<8, F, _>(split_boundary::GatherBlocks::<
                        F,
                        2,
                        BLOCK_LANES,
                    > {
                        src: lanes::<F>(data),
                        dst: eunomia::layout::cast_slice_mut(&mut scratch[..n]),
                    }),
                    _ => hermes_simd::vectorize_lanes::<8, F, _>(split_boundary::GatherBlocks::<
                        F,
                        4,
                        BLOCK_LANES,
                    > {
                        src: lanes::<F>(data),
                        dst: eunomia::layout::cast_slice_mut(&mut scratch[..n]),
                    }),
                }
                .unwrap_or(false);
                if !gathered {
                    return false;
                }
            }
            #[cfg(all(test, windows, target_arch = "x86_64"))]
            let t1 = if MEASURE {
                let t = instance_major::phase_meter::stamp();
                instance_major::phase_meter::add_outer(0, t - t0);
                t
            } else {
                0
            };
            let ran = match (blocks, gathered_width) {
                (2, false) => two_blocks_direct::<
                    F,
                    INVERSE,
                    MEASURE,
                    ROW_LEN,
                    BASE,
                    BLOCK_LANES,
                    SINK_LANES,
                    TABLE_LANES,
                >(data, scratch, plan, sinks),
                (2, true) => two_blocks_gathered::<
                    F,
                    INVERSE,
                    MEASURE,
                    ROW_LEN,
                    BASE,
                    BLOCK_LANES,
                    SINK_LANES,
                    TABLE_LANES,
                >(data, scratch, plan, sinks),
                (_, false) => four_blocks_direct::<
                    F,
                    INVERSE,
                    MEASURE,
                    ROW_LEN,
                    BASE,
                    BLOCK_LANES,
                    SINK_LANES,
                    TABLE_LANES,
                >(data, scratch, plan, sinks),
                (_, true) => four_blocks_gathered::<
                    F,
                    INVERSE,
                    MEASURE,
                    ROW_LEN,
                    BASE,
                    BLOCK_LANES,
                    SINK_LANES,
                    TABLE_LANES,
                >(data, scratch, plan, sinks),
            };
            #[cfg(all(test, windows, target_arch = "x86_64"))]
            if MEASURE {
                let t = instance_major::phase_meter::stamp();
                instance_major::phase_meter::add_outer(1, t - t1);
                instance_major::phase_meter::OUTER_CALLS
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            ran
        },
    )
}

/// One base block from `source` through `sink` into `out`.
fn block<
    F,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROW_LEN: usize,
    const BLOCK_LANES: usize,
    const TABLE_LANES: usize,
    Src,
    S,
>(
    out: &mut [F::Complex],
    source: Src,
    plan: &instance_major::BasePlan<F, 8, ROW_LEN, TABLE_LANES>,
    sink: S,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
    Src: instance_major::BlockSource<F>,
    S: instance_major::StoreSink<F>,
{
    instance_major::transform_base::<
        F,
        INVERSE,
        MEASURE,
        8,
        ROW_LEN,
        BLOCK_LANES,
        TABLE_LANES,
        Src,
        S,
    >(out, source, plan, sink)
}

/// Two blocks, each reading the parent: the even block into scratch, the
/// odd block over the parent with the combining sink.
fn two_blocks_direct<
    F,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROW_LEN: usize,
    const BASE: usize,
    const BLOCK_LANES: usize,
    const SINK_LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [F::Complex],
    scratch: &mut [F::Complex],
    plan: &instance_major::BasePlan<F, 8, ROW_LEN, TABLE_LANES>,
    sinks: &instance_major::SplitSinks<F>,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let inner = lane_array::<F, SINK_LANES>(sinks.inner());
    let even = &mut scratch[..BASE];
    block::<F, INVERSE, MEASURE, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        even,
        instance_major::ParentSplit::<F, 2, 0>(lanes::<F>(data)),
        plan,
        instance_major::DirectSink,
    ) && block::<F, INVERSE, MEASURE, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        data,
        instance_major::SelfSplit::<2, 1>,
        plan,
        instance_major::CombineSink {
            peer: base_lanes::<F, BLOCK_LANES>(even),
            tw: inner,
        },
    )
}

/// Two gathered blocks in scratch: the even block in place, the odd block
/// over the parent with the combining sink.
fn two_blocks_gathered<
    F,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROW_LEN: usize,
    const BASE: usize,
    const BLOCK_LANES: usize,
    const SINK_LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [F::Complex],
    scratch: &mut [F::Complex],
    plan: &instance_major::BasePlan<F, 8, ROW_LEN, TABLE_LANES>,
    sinks: &instance_major::SplitSinks<F>,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let inner = lane_array::<F, SINK_LANES>(sinks.inner());
    let (even, odd) = scratch.split_at_mut(BASE);
    block::<F, INVERSE, MEASURE, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        even,
        instance_major::SelfSplit::<1, 0>,
        plan,
        instance_major::DirectSink,
    ) && block::<F, INVERSE, MEASURE, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        data,
        instance_major::ParentSplit::<F, 1, 0>(lanes::<F>(&odd[..BASE])),
        plan,
        instance_major::CombineSink {
            peer: base_lanes::<F, BLOCK_LANES>(even),
            tw: inner,
        },
    )
}

/// Four blocks, each reading the parent: subsequences 0, 1, and 2 into
/// scratch, subsequence 3 over the parent with the radix-4 sink.
fn four_blocks_direct<
    F,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROW_LEN: usize,
    const BASE: usize,
    const BLOCK_LANES: usize,
    const SINK_LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [F::Complex],
    scratch: &mut [F::Complex],
    plan: &instance_major::BasePlan<F, 8, ROW_LEN, TABLE_LANES>,
    sinks: &instance_major::SplitSinks<F>,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let (sub0, rest) = scratch.split_at_mut(BASE);
    let (sub1, sub2) = rest.split_at_mut(BASE);
    let sub2 = &mut sub2[..BASE];
    block::<F, INVERSE, MEASURE, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        sub0,
        instance_major::ParentSplit::<F, 4, 0>(lanes::<F>(data)),
        plan,
        instance_major::DirectSink,
    ) && block::<F, INVERSE, MEASURE, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        sub1,
        instance_major::ParentSplit::<F, 4, 1>(lanes::<F>(data)),
        plan,
        instance_major::DirectSink,
    ) && block::<F, INVERSE, MEASURE, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        sub2,
        instance_major::ParentSplit::<F, 4, 2>(lanes::<F>(data)),
        plan,
        instance_major::DirectSink,
    ) && block::<F, INVERSE, MEASURE, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        data,
        instance_major::SelfSplit::<4, 3>,
        plan,
        radix4_sink::<F, INVERSE, BASE, BLOCK_LANES, SINK_LANES>(sub0, sub1, sub2, sinks),
    )
}

/// Four gathered blocks in scratch, in the gather's bit-reversed order
/// (subsequences 0, 2, 1, 3): the first three in place, subsequence 3 over
/// the output with the radix-4 sink.
fn four_blocks_gathered<
    F,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROW_LEN: usize,
    const BASE: usize,
    const BLOCK_LANES: usize,
    const SINK_LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [F::Complex],
    scratch: &mut [F::Complex],
    plan: &instance_major::BasePlan<F, 8, ROW_LEN, TABLE_LANES>,
    sinks: &instance_major::SplitSinks<F>,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let (sub0, rest) = scratch.split_at_mut(BASE);
    let (sub2, rest) = rest.split_at_mut(BASE);
    let (sub1, sub3) = rest.split_at_mut(BASE);
    let sub3 = &sub3[..BASE];
    block::<F, INVERSE, MEASURE, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        sub0,
        instance_major::SelfSplit::<1, 0>,
        plan,
        instance_major::DirectSink,
    ) && block::<F, INVERSE, MEASURE, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        sub1,
        instance_major::SelfSplit::<1, 0>,
        plan,
        instance_major::DirectSink,
    ) && block::<F, INVERSE, MEASURE, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        sub2,
        instance_major::SelfSplit::<1, 0>,
        plan,
        instance_major::DirectSink,
    ) && block::<F, INVERSE, MEASURE, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        data,
        instance_major::ParentSplit::<F, 1, 0>(lanes::<F>(sub3)),
        plan,
        radix4_sink::<F, INVERSE, BASE, BLOCK_LANES, SINK_LANES>(sub0, sub1, sub2, sinks),
    )
}

/// The radix-4 sink over three transformed blocks and the split's tables.
fn radix4_sink<
    'a,
    F,
    const INVERSE: bool,
    const BASE: usize,
    const BLOCK_LANES: usize,
    const SINK_LANES: usize,
>(
    sub0: &'a [F::Complex],
    sub1: &'a [F::Complex],
    sub2: &'a [F::Complex],
    sinks: &'a instance_major::SplitSinks<F>,
) -> instance_major::FinalRadix4Sink<'a, F, BLOCK_LANES, SINK_LANES, INVERSE>
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    instance_major::FinalRadix4Sink {
        sub0: base_lanes::<F, BLOCK_LANES>(sub0),
        sub2: base_lanes::<F, BLOCK_LANES>(sub2),
        sub1: base_lanes::<F, BLOCK_LANES>(sub1),
        inner_tw: lane_array::<F, SINK_LANES>(sinks.inner()),
        outer_tw: lane_array::<F, BLOCK_LANES>(sinks.outer()),
    }
}
