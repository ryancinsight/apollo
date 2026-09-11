//! The register-resident bases and the radix step over them —
//! `ATLAS-APOLLO-BASE-BUTTERFLY-128`, ADR 0061.
//!
//! One instance-major kernel serves 64, 128, 256, and 512 points (128 as
//! eight 16-sample rows at four lanes and four 32-sample rows at eight,
//! 512 as sixteen 32-sample rows, the column pass a sixteen-point DIF);
//! 1024 is four 256-blocks and 2048 four 512-blocks, the radix-4 step
//! riding the last block's column pass in both.
//! At four lanes every block loads its samples straight out of the
//! parent, so the route is the blocks' own passes and nothing else; at
//! eight lanes the blocks are gathered first, the measured better of the
//! two (`transform_via_base`).
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

/// The 256 base and one radix step over it for 256, 1024 (four blocks,
/// radix 4) and 2048 samples (eight blocks, radix 8), from the plan state
/// that owns its tables.
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
        8,
        32,
        256,
        512,
        1024,
        { instance_major::table_lanes(8, 32) },
    >(data, plan, sinks)
}

/// The 128 base in the shape its state selected and the radix-3 step over
/// three of its blocks for 384 samples, the blocks reading the parent at
/// either width.
pub(crate) fn transform_via_base_128<F, const INVERSE: bool, const MEASURE: bool>(
    data: &mut [F::Complex],
    state: &instance_major::State128<F>,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    match state {
        instance_major::State128::EightRows(state) => {
            let (plan, sinks) = if INVERSE {
                (state.inverse(), state.inverse_sinks())
            } else {
                (state.forward(), state.sinks())
            };
            transform_via_base::<
                F,
                INVERSE,
                MEASURE,
                8,
                16,
                128,
                256,
                512,
                { instance_major::table_lanes(8, 16) },
            >(data, plan, sinks)
        }
        instance_major::State128::FourRows(state) => {
            let (plan, sinks) = if INVERSE {
                (state.inverse(), state.inverse_sinks())
            } else {
                (state.forward(), state.sinks())
            };
            transform_via_base::<
                F,
                INVERSE,
                MEASURE,
                4,
                32,
                128,
                256,
                512,
                { instance_major::table_lanes(4, 32) },
            >(data, plan, sinks)
        }
    }
}

/// The 512 base and one radix-4 step over it for 512 and 2048 samples,
/// from the plan state that owns its tables.
pub(crate) fn transform_via_base_512<F, const INVERSE: bool, const MEASURE: bool>(
    data: &mut [F::Complex],
    state: &instance_major::State512<F>,
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
        16,
        32,
        512,
        1024,
        2048,
        { instance_major::table_lanes(16, 32) },
    >(data, plan, sinks)
}

/// The base and one radix step over it: `n = BASE` runs the kernel in
/// place, `3 BASE`, `4 BASE` and `8 BASE` run the radix-3, radix-4 and
/// radix-8 steps over their subsequences' spectra, the combining
/// butterflies riding the last block's column pass
/// ([`instance_major::FinalRadix4Sink`], [`instance_major::FinalRadix8Sink`]);
/// the twiddle table is stage-major, the level of half-length `len`
/// reading `len` entries from `len - 1`. (`2 BASE` was a pair of blocks
/// under a combining sink until the sixteen-row base took 512 at both
/// widths, ADR 0061.)
///
/// Where a block reads its samples is the plan width's measured choice
/// (ADR 0061). At four lanes every block loads its stride-`blocks`
/// subsequence out of the parent directly ([`instance_major::BlockSource`]):
/// two loads and one shuffle a register, the gather pass gone, and the
/// parent read by the last block's rows before its sink writes it — `f64`
/// 1024 from 1.33 to 1.15 of RustFFT. At eight lanes the same loads cost
/// two or four windows and up to three shuffles a register, more than the
/// gather they replace (`f32` 1024 measured 2 to 4% slower), so that
/// width gathers the subsequences into scratch first
/// ([`split_boundary::GatherBlocks`]) and every block reads contiguously.
/// Both forms run the first three blocks into scratch and the last
/// through the radix-4 sink, so no intermediate pair is written. Eight
/// blocks take the same two forms — the parent read at stride eight at
/// four lanes, gathered at eight — under the radix-8 sink, the first
/// seven into scratch.
///
/// Reports whether the dispatched width ran it.
fn transform_via_base<
    F,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const BASE: usize,
    const BLOCK_LANES: usize,
    const SINK_LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [F::Complex],
    plan: &instance_major::BasePlan<F, ROWS, ROW_LEN, TABLE_LANES>,
    sinks: &instance_major::SplitSinks<F>,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let n = data.len();
    debug_assert!(
        BASE == ROWS * ROW_LEN && BLOCK_LANES == 2 * BASE && SINK_LANES == 2 * BLOCK_LANES
    );
    debug_assert!(n == BASE || n == 3 * BASE || n == 4 * BASE || n == 8 * BASE);
    if n == BASE {
        return instance_major::transform_block::<
            F,
            INVERSE,
            MEASURE,
            ROWS,
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
    let blocks = n / BASE;
    debug_assert!(if blocks == 8 {
        sinks.rows().len() == 7 * BLOCK_LANES
    } else {
        sinks.inner().len() == SINK_LANES
    });
    // Three blocks read the parent directly at either width (a stride-three
    // register from three-sample-apart windows); four and eight gather at
    // eight.
    let gathered_width = (blocks == 4 || blocks == 8) && plan.native_eight_lanes();
    let scratch_len = n;
    <F as crate::application::execution::kernel::mixed_radix::MixedRadixScalar>::with_scratch(
        scratch_len,
        |scratch| {
            let scratch = &mut scratch[..scratch_len];
            if gathered_width {
                let gathered = if blocks == 8 {
                    hermes_simd::vectorize_lanes::<8, F, _>(split_boundary::GatherBlocks::<
                        F,
                        8,
                        BLOCK_LANES,
                    > {
                        src: lanes::<F>(data),
                        dst: eunomia::layout::cast_slice_mut(&mut scratch[..n]),
                    })
                } else {
                    hermes_simd::vectorize_lanes::<8, F, _>(split_boundary::GatherBlocks::<
                        F,
                        4,
                        BLOCK_LANES,
                    > {
                        src: lanes::<F>(data),
                        dst: eunomia::layout::cast_slice_mut(&mut scratch[..n]),
                    })
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
            let ran = if blocks == 8 && gathered_width {
                eight_blocks_gathered::<
                    F,
                    INVERSE,
                    MEASURE,
                    ROWS,
                    ROW_LEN,
                    BASE,
                    BLOCK_LANES,
                    TABLE_LANES,
                >(data, scratch, plan, sinks)
            } else if blocks == 8 {
                eight_blocks_direct::<
                    F,
                    INVERSE,
                    MEASURE,
                    ROWS,
                    ROW_LEN,
                    BASE,
                    BLOCK_LANES,
                    TABLE_LANES,
                >(data, scratch, plan, sinks)
            } else if blocks == 3 {
                three_blocks_direct::<
                    F,
                    INVERSE,
                    MEASURE,
                    ROWS,
                    ROW_LEN,
                    BASE,
                    BLOCK_LANES,
                    SINK_LANES,
                    TABLE_LANES,
                >(data, scratch, plan, sinks)
            } else if gathered_width {
                four_blocks_gathered::<
                    F,
                    INVERSE,
                    MEASURE,
                    ROWS,
                    ROW_LEN,
                    BASE,
                    BLOCK_LANES,
                    SINK_LANES,
                    TABLE_LANES,
                >(data, scratch, plan, sinks)
            } else {
                four_blocks_direct::<
                    F,
                    INVERSE,
                    MEASURE,
                    ROWS,
                    ROW_LEN,
                    BASE,
                    BLOCK_LANES,
                    SINK_LANES,
                    TABLE_LANES,
                >(data, scratch, plan, sinks)
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
    const ROWS: usize,
    const ROW_LEN: usize,
    const BLOCK_LANES: usize,
    const TABLE_LANES: usize,
    Src,
    S,
>(
    out: &mut [F::Complex],
    source: Src,
    plan: &instance_major::BasePlan<F, ROWS, ROW_LEN, TABLE_LANES>,
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
        ROWS,
        ROW_LEN,
        BLOCK_LANES,
        TABLE_LANES,
        Src,
        S,
    >(out, source, plan, sink)
}

/// Eight blocks, each reading the parent: subsequences 0 to 6 into
/// scratch, subsequence 7 over the parent with the radix-8 sink.
fn eight_blocks_direct<
    F,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const BASE: usize,
    const BLOCK_LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [F::Complex],
    scratch: &mut [F::Complex],
    plan: &instance_major::BasePlan<F, ROWS, ROW_LEN, TABLE_LANES>,
    sinks: &instance_major::SplitSinks<F>,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let (s0, rest) = scratch.split_at_mut(BASE);
    let (s1, rest) = rest.split_at_mut(BASE);
    let (s2, rest) = rest.split_at_mut(BASE);
    let (s3, rest) = rest.split_at_mut(BASE);
    let (s4, rest) = rest.split_at_mut(BASE);
    let (s5, rest) = rest.split_at_mut(BASE);
    let s6 = &mut rest[..BASE];
    let parent = lanes::<F>(data);
    let transformed =
        block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
            s0,
            instance_major::ParentSplit::<F, 8, 0>(parent),
            plan,
            instance_major::DirectSink,
        ) && block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
            s1,
            instance_major::ParentSplit::<F, 8, 1>(parent),
            plan,
            instance_major::DirectSink,
        ) && block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
            s2,
            instance_major::ParentSplit::<F, 8, 2>(parent),
            plan,
            instance_major::DirectSink,
        ) && block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
            s3,
            instance_major::ParentSplit::<F, 8, 3>(parent),
            plan,
            instance_major::DirectSink,
        ) && block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
            s4,
            instance_major::ParentSplit::<F, 8, 4>(parent),
            plan,
            instance_major::DirectSink,
        ) && block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
            s5,
            instance_major::ParentSplit::<F, 8, 5>(parent),
            plan,
            instance_major::DirectSink,
        ) && block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
            s6,
            instance_major::ParentSplit::<F, 8, 6>(parent),
            plan,
            instance_major::DirectSink,
        );
    transformed
        && block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
            data,
            instance_major::SelfSplit::<8, 7>,
            plan,
            radix8_sink::<F, INVERSE, BLOCK_LANES>([s0, s1, s2, s3, s4, s5, s6], sinks),
        )
}

/// Eight gathered blocks in scratch, in natural order: the first seven in
/// place, subsequence 7 over the output with the radix-8 sink.
fn eight_blocks_gathered<
    F,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const BASE: usize,
    const BLOCK_LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [F::Complex],
    scratch: &mut [F::Complex],
    plan: &instance_major::BasePlan<F, ROWS, ROW_LEN, TABLE_LANES>,
    sinks: &instance_major::SplitSinks<F>,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let (seven, s7) = scratch.split_at_mut(7 * BASE);
    let s7 = &s7[..BASE];
    let transformed = seven.chunks_exact_mut(BASE).all(|sub| {
        block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
            sub,
            instance_major::SelfSplit::<1, 0>,
            plan,
            instance_major::DirectSink,
        )
    });
    let (s0, rest) = seven.split_at(BASE);
    let (s1, rest) = rest.split_at(BASE);
    let (s2, rest) = rest.split_at(BASE);
    let (s3, rest) = rest.split_at(BASE);
    let (s4, rest) = rest.split_at(BASE);
    let (s5, s6) = rest.split_at(BASE);
    transformed
        && block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
            data,
            instance_major::ParentSplit::<F, 1, 0>(lanes::<F>(s7)),
            plan,
            radix8_sink::<F, INVERSE, BLOCK_LANES>([s0, s1, s2, s3, s4, s5, s6], sinks),
        )
}

/// The radix-8 sink over seven transformed blocks and the split's rows.
fn radix8_sink<'a, F, const INVERSE: bool, const BLOCK_LANES: usize>(
    subs: [&'a [F::Complex]; 7],
    sinks: &'a instance_major::SplitSinks<F>,
) -> instance_major::FinalRadix8Sink<'a, F, BLOCK_LANES, INVERSE>
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let dir = if INVERSE { 1.0_f64 } else { -1.0_f64 };
    let eighth = |j: f64| -> [F; 2] {
        let (s, c) = (dir * core::f64::consts::TAU * j / 8.0).sin_cos();
        [F::from_precise(c), F::from_precise(s)]
    };
    let rows = sinks.rows();
    let row =
        |j: usize| lane_array::<F, BLOCK_LANES>(&rows[j * BLOCK_LANES..(j + 1) * BLOCK_LANES]);
    instance_major::FinalRadix8Sink {
        subs: subs.map(|sub| base_lanes::<F, BLOCK_LANES>(sub)),
        rows: [row(0), row(1), row(2), row(3), row(4), row(5), row(6)],
        eighth_one: eighth(1.0),
        eighth_three: eighth(3.0),
    }
}

/// Three blocks, each reading the parent: subsequences 0 and 1 into
/// scratch, subsequence 2 over the parent with the radix-3 sink.
fn three_blocks_direct<
    F,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const BASE: usize,
    const BLOCK_LANES: usize,
    const SINK_LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [F::Complex],
    scratch: &mut [F::Complex],
    plan: &instance_major::BasePlan<F, ROWS, ROW_LEN, TABLE_LANES>,
    sinks: &instance_major::SplitSinks<F>,
) -> bool
where
    F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
        Complex = eunomia::Complex<F>,
    >,
    eunomia::Complex<F>: eunomia::layout::Pod,
{
    let (sub0, rest) = scratch.split_at_mut(BASE);
    let sub1 = &mut rest[..BASE];
    block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        sub0,
        instance_major::ParentSplit::<F, 3, 0>(lanes::<F>(data)),
        plan,
        instance_major::DirectSink,
    ) && block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        sub1,
        instance_major::ParentSplit::<F, 3, 1>(lanes::<F>(data)),
        plan,
        instance_major::DirectSink,
    ) && block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        data,
        instance_major::SelfSplit::<3, 2>,
        plan,
        instance_major::FinalRadix3Sink::<F, BLOCK_LANES, SINK_LANES, INVERSE> {
            sub0: base_lanes::<F, BLOCK_LANES>(sub0),
            sub1: base_lanes::<F, BLOCK_LANES>(sub1),
            first_tw: lane_array::<F, SINK_LANES>(sinks.inner()),
            second_tw: lane_array::<F, SINK_LANES>(sinks.second()),
            half_negative: F::from_precise(-0.5),
            sine: F::from_precise(0.866_025_403_784_438_6),
        },
    )
}

/// Four blocks, each reading the parent: subsequences 0, 1, and 2 into
/// scratch, subsequence 3 over the parent with the radix-4 sink.
fn four_blocks_direct<
    F,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const BASE: usize,
    const BLOCK_LANES: usize,
    const SINK_LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [F::Complex],
    scratch: &mut [F::Complex],
    plan: &instance_major::BasePlan<F, ROWS, ROW_LEN, TABLE_LANES>,
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
    block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        sub0,
        instance_major::ParentSplit::<F, 4, 0>(lanes::<F>(data)),
        plan,
        instance_major::DirectSink,
    ) && block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        sub1,
        instance_major::ParentSplit::<F, 4, 1>(lanes::<F>(data)),
        plan,
        instance_major::DirectSink,
    ) && block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        sub2,
        instance_major::ParentSplit::<F, 4, 2>(lanes::<F>(data)),
        plan,
        instance_major::DirectSink,
    ) && block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
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
    const ROWS: usize,
    const ROW_LEN: usize,
    const BASE: usize,
    const BLOCK_LANES: usize,
    const SINK_LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [F::Complex],
    scratch: &mut [F::Complex],
    plan: &instance_major::BasePlan<F, ROWS, ROW_LEN, TABLE_LANES>,
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
    block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        sub0,
        instance_major::SelfSplit::<1, 0>,
        plan,
        instance_major::DirectSink,
    ) && block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        sub1,
        instance_major::SelfSplit::<1, 0>,
        plan,
        instance_major::DirectSink,
    ) && block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
        sub2,
        instance_major::SelfSplit::<1, 0>,
        plan,
        instance_major::DirectSink,
    ) && block::<F, INVERSE, MEASURE, ROWS, ROW_LEN, BLOCK_LANES, TABLE_LANES, _, _>(
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
