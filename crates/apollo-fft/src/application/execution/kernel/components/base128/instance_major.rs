//! The 128-point base butterfly: mixed radix 8 x 16, register-resident
//! stages, one 2 KB staging buffer, no gathers and no scattered accesses.
//!
//! Decomposition `x[8b + a]`, `a = 0..8`, `b = 0..16`:
//!
//! 1. **Redistribute.** The eight stride-8 subsequences move into staging
//!    rows through whole-register concatenations: block pair `(m, m + 8)`
//!    loads eight contiguous registers and emits, per `a`, the pair
//!    `[x[8m + a], x[8(m + 8) + a]]` with one `swap_pairs` + `blend` each.
//!    Those pairs are exactly the first-stage operand pairs of a
//!    decimation-in-time 16-point transform, so writing them at position
//!    `rev3(m)` leaves every staging row already in DIT order — the bit
//!    reversal costs nothing.
//! 2. **Rows.** Each staging row runs the register-resident DIT-16 (the
//!    N = 16 codelet's stage network: an in-register sample butterfly, then
//!    three whole-register twiddled stages), producing natural spectral
//!    order in place.
//! 3. **Columns.** For each natural column pair, eight registers load
//!    contiguously, rows `1..8` multiply by the mixed-radix twiddle
//!    `W_128^{a * k2}`, a lane-wise 8-point DIF runs across the row index,
//!    and register `q` stores to output row `rev3(q)` — natural output
//!    order, 32-byte contiguous stores.
//!
//! The register map has native layouts for four scalar lanes (two interleaved
//! complex samples) and eight scalar lanes (four interleaved complex samples).
//! The plan selects the widest supported layout once, before execution; other
//! native widths decline before touching the input so production routing keeps
//! its incumbent path.

use crate::application::execution::kernel::components::register_butterfly::{
    radix4, radix8, root2_twiddle, rot90,
};
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use core::mem::size_of;
use eunomia::Complex;
use hermes_simd::{ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage};

mod column;
mod plan;
mod store;
mod wide;

use plan::BaseLaneWidth;
pub(crate) use plan::{BasePlan, BasePlanState};
pub(crate) use store::{CombineSink, FinalCombineSink};
use store::{DirectSink, StoreSink};

/// Per-phase TSC accumulators for the separately instantiated attribution
/// instrument.
#[cfg(all(test, windows, target_arch = "x86_64"))]
pub(crate) mod phase_meter {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub(crate) static PHASES: [AtomicU64; 3] =
        [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    pub(crate) static CALLS: AtomicU64 = AtomicU64::new(0);
    /// The split construction's phases per whole transform: the gather,
    /// the base transforms with their sinks, the combine levels past them.
    pub(crate) static OUTER: [AtomicU64; 3] =
        [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    pub(crate) static OUTER_CALLS: AtomicU64 = AtomicU64::new(0);
    #[expect(
        clippy::inline_always,
        reason = "a timing stamp outlined from the measured region distorts it"
    )]
    #[inline(always)]
    pub(crate) fn stamp() -> u64 {
        // SAFETY: x86_64 guarantees SSE2, so LFENCE is available. It orders
        // prior work before RDTSC; these stamps exist only in the separately
        // monomorphized attribution variant, never in the comparison kernel.
        unsafe {
            core::arch::x86_64::_mm_lfence();
            core::arch::x86_64::_rdtsc()
        }
    }
    #[expect(
        clippy::inline_always,
        reason = "a timing accumulator outlined from the measured region distorts it"
    )]
    #[inline(always)]
    pub(crate) fn add(phase: usize, dt: u64) {
        PHASES[phase].fetch_add(dt, Ordering::Relaxed);
    }
    #[expect(
        clippy::inline_always,
        reason = "a timing accumulator outlined from the measured region distorts it"
    )]
    #[inline(always)]
    pub(crate) fn add_outer(phase: usize, dt: u64) {
        OUTER[phase].fetch_add(dt, Ordering::Relaxed);
    }
}

/// Output row permutation of the column pass: DIF register `q` holds
/// spectral `REV3[q]`.
const REV3: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];
/// Two-bit reversal for the four-row column pass.
const REV2: [usize; 4] = [0, 2, 1, 3];

/// Twiddle tables for one direction, in dup-split form: every pair twiddle
/// `[W_a, W_b]` is stored as two four-lane chunks `[re_a, re_a, re_b, re_b]`
/// and `[im_a, im_a, im_b, im_b]`, so a complex multiply is one
/// `swap_adjacent`, one multiply, and one `fmaddsub` — a single shuffle
/// where the general interleaved multiply pays three. Chunk layout:
/// `W_8^{0..4}` pairs at chunk 0, `W_16^{0..8}` pairs at chunk 4, then the
/// mixed-radix pairs `[W_128^{a*2g}, W_128^{a*(2g+1)}]` for `a = 1..8`,
/// `g = 0..8` at chunk 12.
/// First chunk of the mixed-radix twiddles: the table opens with them.
const MIX_CH: usize = 0;

/// Chunk of the row layer's first broadcast twiddle (`W_16^1` for
/// sixteen-sample rows; `W_16^3`, `-W_16^1`, and the real `sqrt(2)/2`
/// broadcast follow at `+2`, `+4`, and `+6`), after the `(rows - 1) *
/// row_len` mixed-radix chunks.
const fn layer_ch(rows: usize, row_len: usize) -> usize {
    (rows - 1) * row_len
}

/// Chunks of the row layer's broadcasts: sixteen-sample rows carry
/// `W_16^1`, `W_16^3`, `-W_16^1` and the real `sqrt(2)/2`; 32-sample rows
/// carry `W_32^1`, `W_32^3`, `W_32^5`, `W_32^7`, `W_16^1`, `W_16^3` and the
/// real `sqrt(2)/2`, every other `W_32^{b0 m}` being one of those under a
/// rotation or a sign.
const fn layer_chunks(row_len: usize) -> usize {
    if row_len == 32 {
        13
    } else {
        7
    }
}

/// Lane count of the table for `rows` subsequences of `row_len` samples:
/// `(rows - 1) * row_len / 2` mixed-radix dup-split pairs, then the row
/// layer's broadcasts ([`layer_chunks`]).
pub(crate) const fn table_lanes(rows: usize, row_len: usize) -> usize {
    ((rows - 1) * row_len + layer_chunks(row_len)) * 4
}

/// The base transform as a lane kernel over interleaved samples: `ROWS`
/// stride-`ROWS` subsequences of `ROW_LEN`, so `ROWS = 8` over sixteen is
/// the 128-point transform, `ROWS = 4` the 64-point one, and `ROWS = 8`
/// over thirty-two the 256-point one (ADR 0061). The sixteen-sample row
/// machinery is identical at both; the column pass is a lane-wise DIF of
/// length `ROWS`.
pub(crate) struct BaseTransform<
    'a,
    T,
    const INVERSE: bool,
    const MEASURE_PHASES: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const LANES: usize,
    const TABLE_LANES: usize,
    S,
> {
    /// Interleaved samples. Fixed-size for the reason [`BasePlan::table`]
    /// documents: the phase-one loads index this from inside a loop.
    pub(crate) data: &'a mut [T; LANES],
    pub(crate) plan: &'a BasePlan<T, ROWS, ROW_LEN, TABLE_LANES>,
    /// Type-selected output strategy. Direct, pair, and four-block-final
    /// stores are separate monomorphizations, so no mode branch reaches the
    /// column loop.
    sink: S,
}

impl<
        T,
        const INVERSE: bool,
        const MEASURE_PHASES: bool,
        const ROWS: usize,
        const ROW_LEN: usize,
        const LANES: usize,
        const TABLE_LANES: usize,
        S,
    > LaneKernel<T>
    for BaseTransform<'_, T, INVERSE, MEASURE_PHASES, ROWS, ROW_LEN, LANES, TABLE_LANES, S>
where
    T: LaneScalar + MixedRadixScalar,
    S: StoreSink<T>,
{
    /// Whether the dispatched width handled the transform.
    type Output = bool;

    #[expect(
        clippy::inline_always,
        reason = "the body must inline into the dispatcher's target-feature \
                  frame (hermes LaneKernel contract for large bodies)"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "one three-phase register-resident transform; splitting it \
                  moves live registers across call boundaries"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) -> bool {
        if <A as SimdStorage<T>>::LANE_COUNT != 4 {
            return false;
        }
        // The dispatch token proves support before this kernel begins, while
        // views hoist bounds reasoning once per slice. Every offset below is
        // a multiple of the four-lane width, so chunk indices are exact. The
        // checked-slice load form spends a visible share of the transform in
        // repeated probes, bounds checks, and Result branches.
        let tab_view = simd.view(self.plan.table.as_slice());
        // Dup-split complex multiply: one shuffle, one multiply, one
        // alternating FMA (see the plan-layout doc).
        let cmul = |v: ComplexReg<T, A>, ch: usize| super::cmul::cmul_chunk(&tab_view, v, ch);
        let zero = T::from_precise(0.0);
        let one = T::from_precise(1.0);
        let neg = T::from_precise(-1.0);
        // Sign vector of the in-register sample butterfly and the blend mask
        // selecting the high complex of a register.
        let constants = [one, one, neg, neg, zero, zero, neg, neg];
        let constants = simd.view(&constants);
        let _sgn = hermes_simd::Vector::<T, A>::from_view_chunk(&constants, 0);
        let hi_mask = hermes_simd::Vector::<T, A>::from_view_chunk(&constants, 1);

        #[cfg(all(test, windows, target_arch = "x86_64"))]
        let t0 = if MEASURE_PHASES {
            phase_meter::stamp()
        } else {
            0
        };
        // Phases 1 and 2, across FFT instances. A register holds sample `b`
        // of two *different* rows — `[x[8b + 2p], x[8b + 2p + 1]]` — which is
        // one contiguous source load, so the separate redistribution pass is
        // gone. Every row twiddle is then a broadcast scalar rather than a
        // per-sample pair, which is what makes the trivial ones free: the
        // radix-4 step's only internal twiddle is a rotation, and `W_16^2`
        // and `W_16^6` reduce to the `sqrt(2)/2` identity. Four general
        // multiplies per pair of instances replace the sixteen the
        // sample-major layout required (gap_audit.md#base128-arithmetic-count).
        //
        // 16 = 4 x 4 with `b = 4*b1 + b0` and output `k2 = 4q + m`: radix-4
        // over `b1` within each stride-4 group, the `W_16^{b0*m}` layer, then
        // radix-4 over `b0`. Natural order in and out, so no bit reversal.
        // The staging buffer is written in full by phase 1 before phase 2
        // reads a lane of it, so zero-filling it first was pure waste — a
        // 2 KB `memset` the disassembly showed costing about 7% of the
        // transform at every size this kernel serves
        // (gap_audit.md#base-kernel-memset).
        debug_assert!(LANES == 2 * ROW_LEN * ROWS && TABLE_LANES == table_lanes(ROWS, ROW_LEN));
        let b16_1 = layer_ch(ROWS, ROW_LEN);
        let chunks_per_row = ROW_LEN / 2;
        let mut staging_uninit = core::mem::MaybeUninit::<[T; LANES]>::uninit();
        // SAFETY: the reference is used only for writes until every lane is
        // initialized. Phase 1 stores chunk `2p*8 + g` and `(2p+1)*8 + g`
        // for `p in 0..ROWS/2` and `g = 2q + mh` with `q in 0..4,
        // mh in 0..2` — rows `0..ROWS` times chunks 0..8, all `8*ROWS`
        // four-lane chunks (`LANES` lanes), before phase 2 performs the
        // first read. Every `LaneScalar` implementor is an
        // IEEE float with no validity niche. Coverage is enforced, not just
        // argued: debug builds poison the buffer with NaN below, so a lane
        // read before it is written poisons the output and fails every
        // value-semantic oracle in the debug suite. Miri cannot reach this
        // body (the dispatcher only selects it on AVX2 hardware); the NaN
        // poison plus the analytical oracles are the substitute coverage.
        let staging: &mut [T; LANES] = unsafe { &mut *staging_uninit.as_mut_ptr() };
        #[cfg(debug_assertions)]
        staging.fill(T::from_precise(f64::NAN));
        {
            let data_view = simd.view(self.data.as_slice());
            let mut stg = simd.view_mut(staging.as_mut_slice());
            let half_root2 = hermes_simd::Vector::<T, A>::from_view_chunk(
                &tab_view,
                b16_1 + layer_chunks(ROW_LEN) - 1,
            );
            // The row's second stage is over `b0`, `ROW_LEN / 4` values per
            // `m`; `zbuf` holds `z[b0][m]` at chunk `per_m * m + b0`.
            let per_m = ROW_LEN / 4;

            // The two radix-4 stages run as separate passes over a
            // 512-byte spill plane. Holding all sixteen twiddled values in
            // registers across the second stage was measured at 4x the
            // baseline: sixteen live registers are the whole AVX2 file, and
            // the allocator spilled the unchanged column pass along with it.
            // Staging them deliberately costs the same traffic the deleted
            // redistribution pass used to pay, and keeps each stage inside
            // roughly a dozen live values.
            let mut zbuf = [T::from_precise(0.0); 128];
            for p in 0..ROWS / 2 {
                {
                    let mut zv = simd.view_mut(&mut zbuf);
                    let load_r = |b: usize| {
                        ComplexReg::<T, A>::from_interleaved(hermes_simd::Vector::from_view_chunk(
                            &data_view,
                            (ROWS / 2) * b + p,
                        ))
                    };
                    if ROW_LEN == 32 {
                        // 32 = 4 x 8 with `b = 8 b1 + b0`: radix-4 over `b1`
                        // within each stride-8 group, the `W_32^{b0 m}` layer,
                        // then radix-8 over `b0`. The layer's general
                        // multiplies are the six broadcasts of the table; the
                        // others are those under `W_32^8` (a rotation) or
                        // `W_32^16` (a sign), and the eighths are the
                        // `sqrt(2)/2` scalings.
                        let rot = rot90::<T, A, INVERSE>;
                        let neg = |v: ComplexReg<T, A>| {
                            ComplexReg::from_interleaved(-v.into_interleaved())
                        };
                        for b0 in 0..8usize {
                            let y = radix4::<T, A, INVERSE>([
                                load_r(b0),
                                load_r(b0 + 8),
                                load_r(b0 + 16),
                                load_r(b0 + 24),
                            ]);
                            let (w1, w3, w5, w7, v1, v3) = (
                                b16_1,
                                b16_1 + 2,
                                b16_1 + 4,
                                b16_1 + 6,
                                b16_1 + 8,
                                b16_1 + 10,
                            );
                            let z = match b0 {
                                0 => y,
                                1 => [y[0], cmul(y[1], w1), cmul(y[2], v1), cmul(y[3], w3)],
                                2 => [
                                    y[0],
                                    cmul(y[1], v1),
                                    root2_twiddle::<T, A, INVERSE, false>(y[2], half_root2),
                                    cmul(y[3], v3),
                                ],
                                3 => [y[0], cmul(y[1], w3), cmul(y[2], v3), rot(cmul(y[3], w1))],
                                4 => [
                                    y[0],
                                    root2_twiddle::<T, A, INVERSE, false>(y[1], half_root2),
                                    rot(y[2]),
                                    root2_twiddle::<T, A, INVERSE, true>(y[3], half_root2),
                                ],
                                5 => [
                                    y[0],
                                    cmul(y[1], w5),
                                    rot(cmul(y[2], v1)),
                                    rot(cmul(y[3], w7)),
                                ],
                                6 => [
                                    y[0],
                                    cmul(y[1], v3),
                                    root2_twiddle::<T, A, INVERSE, true>(y[2], half_root2),
                                    neg(cmul(y[3], v1)),
                                ],
                                _ => [
                                    y[0],
                                    cmul(y[1], w7),
                                    rot(cmul(y[2], v3)),
                                    neg(cmul(y[3], w5)),
                                ],
                            };
                            for (m, reg) in z.into_iter().enumerate() {
                                reg.into_interleaved()
                                    .store_to_view_chunk(&mut zv, per_m * m + b0);
                            }
                        }
                    }
                    for b0 in 0..(if ROW_LEN == 32 { 0 } else { 4usize }) {
                        let y = radix4::<T, A, INVERSE>([
                            load_r(b0),
                            load_r(b0 + 4),
                            load_r(b0 + 8),
                            load_r(b0 + 12),
                        ]);
                        // The `W_16^{b0*m}` layer: one general multiply only
                        // where the twiddle is neither unity, a rotation, nor
                        // a `sqrt(2)/2` scaling.
                        let z = match b0 {
                            0 => y,
                            1 => [
                                y[0],
                                cmul(y[1], b16_1),
                                root2_twiddle::<T, A, INVERSE, false>(y[2], half_root2),
                                cmul(y[3], b16_1 + 2),
                            ],
                            2 => [
                                y[0],
                                root2_twiddle::<T, A, INVERSE, false>(y[1], half_root2),
                                rot90::<T, A, INVERSE>(y[2]),
                                root2_twiddle::<T, A, INVERSE, true>(y[3], half_root2),
                            ],
                            _ => [
                                y[0],
                                cmul(y[1], b16_1 + 2),
                                root2_twiddle::<T, A, INVERSE, true>(y[2], half_root2),
                                cmul(y[3], b16_1 + 4),
                            ],
                        };
                        for (m, reg) in z.into_iter().enumerate() {
                            reg.into_interleaved()
                                .store_to_view_chunk(&mut zv, per_m * m + b0);
                        }
                    }
                }

                // Radix-4 over `b0`, then the pair transpose that hands
                // phase 3 its sample-major registers.
                let zv = simd.view(&zbuf);
                let load_z = |m: usize, b0: usize| {
                    ComplexReg::<T, A>::from_interleaved(hermes_simd::Vector::from_view_chunk(
                        &zv,
                        per_m * m + b0,
                    ))
                };
                for mh in 0..2usize {
                    let m0 = 2 * mh;
                    // Output `4 q + m` for `q` below `ROW_LEN / 4`; the two
                    // `m` of a pair land as consecutive samples of a chunk.
                    let mut o0 = [ComplexReg::from_interleaved(simd.zero()); 8];
                    let mut o1 = [ComplexReg::from_interleaved(simd.zero()); 8];
                    if ROW_LEN == 32 {
                        o0 = radix8::<T, A, INVERSE>(
                            core::array::from_fn(|b0| load_z(m0, b0)),
                            half_root2,
                        );
                        o1 = radix8::<T, A, INVERSE>(
                            core::array::from_fn(|b0| load_z(m0 + 1, b0)),
                            half_root2,
                        );
                    } else {
                        o0[..4].copy_from_slice(&radix4::<T, A, INVERSE>([
                            load_z(m0, 0),
                            load_z(m0, 1),
                            load_z(m0, 2),
                            load_z(m0, 3),
                        ]));
                        o1[..4].copy_from_slice(&radix4::<T, A, INVERSE>([
                            load_z(m0 + 1, 0),
                            load_z(m0 + 1, 1),
                            load_z(m0 + 1, 2),
                            load_z(m0 + 1, 3),
                        ]));
                    }
                    for q in 0..per_m {
                        let g = 2 * q + mh;
                        let a = o0[q].into_interleaved();
                        let b = o1[q].into_interleaved();
                        hi_mask
                            .blend(b.swap_pairs(), a)
                            .store_to_view_chunk(&mut stg, 2 * p * chunks_per_row + g);
                        hi_mask
                            .blend(b, a.swap_pairs())
                            .store_to_view_chunk(&mut stg, (2 * p + 1) * chunks_per_row + g);
                    }
                }
            }
        }

        // Counter 0 now carries the fused load-and-rows pass; counter 1 is
        // retired with the separate redistribution it used to time.
        #[cfg(all(test, windows, target_arch = "x86_64"))]
        let t2m = if MEASURE_PHASES {
            let t = phase_meter::stamp();
            phase_meter::add(0, t - t0);
            t
        } else {
            0
        };
        // Phase 3: the shared lane-wise `ROWS`-point DIF column pass, eight
        // groups of two interleaved complex samples at this width.
        column::column_pass::<T, A, S, INVERSE, ROWS, ROW_LEN>(
            simd,
            staging.as_slice(),
            self.plan.table.as_slice(),
            &self.plan.col,
            self.data.as_mut_slice(),
            self.sink,
        );
        #[cfg(all(test, windows, target_arch = "x86_64"))]
        if MEASURE_PHASES {
            let t = phase_meter::stamp();
            phase_meter::add(2, t - t2m);
            phase_meter::CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        true
    }
}

fn transform_base<
    T,
    const INVERSE: bool,
    const MEASURE_PHASES: bool,
    const ROWS: usize,
    const ROW_LEN: usize,
    const LANES: usize,
    const TABLE_LANES: usize,
    S,
>(
    data: &mut [Complex<T>],
    plan: &BasePlan<T, ROWS, ROW_LEN, TABLE_LANES>,
    sink: S,
) -> bool
where
    T: MixedRadixScalar,
    Complex<T>: eunomia::layout::Pod,
    S: StoreSink<T>,
{
    assert_eq!(
        2 * data.len(),
        LANES,
        "the base transform requires LANES / 2 samples"
    );
    let flat: &mut [T; LANES] = eunomia::layout::cast_slice_mut(data)
        .try_into()
        .expect("invariant: the assertion above fixes the lane count");
    // `MixedRadixScalar` is sealed to f32/f64. Keeping the eight-byte route
    // outside the runtime width match preserves its pre-existing monomorphic
    // kernel body; four-byte hosts still select between NEON-width and AVX2-
    // width layouts once per base invocation.
    if size_of::<T>() != 4 {
        return hermes_simd::vectorize_lanes::<4, T, _>(BaseTransform::<
            T,
            INVERSE,
            MEASURE_PHASES,
            ROWS,
            ROW_LEN,
            LANES,
            TABLE_LANES,
            S,
        > {
            data: flat,
            plan,
            sink,
        })
        .unwrap_or(false);
    }
    match plan.lane_width {
        BaseLaneWidth::Four => hermes_simd::vectorize_lanes::<4, T, _>(BaseTransform::<
            T,
            INVERSE,
            MEASURE_PHASES,
            ROWS,
            ROW_LEN,
            LANES,
            TABLE_LANES,
            S,
        > {
            data: flat,
            plan,
            sink,
        })
        .unwrap_or(false),
        BaseLaneWidth::Eight => hermes_simd::vectorize_lanes::<8, T, _>(wide::BaseTransform::<
            T,
            INVERSE,
            MEASURE_PHASES,
            ROWS,
            ROW_LEN,
            LANES,
            TABLE_LANES,
            S,
        > {
            data: flat,
            plan,
            sink,
        })
        .unwrap_or(false),
    }
}

/// Runs the 128-point base butterfly as the odd half of a split pair,
/// combining with `sink.peer` on the way out (see [`CombineSink`]).
///
/// # Panics
///
/// If `data.len() * 2` is not the base lane count.
pub(crate) fn transform_block_combining<
    T,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROW_LEN: usize,
    const LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [Complex<T>],
    plan: &BasePlan<T, 8, ROW_LEN, TABLE_LANES>,
    sink: CombineSink<'_, T, LANES>,
) -> bool
where
    T: MixedRadixScalar,
    Complex<T>: eunomia::layout::Pod,
{
    transform_base::<T, INVERSE, MEASURE, 8, ROW_LEN, LANES, TABLE_LANES, _>(data, plan, sink)
}

/// Runs block three of a four-block split and stores the final four quarters.
///
/// # Panics
///
/// If `data.len() * 2` is not the base lane count.
pub(crate) fn transform_block_combining_final<
    T,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROW_LEN: usize,
    const LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [Complex<T>],
    plan: &BasePlan<T, 8, ROW_LEN, TABLE_LANES>,
    sink: FinalCombineSink<'_, T, LANES>,
) -> bool
where
    T: MixedRadixScalar,
    Complex<T>: eunomia::layout::Pod,
{
    transform_base::<T, INVERSE, MEASURE, 8, ROW_LEN, LANES, TABLE_LANES, _>(data, plan, sink)
}

/// Runs one eight-row base block of `ROW_LEN` samples per row in place:
/// the 128-point form at sixteen, the 256-point form at thirty-two.
pub(crate) fn transform_block<
    T,
    const INVERSE: bool,
    const MEASURE: bool,
    const ROW_LEN: usize,
    const LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [Complex<T>],
    plan: &BasePlan<T, 8, ROW_LEN, TABLE_LANES>,
) -> bool
where
    T: MixedRadixScalar,
    Complex<T>: eunomia::layout::Pod,
{
    transform_base::<T, INVERSE, MEASURE, 8, ROW_LEN, LANES, TABLE_LANES, _>(data, plan, DirectSink)
}

/// The 128-point base plan: eight rows of sixteen.
pub(crate) type Plan256<T> = BasePlan<T, 8, 32, { table_lanes(8, 32) }>;
pub(crate) type Plan128<T> = BasePlan<T, 8, 16, { table_lanes(8, 16) }>;
/// The 64-point base plan: four rows of sixteen.
pub(crate) type Plan64<T> = BasePlan<T, 4, 16, { table_lanes(4, 16) }>;
/// Directional state for the 128-point base.
pub(crate) type State256<T> = BasePlanState<T, 8, 32, { table_lanes(8, 32) }>;
pub(crate) type State128<T> = BasePlanState<T, 8, 16, { table_lanes(8, 16) }>;
/// Directional state for the 64-point base.
pub(crate) type State64<T> = BasePlanState<T, 4, 16, { table_lanes(4, 16) }>;

#[cfg(test)]
/// Runs the 128-point base butterfly when a supported native layout is
/// available.
///
/// # Panics
///
/// If `data` is not exactly 128 samples.
pub(crate) fn transform_128<T, const INVERSE: bool, const MEASURE: bool>(
    data: &mut [Complex<T>],
    plan: &Plan128<T>,
) -> bool
where
    T: MixedRadixScalar,
    Complex<T>: eunomia::layout::Pod,
{
    transform_base::<T, INVERSE, MEASURE, 8, 16, 256, { table_lanes(8, 16) }, _>(
        data, plan, DirectSink,
    )
}

#[cfg(test)]
/// Runs the 256-point base butterfly: eight stride-8 subsequences of
/// thirty-two, the row phases over `4 x 8` and the eight-point column pass
/// (ADR 0061, the two-pass base under one radix step).
///
/// # Panics
///
/// If `data` is not exactly 256 samples.
pub(crate) fn transform_256<T, const INVERSE: bool, const MEASURE: bool>(
    data: &mut [Complex<T>],
    plan: &Plan256<T>,
) -> bool
where
    T: MixedRadixScalar,
    Complex<T>: eunomia::layout::Pod,
{
    transform_base::<T, INVERSE, MEASURE, 8, 32, 512, { table_lanes(8, 32) }, _>(
        data, plan, DirectSink,
    )
}

/// Runs the 64-point base butterfly: the same construction over four
/// stride-4 subsequences of sixteen.
///
/// # Panics
///
/// If `data` is not exactly 64 samples.
pub(crate) fn transform_64<T, const INVERSE: bool>(
    data: &mut [Complex<T>],
    plan: &Plan64<T>,
) -> bool
where
    T: MixedRadixScalar,
    Complex<T>: eunomia::layout::Pod,
{
    transform_base::<T, INVERSE, false, 4, 16, 128, { table_lanes(4, 16) }, _>(
        data, plan, DirectSink,
    )
}
