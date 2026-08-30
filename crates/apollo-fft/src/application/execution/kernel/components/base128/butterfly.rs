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
//! The register map is defined for four native scalar lanes (two interleaved
//! complex samples). Other native widths decline before touching the input;
//! production routing must retain its incumbent path until Hermes can select
//! this exact width or this kernel gains a native-width variant.

use crate::application::execution::kernel::components::lane_capability::exact_lanes_supported;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::Complex;
use hermes_simd::{ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage};
use std::sync::OnceLock;

/// Per-phase TSC accumulators for the separately instantiated attribution
/// instrument.
#[cfg(all(test, windows, target_arch = "x86_64"))]
pub(crate) mod phase_meter {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub(crate) static PHASES: [AtomicU64; 3] =
        [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    pub(crate) static CALLS: AtomicU64 = AtomicU64::new(0);
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
}

/// Output row permutation of the column pass: DIF register `q` holds
/// spectral `REV3[q]`.
const REV3: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];

/// Phase-one placement for an eight-sample row: four positions over two
/// bits, tabled for the reason recorded beside its use.
const REV2: [usize; 4] = [0, 2, 1, 3];

/// Twiddle tables for one direction, in dup-split form: every pair twiddle
/// `[W_a, W_b]` is stored as two four-lane chunks `[re_a, re_a, re_b, re_b]`
/// and `[im_a, im_a, im_b, im_b]`, so a complex multiply is one
/// `swap_adjacent`, one multiply, and one `fmaddsub` — a single shuffle
/// where the general interleaved multiply pays three. Chunk layout:
/// `W_8^{0..4}` pairs at chunk 0, `W_16^{0..8}` pairs at chunk 4, then the
/// mixed-radix pairs `[W_128^{a*2g}, W_128^{a*(2g+1)}]` for `a = 1..8`,
/// `g = 0..8` at chunk 12.
pub(crate) struct BasePlan<T, const REGS: usize, const TABLE_LANES: usize> {
    /// Dup-split twiddles, 496 lanes (124 chunks).
    ///
    /// Boxed as a fixed-size array rather than a slice for the same reason
    /// the sample buffer is: the length reaches the kernel in the type, so
    /// the chunk-access bounds discharge at compile time instead of costing
    /// a compare and a branch on each of the transform's ~144 twiddle loads.
    table: Box<[T; TABLE_LANES]>,
    /// `W_8^1` and `W_8^3` as complex values for the column-pass splats.
    col: [[T; 2]; 2],
}

/// Chunk offsets into [`BasePlan::table`]. The `W_16` block exists only for
/// row length 16, so the mixed-radix block starts where it ends.
const T3_CH: usize = 0;
const T4_CH: usize = 4;

/// First chunk of the mixed-radix twiddles for a row of `2 * REGS` samples.
const fn mix_chunk(regs: usize) -> usize {
    if regs >= 8 {
        12
    } else {
        4
    }
}

/// Lane count of the table for a row of `2 * REGS` samples: the stage
/// twiddles, then `7 * REGS` mixed-radix pairs of two chunks each.
pub(crate) const fn table_lanes(regs: usize) -> usize {
    (mix_chunk(regs) + 14 * regs) * 4
}

impl<T: MixedRadixScalar, const REGS: usize, const TABLE_LANES: usize>
    BasePlan<T, REGS, TABLE_LANES>
{
    /// Builds the immutable plan when the exact four-lane capability exists.
    pub(crate) fn new_if_supported<const INVERSE: bool>() -> Option<Self> {
        exact_lanes_supported::<4, T>().then(Self::new::<INVERSE>)
    }

    fn new<const INVERSE: bool>() -> Self {
        let dir = if INVERSE { 1.0_f64 } else { -1.0_f64 };
        let w = |j: usize, n: usize| -> [f64; 2] {
            let (s, c) = (dir * core::f64::consts::TAU * j as f64 / n as f64).sin_cos();
            [c, s]
        };
        debug_assert_eq!(TABLE_LANES, table_lanes(REGS));
        let row = 2 * REGS;
        let n = 8 * row;
        let mut table = Vec::with_capacity(TABLE_LANES);
        let mut push_pair = |a: [f64; 2], b: [f64; 2]| {
            for v in [[a[0], a[0], b[0], b[0]], [a[1], a[1], b[1], b[1]]] {
                table.extend(v.map(T::from_precise));
            }
        };
        // Stage 3 always needs `W_8`; stage 4 exists only for a 16-sample
        // row, and its `W_16` block is omitted with it.
        for j in [0usize, 2] {
            push_pair(w(j, 8), w(j + 1, 8));
        }
        if REGS >= 8 {
            for j in [0usize, 2, 4, 6] {
                push_pair(w(j, 16), w(j + 1, 16));
            }
        }
        for a in 1..8 {
            for g in 0..REGS {
                push_pair(w((a * 2 * g) % n, n), w((a * (2 * g + 1)) % n, n));
            }
        }
        let col = [w(1, 8), w(3, 8)].map(|v| [T::from_precise(v[0]), T::from_precise(v[1])]);
        Self {
            table: table
                .into_boxed_slice()
                .try_into()
                .unwrap_or_else(|_| unreachable!("the builder pushes exactly TABLE_LANES lanes")),
            col,
        }
    }
}

/// Plan-owned directional state for a selected base route.
pub(crate) struct BasePlanState<T, const REGS: usize, const TABLE_LANES: usize> {
    forward: BasePlan<T, REGS, TABLE_LANES>,
    inverse: OnceLock<BasePlan<T, REGS, TABLE_LANES>>,
}

impl<T: MixedRadixScalar, const REGS: usize, const TABLE_LANES: usize>
    BasePlanState<T, REGS, TABLE_LANES>
{
    /// Builds the forward plan when the exact-width route is available.
    pub(crate) fn new_if_supported() -> Option<Self> {
        // The 8x8 route is selected per scalar, not universally. Its four
        // lanes fill a whole vector register for a scalar whose widest
        // supported width is four, and half of one otherwise -- where the
        // generic route, which uses the full width, is the faster of the two.
        // `USE_BASE_64` carries that measurement per scalar; the wider 8x16
        // route (REGS = 8) is profitable regardless and is not gated.
        if REGS == 4 && !T::USE_BASE_64 {
            return None;
        }
        BasePlan::new_if_supported::<false>().map(|forward| Self {
            forward,
            inverse: OnceLock::new(),
        })
    }

    /// Borrows the immutable forward plan.
    pub(crate) fn forward(&self) -> &BasePlan<T, REGS, TABLE_LANES> {
        &self.forward
    }

    /// Borrows the immutable inverse plan, initializing it once across clones.
    pub(crate) fn inverse(&self) -> &BasePlan<T, REGS, TABLE_LANES> {
        self.inverse.get_or_init(BasePlan::new::<true>)
    }

    #[cfg(test)]
    pub(crate) fn inverse_is_initialized(&self) -> bool {
        self.inverse.get().is_some()
    }
}

/// A base transform as a lane kernel over interleaved samples.
pub(crate) struct BaseTransform<
    'a,
    T,
    const INVERSE: bool,
    const MEASURE_PHASES: bool,
    const REGS: usize,
    const LANES: usize,
    const TABLE_LANES: usize,
> {
    /// Interleaved samples, with the exact lane count carried in the type.
    ///
    /// The fixed-size reference is load-bearing, not decoration. `SimdView`
    /// chunk access asserts `offset + LANE_COUNT <= len()`, and against a
    /// `&mut [T]` that length is opaque, so every one of the kernel's ~130
    /// accesses emitted a compare and a branch to a panic block — which also
    /// kept the register arrays pinned to the stack. With the length in the
    /// type the compiler discharges every bound at compile time.
    pub(crate) data: &'a mut [T; LANES],
    pub(crate) plan: &'a BasePlan<T, REGS, TABLE_LANES>,
}

impl<
        T,
        const INVERSE: bool,
        const MEASURE_PHASES: bool,
        const REGS: usize,
        const LANES: usize,
        const TABLE_LANES: usize,
    > LaneKernel<T> for BaseTransform<'_, T, INVERSE, MEASURE_PHASES, REGS, LANES, TABLE_LANES>
where
    T: LaneScalar + MixedRadixScalar,
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
        let sgn = hermes_simd::Vector::<T, A>::from_view_chunk(&constants, 0);
        let hi_mask = hermes_simd::Vector::<T, A>::from_view_chunk(&constants, 1);
        let zero_complex = ComplexReg::from_interleaved(simd.zero());
        let complex_splat = |sample: Complex<T>| {
            let (interleaved, _) = simd.splat(sample.re).interleave(simd.splat(sample.im));
            ComplexReg::from_interleaved(interleaved)
        };

        #[cfg(all(test, windows, target_arch = "x86_64"))]
        let t0 = if MEASURE_PHASES {
            phase_meter::stamp()
        } else {
            0
        };
        // Phase 1: redistribute into DIT-ordered staging rows. Block pair
        // (m, m + REGS) yields, per row `a`, the sample pair
        // [x[8m + a], x[8m + 8*REGS + a]] for the first DIT stage.
        // `LANES = 32 * REGS`: eight rows of `2 * REGS` interleaved samples.
        debug_assert_eq!(LANES, 32 * REGS);
        debug_assert_eq!(TABLE_LANES, table_lanes(REGS));
        let mut staging = [T::from_precise(0.0); LANES];
        {
            let data_view = simd.view(self.data.as_slice());
            let mut stg = simd.view_mut(&mut staging);
            // Row position `m` lands at register `rev(m)` over
            // `log2(REGS)` bits, which is what leaves each staging row in
            // decimation-in-time order for free. Tabled rather than
            // computed: x86 has no bit-reverse instruction, and
            // `usize::reverse_bits` lowers to a shift-and-mask sequence that
            // cost this phase 60% (gap_audit.md#base-row-length).
            for m in 0..REGS {
                let dst = if REGS >= 8 { REV3[m] } else { REV2[m] };
                for r in 0..4usize {
                    let lo = hermes_simd::Vector::from_view_chunk(&data_view, 4 * m + r);
                    let hi = hermes_simd::Vector::from_view_chunk(&data_view, 4 * REGS + 4 * m + r);
                    // concat_low = [lo.c0, hi.c0], concat_high = [lo.c1, hi.c1].
                    let even = hi_mask.blend(hi.swap_pairs(), lo);
                    let odd = hi_mask.blend(hi, lo.swap_pairs());
                    even.store_to_view_chunk(&mut stg, (2 * r) * REGS + dst);
                    odd.store_to_view_chunk(&mut stg, (2 * r + 1) * REGS + dst);
                }
            }
        }

        #[cfg(all(test, windows, target_arch = "x86_64"))]
        let t1 = if MEASURE_PHASES {
            let t = phase_meter::stamp();
            phase_meter::add(0, t - t0);
            t
        } else {
            0
        };
        // Phase 2: register-resident DIT-16 per staging row; bit reversal was
        // absorbed by phase 1, so output lands in natural spectral order.
        let mut stg_rows = simd.view_mut(&mut staging);
        for a in 0..8usize {
            let row = a * REGS;
            let mut r = [zero_complex; REGS];
            for (k, reg) in r.iter_mut().enumerate() {
                *reg = ComplexReg::from_interleaved(hermes_simd::Vector::from_view_chunk(
                    &stg_rows,
                    row + k,
                ));
            }
            // Stage 1 (distance 1, twiddle 1): [s0 + s1, s0 - s1] per
            // register — the sample swap feeding one alternating-sign FMA.
            for reg in &mut r {
                let vi = reg.into_interleaved();
                *reg = ComplexReg::from_interleaved(vi.mul_add(sgn, vi.swap_pairs()));
            }
            // Stage 2 (distance 2): the twiddle [1, -+i] is a high-sample
            // rotation and a blend, not a general complex multiply.
            for base in (0..REGS).step_by(2) {
                let v = r[base + 1];
                let rot = if INVERSE { v.mul_i() } else { v.mul_neg_i() };
                let wb = ComplexReg::from_interleaved(
                    hi_mask.blend(rot.into_interleaved(), v.into_interleaved()),
                );
                let (lo, hi) = r[base].butterfly(wb);
                (r[base], r[base + 1]) = (lo, hi);
            }
            // Stage 3 (distance 4): twiddles W8^{0..4} across two registers.
            for base in (0..REGS).step_by(4) {
                for offset in 0..2 {
                    let wb = cmul(r[base + 2 + offset], T3_CH + 2 * offset);
                    let (lo, hi) = r[base + offset].butterfly(wb);
                    (r[base + offset], r[base + 2 + offset]) = (lo, hi);
                }
            }
            // Stage 4 (distance 8): twiddles W16^{0..8} across four
            // registers. A row of eight samples ends at stage 3.
            if REGS >= 8 {
                for offset in 0..4 {
                    let wb = cmul(r[offset + 4], T4_CH + 2 * offset);
                    let (lo, hi) = r[offset].butterfly(wb);
                    (r[offset], r[offset + 4]) = (lo, hi);
                }
            }
            for (k, reg) in r.iter().enumerate() {
                reg.into_interleaved()
                    .store_to_view_chunk(&mut stg_rows, row + k);
            }
        }
        let _ = stg_rows;

        #[cfg(all(test, windows, target_arch = "x86_64"))]
        let t2m = if MEASURE_PHASES {
            let t = phase_meter::stamp();
            phase_meter::add(1, t - t1);
            t
        } else {
            0
        };
        // Phase 3: mixed-radix twiddle and the lane-wise 8-point DIF across
        // rows, one natural column pair per group.
        let w8_1 = complex_splat(Complex::new(self.plan.col[0][0], self.plan.col[0][1]));
        let w8_3 = complex_splat(Complex::new(self.plan.col[1][0], self.plan.col[1][1]));
        let stg = simd.view(&staging);
        let mut out = simd.view_mut(self.data.as_mut_slice());
        let mix = mix_chunk(REGS);
        for g in 0..REGS {
            let mut c = [zero_complex; 8];
            for (a, reg) in c.iter_mut().enumerate() {
                *reg = ComplexReg::from_interleaved(hermes_simd::Vector::from_view_chunk(
                    &stg,
                    a * REGS + g,
                ));
            }
            for a in 1..8usize {
                c[a] = cmul(c[a], mix + 2 * ((a - 1) * REGS + g));
            }
            // Distance 4 (L = 8): v * W8^a on the difference.
            for a in 0..4usize {
                let (u, v) = c[a].butterfly(c[a + 4]);
                c[a] = u;
                c[a + 4] = match a {
                    0 => v,
                    1 => v * w8_1,
                    2 => {
                        if INVERSE {
                            v.mul_i()
                        } else {
                            v.mul_neg_i()
                        }
                    }
                    _ => v * w8_3,
                };
            }
            // Distance 2 (L = 4): twiddles [1, -+i].
            for base in [0usize, 4] {
                for j in 0..2usize {
                    let (u, v) = c[base + j].butterfly(c[base + j + 2]);
                    c[base + j] = u;
                    c[base + j + 2] = if j == 0 {
                        v
                    } else if INVERSE {
                        v.mul_i()
                    } else {
                        v.mul_neg_i()
                    };
                }
            }
            // Distance 1 (L = 2): twiddle one.
            for base in [0usize, 2, 4, 6] {
                let (u, v) = c[base].butterfly(c[base + 1]);
                (c[base], c[base + 1]) = (u, v);
            }
            for (q, reg) in c.iter().enumerate() {
                reg.into_interleaved()
                    .store_to_view_chunk(&mut out, REV3[q] * REGS + g);
            }
        }
        #[cfg(all(test, windows, target_arch = "x86_64"))]
        if MEASURE_PHASES {
            let t = phase_meter::stamp();
            phase_meter::add(2, t - t2m);
            phase_meter::CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        true
    }
}

/// The 64-point base plan: eight rows of eight.
pub(crate) type Plan64<T> = BasePlan<T, 4, { table_lanes(4) }>;
/// Directional state for the 64-point base.
pub(crate) type State64<T> = BasePlanState<T, 4, { table_lanes(4) }>;

/// Runs the base transform for a row of `2 * REGS` samples.
fn transform_base<
    T,
    const INVERSE: bool,
    const MEASURE_PHASES: bool,
    const REGS: usize,
    const LANES: usize,
    const TABLE_LANES: usize,
>(
    data: &mut [Complex<T>],
    plan: &BasePlan<T, REGS, TABLE_LANES>,
) -> bool
where
    T: MixedRadixScalar,
    Complex<T>: bytemuck::Pod,
{
    assert_eq!(
        2 * data.len(),
        LANES,
        "the base transform requires LANES / 2 samples"
    );
    let flat: &mut [T; LANES] = bytemuck::cast_slice_mut(data)
        .try_into()
        .expect("invariant: the assertion above fixes the lane count");
    hermes_simd::vectorize_lanes::<4, T, _>(BaseTransform::<
        T,
        INVERSE,
        MEASURE_PHASES,
        REGS,
        LANES,
        TABLE_LANES,
    > {
        data: flat,
        plan,
    })
    .unwrap_or(false)
}

/// Runs the 64-point base butterfly: the same construction at half the row
/// length, its row transform being the sixteen-sample one without the last
/// stage.
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
    Complex<T>: bytemuck::Pod,
{
    transform_base::<T, INVERSE, false, 4, 128, { table_lanes(4) }>(data, plan)
}
