//! Four-step with register-resident row transforms.
//!
//! **Status: correct, measured at true cost, slower than the batched route**
//! — compiled for tests only. All five oracles pass. The hermes dispatch
//! defect this module exposed (large kernel bodies outlined from the
//! `#[target_feature]` frame to baseline codegen) is fixed upstream, which
//! took the kernel from ~26.5 us to 6.5 us at N = 1024 pinned; the row
//! passes now run near their port-limited bound (~171 cycles per 32-point
//! row). What that reveals is structural: the two row passes alone cost more
//! TSC cycles than the batched route's entire transform, and the shape pays
//! ~38% more in the two transposes and the closing involution
//! (`RESIDENT_SECTIONS=1` prints the per-pass attribution). The interleaved
//! butterfly is shuffle-port-bound where the batched planar arithmetic is
//! FMA-bound, so beating batched from this shape needs planar-register rows
//! and transposes fused into the row load/store networks — the RustFFT
//! construction — not incremental tuning here. The same-process probe beside
//! this module is independently valuable: it is the four-engine pinned
//! comparison that corrected the cross-instrument gap arithmetic.
//!
//! The batched kernel streams the array once per fused stage pair; at
//! N = 1024 that is ten-plus passes, and every alternative inside that shape
//! measured out (radix-8 spills, interleaved streaming loses, boundary
//! vectorization loses). The reference engines take a different shape
//! entirely: whole sub-transforms held in vector registers, so a 32-point FFT
//! is one load, five in-register stages, one store.
//!
//! Interleaved data makes that fit where planar cannot: 32 complex samples
//! are 64 lanes — exactly sixteen AVX2 registers — where planar needs
//! thirty-two. The interleaved-kernel experiment lost when it *streamed*
//! stages; register residency is where the layout's density pays.
//!
//! ## Structure, five passes for N = m²
//!
//! 1. In-place square transpose, so each column becomes a contiguous row.
//! 2. Row pass: every row through a register-resident **DIF** transform —
//!    natural input order, bit-reversed output order. DIF is what makes the
//!    ordering free: every stage pairs whole registers except the last, whose
//!    twiddle is one, so it is the sample-swap-and-sign step and no
//!    two-register sample shuffle is ever needed.
//! 3. In-place transpose again (pure exchange).
//! 4. Row pass: the four-step twiddle multiplied at load — its matrix is
//!    stored interleaved with **bit-reversed row indices baked in**, so the
//!    bit-reversed order left by pass 2 is absorbed rather than repaired —
//!    then the same DIF transform.
//! 5. One in-place involution combining the final transpose with both
//!    bit-reversals: destination `(a, b)` exchanges with `(rev(b), rev(a))`,
//!    which is its own inverse, so it is pair swaps with no scratch.
//!
//! No scratch buffer exists anywhere in the driver: every pass is in place.
//!
//! ## Scope
//!
//! Row length 32 at four f64 lanes per register — N = 1024, the measured
//! centre of the mid-size gap — with other widths and lengths reporting
//! unhandled so the caller falls back. Generalizing the row length changes
//! the register count, which is exactly what must not be generic on a
//! sixteen-register file.

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::Complex;
use hermes_simd::{ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// Row length: 32 complex samples, sixteen two-sample registers.
const ROW: usize = 32;

/// Twiddle tables for the resident driver, one per `(n, direction)`.
pub(crate) struct ResidentPlan<T> {
    /// Flat interleaved DIF stage twiddles, stage-descending: `W_L^j` for
    /// `L = 32, 16, 8, 4`, each stage contributing `L` scalars (`L/2`
    /// complex entries).
    stages: Vec<T>,
    /// Interleaved four-step matrix with the row bit-reversal baked in:
    /// row `p`, column `b` holds `W_n^(rev(p) * b)`.
    matrix: Vec<T>,
}

impl<T: MixedRadixScalar> ResidentPlan<T> {
    fn new<const INVERSE: bool>(n: usize) -> Self {
        let m = ROW;
        debug_assert_eq!(n, m * m);
        let sign = if INVERSE { 1.0_f64 } else { -1.0_f64 };

        let mut stages = Vec::with_capacity(2 * (m + m / 2 + m / 4 + m / 8));
        let mut l = m;
        while l >= 4 {
            for j in 0..l / 2 {
                let (s, c) = (sign * core::f64::consts::TAU * j as f64 / l as f64).sin_cos();
                stages.push(T::from_precise(c));
                stages.push(T::from_precise(s));
            }
            l >>= 1;
        }

        let bits = m.trailing_zeros();
        let mut matrix = Vec::with_capacity(2 * n);
        for p in 0..m {
            let k1 = p.reverse_bits() >> (usize::BITS - bits);
            for b in 0..m {
                let e = (k1 * b) % n;
                let (s, c) = (sign * core::f64::consts::TAU * e as f64 / n as f64).sin_cos();
                matrix.push(T::from_precise(c));
                matrix.push(T::from_precise(s));
            }
        }
        Self { stages, matrix }
    }
}

type ResidentCache<T> = RefCell<HashMap<(usize, bool), Arc<ResidentPlan<T>>>>;

thread_local! {
    static RESIDENT_CACHE_F64: ResidentCache<f64> = RefCell::new(HashMap::new());
    static RESIDENT_CACHE_F32: ResidentCache<f32> = RefCell::new(HashMap::new());
}

/// Scalars whose resident plans are cached per thread.
pub(crate) trait ResidentPlanCache:
    MixedRadixScalar + LaneScalar + bytemuck::Pod + Sized
{
    fn cached_resident_plan<const INVERSE: bool>(n: usize) -> Arc<ResidentPlan<Self>>;
}

macro_rules! impl_resident_cache {
    ($t:ty, $cache:ident) => {
        impl ResidentPlanCache for $t {
            fn cached_resident_plan<const INVERSE: bool>(n: usize) -> Arc<ResidentPlan<Self>> {
                $cache.with(|c| {
                    let key = (n, INVERSE);
                    if let Some(plan) = c.borrow().get(&key) {
                        return Arc::clone(plan);
                    }
                    let plan = Arc::new(ResidentPlan::<$t>::new::<INVERSE>(n));
                    c.borrow_mut().insert(key, Arc::clone(&plan));
                    plan
                })
            }
        }
    };
}

impl_resident_cache!(f64, RESIDENT_CACHE_F64);
impl_resident_cache!(f32, RESIDENT_CACHE_F32);

#[expect(
    clippy::inline_always,
    reason = "must fold into the caller's target-feature scope; an outlined \
              call would re-enter baseline codegen per load"
)]
#[inline(always)]
fn load<T, A>(data: &[T], at: usize) -> ComplexReg<T, A>
where
    T: LaneScalar,
    A: SimdArch + SimdKernel<T>,
{
    ComplexReg::from_interleaved(
        hermes_simd::Vector::load_unaligned_from_slice(
            &data[at..at + <A as SimdStorage<T>>::LANE_COUNT],
        )
        .expect("invariant: the caller bounds `at` by the slice length"),
    )
}

/// Every row of the `m x m` matrix through the register-resident DIF
/// transform; `APPLY_MATRIX` multiplies the four-step matrix in at load.
struct ResidentRows<'a, T, const APPLY_MATRIX: bool> {
    /// Interleaved samples, `2 * m * m` lanes.
    data: &'a mut [T],
    plan: &'a ResidentPlan<T>,
}

impl<T, const APPLY_MATRIX: bool> LaneKernel<T> for ResidentRows<'_, T, APPLY_MATRIX>
where
    T: LaneScalar + MixedRadixScalar,
{
    /// Whether the dispatched width handled the rows.
    type Output = bool;

    #[expect(
        clippy::inline_always,
        reason = "the fully unrolled body exceeds the inline budget, and an                   outlined kernel falls out of the dispatcher's                   target-feature scope to baseline codegen — measured at                   thirty times slower before this attribute"
    )]
    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, _simd: Simd<T, A>) -> bool {
        // Two samples per register is the shape the sixteen-register budget
        // is derived for; other widths fall back to the batched route.
        if <A as SimdStorage<T>>::LANE_COUNT != 4 {
            return false;
        }
        let row_lanes = 2 * ROW;
        let sign_pattern = [
            T::from_precise(1.0),
            T::from_precise(1.0),
            T::from_precise(-1.0),
            T::from_precise(-1.0),
        ];
        let sign = hermes_simd::Vector::<T, A>::load_unaligned_from_slice(&sign_pattern)
            .expect("invariant: pattern holds one register");

        for row in 0..ROW {
            let base = row * row_lanes;
            // Sixteen named locals: the array form kept the working set on
            // the stack twice over (indexed array, then a closure taking it
            // by reference), measured at 2x and then 6x slower than the
            // streaming kernel. Named bindings are the one form the
            // register allocator reliably keeps in registers.
            let mut r0 = load::<T, A>(self.data, base);
            let mut r1 = load::<T, A>(self.data, base + 4);
            let mut r2 = load::<T, A>(self.data, base + 8);
            let mut r3 = load::<T, A>(self.data, base + 12);
            let mut r4 = load::<T, A>(self.data, base + 16);
            let mut r5 = load::<T, A>(self.data, base + 20);
            let mut r6 = load::<T, A>(self.data, base + 24);
            let mut r7 = load::<T, A>(self.data, base + 28);
            let mut r8 = load::<T, A>(self.data, base + 32);
            let mut r9 = load::<T, A>(self.data, base + 36);
            let mut r10 = load::<T, A>(self.data, base + 40);
            let mut r11 = load::<T, A>(self.data, base + 44);
            let mut r12 = load::<T, A>(self.data, base + 48);
            let mut r13 = load::<T, A>(self.data, base + 52);
            let mut r14 = load::<T, A>(self.data, base + 56);
            let mut r15 = load::<T, A>(self.data, base + 60);
            if APPLY_MATRIX {
                r0 = r0 * load::<T, A>(&self.plan.matrix, base);
                r1 = r1 * load::<T, A>(&self.plan.matrix, base + 4);
                r2 = r2 * load::<T, A>(&self.plan.matrix, base + 8);
                r3 = r3 * load::<T, A>(&self.plan.matrix, base + 12);
                r4 = r4 * load::<T, A>(&self.plan.matrix, base + 16);
                r5 = r5 * load::<T, A>(&self.plan.matrix, base + 20);
                r6 = r6 * load::<T, A>(&self.plan.matrix, base + 24);
                r7 = r7 * load::<T, A>(&self.plan.matrix, base + 28);
                r8 = r8 * load::<T, A>(&self.plan.matrix, base + 32);
                r9 = r9 * load::<T, A>(&self.plan.matrix, base + 36);
                r10 = r10 * load::<T, A>(&self.plan.matrix, base + 40);
                r11 = r11 * load::<T, A>(&self.plan.matrix, base + 44);
                r12 = r12 * load::<T, A>(&self.plan.matrix, base + 48);
                r13 = r13 * load::<T, A>(&self.plan.matrix, base + 52);
                r14 = r14 * load::<T, A>(&self.plan.matrix, base + 56);
                r15 = r15 * load::<T, A>(&self.plan.matrix, base + 60);
            }

            // Sample distance 16: stage table at offset 0.
            let (u0_8, v0_8) = (r0 + r8, r0 - r8);
            r0 = u0_8;
            r8 = v0_8 * load::<T, A>(&self.plan.stages, 0);
            let (u1_9, v1_9) = (r1 + r9, r1 - r9);
            r1 = u1_9;
            r9 = v1_9 * load::<T, A>(&self.plan.stages, 4);
            let (u2_10, v2_10) = (r2 + r10, r2 - r10);
            r2 = u2_10;
            r10 = v2_10 * load::<T, A>(&self.plan.stages, 8);
            let (u3_11, v3_11) = (r3 + r11, r3 - r11);
            r3 = u3_11;
            r11 = v3_11 * load::<T, A>(&self.plan.stages, 12);
            let (u4_12, v4_12) = (r4 + r12, r4 - r12);
            r4 = u4_12;
            r12 = v4_12 * load::<T, A>(&self.plan.stages, 16);
            let (u5_13, v5_13) = (r5 + r13, r5 - r13);
            r5 = u5_13;
            r13 = v5_13 * load::<T, A>(&self.plan.stages, 20);
            let (u6_14, v6_14) = (r6 + r14, r6 - r14);
            r6 = u6_14;
            r14 = v6_14 * load::<T, A>(&self.plan.stages, 24);
            let (u7_15, v7_15) = (r7 + r15, r7 - r15);
            r7 = u7_15;
            r15 = v7_15 * load::<T, A>(&self.plan.stages, 28);

            // Sample distance 8: stage table at offset 32.
            let (u0_4, v0_4) = (r0 + r4, r0 - r4);
            r0 = u0_4;
            r4 = v0_4 * load::<T, A>(&self.plan.stages, 32);
            let (u1_5, v1_5) = (r1 + r5, r1 - r5);
            r1 = u1_5;
            r5 = v1_5 * load::<T, A>(&self.plan.stages, 36);
            let (u2_6, v2_6) = (r2 + r6, r2 - r6);
            r2 = u2_6;
            r6 = v2_6 * load::<T, A>(&self.plan.stages, 40);
            let (u3_7, v3_7) = (r3 + r7, r3 - r7);
            r3 = u3_7;
            r7 = v3_7 * load::<T, A>(&self.plan.stages, 44);
            let (u8_12, v8_12) = (r8 + r12, r8 - r12);
            r8 = u8_12;
            r12 = v8_12 * load::<T, A>(&self.plan.stages, 32);
            let (u9_13, v9_13) = (r9 + r13, r9 - r13);
            r9 = u9_13;
            r13 = v9_13 * load::<T, A>(&self.plan.stages, 36);
            let (u10_14, v10_14) = (r10 + r14, r10 - r14);
            r10 = u10_14;
            r14 = v10_14 * load::<T, A>(&self.plan.stages, 40);
            let (u11_15, v11_15) = (r11 + r15, r11 - r15);
            r11 = u11_15;
            r15 = v11_15 * load::<T, A>(&self.plan.stages, 44);

            // Sample distance 4: stage table at offset 48.
            let (u0_2, v0_2) = (r0 + r2, r0 - r2);
            r0 = u0_2;
            r2 = v0_2 * load::<T, A>(&self.plan.stages, 48);
            let (u1_3, v1_3) = (r1 + r3, r1 - r3);
            r1 = u1_3;
            r3 = v1_3 * load::<T, A>(&self.plan.stages, 52);
            let (u4_6, v4_6) = (r4 + r6, r4 - r6);
            r4 = u4_6;
            r6 = v4_6 * load::<T, A>(&self.plan.stages, 48);
            let (u5_7, v5_7) = (r5 + r7, r5 - r7);
            r5 = u5_7;
            r7 = v5_7 * load::<T, A>(&self.plan.stages, 52);
            let (u8_10, v8_10) = (r8 + r10, r8 - r10);
            r8 = u8_10;
            r10 = v8_10 * load::<T, A>(&self.plan.stages, 48);
            let (u9_11, v9_11) = (r9 + r11, r9 - r11);
            r9 = u9_11;
            r11 = v9_11 * load::<T, A>(&self.plan.stages, 52);
            let (u12_14, v12_14) = (r12 + r14, r12 - r14);
            r12 = u12_14;
            r14 = v12_14 * load::<T, A>(&self.plan.stages, 48);
            let (u13_15, v13_15) = (r13 + r15, r13 - r15);
            r13 = u13_15;
            r15 = v13_15 * load::<T, A>(&self.plan.stages, 52);

            // Sample distance 2: one shared [W_4^0, W_4^1] register.
            let tw2 = load::<T, A>(&self.plan.stages, 56);
            let (p0, q0) = (r0 + r1, r0 - r1);
            r0 = p0;
            r1 = q0 * tw2;
            let (p2, q2) = (r2 + r3, r2 - r3);
            r2 = p2;
            r3 = q2 * tw2;
            let (p4, q4) = (r4 + r5, r4 - r5);
            r4 = p4;
            r5 = q4 * tw2;
            let (p6, q6) = (r6 + r7, r6 - r7);
            r6 = p6;
            r7 = q6 * tw2;
            let (p8, q8) = (r8 + r9, r8 - r9);
            r8 = p8;
            r9 = q8 * tw2;
            let (p10, q10) = (r10 + r11, r10 - r11);
            r10 = p10;
            r11 = q10 * tw2;
            let (p12, q12) = (r12 + r13, r12 - r13);
            r12 = p12;
            r13 = q12 * tw2;
            let (p14, q14) = (r14 + r15, r14 - r15);
            r14 = p14;
            r15 = q14 * tw2;

            // Sample distance 1, twiddle one: swap-and-sign per register.
            r0 = r0.swap_samples() + ComplexReg::from_interleaved(r0.into_interleaved() * sign);
            r1 = r1.swap_samples() + ComplexReg::from_interleaved(r1.into_interleaved() * sign);
            r2 = r2.swap_samples() + ComplexReg::from_interleaved(r2.into_interleaved() * sign);
            r3 = r3.swap_samples() + ComplexReg::from_interleaved(r3.into_interleaved() * sign);
            r4 = r4.swap_samples() + ComplexReg::from_interleaved(r4.into_interleaved() * sign);
            r5 = r5.swap_samples() + ComplexReg::from_interleaved(r5.into_interleaved() * sign);
            r6 = r6.swap_samples() + ComplexReg::from_interleaved(r6.into_interleaved() * sign);
            r7 = r7.swap_samples() + ComplexReg::from_interleaved(r7.into_interleaved() * sign);
            r8 = r8.swap_samples() + ComplexReg::from_interleaved(r8.into_interleaved() * sign);
            r9 = r9.swap_samples() + ComplexReg::from_interleaved(r9.into_interleaved() * sign);
            r10 = r10.swap_samples() + ComplexReg::from_interleaved(r10.into_interleaved() * sign);
            r11 = r11.swap_samples() + ComplexReg::from_interleaved(r11.into_interleaved() * sign);
            r12 = r12.swap_samples() + ComplexReg::from_interleaved(r12.into_interleaved() * sign);
            r13 = r13.swap_samples() + ComplexReg::from_interleaved(r13.into_interleaved() * sign);
            r14 = r14.swap_samples() + ComplexReg::from_interleaved(r14.into_interleaved() * sign);
            r15 = r15.swap_samples() + ComplexReg::from_interleaved(r15.into_interleaved() * sign);

            r0.into_interleaved()
                .store_unaligned_to_slice(&mut self.data[base..base + 4])
                .expect("invariant: one register per slot");
            r1.into_interleaved()
                .store_unaligned_to_slice(&mut self.data[base + 4..base + 8])
                .expect("invariant: one register per slot");
            r2.into_interleaved()
                .store_unaligned_to_slice(&mut self.data[base + 8..base + 12])
                .expect("invariant: one register per slot");
            r3.into_interleaved()
                .store_unaligned_to_slice(&mut self.data[base + 12..base + 16])
                .expect("invariant: one register per slot");
            r4.into_interleaved()
                .store_unaligned_to_slice(&mut self.data[base + 16..base + 20])
                .expect("invariant: one register per slot");
            r5.into_interleaved()
                .store_unaligned_to_slice(&mut self.data[base + 20..base + 24])
                .expect("invariant: one register per slot");
            r6.into_interleaved()
                .store_unaligned_to_slice(&mut self.data[base + 24..base + 28])
                .expect("invariant: one register per slot");
            r7.into_interleaved()
                .store_unaligned_to_slice(&mut self.data[base + 28..base + 32])
                .expect("invariant: one register per slot");
            r8.into_interleaved()
                .store_unaligned_to_slice(&mut self.data[base + 32..base + 36])
                .expect("invariant: one register per slot");
            r9.into_interleaved()
                .store_unaligned_to_slice(&mut self.data[base + 36..base + 40])
                .expect("invariant: one register per slot");
            r10.into_interleaved()
                .store_unaligned_to_slice(&mut self.data[base + 40..base + 44])
                .expect("invariant: one register per slot");
            r11.into_interleaved()
                .store_unaligned_to_slice(&mut self.data[base + 44..base + 48])
                .expect("invariant: one register per slot");
            r12.into_interleaved()
                .store_unaligned_to_slice(&mut self.data[base + 48..base + 52])
                .expect("invariant: one register per slot");
            r13.into_interleaved()
                .store_unaligned_to_slice(&mut self.data[base + 52..base + 56])
                .expect("invariant: one register per slot");
            r14.into_interleaved()
                .store_unaligned_to_slice(&mut self.data[base + 56..base + 60])
                .expect("invariant: one register per slot");
            r15.into_interleaved()
                .store_unaligned_to_slice(&mut self.data[base + 60..base + 64])
                .expect("invariant: one register per slot");
        }
        true
    }
}

/// In-place square transpose of the `m x m` sample matrix, tiled.
fn transpose_samples<T: Copy>(data: &mut [Complex<T>], m: usize) {
    const TILE: usize = 8;
    for ib in (0..m).step_by(TILE) {
        let ie = (ib + TILE).min(m);
        for jb in (ib..m).step_by(TILE) {
            let je = (jb + TILE).min(m);
            for i in ib..ie {
                let start = if jb == ib { i + 1 } else { jb };
                for j in start.max(jb)..je {
                    data.swap(i * m + j, j * m + i);
                }
            }
        }
    }
}

/// The closing pass: transpose combined with both bit-reversals, in place.
///
/// Destination `(a, b)` takes its value from `(rev(b), rev(a))`; applying the
/// map twice is the identity, so it is pair swaps with the diagonal-like
/// fixed points (`b == rev(a)`) untouched.
fn untangle_output<T: Copy>(data: &mut [Complex<T>], m: usize) {
    let bits = m.trailing_zeros();
    let rev = |x: usize| x.reverse_bits() >> (usize::BITS - bits);
    for a in 0..m {
        for b in 0..m {
            let (c, d) = (rev(b), rev(a));
            if (a, b) < (c, d) {
                data.swap(a * m + b, c * m + d);
            }
        }
    }
}

/// In-place transform for `n = 1024` with register-resident rows, reporting
/// whether the dispatched width ran it.
pub(crate) fn four_step_resident<T, const INVERSE: bool>(data: &mut [Complex<T>]) -> bool
where
    T: ResidentPlanCache,
    Complex<T>: bytemuck::Pod,
{
    if data.len() != ROW * ROW {
        return false;
    }
    let plan = T::cached_resident_plan::<INVERSE>(ROW * ROW);

    #[cfg(all(test, windows, target_arch = "x86_64"))]
    macro_rules! sect {
        ($label:literal, $body:block) => {{
            let t0 = unsafe { core::arch::x86_64::_rdtsc() };
            let out = $body;
            let t1 = unsafe { core::arch::x86_64::_rdtsc() };
            if std::env::var_os("RESIDENT_SECTIONS").is_some() {
                eprintln!("RSECT {} {}", $label, t1 - t0);
            }
            out
        }};
    }
    #[cfg(not(all(test, windows, target_arch = "x86_64")))]
    macro_rules! sect {
        ($label:literal, $body:block) => {
            $body
        };
    }

    // The width check runs on an untouched buffer: the first row pass reports
    // before mutating only after the transpose, so probe the dispatch first
    // with a no-op row pass over a stack copy? Cheaper and exact: the row
    // kernels themselves gate on width, and the transpose that precedes them
    // is undone if they decline.
    sect!("t1", { transpose_samples(data, ROW) });
    {
        let flat: &mut [T] = bytemuck::cast_slice_mut(data);
        if !sect!("rows1", {
            hermes_simd::vectorize(ResidentRows::<T, false> {
                data: flat,
                plan: plan.as_ref(),
            })
        }) {
            transpose_samples(data, ROW);
            return false;
        }
    }
    sect!("t2", { transpose_samples(data, ROW) });
    {
        let flat: &mut [T] = bytemuck::cast_slice_mut(data);
        let handled = sect!("rows2", {
            hermes_simd::vectorize(ResidentRows::<T, true> {
                data: flat,
                plan: plan.as_ref(),
            })
        });
        debug_assert!(handled, "width accepted the first pass");
    }
    sect!("untangle", { untangle_output(data, ROW) });
    true
}

// Windows-gated: pins threads through Win32 to control the hybrid scheduler.
#[cfg(all(test, windows))]
mod pinned_probe;
#[cfg(test)]
mod tests;
