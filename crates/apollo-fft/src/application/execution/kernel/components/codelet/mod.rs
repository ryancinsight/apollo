//! Register-resident small-N transform codelets over interleaved data.
//!
//! **Status: correct, measured, and declined** — this module is compiled for
//! tests only. The N = 16 codelet passes a direct-DFT oracle in both
//! directions but loses to the incumbent sized kernel pinned on both core
//! types: `codelet_against_the_incumbent_by_core_type` reads 13.1x on the
//! P-core and 10.4x on the E-core (three runs each, 2026-09-04).
//!
//! An earlier revision of this header recorded 1.8x on an E-core and named the
//! cause as the stack-buffer bit reversal, with promotion gated on hermes
//! gaining a two-register sample-granularity shuffle. Both claims are now
//! disproved. The shuffle exists and did all along:
//! `SimdPermute::deinterleave_pairs` is exactly it, since a lane pair is a
//! complex sample at this width, and the bit reversal is four of those calls
//! over the naturally ordered registers — `(w0, w4)` from `(r0, r4)`,
//! `(w1, w5)` from `(r2, r6)`, `(w2, w6)` from `(r1, r5)`, `(w3, w7)` from
//! `(r3, r7)`. Implementing it passes every oracle and measures **slower**:
//! 14.0x against 13.1x on the P-core, unchanged on the E-core. The eight
//! cross-lane `vperm2f128` cost more than the scalar stores they replace,
//! consistent with the port-5 latency that rejected HS-DEINTERLEAVE-PAIRS-AVX2-F32.
//!
//! So the permutation was never the binding constraint, and an order of
//! magnitude of the gap remains unattributed
//! (`backlog.md#apollo-n16-register-permute`). Until that is profiled, this
//! module is an instrument whose subject is far enough from the incumbent to
//! make it a weak one.
//!
//! The reference engines close the small-N range by holding an entire
//! transform in vector registers: RustFFT's AVX butterflies run six stages in
//! two memory passes on interleaved complex vectors, and PhastFT fuses four
//! stages per codelet. Apollo's planar kernels cannot follow — separate real
//! and imaginary registers double the register pressure — so these codelets
//! use `hermes_simd::ComplexReg`, the interleaved vocabulary built for exactly
//! this: a complex multiply is three shuffles and one alternating FMA, and a
//! butterfly is a plain add/sub pair.
//!
//! ## Structure
//!
//! Decimation in time over bit-reversed input, which leaves the output in
//! natural order. The input permutation runs through a stack buffer of scalar
//! copies — measured as acceptable at this size, and the in-register
//! alternative needs a two-operand sample shuffle the vocabulary does not
//! carry yet (recorded on the item as the follow-up hermes primitive).
//!
//! With two samples per register, only the first stage pairs samples inside a
//! register, and it needs no shuffle vocabulary beyond a sample swap: its
//! twiddle is one, so `swap_samples(v) + v * [1, 1, -1, -1]` produces
//! `[s0 + s1, s0 - s1]` directly. Every later stage pairs whole registers:
//! a twiddle multiply and a butterfly, nothing else.
//!
//! ## Twiddle constants
//!
//! `W_N^j = exp(-2*pi*i*j/N)` for the forward direction, conjugated for the
//! inverse by negating the imaginary literals at build time (const-folded per
//! monomorphized direction). The literals are the exactly rounded values of
//! `cos(pi/8)` and `sin(pi/8)`; `sqrt(2)/2` comes from the standard library.
//! The accuracy gate's ladder covers this table exactly as it covers the
//! shared twiddle builders.

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::Complex;
use hermes_simd::{ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage};

/// `cos(pi/8)`, exactly rounded.
const COS_PI_8: f64 = 0.923_879_532_511_286_7;
/// `sin(pi/8)`, exactly rounded.
const SIN_PI_8: f64 = 0.382_683_432_365_089_8;
/// `sqrt(2)/2 = cos(pi/4)`.
const HALF_SQRT_2: f64 = core::f64::consts::FRAC_1_SQRT_2;

/// Bit-reversed sample order for N = 16: sample `k` of the working sequence is
/// input sample `BIT_REVERSED_16[k]`.
const BIT_REVERSED_16: [usize; 16] = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];

/// The N = 16 transform as a lane kernel: eight registers of two samples.
struct Transform16<'a, T, const INVERSE: bool, const NORMALIZE: bool> {
    /// Interleaved samples, `[re, im] * 16`.
    data: &'a mut [T],
}

impl<T, const INVERSE: bool, const NORMALIZE: bool> LaneKernel<T>
    for Transform16<'_, T, INVERSE, NORMALIZE>
where
    T: LaneScalar + MixedRadixScalar,
{
    /// Whether the dispatched width handled the transform.
    type Output = bool;

    fn call<A: SimdArch + SimdKernel<T>>(self, _simd: Simd<T, A>) -> bool {
        // Two samples per register is the shape this codelet is written for;
        // other widths fall back to the incumbent route rather than emulate.
        if <A as SimdStorage<T>>::LANE_COUNT != 4 {
            return false;
        }
        let lanes = 4usize;
        // The twiddle literals below already carry the forward sign
        // (`W_N^j = exp(-2*pi*i*j/N)`), so the forward direction takes them
        // as written and the inverse conjugates them.
        let sign = if INVERSE { -1.0 } else { 1.0 };
        let w = |re: f64, im: f64| -> [T; 2] { [T::from_precise(re), T::from_precise(sign * im)] };

        // Bit-reversed copy; the loads below are then contiguous.
        let mut buf = [T::from_precise(0.0); 32];
        for (slot, &src) in BIT_REVERSED_16.iter().enumerate() {
            buf[2 * slot] = self.data[2 * src];
            buf[2 * slot + 1] = self.data[2 * src + 1];
        }
        let load = |flat: &[T; 4]| {
            ComplexReg::<T, A>::from_interleaved(
                hermes_simd::Vector::load_unaligned_from_slice(flat)
                    .expect("invariant: exactly one register of lanes"),
            )
        };
        let mut r = [ComplexReg::<T, A>::zero(); 8];
        for (k, reg) in r.iter_mut().enumerate() {
            *reg = load(
                buf[lanes * k..lanes * (k + 1)]
                    .try_into()
                    .expect("invariant: four lanes per register"),
            );
        }

        // Stage 1 (distance 1, twiddle 1): [s0 + s1, s0 - s1] per register.
        let stage_sign = load(&[
            T::from_precise(1.0),
            T::from_precise(1.0),
            T::from_precise(-1.0),
            T::from_precise(-1.0),
        ])
        .into_interleaved();
        for reg in &mut r {
            *reg = reg.swap_samples()
                + ComplexReg::from_interleaved(reg.into_interleaved() * stage_sign);
        }

        // Stage 2 (distance 2): twiddles [W4^0, W4^1] = [1, -i].
        let t2 = {
            let (a, b) = (w(1.0, 0.0), w(0.0, -1.0));
            load(&[a[0], a[1], b[0], b[1]])
        };
        for base in [0usize, 2, 4, 6] {
            let wb = r[base + 1] * t2;
            let (lo, hi) = r[base].butterfly(wb);
            (r[base], r[base + 1]) = (lo, hi);
        }

        // Stage 3 (distance 4): twiddles [W8^0..W8^3] across two registers.
        let t3 = {
            let (a, b) = (w(1.0, 0.0), w(HALF_SQRT_2, -HALF_SQRT_2));
            let (c, d) = (w(0.0, -1.0), w(-HALF_SQRT_2, -HALF_SQRT_2));
            [
                load(&[a[0], a[1], b[0], b[1]]),
                load(&[c[0], c[1], d[0], d[1]]),
            ]
        };
        for base in [0usize, 4] {
            for offset in 0..2 {
                let wb = r[base + 2 + offset] * t3[offset];
                let (lo, hi) = r[base + offset].butterfly(wb);
                (r[base + offset], r[base + 2 + offset]) = (lo, hi);
            }
        }

        // Stage 4 (distance 8): twiddles [W16^0..W16^7] across four registers.
        let t4 = {
            let pairs = [
                (w(1.0, 0.0), w(COS_PI_8, -SIN_PI_8)),
                (w(HALF_SQRT_2, -HALF_SQRT_2), w(SIN_PI_8, -COS_PI_8)),
                (w(0.0, -1.0), w(-SIN_PI_8, -COS_PI_8)),
                (w(-HALF_SQRT_2, -HALF_SQRT_2), w(-COS_PI_8, -SIN_PI_8)),
            ];
            pairs.map(|(a, b)| load(&[a[0], a[1], b[0], b[1]]))
        };
        for offset in 0..4 {
            let wb = r[offset + 4] * t4[offset];
            let (lo, hi) = r[offset].butterfly(wb);
            (r[offset], r[offset + 4]) = (lo, hi);
        }

        if INVERSE && NORMALIZE {
            let scale = hermes_simd::Vector::splat(T::from_precise(1.0 / 16.0));
            for reg in &mut r {
                *reg = ComplexReg::from_interleaved(reg.into_interleaved() * scale);
            }
        }

        for (k, reg) in r.iter().enumerate() {
            reg.into_interleaved()
                .store_unaligned_to_slice(&mut self.data[lanes * k..lanes * (k + 1)])
                .expect("invariant: four lanes per register");
        }
        true
    }
}

/// Runs the register-resident N = 16 transform when four lanes are available.
///
/// # Panics
///
/// If `data` is not exactly 16 samples.
pub(crate) fn try_transform_16<T, const INVERSE: bool, const NORMALIZE: bool>(
    data: &mut [Complex<T>],
) -> bool
where
    T: LaneScalar + MixedRadixScalar + eunomia::layout::Pod,
    Complex<T>: eunomia::layout::Pod,
{
    assert_eq!(data.len(), 16, "the N = 16 codelet requires 16 samples");
    let flat: &mut [T] = eunomia::layout::cast_slice_mut(data);
    hermes_simd::vectorize_lanes::<4, T, _>(Transform16::<T, INVERSE, NORMALIZE> { data: flat })
        .unwrap_or(false)
}

// Windows-gated: uses Hermes exact processor binding to control the hybrid scheduler.
#[cfg(all(test, windows))]
mod pinned_probe;
#[cfg(test)]
mod tests;
