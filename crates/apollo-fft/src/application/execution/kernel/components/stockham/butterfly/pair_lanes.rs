//! The Stockham pair stage as one hermes lane kernel.
//!
//! [`stage_pair_impl`] fuses two radix-2 Stockham passes into one traversal
//! (its docs derive the four-output recurrence). This module is that
//! recurrence vectorised: each group's `k` loop runs `A::LANE_COUNT / 2`
//! complex samples per register through [`ComplexReg`] arithmetic, and the
//! ragged tail (fewer samples than a register holds) runs the scalar
//! recurrence. One generic body serves every lane width hermes offers, so
//! the per-ISA intrinsic copies of this stage are retired; the caller picks
//! the lane count its route was tuned for and falls back to
//! [`stage_pair_impl`] when no hardware backend serves it.
//!
//! Rounding matches the retired AVX/FMA stage bit-for-bit: [`ComplexReg`]'s
//! multiply is the same dup/swap/`fmaddsub` sequence with the same operand
//! order, and adds and subtracts are lane-wise IEEE operations in both.
//!
//! [`stage_pair_impl`]: super::stage::stage_pair_impl

use core::ops::{Add, Mul, Sub};

use eunomia::Complex;
use hermes_simd::{ComplexReg, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, Vector};

/// The general pair stage: `radix >= 2`, `src.len() / (2 * radix)` groups.
struct PairStage<'a, T> {
    src: &'a [Complex<T>],
    dst: &'a mut [Complex<T>],
    radix: usize,
    first_twiddles: &'a [Complex<T>],
    second_twiddles: &'a [Complex<T>],
}

/// The `radix == 1` pair stage, which needs only the second pass's twiddle.
struct PairStageRadixOne<'a, T> {
    src: &'a [Complex<T>],
    dst: &'a mut [Complex<T>],
    second_twiddles: &'a [Complex<T>],
}

impl<T> LaneKernel<T> for PairStage<'_, T>
where
    T: LaneScalar + bytemuck::Pod,
    Complex<T>: bytemuck::Pod
        + Add<Output = Complex<T>>
        + Sub<Output = Complex<T>>
        + Mul<Output = Complex<T>>,
{
    type Output = ();

    #[inline]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
        let Self {
            src,
            dst,
            radix,
            first_twiddles,
            second_twiddles,
        } = self;
        let n = src.len();
        let groups = n / (radix << 1);
        let half_groups = groups >> 1;
        let quarter_n = n >> 2;
        let half_n = n >> 1;
        let per_register = A::LANE_COUNT / 2;
        let vector_end = half_groups & !(per_register - 1);
        let src_view = simd.view(bytemuck::cast_slice::<Complex<T>, T>(src));

        for j in 0..radix {
            let w1 = first_twiddles[j];
            let w2 = second_twiddles[j];
            let w3 = second_twiddles[j + radix];
            let src_base = j * groups * 2;
            let dst_base = j * half_groups;

            if vector_end > 0 {
                // The vector loop runs only when `half_groups >= per_register`;
                // both are powers of two, so every offset below — multiples of
                // `half_groups` plus a `k` stepped by `per_register` — is a
                // whole register index.
                let w1v = ComplexReg::<T, A>::splat(w1);
                let w2v = ComplexReg::<T, A>::splat(w2);
                let w3v = ComplexReg::<T, A>::splat(w3);
                let load = |offset: usize| {
                    ComplexReg::from_interleaved(Vector::from_view_chunk(
                        &src_view,
                        offset / per_register,
                    ))
                };
                let mut dst_view = simd.view_mut(bytemuck::cast_slice_mut::<Complex<T>, T>(dst));
                for k in (0..vector_end).step_by(per_register) {
                    let x0 = load(src_base + k);
                    let x1 = load(src_base + half_groups + k);
                    let x2 = load(src_base + groups + k) * w1v;
                    let x3 = load(src_base + groups + half_groups + k) * w1v;
                    let a0 = x0 + x2;
                    let a1 = x1 + x3;
                    let b0 = x0 - x2;
                    let b1 = x1 - x3;
                    let c0 = a1 * w2v;
                    let c1 = b1 * w3v;
                    let mut store = |offset: usize, value: ComplexReg<T, A>| {
                        value
                            .into_interleaved()
                            .store_to_view_chunk(&mut dst_view, offset / per_register);
                    };
                    store(dst_base + k, a0 + c0);
                    store(dst_base + half_n + k, a0 - c0);
                    store(dst_base + quarter_n + k, b0 + c1);
                    store(dst_base + half_n + quarter_n + k, b0 - c1);
                }
            }

            for k in vector_end..half_groups {
                let x0 = src[src_base + k];
                let x1 = src[src_base + half_groups + k];
                let x2 = src[src_base + groups + k] * w1;
                let x3 = src[src_base + groups + half_groups + k] * w1;
                let a0 = x0 + x2;
                let a1 = x1 + x3;
                let b0 = x0 - x2;
                let b1 = x1 - x3;
                let c0 = a1 * w2;
                let c1 = b1 * w3;
                dst[dst_base + k] = a0 + c0;
                dst[dst_base + half_n + k] = a0 - c0;
                dst[dst_base + quarter_n + k] = b0 + c1;
                dst[dst_base + half_n + quarter_n + k] = b0 - c1;
            }
        }
    }
}

impl<T> LaneKernel<T> for PairStageRadixOne<'_, T>
where
    T: LaneScalar + bytemuck::Pod,
    Complex<T>: bytemuck::Pod
        + Add<Output = Complex<T>>
        + Sub<Output = Complex<T>>
        + Mul<Output = Complex<T>>,
{
    type Output = ();

    #[inline]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
        let Self {
            src,
            dst,
            second_twiddles,
        } = self;
        let n = src.len();
        let quarter_n = n >> 2;
        let half_n = n >> 1;
        let per_register = A::LANE_COUNT / 2;
        let vector_end = quarter_n & !(per_register - 1);
        let w3 = second_twiddles[1];

        if vector_end > 0 {
            // As in `PairStage`: `quarter_n >= per_register`, both powers of
            // two, so `quarter_n`, `half_n`, and each `k` are register indices.
            let w3v = ComplexReg::<T, A>::splat(w3);
            let src_view = simd.view(bytemuck::cast_slice::<Complex<T>, T>(src));
            let mut dst_view = simd.view_mut(bytemuck::cast_slice_mut::<Complex<T>, T>(dst));
            let load = |offset: usize| {
                ComplexReg::from_interleaved(Vector::from_view_chunk(
                    &src_view,
                    offset / per_register,
                ))
            };
            for k in (0..vector_end).step_by(per_register) {
                let x0 = load(k);
                let x1 = load(quarter_n + k);
                let x2 = load(half_n + k);
                let x3 = load(half_n + quarter_n + k);
                let a0 = x0 + x2;
                let a1 = x1 + x3;
                let b0 = x0 - x2;
                let b1 = x1 - x3;
                let c1 = b1 * w3v;
                let mut store = |offset: usize, value: ComplexReg<T, A>| {
                    value
                        .into_interleaved()
                        .store_to_view_chunk(&mut dst_view, offset / per_register);
                };
                store(k, a0 + a1);
                store(half_n + k, a0 - a1);
                store(quarter_n + k, b0 + c1);
                store(half_n + quarter_n + k, b0 - c1);
            }
        }

        for k in vector_end..quarter_n {
            let x0 = src[k];
            let x1 = src[quarter_n + k];
            let x2 = src[half_n + k];
            let x3 = src[half_n + quarter_n + k];
            let a0 = x0 + x2;
            let a1 = x1 + x3;
            let b0 = x0 - x2;
            let b1 = x1 - x3;
            let c1 = b1 * w3;
            dst[k] = a0 + a1;
            dst[half_n + k] = a0 - a1;
            dst[quarter_n + k] = b0 + c1;
            dst[half_n + quarter_n + k] = b0 - c1;
        }
    }
}

/// Runs the general pair stage (`radix >= 2`) at `LANES` scalar lanes per
/// register on the hardware backend of that width.
///
/// Returns `false`, having touched nothing, when this host has no hardware
/// backend at `LANES`; the caller then takes its scalar route.
#[inline]
pub(crate) fn stage_pair_lanes<T, const LANES: usize>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    radix: usize,
    first_twiddles: &[Complex<T>],
    second_twiddles: &[Complex<T>],
) -> bool
where
    T: LaneScalar + bytemuck::Pod,
    Complex<T>: bytemuck::Pod
        + Add<Output = Complex<T>>
        + Sub<Output = Complex<T>>
        + Mul<Output = Complex<T>>,
{
    hermes_simd::vectorize_hardware_lanes::<LANES, T, _>(PairStage {
        src,
        dst,
        radix,
        first_twiddles,
        second_twiddles,
    })
    .is_some()
}

/// [`stage_pair_lanes`] for `radix == 1`, where only `second_twiddles[1]`
/// enters the recurrence.
#[inline]
pub(crate) fn stage_pair_radix_one_lanes<T, const LANES: usize>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    second_twiddles: &[Complex<T>],
) -> bool
where
    T: LaneScalar + bytemuck::Pod,
    Complex<T>: bytemuck::Pod
        + Add<Output = Complex<T>>
        + Sub<Output = Complex<T>>
        + Mul<Output = Complex<T>>,
{
    hermes_simd::vectorize_hardware_lanes::<LANES, T, _>(PairStageRadixOne {
        src,
        dst,
        second_twiddles,
    })
    .is_some()
}

#[cfg(test)]
mod tests {
    use super::super::stage::stage_pair_impl;
    use super::{stage_pair_lanes, stage_pair_radix_one_lanes};
    use eunomia::Complex;

    /// Deterministic samples in `[-1, 1]` from a 64-bit LCG (Knuth MMIX
    /// constants), so a failure replays without a seed file.
    fn samples<T: From<f32>>(count: usize, seed: u64) -> Vec<Complex<T>> {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // The top 24 bits give a uniform f32 in [0, 1); map to [-1, 1].
            ((state >> 40) as f32 / (1u32 << 24) as f32).mul_add(2.0, -1.0)
        };
        (0..count)
            .map(|_| Complex::new(T::from(next()), T::from(next())))
            .collect()
    }

    /// Unit-modulus twiddles `exp(-2πi k / (2 * radix))`, `count` of them.
    fn twiddles<T: From<f32>>(radix: usize, count: usize) -> Vec<Complex<T>> {
        (0..count)
            .map(|k| {
                let angle = -core::f64::consts::TAU * k as f64 / (2 * radix) as f64;
                Complex::new(T::from(angle.cos() as f32), T::from(angle.sin() as f32))
            })
            .collect()
    }

    /// Each output is `a ± w·b` with `a`, `b` themselves `x ± w'·y`: at most
    /// four unit-twiddled inputs, so `|out| ≤ 4·max|x| = 4`. The lane route
    /// spends at most six roundings per output (two complex products, each
    /// one product plus one FMA, and four adds) and the scalar route at most
    /// eight; the routes' difference is bounded by the sum, 14 · ε · 4.
    fn tolerance<T: Into<f64>>(epsilon: T) -> f64 {
        14.0 * 4.0 * epsilon.into()
    }

    fn assert_close<T: Copy + Into<f64>>(lanes: &[Complex<T>], scalar: &[Complex<T>], tol: f64) {
        for (index, (got, want)) in lanes.iter().zip(scalar).enumerate() {
            let dre = (got.re.into() - want.re.into()).abs();
            let dim = (got.im.into() - want.im.into()).abs();
            assert!(
                dre <= tol && dim <= tol,
                "sample {index}: lanes {:?} scalar {:?} (tolerance {tol:e})",
                (got.re.into(), got.im.into()),
                (want.re.into(), want.im.into())
            );
        }
    }

    /// `(n, radix)` pairs spanning the vector loop with and without a tail,
    /// and the all-tail case (`half_groups < per_register` at both widths).
    const GENERAL_CASES: &[(usize, usize)] = &[
        (16, 2),
        (32, 2),
        (64, 2),
        (64, 4),
        (128, 8),
        (1024, 2),
        (1024, 16),
        (1024, 128),
        (4096, 64),
    ];

    const RADIX_ONE_SIZES: &[usize] = &[8, 16, 32, 64, 1024, 4096];

    #[test]
    fn general_stage_matches_scalar_recurrence_at_both_precisions() {
        for &(n, radix) in GENERAL_CASES {
            let src32 = samples::<f32>(n, 0x9E37_79B9_7F4A_7C15 ^ n as u64);
            let first32 = twiddles::<f32>(radix, radix);
            let second32 = twiddles::<f32>(2 * radix, 2 * radix);
            let mut want32 = vec![Complex::new(0.0f32, 0.0); n];
            stage_pair_impl::<_, 1024>(&src32, &mut want32, radix, &first32, &second32);
            let mut got32 = vec![Complex::new(0.0f32, 0.0); n];
            assert!(
                stage_pair_lanes::<f32, 8>(&src32, &mut got32, radix, &first32, &second32),
                "eight-lane f32 backend absent on this host"
            );
            assert_close(&got32, &want32, tolerance(f32::EPSILON));

            let src64 = samples::<f64>(n, 0xD1B5_4A32_D192_ED03 ^ n as u64);
            let first64 = twiddles::<f64>(radix, radix);
            let second64 = twiddles::<f64>(2 * radix, 2 * radix);
            let mut want64 = vec![Complex::new(0.0f64, 0.0); n];
            stage_pair_impl::<_, 512>(&src64, &mut want64, radix, &first64, &second64);
            let mut got64 = vec![Complex::new(0.0f64, 0.0); n];
            assert!(
                stage_pair_lanes::<f64, 4>(&src64, &mut got64, radix, &first64, &second64),
                "four-lane f64 backend absent on this host"
            );
            assert_close(&got64, &want64, tolerance(f64::EPSILON));

            // The AVX-512 width is host-dependent; when served it must agree too.
            let mut got512 = vec![Complex::new(0.0f64, 0.0); n];
            if stage_pair_lanes::<f64, 8>(&src64, &mut got512, radix, &first64, &second64) {
                assert_close(&got512, &want64, tolerance(f64::EPSILON));
            }
        }
    }

    #[test]
    fn radix_one_stage_matches_scalar_recurrence_at_both_precisions() {
        for &n in RADIX_ONE_SIZES {
            let src32 = samples::<f32>(n, 0x2545_F491_4F6C_DD1D ^ n as u64);
            let second32 = twiddles::<f32>(2, 2);
            let mut want32 = vec![Complex::new(0.0f32, 0.0); n];
            stage_pair_impl::<_, 1024>(&src32, &mut want32, 1, &second32, &second32);
            let mut got32 = vec![Complex::new(0.0f32, 0.0); n];
            assert!(stage_pair_radix_one_lanes::<f32, 8>(
                &src32, &mut got32, &second32
            ));
            assert_close(&got32, &want32, tolerance(f32::EPSILON));

            let src64 = samples::<f64>(n, 0x853C_49E6_748F_EA9B ^ n as u64);
            let second64 = twiddles::<f64>(2, 2);
            let mut want64 = vec![Complex::new(0.0f64, 0.0); n];
            stage_pair_impl::<_, 512>(&src64, &mut want64, 1, &second64, &second64);
            let mut got64 = vec![Complex::new(0.0f64, 0.0); n];
            assert!(stage_pair_radix_one_lanes::<f64, 4>(
                &src64, &mut got64, &second64
            ));
            assert_close(&got64, &want64, tolerance(f64::EPSILON));
        }
    }

    #[test]
    fn unserved_lane_width_touches_nothing() {
        let src = samples::<f32>(64, 7);
        let second = twiddles::<f32>(2, 2);
        let mut dst = vec![Complex::new(7.0f32, -7.0); 64];
        // No backend offers 64 f32 lanes; the caller keeps its scalar route.
        assert!(!stage_pair_radix_one_lanes::<f32, 64>(
            &src, &mut dst, &second
        ));
        assert!(dst.iter().all(|c| c.re == 7.0 && c.im == -7.0));
    }
}
