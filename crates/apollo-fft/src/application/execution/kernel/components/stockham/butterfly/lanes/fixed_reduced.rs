//! The fixed 64-point f32 leaf as a lane kernel.
//!
//! The same `N = 8 · 8` factorisation as the f64 leaf: phase one runs a
//! register-resident DFT-8 down each column, multiplies by `W_64^{pc}`, and
//! stores row-major; phase two runs a DFT-8 across each row. A register holds
//! four adjacent complex samples, so phase one covers four columns at once and
//! phase two four rows; the transpose between the phases is one 4×4 sample
//! transpose per register quad. The twiddles are the exact constant tables
//! rounded once to `f32`.

use eunomia::{Complex32, Complex64};
use hermes_simd::{ComplexReg, LaneKernel, Simd, SimdArch, SimdKernel, Vector};

use crate::application::execution::kernel::components::register_butterfly::radix8;
use crate::application::execution::kernel::mixed_radix::scalar::twiddle_constants::{
    TWIDDLES_64_FWD, TWIDDLES_64_INV,
};

struct Dft64Reduced<'data, const INVERSE: bool> {
    data: &'data mut [Complex32; 64],
}

#[expect(
    clippy::inline_always,
    reason = "the leaf must remain in the selected target-feature frame"
)]
#[inline(always)]
fn load<A>(simd: Simd<f32, A>, values: &[Complex32]) -> ComplexReg<f32, A>
where
    A: SimdArch + SimdKernel<f32>,
{
    let view = simd.view(eunomia::layout::cast_slice(values));
    ComplexReg::from_interleaved(Vector::from_view_chunk(&view, 0))
}

#[expect(
    clippy::inline_always,
    reason = "stores must remain in the selected target-feature frame"
)]
#[inline(always)]
fn store<A>(value: ComplexReg<f32, A>, destination: &mut [Complex32])
where
    A: SimdArch + SimdKernel<f32>,
{
    value
        .into_interleaved()
        .store_unaligned_to_slice(eunomia::layout::cast_slice_mut(destination))
        .expect("invariant: four complex samples fill eight lanes");
}

/// `W_64^k` from the exact table, rounded once to `f32`.
#[expect(
    clippy::inline_always,
    reason = "constant twiddles must fold into the selected target-feature frame"
)]
#[inline(always)]
fn twiddle<const INVERSE: bool>(k: usize) -> Complex32 {
    let table: &[Complex64; 64] = if INVERSE {
        &TWIDDLES_64_INV
    } else {
        &TWIDDLES_64_FWD
    };
    let value = table[k & 63];
    Complex32::new(value.re as f32, value.im as f32)
}

/// `x[8r + c]`, eight rows by eight columns, four columns per register.
#[expect(
    clippy::inline_always,
    reason = "the leaf must remain in the selected target-feature frame"
)]
#[inline(always)]
fn dft64_kernel<A, const INVERSE: bool>(simd: Simd<f32, A>, data: &mut [Complex32; 64])
where
    A: SimdArch + SimdKernel<f32>,
{
    let half_root2 = simd.splat(core::f32::consts::FRAC_1_SQRT_2);
    let mut mid = [Complex32::new(0.0, 0.0); 64];
    for quad in 0..2 {
        let c0 = 4 * quad;
        let rows = core::array::from_fn(|r| load(simd, &data[8 * r + c0..8 * r + c0 + 4]));
        let mut column = radix8::<f32, A, INVERSE>(rows, half_root2);
        for (p, value) in column.iter_mut().enumerate().skip(1) {
            let twiddles = [
                twiddle::<INVERSE>(p * c0),
                twiddle::<INVERSE>(p * (c0 + 1)),
                twiddle::<INVERSE>(p * (c0 + 2)),
                twiddle::<INVERSE>(p * (c0 + 3)),
            ];
            *value = *value * load(simd, &twiddles);
        }
        for (p, value) in column.into_iter().enumerate() {
            store(value, &mut mid[8 * p + c0..8 * p + c0 + 4]);
        }
    }
    for quad in 0..2 {
        let p0 = 4 * quad;
        let mut columns = [ComplexReg::<f32, A>::zero(); 8];
        for j in 0..2 {
            let c0 = 4 * j;
            let mut block: [ComplexReg<f32, A>; 4] = core::array::from_fn(|i| {
                load(simd, &mid[8 * (p0 + i) + c0..8 * (p0 + i) + c0 + 4])
            });
            ComplexReg::transpose_square(&mut block);
            columns[c0..c0 + 4].copy_from_slice(&block);
        }
        let out = radix8::<f32, A, INVERSE>(columns, half_root2);
        for (q, value) in out.into_iter().enumerate() {
            store(value, &mut data[8 * q + p0..8 * q + p0 + 4]);
        }
    }
}

impl<const INVERSE: bool> LaneKernel<f32> for Dft64Reduced<'_, INVERSE> {
    type Output = ();

    #[inline]
    fn call<A: SimdArch + SimdKernel<f32>>(self, simd: Simd<f32, A>) {
        debug_assert_eq!(
            A::LANE_COUNT,
            8,
            "invariant: the f32 leaf holds four samples per register"
        );
        dft64_kernel::<A, INVERSE>(simd, self.data);
    }
}

/// Runs the 64-point f32 leaf on the eight-lane hardware backend.
///
/// The direction follows the plan's twiddle table exactly as the staged route
/// reads it. Returns `false`, having touched nothing, when the host has no
/// eight-lane f32 backend or `data` is not 64 samples.
#[inline]
pub(crate) fn fixed_len64_reduced_lanes(data: &mut [Complex32], twiddles: &[Complex32]) -> bool {
    let Ok(data) = <&mut [Complex32; 64]>::try_from(data) else {
        return false;
    };
    if twiddles.get(2).is_some_and(|w| w.im > 0.0) {
        hermes_simd::vectorize_hardware_lanes::<8, f32, _>(Dft64Reduced::<true> { data }).is_some()
    } else {
        hermes_simd::vectorize_hardware_lanes::<8, f32, _>(Dft64Reduced::<false> { data }).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::fixed_len64_reduced_lanes;
    use eunomia::Complex32;

    /// Deterministic samples in `[-1, 1]` (Knuth MMIX LCG).
    fn samples(count: usize, seed: u64) -> Vec<Complex32> {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u32 << 24) as f32).mul_add(2.0, -1.0)
        };
        (0..count).map(|_| Complex32::new(next(), next())).collect()
    }

    /// The plan's f32 table for `direction` (`-1.0` forward, `+1.0` inverse).
    fn twiddles(n: usize, direction: f64) -> Vec<Complex32> {
        (0..n)
            .map(|k| {
                let angle = direction * core::f64::consts::TAU * k as f64 / n as f64;
                Complex32::new(angle.cos() as f32, angle.sin() as f32)
            })
            .collect()
    }

    /// The DFT sum in f64 with exact per-term angles: an oracle independent of
    /// any factorisation and of f32 rounding.
    fn naive_dft(x: &[Complex32], direction: f64) -> Vec<(f64, f64)> {
        let n = x.len();
        (0..n)
            .map(|k| {
                x.iter()
                    .enumerate()
                    .fold((0.0f64, 0.0f64), |(re, im), (j, v)| {
                        let angle =
                            direction * core::f64::consts::TAU * ((j * k) % n) as f64 / n as f64;
                        let (s, c) = angle.sin_cos();
                        let (vr, vi) = (f64::from(v.re), f64::from(v.im));
                        (re + vr * c - vi * s, im + vr * s + vi * c)
                    })
            })
            .collect()
    }

    /// `|X| ≤ 64`; the leaf spends at most 12 f32 roundings per output (two
    /// DFT-8 levels of three butterflies with one twiddle product each, plus
    /// the middle twiddle) against an f64 oracle, so `12 · ε · 64`.
    fn tolerance() -> f64 {
        12.0 * f64::from(f32::EPSILON) * 64.0
    }

    #[test]
    fn leaf_matches_the_naive_dft_in_both_directions() {
        for direction in [-1.0f64, 1.0] {
            let src = samples(64, 0x5DEE_CE66_D1B5_4A32 ^ direction.to_bits());
            let table = twiddles(64, direction);
            let want = naive_dft(&src, direction);
            let mut got = src.clone();
            assert!(
                fixed_len64_reduced_lanes(&mut got, &table),
                "eight-lane f32 backend absent on this host"
            );
            for (index, (g, (wr, wi))) in got.iter().zip(&want).enumerate() {
                assert!(
                    (f64::from(g.re) - wr).abs() <= tolerance()
                        && (f64::from(g.im) - wi).abs() <= tolerance(),
                    "sample {index}: lanes ({}, {}) oracle ({wr}, {wi}) (tolerance {:e})",
                    g.re,
                    g.im,
                    tolerance()
                );
            }
        }
    }

    #[test]
    fn wrong_length_touches_nothing() {
        let table = twiddles(64, -1.0);
        let mut data = vec![Complex32::new(3.0, -3.0); 48];
        assert!(!fixed_len64_reduced_lanes(&mut data, &table));
        assert!(data.iter().all(|c| c.re == 3.0 && c.im == -3.0));
    }
}
