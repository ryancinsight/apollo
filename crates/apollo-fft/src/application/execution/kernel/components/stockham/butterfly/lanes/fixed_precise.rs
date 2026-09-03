//! The fixed 32- and 64-point f64 leaves as lane kernels.
//!
//! Both evaluate the DFT by the Cooley–Tukey identity with `N = 8 · C`
//! (`C = 4` or `8`): for input `x[8r + c]`, phase one computes
//! `A[p, c] = Σ_r x[8r + c] · W_8^{pr}` down each column with a
//! register-resident DFT-8, multiplies by `W_N^{pc}`, and stores `A'` row-major;
//! phase two computes `X[Cq + p] = Σ_c A'[p, c] · W_C^{qc}` across each row
//! pair. A register holds two adjacent complex samples, so every butterfly is
//! two independent columns (phase one) or two independent rows (phase two);
//! the only lane movement is the 2×2 sample transpose between the phases,
//! done with one pair-deinterleave per register pair.
//!
//! The twiddles are the exact constant tables the retired intrinsic leaves
//! used, so the two agree to rounding; the differential tests bound the
//! leaves against a naive DFT rather than against each other.

use eunomia::Complex64;
use hermes_simd::{ComplexReg, LaneKernel, Simd, SimdArch, SimdKernel, Vector};

use crate::application::execution::kernel::components::register_butterfly::{radix4, radix8};
use crate::application::execution::kernel::mixed_radix::scalar::twiddle_constants::{
    TWIDDLES_32_FWD, TWIDDLES_32_INV, TWIDDLES_64_FWD, TWIDDLES_64_INV,
};

struct Dft64<'data, const INVERSE: bool> {
    data: &'data mut [Complex64; 64],
}

struct Dft32<'data, const INVERSE: bool> {
    data: &'data mut [Complex64; 32],
}

#[expect(
    clippy::inline_always,
    reason = "the leaf must remain in the selected target-feature frame"
)]
#[inline(always)]
fn load<A>(simd: Simd<f64, A>, values: &[Complex64]) -> ComplexReg<f64, A>
where
    A: SimdArch + SimdKernel<f64>,
{
    let view = simd.view(eunomia::layout::cast_slice(values));
    ComplexReg::from_interleaved(Vector::from_view_chunk(&view, 0))
}

#[expect(
    clippy::inline_always,
    reason = "stores must remain in the selected target-feature frame"
)]
#[inline(always)]
fn store<A>(value: ComplexReg<f64, A>, destination: &mut [Complex64])
where
    A: SimdArch + SimdKernel<f64>,
{
    value
        .into_interleaved()
        .store_unaligned_to_slice(eunomia::layout::cast_slice_mut(destination))
        .expect("invariant: two complex samples fill four lanes");
}

/// The register `(W_N^{k0}, W_N^{k1})` from the exact constant table.
#[expect(
    clippy::inline_always,
    reason = "constant twiddle loads must fold into the selected target-feature frame"
)]
#[inline(always)]
fn twiddle_pair<A, const INVERSE: bool, const N: usize>(
    simd: Simd<f64, A>,
    k0: usize,
    k1: usize,
) -> ComplexReg<f64, A>
where
    A: SimdArch + SimdKernel<f64>,
{
    let table: &[Complex64] = match (N, INVERSE) {
        (64, false) => &TWIDDLES_64_FWD,
        (64, true) => &TWIDDLES_64_INV,
        (32, false) => &TWIDDLES_32_FWD,
        (32, true) => &TWIDDLES_32_INV,
        _ => unreachable!("invariant: the fixed leaves are 32 or 64 points"),
    };
    // `k < N` for every product `p · c` the phases form; the mask makes the
    // bound a type fact so the lookup carries no check.
    load(simd, &[table[k0 & (N - 1)], table[k1 & (N - 1)]])
}

/// `(a0, a1), (b0, b1)` → `(a0, b0), (a1, b1)`: the 2×2 sample transpose.
#[expect(
    clippy::inline_always,
    reason = "the transpose must remain in the selected target-feature frame"
)]
#[inline(always)]
fn transpose_pair<A>(
    a: ComplexReg<f64, A>,
    b: ComplexReg<f64, A>,
) -> (ComplexReg<f64, A>, ComplexReg<f64, A>)
where
    A: SimdArch + SimdKernel<f64>,
{
    let (even, odd) = a
        .into_interleaved()
        .deinterleave_pairs(b.into_interleaved());
    (
        ComplexReg::from_interleaved(even),
        ComplexReg::from_interleaved(odd),
    )
}

/// `x[8r + c]`, eight rows by eight columns, two columns per register.
#[expect(
    clippy::inline_always,
    reason = "the leaf must remain in the selected target-feature frame"
)]
#[inline(always)]
fn dft64_kernel<A, const INVERSE: bool>(simd: Simd<f64, A>, data: &mut [Complex64; 64])
where
    A: SimdArch + SimdKernel<f64>,
{
    let half_root2 = simd.splat(core::f64::consts::FRAC_1_SQRT_2);
    let mut mid = [Complex64::new(0.0, 0.0); 64];
    for pair in 0..4 {
        let c0 = 2 * pair;
        let rows = core::array::from_fn(|r| load(simd, &data[8 * r + c0..8 * r + c0 + 2]));
        let mut column = radix8::<f64, A, INVERSE>(rows, half_root2);
        for (p, value) in column.iter_mut().enumerate().skip(1) {
            *value = *value * twiddle_pair::<A, INVERSE, 64>(simd, p * c0, p * (c0 + 1));
        }
        for (p, value) in column.into_iter().enumerate() {
            store(value, &mut mid[8 * p + c0..8 * p + c0 + 2]);
        }
    }
    for pair in 0..4 {
        let p0 = 2 * pair;
        let mut columns = [ComplexReg::<f64, A>::zero(); 8];
        for j in 0..4 {
            let upper = load(simd, &mid[8 * p0 + 2 * j..8 * p0 + 2 * j + 2]);
            let lower = load(simd, &mid[8 * (p0 + 1) + 2 * j..8 * (p0 + 1) + 2 * j + 2]);
            let (even, odd) = transpose_pair(upper, lower);
            columns[2 * j] = even;
            columns[2 * j + 1] = odd;
        }
        let out = radix8::<f64, A, INVERSE>(columns, half_root2);
        for (q, value) in out.into_iter().enumerate() {
            store(value, &mut data[8 * q + p0..8 * q + p0 + 2]);
        }
    }
}

/// `x[8r + c]`, eight rows by four columns, two columns per register.
#[expect(
    clippy::inline_always,
    reason = "the leaf must remain in the selected target-feature frame"
)]
#[inline(always)]
fn dft32_kernel<A, const INVERSE: bool>(simd: Simd<f64, A>, data: &mut [Complex64; 32])
where
    A: SimdArch + SimdKernel<f64>,
{
    let half_root2 = simd.splat(core::f64::consts::FRAC_1_SQRT_2);
    let mut mid = [Complex64::new(0.0, 0.0); 32];
    for pair in 0..2 {
        let c0 = 2 * pair;
        let rows = core::array::from_fn(|r| load(simd, &data[4 * r + c0..4 * r + c0 + 2]));
        let mut column = radix8::<f64, A, INVERSE>(rows, half_root2);
        for (p, value) in column.iter_mut().enumerate().skip(1) {
            *value = *value * twiddle_pair::<A, INVERSE, 32>(simd, p * c0, p * (c0 + 1));
        }
        for (p, value) in column.into_iter().enumerate() {
            store(value, &mut mid[4 * p + c0..4 * p + c0 + 2]);
        }
    }
    for pair in 0..4 {
        let p0 = 2 * pair;
        let mut columns = [ComplexReg::<f64, A>::zero(); 4];
        for j in 0..2 {
            let upper = load(simd, &mid[4 * p0 + 2 * j..4 * p0 + 2 * j + 2]);
            let lower = load(simd, &mid[4 * (p0 + 1) + 2 * j..4 * (p0 + 1) + 2 * j + 2]);
            let (even, odd) = transpose_pair(upper, lower);
            columns[2 * j] = even;
            columns[2 * j + 1] = odd;
        }
        let out = radix4::<f64, A, INVERSE>(columns);
        for (q, value) in out.into_iter().enumerate() {
            store(value, &mut data[8 * q + p0..8 * q + p0 + 2]);
        }
    }
}

impl<const INVERSE: bool> LaneKernel<f64> for Dft64<'_, INVERSE> {
    type Output = ();

    #[inline]
    fn call<A: SimdArch + SimdKernel<f64>>(self, simd: Simd<f64, A>) {
        debug_assert_eq!(
            A::LANE_COUNT,
            4,
            "invariant: the f64 leaves hold two samples per register"
        );
        dft64_kernel::<A, INVERSE>(simd, self.data);
    }
}

impl<const INVERSE: bool> LaneKernel<f64> for Dft32<'_, INVERSE> {
    type Output = ();

    #[inline]
    fn call<A: SimdArch + SimdKernel<f64>>(self, simd: Simd<f64, A>) {
        debug_assert_eq!(
            A::LANE_COUNT,
            4,
            "invariant: the f64 leaves hold two samples per register"
        );
        dft32_kernel::<A, INVERSE>(simd, self.data);
    }
}

/// Runs the 64-point f64 leaf on the four-lane hardware backend.
///
/// The direction follows the plan's twiddle table exactly as the scalar
/// route reads it. Returns `false`, having touched nothing, when the host has
/// no four-lane f64 backend or `data` is not 64 samples; the caller then takes
/// its staged route.
#[inline]
pub(crate) fn fixed_len64_lanes(data: &mut [Complex64], twiddles: &[Complex64]) -> bool {
    let Ok(data) = <&mut [Complex64; 64]>::try_from(data) else {
        return false;
    };
    if twiddles.get(2).is_some_and(|w| w.im > 0.0) {
        hermes_simd::vectorize_hardware_lanes::<4, f64, _>(Dft64::<true> { data }).is_some()
    } else {
        hermes_simd::vectorize_hardware_lanes::<4, f64, _>(Dft64::<false> { data }).is_some()
    }
}

/// [`fixed_len64_lanes`] for 32 samples.
#[inline]
pub(crate) fn fixed_len32_lanes(data: &mut [Complex64], twiddles: &[Complex64]) -> bool {
    let Ok(data) = <&mut [Complex64; 32]>::try_from(data) else {
        return false;
    };
    if twiddles.get(2).is_some_and(|w| w.im > 0.0) {
        hermes_simd::vectorize_hardware_lanes::<4, f64, _>(Dft32::<true> { data }).is_some()
    } else {
        hermes_simd::vectorize_hardware_lanes::<4, f64, _>(Dft32::<false> { data }).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::{fixed_len32_lanes, fixed_len64_lanes};
    use eunomia::Complex64;

    /// Deterministic samples in `[-1, 1]` (Knuth MMIX LCG).
    fn samples(count: usize, seed: u64) -> Vec<Complex64> {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / (1u64 << 53) as f64).mul_add(2.0, -1.0)
        };
        (0..count).map(|_| Complex64::new(next(), next())).collect()
    }

    /// The plan's twiddle table for a length-`n` transform in `direction`
    /// (`-1.0` forward, `+1.0` inverse): `W_n^k = exp(direction · 2πi k / n)`.
    fn twiddles(n: usize, direction: f64) -> Vec<Complex64> {
        (0..n)
            .map(|k| {
                let angle = direction * core::f64::consts::TAU * k as f64 / n as f64;
                Complex64::new(angle.cos(), angle.sin())
            })
            .collect()
    }

    /// An independent oracle: the DFT sum evaluated directly with exact
    /// per-term angles, so the leaf's factorisation is checked against the
    /// definition rather than against another factorisation.
    fn naive_dft(x: &[Complex64], direction: f64) -> Vec<Complex64> {
        let n = x.len();
        (0..n)
            .map(|k| {
                x.iter()
                    .enumerate()
                    .fold(Complex64::new(0.0, 0.0), |acc, (j, v)| {
                        let angle =
                            direction * core::f64::consts::TAU * ((j * k) % n) as f64 / n as f64;
                        acc + *v * Complex64::new(angle.cos(), angle.sin())
                    })
            })
            .collect()
    }

    /// `|X| ≤ n · max|x| = n`. The naive sum spends at most `2n` roundings
    /// per output and the leaf at most 12 (two DFT-8 levels of three
    /// butterflies with one twiddle product each, plus the middle twiddle),
    /// so the difference is bounded by `(2n + 12) · ε · n`.
    fn tolerance(n: usize) -> f64 {
        (2.0 * n as f64 + 12.0) * f64::EPSILON * n as f64
    }

    fn assert_close(got: &[Complex64], want: &[Complex64], tol: f64) {
        for (index, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g.re - w.re).abs() <= tol && (g.im - w.im).abs() <= tol,
                "sample {index}: lanes ({}, {}) oracle ({}, {}) (tolerance {tol:e})",
                g.re,
                g.im,
                w.re,
                w.im
            );
        }
    }

    #[test]
    fn leaves_match_the_naive_dft_in_both_directions() {
        for direction in [-1.0, 1.0] {
            for (n, seed) in [
                (32usize, 0x1234_5678_9ABC_DEF0u64),
                (64, 0x0FED_CBA9_8765_4321),
            ] {
                let src = samples(n, seed ^ n as u64);
                let table = twiddles(n, direction);
                let want = naive_dft(&src, direction);
                let mut got = src.clone();
                let ran = if n == 32 {
                    fixed_len32_lanes(&mut got, &table)
                } else {
                    fixed_len64_lanes(&mut got, &table)
                };
                assert!(ran, "four-lane f64 backend absent on this host");
                assert_close(&got, &want, tolerance(n));
            }
        }
    }

    #[test]
    fn wrong_length_touches_nothing() {
        let table = twiddles(64, -1.0);
        let mut data = vec![Complex64::new(3.0, -3.0); 48];
        assert!(!fixed_len64_lanes(&mut data, &table));
        assert!(!fixed_len32_lanes(&mut data, &table));
        assert!(data.iter().all(|c| c.re == 3.0 && c.im == -3.0));
    }
}
