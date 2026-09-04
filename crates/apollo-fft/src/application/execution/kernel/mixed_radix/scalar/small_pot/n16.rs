//! Length-16 precise codelet as a four-by-four four-step in registers.
//!
//! Sixteen `Complex64` is eight YMM registers, half the AVX2 file, so unlike
//! the length-32 arm this shape has room for its own temporaries and needs no
//! designed spill.
//!
//! The factorisation is chosen for the register layout rather than for the
//! arithmetic: with two complex samples per register, register `j` holds
//! samples `2j` and `2j + 1`, so the four samples a first-stage radix-4
//! butterfly needs — `n1`, `n1 + 4`, `n1 + 8`, `n1 + 12` — sit in the *same*
//! slot of four registers four apart. [`avx_fft4_parallel_precise`] is exactly
//! that lanewise butterfly, so the first stage runs on the natural load order
//! with no shuffle at all, two calls covering all four `n1`. Only the second
//! stage needs the four-step's transpose, and that is eight `vperm2f128`
//! because a transpose at this size is a half-swap.

use eunomia::Complex64;

#[cfg(target_arch = "x86_64")]
use super::super::simd::avx::{avx_cmul_precise, avx_fft4_parallel_precise};

/// `cos(pi/8)`, exactly rounded.
#[cfg(target_arch = "x86_64")]
const COS_PI_8: f64 = 0.923_879_532_511_286_7;
/// `sin(pi/8)`, exactly rounded.
#[cfg(target_arch = "x86_64")]
const SIN_PI_8: f64 = 0.382_683_432_365_089_8;
/// `cos(pi/4) = sin(pi/4)`.
#[cfg(target_arch = "x86_64")]
const COS_PI_4: f64 = core::f64::consts::FRAC_1_SQRT_2;

/// Second-stage twiddles, one register per `(k2, half)` pair.
///
/// Row `k2 - 1` carries `W^0` and `W^{k2}` for the registers holding `n1 = 0`
/// and `n1 = 1`; row `k2 + 2` carries `W^{2 k2}` and `W^{3 k2}` for `n1 = 2`
/// and `n1 = 3`. `k2 = 0` is the identity and is not stored.
/// `W = exp(-2 pi i / 16)` forward; `n16_twiddles_match_the_analytic_values`
/// checks every entry against that definition.
#[cfg(target_arch = "x86_64")]
const TWIDDLES_FWD_16: [[f64; 4]; 6] = [
    [1.0, 0.0, COS_PI_8, -SIN_PI_8],
    [1.0, 0.0, COS_PI_4, -COS_PI_4],
    [1.0, 0.0, SIN_PI_8, -COS_PI_8],
    [COS_PI_4, -COS_PI_4, SIN_PI_8, -COS_PI_8],
    [0.0, -1.0, -COS_PI_4, -COS_PI_4],
    [-COS_PI_4, -COS_PI_4, -COS_PI_8, SIN_PI_8],
];

/// [`TWIDDLES_FWD_16`] conjugated, for the inverse direction.
#[cfg(target_arch = "x86_64")]
const TWIDDLES_INV_16: [[f64; 4]; 6] = [
    [1.0, 0.0, COS_PI_8, SIN_PI_8],
    [1.0, 0.0, COS_PI_4, COS_PI_4],
    [1.0, 0.0, SIN_PI_8, COS_PI_8],
    [COS_PI_4, COS_PI_4, SIN_PI_8, COS_PI_8],
    [0.0, 1.0, -COS_PI_4, COS_PI_4],
    [-COS_PI_4, COS_PI_4, -COS_PI_8, -SIN_PI_8],
];

/// Runs the AVX/FMA length-16 codelet when the host supports its instructions.
pub(super) fn try_inplace<const INVERSE: bool, const NORMALIZE: bool>(
    data: &mut [Complex64],
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if super::super::simd::avx::avx_fma_available() {
            // SAFETY: the capability probe establishes AVX and FMA support, and
            // the length-16 caller supplies the sixteen samples read below.
            unsafe { vector_arm::<INVERSE, NORMALIZE>(data) };
            return true;
        }
    }

    let _ = data;
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn vector_arm<const INVERSE: bool, const NORMALIZE: bool>(data: &mut [Complex64]) {
    use std::arch::x86_64::{
        _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute2f128_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };

    let ptr = data.as_mut_ptr().cast::<f64>();
    let twiddles = if INVERSE {
        &TWIDDLES_INV_16
    } else {
        &TWIDDLES_FWD_16
    };

    // SAFETY: the caller's sixteen samples are sixty-four contiguous doubles,
    // so every offset below lies inside the span.
    let (low, high) = unsafe {
        (
            avx_fft4_parallel_precise::<INVERSE>(
                _mm256_loadu_pd(ptr),
                _mm256_loadu_pd(ptr.add(8)),
                _mm256_loadu_pd(ptr.add(16)),
                _mm256_loadu_pd(ptr.add(24)),
            ),
            avx_fft4_parallel_precise::<INVERSE>(
                _mm256_loadu_pd(ptr.add(4)),
                _mm256_loadu_pd(ptr.add(12)),
                _mm256_loadu_pd(ptr.add(20)),
                _mm256_loadu_pd(ptr.add(28)),
            ),
        )
    };

    // `k2 = 0` multiplies by one in every slot, so only three of the four
    // columns carry a twiddle.
    let mut columns = [[low[0], high[0]]; 4];
    for (k2, column) in columns.iter_mut().enumerate().skip(1) {
        // SAFETY: the table holds six rows of four doubles and `k2` is below
        // four, so both offsets are in bounds.
        unsafe {
            *column = [
                avx_cmul_precise(low[k2], _mm256_loadu_pd(twiddles[k2 - 1].as_ptr())),
                avx_cmul_precise(high[k2], _mm256_loadu_pd(twiddles[k2 + 2].as_ptr())),
            ];
        }
    }

    // Regroup before the second stage rather than butterflying across slots.
    //
    // After the first stage `t[n1][k2]` sits in slot `n1 % 2` of
    // `columns[k2][n1 / 2]`, so a radix-4 over `n1` has to cross slots.
    // `avx_fft4_precise` does that directly and costs five `vperm2f128` per
    // call, plus eight more to place its output: twenty-eight cross-lane
    // operations. Transposing first costs eight and leaves the second stage
    // lanewise like the first, which is what the efficiency cores care about —
    // cross-lane movement is the expensive kind there, and the direct form
    // measured 18.8 ns against the scalar codelet's 16.5 ns on one.
    //
    // `rows[n1]` holds `t[n1][0], t[n1][1]` and `tail[n1]` holds
    // `t[n1][2], t[n1][3]`, so one lanewise butterfly per group finishes.
    let rows = [
        _mm256_permute2f128_pd::<0x20>(columns[0][0], columns[1][0]),
        _mm256_permute2f128_pd::<0x31>(columns[0][0], columns[1][0]),
        _mm256_permute2f128_pd::<0x20>(columns[0][1], columns[1][1]),
        _mm256_permute2f128_pd::<0x31>(columns[0][1], columns[1][1]),
    ];
    let tail = [
        _mm256_permute2f128_pd::<0x20>(columns[2][0], columns[3][0]),
        _mm256_permute2f128_pd::<0x31>(columns[2][0], columns[3][0]),
        _mm256_permute2f128_pd::<0x20>(columns[2][1], columns[3][1]),
        _mm256_permute2f128_pd::<0x31>(columns[2][1], columns[3][1]),
    ];

    let scale = _mm256_set1_pd(0.0625);
    // SAFETY: the target feature frame established above covers both calls, and
    // the eight stores below cover the caller's sixty-four-double span exactly
    // once.
    unsafe {
        let low_out = avx_fft4_parallel_precise::<INVERSE>(rows[0], rows[1], rows[2], rows[3]);
        let high_out = avx_fft4_parallel_precise::<INVERSE>(tail[0], tail[1], tail[2], tail[3]);
        for (k1, (mut left, mut right)) in low_out.into_iter().zip(high_out).enumerate() {
            if INVERSE && NORMALIZE {
                left = _mm256_mul_pd(left, scale);
                right = _mm256_mul_pd(right, scale);
            }
            // Output `k = k2 + 4 * k1`: `low_out[k1]` carries `k2 = 0, 1` and
            // `high_out[k1]` carries `k2 = 2, 3`, the four consecutive outputs
            // starting at `4 * k1`.
            let base = ptr.add(8 * k1);
            _mm256_storeu_pd(base, left);
            _mm256_storeu_pd(base.add(4), right);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Complex64;

    fn assert_matches_reference<const INVERSE: bool, const NORMALIZE: bool>() {
        let input: [Complex64; 16] = core::array::from_fn(|index| {
            let value = index as f64;
            Complex64::new((value * 0.23).sin(), (value * 0.41).cos())
        });
        let mut got = input;
        let mut expected = input;

        if !super::try_inplace::<INVERSE, NORMALIZE>(&mut got) {
            return;
        }
        crate::application::execution::kernel::components::winograd::dft16_impl::<f64, INVERSE>(
            &mut expected,
        );
        if INVERSE && NORMALIZE {
            let scale = Complex64::new(0.0625, 0.0);
            for value in &mut expected {
                *value *= scale;
            }
        }

        let error = got
            .iter()
            .zip(expected.iter())
            .map(|(actual, reference)| (*actual - *reference).norm())
            .fold(0.0, f64::max);
        // Four stages of pairwise addition over unit-scale inputs, so the
        // accumulated bound is a few multiples of `16 * f64::EPSILON`; 1e-12
        // sits far above that and far below any routing error, which would
        // show as an O(1) difference.
        assert!(error < 1.0e-12, "n=16 f64 codelet error={error:e}");
    }

    #[test]
    fn forward_matches_reference() {
        assert_matches_reference::<false, false>();
    }

    #[test]
    fn inverse_matches_reference() {
        assert_matches_reference::<true, false>();
    }

    #[test]
    fn normalized_inverse_matches_reference() {
        assert_matches_reference::<true, true>();
    }

    /// An impulse names a wrong output slot directly, where the smooth signal
    /// above would only show it as a diffuse mismatch.
    #[test]
    fn forward_impulse_matches_reference() {
        for position in 0..16usize {
            let mut data = [Complex64::default(); 16];
            data[position] = Complex64::new(1.0, 0.0);
            let mut expected = data;
            crate::application::execution::kernel::components::winograd::dft16_impl::<f64, false>(
                &mut expected,
            );
            if !super::try_inplace::<false, false>(&mut data) {
                return;
            }
            for (index, value) in data.into_iter().enumerate() {
                assert!(
                    (value - expected[index]).norm() < 1.0e-12,
                    "impulse at {position}: output {index} is ({}, {})",
                    value.re,
                    value.im
                );
            }
        }
    }

    /// The table is literals, so it needs an oracle that is not itself a table.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn n16_twiddles_match_the_analytic_values() {
        // Row `k2 - 1` holds `W^0, W^{k2}`; row `k2 + 2` holds
        // `W^{2 k2}, W^{3 k2}`.
        let exponents: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [2, 3], [4, 6], [6, 9]];
        for (inverse, table) in [
            (false, &super::TWIDDLES_FWD_16),
            (true, &super::TWIDDLES_INV_16),
        ] {
            for (row, powers) in exponents.iter().enumerate() {
                for (slot, &power) in powers.iter().enumerate() {
                    let angle = core::f64::consts::TAU * (power as f64) / 16.0;
                    let sign = if inverse { 1.0 } else { -1.0 };
                    let (want_re, want_im) = (angle.cos(), sign * angle.sin());
                    let (got_re, got_im) = (table[row][2 * slot], table[row][2 * slot + 1]);
                    assert!(
                        (got_re - want_re).abs() < 4.0 * f64::EPSILON
                            && (got_im - want_im).abs() < 4.0 * f64::EPSILON,
                        "inverse={inverse} row={row} slot={slot}: \
                         table ({got_re}, {got_im}) against W^{power} ({want_re}, {want_im})"
                    );
                }
            }
        }
    }
}
