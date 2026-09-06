//! The declined length-8 codelet: a four-by-two four-step in registers.
//!
//! This module is `cfg(test)` and has no production call site. It is correct —
//! the oracles below check it against the direct DFT in both directions and at
//! every impulse position — and it is *slower* than the scalar Winograd
//! codelet `precise.rs` ships at N = 8. It stays in the tree as the subject of
//! the `small_pot_arms` probe, so the comparison that declined it can be re-run
//! rather than believed; the note it replaces asserted a number nobody could
//! reproduce, which is how it survived unexamined.
//!
//! # The construction
//!
//! Eight `Complex64` is four YMM registers, a quarter of the AVX2 file, so the
//! shape has room for its temporaries and never spills.
//!
//! Writing `n = n1 + 4 n2` and `k = k2 + 2 k1` puts both stages on the natural
//! load order. With two complex samples per register, register `j` holds
//! samples `2j` and `2j + 1`, so the first stage's radix-2 pair `n1` and
//! `n1 + 4` sits in the *same* slot of two registers two apart: one lanewise
//! add/sub covers `n1 = 0, 1` and one covers `n1 = 2, 3`, with no shuffle.
//! Only the second stage needs the four-step's transpose, and that is four
//! `vperm2f128` because a transpose at this size is a half-swap. The output
//! ordering falls out of the same choice: the second stage's `k1`-th result
//! holds `X[2 k1]` and `X[2 k1 + 1]`, which is output register `k1` in place,
//! so nothing is permuted on the way out.
//!
//! # Why it loses anyway
//!
//! Measured round trip, forward plus normalized inverse: 15.08 ns against the
//! scalar codelet's 11.33 on a performance core, 21.58 against 15.81 on an
//! efficiency core, intervals disjoint over three runs.
//!
//! N = 8 is the length at which every twiddle is a trivial rotation — `1`,
//! `-i`, `(+-1 - i)/sqrt(2)`. The scalar codelet spends those as sign flips,
//! part swaps and one real multiply. This form must still pay four cross-lane
//! permutes to make its second stage lanewise, and eight points is not enough
//! arithmetic to amortise them. The same construction one size up *wins* — see
//! [`super::n16`], where the first stage is a radix-4 over four times the work
//! against only twice the shuffle.
//!
//! The alternative factorisation was checked and is worse, not better: taking
//! `n = n1 + 2 n2` makes the first stage a lanewise radix-4 and the second a
//! radix-2, but the outputs then need the same four permutes *and* a third
//! twiddled register, so it trades one multiply in for nothing.

use eunomia::Complex64;

#[cfg(target_arch = "x86_64")]
use super::super::simd::avx::avx_fft4_parallel_precise;

/// `cos(pi/4) = sin(pi/4)`.
#[cfg(target_arch = "x86_64")]
const COS_PI_4: f64 = core::f64::consts::FRAC_1_SQRT_2;

/// Second-stage twiddles for `k2 = 1`, in the `(direct, swapped)` form the
/// multiply below consumes.
///
/// Multiplying `z = (a, b)` by `(c, d)` is `(ca - db, da + cb)`, which is
/// `z * [c, c] + swap(z) * [-d, d]` where `swap` exchanges each sample's real
/// and imaginary parts. Storing those two coefficient vectors instead of the
/// twiddle turns the multiply into a permute, a multiply and an FMA — three
/// instructions against the generic `avx_cmul_precise`'s five. That is worth
/// the less obvious table here because at N = 8 every twiddle is a trivial
/// rotation, so a generic complex multiply is this arm's largest avoidable
/// cost: folding them in bought 13% on the performance core and 15% on the
/// efficiency core.
///
/// Row 0 carries `W^0` and `W^1` for the register holding `n1 = 0` and
/// `n1 = 1`; row 1 carries `W^2` and `W^3` for `n1 = 2` and `n1 = 3`. `k2 = 0`
/// is the identity and is not stored. `W = exp(-2 pi i / 8)` forward;
/// `twiddles_match_the_analytic_values` reconstructs each twiddle from these
/// coefficients and checks it against that definition.
#[cfg(target_arch = "x86_64")]
const COEFFICIENTS_FWD_8: [[[f64; 4]; 2]; 2] = [
    // W^0 = 1; W^1 = (1 - i)/sqrt(2)
    [
        [1.0, 1.0, COS_PI_4, COS_PI_4],
        [0.0, 0.0, COS_PI_4, -COS_PI_4],
    ],
    // W^2 = -i; W^3 = (-1 - i)/sqrt(2)
    [
        [0.0, 0.0, -COS_PI_4, -COS_PI_4],
        [1.0, -1.0, COS_PI_4, -COS_PI_4],
    ],
];

/// [`COEFFICIENTS_FWD_8`] for the inverse direction.
///
/// Conjugating `(c, d)` to `(c, -d)` leaves the direct row alone and negates
/// the swapped row.
#[cfg(target_arch = "x86_64")]
const COEFFICIENTS_INV_8: [[[f64; 4]; 2]; 2] = [
    [
        [1.0, 1.0, COS_PI_4, COS_PI_4],
        [0.0, 0.0, -COS_PI_4, COS_PI_4],
    ],
    [
        [0.0, 0.0, -COS_PI_4, -COS_PI_4],
        [-1.0, 1.0, -COS_PI_4, COS_PI_4],
    ],
];

/// Multiplies each of a register's two samples by its own twiddle.
///
/// `coefficients` is the `(direct, swapped)` pair described on
/// [`COEFFICIENTS_FWD_8`].
///
/// # Safety
///
/// Requires AVX and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn twiddle(
    value: std::arch::x86_64::__m256d,
    coefficients: &[[f64; 4]; 2],
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute_pd};
    // SAFETY: both rows are four doubles, which is the load width.
    unsafe {
        let swapped = _mm256_permute_pd::<0x05>(value);
        let cross = _mm256_mul_pd(swapped, _mm256_loadu_pd(coefficients[1].as_ptr()));
        _mm256_fmadd_pd(value, _mm256_loadu_pd(coefficients[0].as_ptr()), cross)
    }
}

/// Runs the AVX/FMA length-8 codelet when the host supports its instructions.
///
/// Returns `false` on a host without AVX and FMA, matching the shape of the
/// arms that did ship ([`super::n16`], [`super::n32`]) so the probe measures
/// the same call structure they would.
pub(super) fn try_inplace<const INVERSE: bool, const NORMALIZE: bool>(
    data: &mut [Complex64],
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if super::super::simd::avx::avx_fma_available() {
            // SAFETY: the capability probe establishes AVX and FMA support, and
            // the length-8 caller supplies the eight samples read below.
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
        _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute2f128_pd, _mm256_set1_pd,
        _mm256_storeu_pd, _mm256_sub_pd,
    };

    let ptr = data.as_mut_ptr().cast::<f64>();
    let coefficients = if INVERSE {
        &COEFFICIENTS_INV_8
    } else {
        &COEFFICIENTS_FWD_8
    };

    // Loaded in butterfly-partner order rather than address order: `first` and
    // `second` are the pair the `n1 = 0, 1` radix-2 needs, `third` and `fourth`
    // the pair for `n1 = 2, 3`.
    // SAFETY: the caller's eight samples are thirty-two contiguous doubles, so
    // every offset below lies inside the span.
    let (first, second, third, fourth) = unsafe {
        (
            _mm256_loadu_pd(ptr),
            _mm256_loadu_pd(ptr.add(8)),
            _mm256_loadu_pd(ptr.add(4)),
            _mm256_loadu_pd(ptr.add(12)),
        )
    };

    // First stage, lanewise: the radix-2 butterfly over `n2`, whose two inputs
    // `n1` and `n1 + 4` share a slot. Direction does not enter — a radix-2
    // butterfly has no twiddle.
    let low_even = _mm256_add_pd(first, second);
    let low_odd = _mm256_sub_pd(first, second);
    let high_even = _mm256_add_pd(third, fourth);
    let high_odd = _mm256_sub_pd(third, fourth);

    // `k2 = 0` multiplies by one in every slot, so only the odd column twiddles.
    // SAFETY: the target-feature frame above covers both calls.
    let (low_odd, high_odd) = unsafe {
        (
            twiddle(low_odd, &coefficients[0]),
            twiddle(high_odd, &coefficients[1]),
        )
    };

    // Transpose so the second stage is lanewise too: `rows[n1]` holds
    // `t[n1][0], t[n1][1]`, which is one half-swap per output.
    let rows = [
        _mm256_permute2f128_pd::<0x20>(low_even, low_odd),
        _mm256_permute2f128_pd::<0x31>(low_even, low_odd),
        _mm256_permute2f128_pd::<0x20>(high_even, high_odd),
        _mm256_permute2f128_pd::<0x31>(high_even, high_odd),
    ];

    let scale = _mm256_set1_pd(0.125);
    // SAFETY: the target feature frame established above covers the call, and
    // the four stores below cover the caller's thirty-two-double span exactly
    // once.
    unsafe {
        let out = avx_fft4_parallel_precise::<INVERSE>(rows[0], rows[1], rows[2], rows[3]);
        for (k1, mut value) in out.into_iter().enumerate() {
            if INVERSE && NORMALIZE {
                value = _mm256_mul_pd(value, scale);
            }
            // Output `k = k2 + 2 * k1`, and `out[k1]` carries `k2 = 0, 1`: the
            // two consecutive outputs starting at `2 * k1`, which is register
            // `k1` unpermuted.
            _mm256_storeu_pd(ptr.add(4 * k1), value);
        }
    }
}

/// Direct entry to the vector arm, bypassing the per-call capability probe.
///
/// The `small_pot_arms` probe uses this to separate the body's cost from the
/// cost of the `OnceLock` capability check in front of it. It does *not*
/// remove the `#[target_feature]` call boundary — this entry does not carry
/// the attribute either, so [`vector_arm`] cannot inline into it — which is
/// why the two arms read the same and the check is what the difference bounds.
///
/// # Safety
///
/// Carries [`vector_arm`]'s contract, and additionally requires the caller to
/// have established AVX and FMA support itself.
#[cfg(target_arch = "x86_64")]
pub(crate) unsafe fn vector_arm_unchecked<const INVERSE: bool, const NORMALIZE: bool>(
    data: &mut [Complex64],
) {
    // SAFETY: the caller carries both the capability and the length contract.
    unsafe { vector_arm::<INVERSE, NORMALIZE>(data) }
}

#[cfg(test)]
mod tests {
    use super::Complex64;

    /// The scalar codelet this arm was measured against, with its normalization.
    fn reference<const INVERSE: bool, const NORMALIZE: bool>(data: &mut [Complex64; 8]) {
        crate::application::execution::kernel::components::winograd::dft8_array_impl::<
            f64,
            INVERSE,
            false,
        >(data);
        if INVERSE && NORMALIZE {
            let scale = Complex64::new(0.125, 0.0);
            for value in data.iter_mut() {
                *value *= scale;
            }
        }
    }

    fn assert_matches_reference<const INVERSE: bool, const NORMALIZE: bool>() {
        let input: [Complex64; 8] = core::array::from_fn(|index| {
            let value = index as f64;
            Complex64::new((value * 0.23).sin(), (value * 0.41).cos())
        });
        let mut got = input;
        let mut expected = input;

        if !super::try_inplace::<INVERSE, NORMALIZE>(&mut got) {
            return;
        }
        reference::<INVERSE, NORMALIZE>(&mut expected);

        let error = got
            .iter()
            .zip(expected.iter())
            .map(|(actual, wanted)| (*actual - *wanted).norm())
            .fold(0.0, f64::max);
        // Three stages of pairwise addition over unit-scale inputs, so the
        // accumulated bound is a few multiples of `8 * f64::EPSILON`; 1e-12
        // sits far above that and far below any routing error, which would
        // show as an O(1) difference.
        assert!(error < 1.0e-12, "n=8 f64 codelet error={error:e}");
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
        for position in 0..8usize {
            let mut data = [Complex64::default(); 8];
            data[position] = Complex64::new(1.0, 0.0);
            let mut expected = data;
            reference::<false, false>(&mut expected);
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

    /// The inverse direction gets its own impulse sweep: the coefficient rows
    /// are conjugated by hand, so a sign error there is invisible forward.
    #[test]
    fn inverse_impulse_matches_reference() {
        for position in 0..8usize {
            let mut data = [Complex64::default(); 8];
            data[position] = Complex64::new(1.0, 0.0);
            let mut expected = data;
            reference::<true, true>(&mut expected);
            if !super::try_inplace::<true, true>(&mut data) {
                return;
            }
            for (index, value) in data.into_iter().enumerate() {
                assert!(
                    (value - expected[index]).norm() < 1.0e-12,
                    "inverse impulse at {position}: output {index} is ({}, {})",
                    value.re,
                    value.im
                );
            }
        }
    }

    /// The table is literals, so it needs an oracle that is not itself a table.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn twiddles_match_the_analytic_values() {
        // The stored form is `(direct, swapped)` coefficients rather than the
        // twiddle, so the oracle reconstructs `(c, d)` from them before
        // comparing: the direct row holds `[c, c]` per sample and the swapped
        // row holds `[-d, d]`. Checking the coefficients against each other
        // would only prove they are self-consistent.
        //
        // Row 0 holds `W^0, W^1`; row 1 holds `W^2, W^3`.
        let exponents: [[usize; 2]; 2] = [[0, 1], [2, 3]];
        for (inverse, table) in [
            (false, &super::COEFFICIENTS_FWD_8),
            (true, &super::COEFFICIENTS_INV_8),
        ] {
            for (row, powers) in exponents.iter().enumerate() {
                let [direct, swapped] = table[row];
                for (slot, &power) in powers.iter().enumerate() {
                    let angle = core::f64::consts::TAU * (power as f64) / 8.0;
                    let sign = if inverse { 1.0 } else { -1.0 };
                    let (want_re, want_im) = (angle.cos(), sign * angle.sin());

                    // `c` appears in both lanes of the direct row, and `d` as
                    // `-d` then `d` in the swapped row; a table that disagreed
                    // with itself would compute neither twiddle.
                    let (c_low, c_high) = (direct[2 * slot], direct[2 * slot + 1]);
                    let (minus_d, d) = (swapped[2 * slot], swapped[2 * slot + 1]);
                    assert!(
                        (c_low - c_high).abs() < f64::EPSILON && (minus_d + d).abs() < f64::EPSILON,
                        "inverse={inverse} row={row} slot={slot}: coefficients are \
                         not one complex multiply ({c_low}, {c_high}, {minus_d}, {d})"
                    );
                    assert!(
                        (c_low - want_re).abs() < 4.0 * f64::EPSILON
                            && (d - want_im).abs() < 4.0 * f64::EPSILON,
                        "inverse={inverse} row={row} slot={slot}: \
                         table ({c_low}, {d}) against W^{power} ({want_re}, {want_im})"
                    );
                }
            }
        }
    }
}
