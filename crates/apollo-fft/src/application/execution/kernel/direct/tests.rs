//! Value-semantic tests of the direct DFT kernel.

mod bound;
mod reference;

use super::*;
use bound::{
    forward_bound, inverse_bound, modulus_mass, round_trip_bound, roundoff, twiddle_error, Pass,
};
use eunomia::{Bf16, Bf8, F16, F32, F64, F8};
use reference::{assert_close, assert_impulse_response, impulse, signal, unit_circle};

fn forward_two_point<T: FloatElement>() {
    let input = signal::<T>(&[(1.0, 0.0), (2.0, 0.0)]);
    let tolerance = forward_bound(input.len(), roundoff::<T>(), modulus_mass(&input));
    assert_close(&dft_forward(&input), &[(3.0, 0.0), (-1.0, 0.0)], tolerance);
}

fn round_trip<T: FloatElement>(values: &[(f64, f64)]) {
    let input = signal::<T>(values);
    let tolerance = round_trip_bound(input.len(), roundoff::<T>(), modulus_mass(&input));
    let expected: Vec<(f64, f64)> = input
        .iter()
        .map(|z| (z.re.to_f64(), z.im.to_f64()))
        .collect();
    assert_close(&dft_inverse(&dft_forward(&input)), &expected, tolerance);
}

/// Bin 0 of either direction on `[L, 1, 1, 1]` against exact constants.
///
/// Output 0 has phase 0 at every term, so each twiddle is exactly
/// `1 ± 0i` and the output is the plain sequential sum `((L + 1) + 1) + 1`
/// in the accumulator (divided by the exact power of two `4` for the
/// inverse), rounded once into `T`. With `L = 2^p` for the significand
/// precision `p` of the accumulator (`f32` and `f64` here), each `+ 1` is
/// a tie that rounds back to even, so the sum stagnates at `L` exactly
/// when, and only when, the accumulator is `p` bits wide. For a
/// reduced-precision `T` the accumulator is `f32`, and `L = 2^{p_T}` pins
/// only that the sum is not carried in `T`: `L + 3` is exact in `f32` and
/// rounds once at `T`'s spacing, while a `T` accumulator would stagnate at
/// `L`. The accumulator width itself is pinned by
/// `sticky_bit_decides_the_rounding`.
fn sums_in_the_accumulator<T: FloatElement>(leading: f64, forward: f64, inverse: f64) {
    let input = signal::<T>(&[(leading, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]);
    assert_eq!(
        dft_forward(&input)[0].re.to_f64().to_bits(),
        forward.to_bits()
    );
    assert_eq!(
        dft_inverse(&input)[0].re.to_f64().to_bits(),
        inverse.to_bits()
    );
}

/// Bin 0 of either direction on `[L, 1, s, 0]` with `L = 2^{p_T}` and the
/// sticky addend `s = 2^{p_T − 24}`, which distinguishes an `f32`
/// accumulator from a wider one.
///
/// In `f32`, `L + 1` is exact and `s` is half its spacing `2^{p_T − 23}`,
/// a tie that rounds to the even neighbour `L + 1`; the sum then sits
/// exactly on the tie between `L` and `L + 2` of `T`, which rounds to the
/// even `L`. An `f64` accumulator keeps `s`, lands above the tie, and
/// rounds to `L + 2`. The inverse divides by the exact `4`: `L/4 + 1/4` is
/// the tie between `L/4` and `L/4 + 1/2` and rounds to `L/4`, against
/// `L/4 + 1/2` for the wider accumulator.
fn sticky_bit_decides_the_rounding<T: FloatElement>(
    leading: f64,
    sticky: f64,
    forward: f64,
    inverse: f64,
) {
    let input = signal::<T>(&[(leading, 0.0), (1.0, 0.0), (sticky, 0.0), (0.0, 0.0)]);
    assert_eq!(
        dft_forward(&input)[0].re.to_f64().to_bits(),
        forward.to_bits()
    );
    assert_eq!(
        dft_inverse(&input)[0].re.to_f64().to_bits(),
        inverse.to_bits()
    );
}

/// A reduced-precision `T` transforms to the `f32` transform of the
/// exactly widened input, rounded once into `T`, bit for bit.
///
/// Its accumulator is `f32`, so both paths run the same `f32` arithmetic
/// on the same `f32` twiddles and the only `T` rounding is the output's.
/// Narrowing a product into `T` before the sum, or rounding a twiddle
/// into `T`, changes the bits of some output on a non-dyadic signal.
fn rounds_the_f32_transform_once<T: FloatElement<Accumulator = f32>>() {
    let values: Vec<(f64, f64)> = (0..16)
        .map(|t| {
            let t = f64::from_count(t);
            (3.0 * (0.37 * t).sin(), 2.0 * (0.23 * t).cos())
        })
        .collect();
    let narrow = signal::<T>(&values);
    let widened: Vec<Complex<f32>> = narrow
        .iter()
        .map(|z| Complex::new(z.re.to_accumulator(), z.im.to_accumulator()))
        .collect();
    let rounded_once = |spectrum: Vec<Complex<f32>>| -> Vec<Complex<T>> {
        spectrum
            .iter()
            .map(|z| Complex::new(T::from_accumulator(z.re), T::from_accumulator(z.im)))
            .collect()
    };
    let bits = |values: &[Complex<T>]| -> Vec<(u64, u64)> {
        values
            .iter()
            .map(|z| (z.re.to_f64().to_bits(), z.im.to_f64().to_bits()))
            .collect()
    };
    assert_eq!(
        bits(&dft_forward(&narrow)),
        bits(&rounded_once(dft_forward(&widened)))
    );
    assert_eq!(
        bits(&dft_inverse(&narrow)),
        bits(&rounded_once(dft_inverse(&widened)))
    );
}

/// Forward and inverse unit-impulse responses at length `n`, for an
/// impulse at index 1 and at the last index.
///
/// The forward response to an impulse at index `j` is `exp(−2π i k j / N)`
/// and the inverse response to one at `j` is `exp(+2π i n j / N) / N`; the
/// last index makes the phase `k·j mod N` wrap on almost every term.
fn impulse_responses<T: FloatElement>(n: usize) {
    for (index, forward_sign) in [(1, -1.0), (n - 1, 1.0)] {
        let input = impulse::<T>(n, index);
        assert_impulse_response(
            &dft_forward(&input),
            &unit_circle(n, forward_sign),
            Pass::Forward,
        );
        assert_impulse_response(
            &dft_inverse(&input),
            &unit_circle(n, -forward_sign),
            Pass::Inverse,
        );
    }
}

/// A unit impulse at index 1 of `N = 4` transforms to the column
/// `exp(∓2π i k / 4)`, pinning the sign convention of both directions
/// against exact constants rather than the library sine and cosine.
fn sign_convention<T: FloatElement>() {
    let roundoff = roundoff::<T>();
    let input = impulse::<T>(4, 1);
    let mass = modulus_mass(&input);
    // exp(-2π i k / 4) = 1, -i, -1, i.
    assert_close(
        &dft_forward(&input),
        &[(1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0)],
        forward_bound(4, roundoff, mass),
    );
    // x_n = exp(+2π i n / 4) / 4 = 1/4 · (1, i, -1, -i).
    assert_close(
        &dft_inverse(&input),
        &[(0.25, 0.0), (0.0, 0.25), (-0.25, 0.0), (0.0, -0.25)],
        inverse_bound(4, roundoff, mass / 4.0),
    );
}

const COMPLEX_SIGNAL: [(f64, f64); 4] = [(1.0, -1.0), (2.0, 0.5), (-0.5, 0.25), (0.75, -0.125)];
const REAL_SIGNAL: [(f64, f64); 4] = [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (-1.0, 0.0)];

/// Prime and composite lengths with a factor other than two, plus a power
/// of two.
const IMPULSE_LENGTHS: [usize; 6] = [3, 4, 5, 6, 7, 16];

#[test]
fn forward_matches_known_two_point_transform() {
    forward_two_point::<f64>();
    forward_two_point::<F64>();
    forward_two_point::<f32>();
    forward_two_point::<F32>();
    forward_two_point::<F16>();
    forward_two_point::<Bf16>();
    forward_two_point::<F8>();
    forward_two_point::<Bf8>();
}

#[test]
fn inverse_recovers_input() {
    round_trip::<f64>(&COMPLEX_SIGNAL);
    round_trip::<F64>(&COMPLEX_SIGNAL);
    round_trip::<f32>(&COMPLEX_SIGNAL);
    round_trip::<F32>(&COMPLEX_SIGNAL);
    round_trip::<F16>(&COMPLEX_SIGNAL);
    round_trip::<Bf16>(&COMPLEX_SIGNAL);
    round_trip::<F8>(&COMPLEX_SIGNAL);
    round_trip::<Bf8>(&COMPLEX_SIGNAL);
}

#[test]
fn forward_inverse_is_identity_on_real_signal() {
    round_trip::<f64>(&REAL_SIGNAL);
    round_trip::<F64>(&REAL_SIGNAL);
    round_trip::<f32>(&REAL_SIGNAL);
    round_trip::<F32>(&REAL_SIGNAL);
    round_trip::<F16>(&REAL_SIGNAL);
    round_trip::<Bf16>(&REAL_SIGNAL);
    round_trip::<F8>(&REAL_SIGNAL);
    round_trip::<Bf8>(&REAL_SIGNAL);
}

#[test]
fn sums_stagnate_at_the_accumulator_precision() {
    // f64 accumulates in f64 (p = 53): 2^53 + 1 ties back to 2^53.
    sums_in_the_accumulator::<f64>(
        9_007_199_254_740_992.0,
        9_007_199_254_740_992.0,
        2_251_799_813_685_248.0,
    );
    sums_in_the_accumulator::<F64>(
        9_007_199_254_740_992.0,
        9_007_199_254_740_992.0,
        2_251_799_813_685_248.0,
    );
    // f32 accumulates in f32 (p = 24): the sum stays 2^24 and the inverse
    // is 2^22. An f64 accumulator would carry 2^24 + 3, which rounds to
    // 2^24 + 4 in f32, and 2^22 + 0.75, which rounds to 2^22 + 1.
    sums_in_the_accumulator::<f32>(16_777_216.0, 16_777_216.0, 4_194_304.0);
    sums_in_the_accumulator::<F32>(16_777_216.0, 16_777_216.0, 4_194_304.0);
    // F16 (p = 11): 2048 + 3 = 2051 is exact in f32 and rounds to even at
    // F16's spacing 2, giving 2052; 2051 / 4 = 512.75 rounds to even at
    // spacing 0.5, giving 513. An F16 accumulator would stagnate at 2048
    // and 512.
    sums_in_the_accumulator::<F16>(2048.0, 2052.0, 513.0);
    // Bf16 (p = 8): 259 rounds to 260 at spacing 2 and 259 / 4 = 64.75 to
    // 65 at spacing 0.5; a Bf16 accumulator would stagnate at 256 and 64.
    sums_in_the_accumulator::<Bf16>(256.0, 260.0, 65.0);
}

#[test]
fn reduced_precision_sums_keep_the_sticky_bit_of_an_f32_accumulator() {
    // F16: L = 2048, s = 2^-13. f32 gives 2048 and 512; f64 gives 2050
    // and 512.5.
    sticky_bit_decides_the_rounding::<F16>(2048.0, 0.000_122_070_312_5, 2048.0, 512.0);
    // Bf16: L = 256, s = 2^-16. f32 gives 256 and 64; f64 gives 258 and
    // 64.5.
    sticky_bit_decides_the_rounding::<Bf16>(256.0, 0.000_015_258_789_062_5, 256.0, 64.0);
}

#[test]
fn reduced_precision_transforms_round_the_f32_transform_once() {
    rounds_the_f32_transform_once::<F16>();
    rounds_the_f32_transform_once::<Bf16>();
}

#[test]
fn impulse_responses_match_the_unit_circle() {
    for n in IMPULSE_LENGTHS {
        impulse_responses::<f64>(n);
        impulse_responses::<F64>(n);
        impulse_responses::<f32>(n);
        impulse_responses::<F32>(n);
        impulse_responses::<F16>(n);
        impulse_responses::<Bf16>(n);
        impulse_responses::<F8>(n);
        impulse_responses::<Bf8>(n);
    }
}

#[test]
fn transforms_use_the_conventional_signs() {
    sign_convention::<f64>();
    sign_convention::<F64>();
    sign_convention::<f32>();
    sign_convention::<F32>();
    sign_convention::<F16>();
    sign_convention::<Bf16>();
    sign_convention::<F8>();
    sign_convention::<Bf8>();
}

/// An impulse at the last index puts every bin on the unit circle at
/// angle `2π k / N` (`exp(-2π i k (N−1) / N) = exp(2π i k / N)`).
///
/// The phase index `k·(N−1)` reaches `(N−1)²`; without the reduction
/// `mod N` the `f64` angle grows to `≈ 2π N`, whose rounding is `N` times
/// the reduced angle's, so the closed form catches a missing reduction.
/// Every other term is `0 · ŵ = 0` and adds exactly, so the sum is a
/// single product and the bound is the one-term `forward_bound`. It adds
/// the reference twiddle's own error, since the closed form is itself
/// evaluated in `f64`.
#[test]
fn phase_index_reduces_modulo_the_length() {
    const LENGTH: usize = 512;
    let input = impulse::<f64>(LENGTH, LENGTH - 1);
    let tolerance = forward_bound(1, roundoff::<f64>(), modulus_mass(&input))
        + twiddle_error(roundoff::<f64>());
    assert_close(&dft_forward(&input), &unit_circle(LENGTH, 1.0), tolerance);
}
