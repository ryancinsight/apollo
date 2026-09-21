use super::*;
use eunomia::RealField;

// Nearest f64 values of the exact orthonormal N=8 DCT-III basis. The entries
// come from the closed-form values sqrt(1/8) and
// sqrt(2/8)·cos(k(2j+1)π/16), whose cosines are constructible by nested square
// roots. They are independent of Eunomia's locked cosine implementation.
const ORTHONORMAL_DCT_III: [[f64; 8]; 8] = [
    [
        3.535_533_905_932_738e-1,
        4.903_926_402_016_152e-1,
        4.619_397_662_556_434e-1,
        4.157_348_061_512_726e-1,
        3.535_533_905_932_738e-1,
        2.777_851_165_098_011e-1,
        1.913_417_161_825_449e-1,
        9.754_516_100_806_414e-2,
    ],
    [
        3.535_533_905_932_738e-1,
        4.157_348_061_512_726e-1,
        1.913_417_161_825_449e-1,
        -9.754_516_100_806_414e-2,
        -3.535_533_905_932_738e-1,
        -4.903_926_402_016_152e-1,
        -4.619_397_662_556_434e-1,
        -2.777_851_165_098_011e-1,
    ],
    [
        3.535_533_905_932_738e-1,
        2.777_851_165_098_011e-1,
        -1.913_417_161_825_449e-1,
        -4.903_926_402_016_152e-1,
        -3.535_533_905_932_738e-1,
        9.754_516_100_806_414e-2,
        4.619_397_662_556_434e-1,
        4.157_348_061_512_726e-1,
    ],
    [
        3.535_533_905_932_738e-1,
        9.754_516_100_806_414e-2,
        -4.619_397_662_556_434e-1,
        -2.777_851_165_098_011e-1,
        3.535_533_905_932_738e-1,
        4.157_348_061_512_726e-1,
        -1.913_417_161_825_449e-1,
        -4.903_926_402_016_152e-1,
    ],
    [
        3.535_533_905_932_738e-1,
        -9.754_516_100_806_414e-2,
        -4.619_397_662_556_434e-1,
        2.777_851_165_098_011e-1,
        3.535_533_905_932_738e-1,
        -4.157_348_061_512_726e-1,
        -1.913_417_161_825_449e-1,
        4.903_926_402_016_152e-1,
    ],
    [
        3.535_533_905_932_738e-1,
        -2.777_851_165_098_011e-1,
        -1.913_417_161_825_449e-1,
        4.903_926_402_016_152e-1,
        -3.535_533_905_932_738e-1,
        -9.754_516_100_806_414e-2,
        4.619_397_662_556_434e-1,
        -4.157_348_061_512_726e-1,
    ],
    [
        3.535_533_905_932_738e-1,
        -4.157_348_061_512_726e-1,
        1.913_417_161_825_449e-1,
        9.754_516_100_806_414e-2,
        -3.535_533_905_932_738e-1,
        4.903_926_402_016_152e-1,
        -4.619_397_662_556_434e-1,
        2.777_851_165_098_011e-1,
    ],
    [
        3.535_533_905_932_738e-1,
        -4.903_926_402_016_152e-1,
        4.619_397_662_556_434e-1,
        -4.157_348_061_512_726e-1,
        3.535_533_905_932_738e-1,
        -2.777_851_165_098_011e-1,
        1.913_417_161_825_449e-1,
        -9.754_516_100_806_414e-2,
    ],
];

fn coefficient_threshold<T: RealField>() -> T {
    // At N=8 the largest exact phase is 105π/16. The represented π and the two
    // products contribute γ3 = 3u/(1-3u); addition by 1/2 and division by eight
    // are exact for the tested integer grid. Two further ulps cover rounding the
    // independently tabulated reference and the returned coefficient. This is
    // the finite N=8 acceptance interval in which the locked cosine provider is
    // verified below; it assumes no global accuracy contract for that provider.
    let unit_roundoff = T::EPSILON / T::from_f64(2.0);
    let gamma_three =
        T::from_f64(3.0) * unit_roundoff / (T::ONE - T::from_f64(3.0) * unit_roundoff);
    let maximum_phase = T::from_f64(20.616_701_789_183_02) * (T::ONE + unit_roundoff);
    maximum_phase * gamma_three + T::from_f64(2.0) * unit_roundoff
}

fn orthogonality_threshold<T: RealField>() -> T {
    let coefficient_error = coefficient_threshold::<T>();
    let unit_roundoff = T::EPSILON / T::from_f64(2.0);
    let length = T::from_f64(8.0);
    let gamma = length * unit_roundoff / (T::ONE - length * unit_roundoff);
    let coefficient_magnitude = T::ONE + coefficient_error;
    let represented_basis_error =
        length * (T::from_f64(2.0) * coefficient_error + coefficient_error * coefficient_error);
    let dot_product_rounding = gamma * length * coefficient_magnitude * coefficient_magnitude;
    let comparison_rounding = T::from_f64(2.0) * unit_roundoff;
    represented_basis_error + dot_product_rounding + comparison_rounding
}

fn abs<T: RealField>(value: T) -> T {
    <T as eunomia::NumericElement>::abs(value)
}

fn assert_close<T: RealField>(actual: T, expected: T, bound: T) {
    assert!(
        abs(actual - expected) <= bound,
        "actual={actual:?}, expected={expected:?}, bound={bound:?}"
    );
}

fn analytical_basis<T: RealField>() {
    let plan = DctIiiPlan::<T, 8>::new(Normalization::Orthonormal).expect("nonzero length");
    for frequency in 0..8 {
        let mut impulse = [T::ZERO; 8];
        impulse[frequency] = T::ONE;
        let mut output = [T::ZERO; 8];
        plan.transform(&impulse, &mut output);
        for (position, sample) in output.into_iter().enumerate() {
            let expected = T::from_f64(ORTHONORMAL_DCT_III[position][frequency]);
            assert_close(sample, expected, coefficient_threshold::<T>());
        }
    }
}

fn orthonormal_basis_vectors<T: RealField>() {
    let plan = DctIiiPlan::<T, 8>::new(Normalization::Orthonormal).expect("nonzero length");
    let mut rows = [[T::ZERO; 8]; 8];
    for frequency in 0..8 {
        let mut impulse = [T::ZERO; 8];
        impulse[frequency] = T::ONE;
        plan.transform(&impulse, &mut rows[frequency]);
    }
    for left in 0..8 {
        for right in 0..8 {
            let product = rows[left]
                .iter()
                .copied()
                .zip(rows[right].iter().copied())
                .fold(T::ZERO, |sum, (a, b)| sum + a * b);
            let expected = if left == right { T::ONE } else { T::ZERO };
            assert_close(product, expected, orthogonality_threshold::<T>());
        }
    }
}

fn cached_matches_direct<T: RealField>() {
    let plan = DctIiiPlan::<T, 8>::new(Normalization::Unnormalized).expect("nonzero length");
    let signal = core::array::from_fn(|index| T::from_f64(index as f64 - 3.0));
    let mut cached = [T::ZERO; 8];
    let mut direct = [T::ZERO; 8];
    plan.transform(&signal, &mut cached);
    crate::dct3(&signal, &mut direct);
    assert_eq!(cached, direct);
}

fn length_one_values<T: RealField>() {
    let unnormalized =
        DctIiiPlan::<T, 1>::new(Normalization::Unnormalized).expect("one is a nonzero length");
    let orthonormal =
        DctIiiPlan::<T, 1>::new(Normalization::Orthonormal).expect("one is a nonzero length");
    let mut output = [T::ZERO];

    unnormalized.transform(&[T::from_f64(6.0)], &mut output);
    assert_eq!(output, [T::from_f64(3.0)]);
    orthonormal.transform(&[T::from_f64(6.0)], &mut output);
    assert_eq!(output, [T::from_f64(6.0)]);

    orthonormal.transform(&[T::NAN], &mut output);
    assert!(<T as eunomia::NumericElement>::is_nan(output[0]));
    orthonormal.transform(&[T::INFINITY], &mut output);
    assert_eq!(output, [T::INFINITY]);
    orthonormal.transform(&[T::from_f64(-0.0)], &mut output);
    assert!(<T as RealField>::is_sign_positive(output[0]));
}

#[test]
fn basis_matches_analytical_values() {
    analytical_basis::<f32>();
    analytical_basis::<f64>();
}

#[test]
fn basis_is_orthonormal() {
    orthonormal_basis_vectors::<f32>();
    orthonormal_basis_vectors::<f64>();
}

#[test]
fn cached_plan_matches_direct_operation() {
    cached_matches_direct::<f32>();
    cached_matches_direct::<f64>();
}

#[test]
fn length_one_normalizations_and_special_values_are_defined() {
    length_one_values::<f32>();
    length_one_values::<f64>();
}

#[test]
fn zero_capacity_is_rejected() {
    assert_eq!(
        DctIiiPlan::<f64, 0>::new(Normalization::Unnormalized),
        Err(PlanError::EmptyLength)
    );
}

#[test]
fn plan_storage_is_exactly_the_precomputed_basis() {
    assert_eq!(
        core::mem::size_of::<DctIiiPlan<f64, 8>>(),
        64 * core::mem::size_of::<f64>()
    );
}
