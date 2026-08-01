use super::*;
use crate::infrastructure::kernel::spherical_harmonic::gauss_legendre_nodes_weights;
use eunomia::assert_abs_diff_eq;

fn assert_low_degree_roundoff(actual: f64, expected: f64) {
    // These degree <= 6 checks execute fewer than 64 recurrence/arithmetic
    // operations. Four ulps per operation cover elementary-function rounding,
    // so 256 * epsilon * max(|expected|, 1) is the forward-error bound.
    let epsilon = 256.0 * f64::EPSILON * expected.abs().max(1.0);
    assert_abs_diff_eq!(actual, expected, epsilon = epsilon);
}

#[test]
fn analytical_low_order_values_match() {
    let r00 = 1.0 / (4.0 * std::f64::consts::PI).sqrt();
    for &(theta, phi) in &[
        (0.0, 0.0),
        (std::f64::consts::FRAC_PI_2, 0.0),
        (std::f64::consts::PI, 1.0),
    ] {
        assert_low_degree_roundoff(
            real_spherical_harmonic(0, 0, theta, phi).expect("valid harmonic"),
            r00,
        );
    }

    let polar_r20 = 0.5 * (5.0 / std::f64::consts::PI).sqrt();
    assert_low_degree_roundoff(
        real_spherical_harmonic(2, 0, 0.0, 0.0).expect("valid harmonic"),
        polar_r20,
    );
    let equatorial_r20 = -0.25 * (5.0 / std::f64::consts::PI).sqrt();
    assert_low_degree_roundoff(
        real_spherical_harmonic(2, 0, std::f64::consts::FRAC_PI_2, 0.0).expect("valid harmonic"),
        equatorial_r20,
    );
}

#[test]
fn real_components_match_complex_basis() {
    let harmonic = spherical_harmonic(2, 2, 0.7, 1.2);
    assert_eq!(
        real_spherical_harmonic(2, 2, 0.7, 1.2).expect("valid harmonic"),
        std::f64::consts::SQRT_2 * harmonic.re,
    );
    assert_eq!(
        real_spherical_harmonic(2, -2, 0.7, 1.2).expect("valid harmonic"),
        std::f64::consts::SQRT_2 * harmonic.im,
    );
}

#[test]
fn negative_odd_order_matches_mrtrix_closed_form() {
    // MRtrix defines R_l^m = sqrt(2) Im(Y_l^{-m}) for m < 0. Therefore
    // R_1^-1(pi/2, pi/2) = -sqrt(3 / (4 pi)); this independent value catches
    // an erroneous (-1)^m factor that an even negative-order case cannot.
    let expected = -(3.0 / (4.0 * std::f64::consts::PI)).sqrt();
    let actual = real_spherical_harmonic(
        1,
        -1,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::FRAC_PI_2,
    )
    .expect("valid MRtrix reference direction");
    assert_low_degree_roundoff(actual, expected);
}

#[test]
fn basis_validates_degree_and_counts_coefficients() {
    assert_eq!(
        RealSphericalHarmonicBasis::new(0).expect_err("zero is not useful"),
        RealShError::TooSmall(0),
    );
    assert_eq!(
        RealSphericalHarmonicBasis::new(3).expect_err("odd degree breaks symmetry"),
        RealShError::OddLMax(3),
    );
    assert_eq!(
        RealSphericalHarmonicBasis::new(86).expect_err("degree exceeds stable range"),
        RealShError::DegreeOutOfRange {
            degree: 86,
            maximum: MAX_REAL_SH_DEGREE,
        },
    );
    for (degree, expected) in [(2, 6), (4, 15), (6, 28)] {
        assert_eq!(
            RealSphericalHarmonicBasis::new(degree)
                .expect("valid even basis")
                .num_coefficients(),
            expected,
        );
    }
}

#[test]
fn coefficient_indices_are_checked_and_ordered() {
    let basis = RealSphericalHarmonicBasis::new(2).expect("valid even basis");
    let expected = [(0, 0), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)];
    for (index, pair) in expected.into_iter().enumerate() {
        assert_eq!(basis.index_to_lm(index), Some(pair));
    }
    assert_eq!(basis.index_to_lm(expected.len()), None);
}

#[test]
fn public_evaluation_rejects_invalid_inputs() {
    assert_eq!(
        real_spherical_harmonic(2, 3, 0.5, 1.0).expect_err("order is invalid"),
        RealShError::InvalidOrder {
            degree: 2,
            order: 3,
        },
    );
    assert!(matches!(
        real_spherical_harmonic(2, 0, f64::NAN, 1.0),
        Err(RealShError::InvalidTheta(value)) if value.is_nan()
    ));
    assert_eq!(
        real_spherical_harmonic(2, 0, -f64::EPSILON, 1.0).expect_err("negative theta is invalid"),
        RealShError::InvalidTheta(-f64::EPSILON),
    );
    assert!(matches!(
        real_spherical_harmonic(2, 0, 0.5, f64::INFINITY),
        Err(RealShError::InvalidPhi(value)) if value == f64::INFINITY
    ));
}

#[test]
fn maximum_supported_degree_remains_finite() {
    for order in -(MAX_REAL_SH_DEGREE as isize)..=MAX_REAL_SH_DEGREE as isize {
        let value = real_spherical_harmonic(MAX_REAL_SH_DEGREE, order, 1.0, 2.0)
            .expect("the documented degree boundary must remain finite");
        assert!(value.is_finite());
    }
}

#[test]
fn direction_evaluation_validates_finite_unit_vectors() {
    let basis = RealSphericalHarmonicBasis::new(2).expect("valid even basis");
    let from_direction = basis
        .evaluate_at_direction(&[1.0, 0.0, 0.0])
        .expect("unit x direction");
    let from_angles = basis
        .evaluate(std::f64::consts::FRAC_PI_2, 0.0)
        .expect("valid angles");
    assert_eq!(from_direction, from_angles);

    let diagonal_component = 3.0_f64.sqrt().recip();
    let diagonal = basis
        .evaluate_at_direction(&[diagonal_component; 3])
        .expect("a binary64-normalized diagonal direction must satisfy the norm bound");
    assert_eq!(diagonal.len(), basis.num_coefficients());
    assert!(diagonal.into_iter().all(f64::is_finite));

    assert!(matches!(
        basis.evaluate_at_direction(&[f64::NAN, 0.0, 1.0]),
        Err(RealShError::NonFiniteDirection { axis: 0, value }) if value.is_nan()
    ));
    assert!(matches!(
        basis.evaluate_at_direction(&[2.0, 0.0, 0.0]),
        Err(RealShError::NonUnitDirection { norm_squared, .. }) if norm_squared == 4.0
    ));
    assert!(matches!(
        basis.evaluate_at_direction(&[0.0, 0.0, 0.0]),
        Err(RealShError::NonUnitDirection { norm_squared, .. }) if norm_squared == 0.0
    ));
}

#[test]
fn design_matrix_matches_direct_evaluation() {
    let basis = RealSphericalHarmonicBasis::new(4).expect("valid even basis");
    let directions = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let matrix = basis
        .design_matrix(&directions)
        .expect("valid direction matrix");
    assert_eq!(matrix.shape(), [directions.len(), basis.num_coefficients()]);

    for (row, direction) in directions.iter().enumerate() {
        let expected = basis
            .evaluate_at_direction(direction)
            .expect("valid unit direction");
        for (column, value) in expected.into_iter().enumerate() {
            assert_eq!(matrix[[row, column]], value);
        }
    }
}

#[test]
fn design_matrix_first_column_is_constant() {
    let basis = RealSphericalHarmonicBasis::new(4).expect("valid even basis");
    let directions = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
    ];
    let matrix = basis
        .design_matrix(&directions)
        .expect("valid direction matrix");
    let expected = 1.0 / (4.0 * std::f64::consts::PI).sqrt();
    for row in 0..directions.len() {
        assert_low_degree_roundoff(matrix[[row, 0]], expected);
    }
}

#[test]
fn gauss_legendre_fourier_quadrature_proves_orthonormality() {
    let basis = RealSphericalHarmonicBasis::new(4).expect("valid even basis");
    // Eight-point Gauss-Legendre integrates degree <= 15 polynomials exactly;
    // products at l_max=4 have degree <= 8 in cos(theta). Sixteen uniform
    // azimuths integrate every Fourier product mode through |m|=8 exactly.
    let (cos_theta, theta_weights) = gauss_legendre_nodes_weights(8);
    let phi_count = 16;
    let phi_weight = std::f64::consts::TAU / phi_count as f64;
    let term_count = cos_theta.len() * phi_count;

    for left in 0..basis.num_coefficients() {
        for right in 0..basis.num_coefficients() {
            let mut inner_product = 0.0;
            let mut absolute_sum = 0.0;
            for (&node, &theta_weight) in cos_theta.iter().zip(&theta_weights) {
                let theta = node.acos();
                for phi_index in 0..phi_count {
                    let phi = std::f64::consts::TAU * phi_index as f64 / phi_count as f64;
                    let row = basis.evaluate(theta, phi).expect("valid quadrature point");
                    let term = row[left] * row[right] * theta_weight * phi_weight;
                    inner_product += term;
                    absolute_sum += term.abs();
                }
            }
            let expected = if left == right { 1.0 } else { 0.0 };
            // Sequential summation contributes gamma_n * sum(|term|). A factor
            // of 64 covers low-degree basis recurrence and trig evaluation.
            let n_epsilon = term_count as f64 * f64::EPSILON;
            let gamma_n = n_epsilon / (1.0 - n_epsilon);
            let epsilon = 64.0 * gamma_n * absolute_sum.max(1.0);
            assert_abs_diff_eq!(inner_product, expected, epsilon = epsilon);
        }
    }
}

#[test]
fn even_and_odd_degrees_have_expected_antipodal_parity() {
    for degree in 0..=6 {
        for order in -(degree as isize)..=degree as isize {
            let value = real_spherical_harmonic(degree, order, 1.0, 2.0).expect("valid harmonic");
            let antipodal = real_spherical_harmonic(
                degree,
                order,
                std::f64::consts::PI - 1.0,
                2.0 + std::f64::consts::PI,
            )
            .expect("valid antipodal harmonic");
            let expected = if degree % 2 == 0 { value } else { -value };
            assert_low_degree_roundoff(antipodal, expected);
        }
    }
}
