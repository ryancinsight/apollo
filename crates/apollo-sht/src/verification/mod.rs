//! Verification modules for SHT.

#[cfg(test)]
mod tests {
    use crate::domain::spectrum::coefficients::SphericalHarmonicCoefficients;
    use crate::infrastructure::kernel::spherical_harmonic::{
        associated_legendre, gauss_legendre_nodes_weights, spherical_harmonic,
    };
    use crate::{ShtError, ShtPlan};
    use approx::assert_abs_diff_eq;
    use leto::Array2;
    use eunomia::Complex64;
    use proptest::prelude::*;
    use proptest::proptest;

    #[test]
    fn plan_preserves_grid_specification() {
        let plan = ShtPlan::new(8, 17, 4).expect("plan");
        assert_eq!(plan.grid().latitudes(), 8);
        assert_eq!(plan.grid().longitudes(), 17);
        assert_eq!(plan.grid().max_degree(), 4);
    }

    #[test]
    fn rejects_invalid_sampling() {
        assert_eq!(
            ShtPlan::new(0, 8, 1).unwrap_err(),
            ShtError::EmptyLatitudeCount
        );
        assert_eq!(
            ShtPlan::new(8, 0, 1).unwrap_err(),
            ShtError::EmptyLongitudeCount
        );
        assert_eq!(
            ShtPlan::new(4, 5, 4).unwrap_err(),
            ShtError::DegreeExceedsSampling
        );
    }

    #[test]
    fn constant_surface_maps_to_l0_m0_coefficient() {
        let plan = ShtPlan::new(12, 25, 4).expect("plan");
        let constant = 1.0 / (4.0 * std::f64::consts::PI).sqrt();
        let samples = Array2::from_elem(
            [plan.grid().latitudes(), plan.grid().longitudes()],
            constant,
        );

        let coefficients = plan.forward_real(&samples).expect("forward");

        assert_abs_diff_eq!(coefficients.get(0, 0).re, 1.0, epsilon = 1.0e-12);
        assert_abs_diff_eq!(coefficients.get(0, 0).im, 0.0, epsilon = 1.0e-12);
        for degree in 1..=plan.grid().max_degree() {
            for order in -(degree as isize)..=(degree as isize) {
                assert_abs_diff_eq!(
                    coefficients.get(degree, order).norm(),
                    0.0,
                    epsilon = 1.0e-12
                );
            }
        }

        let recovered = plan.inverse_real(&coefficients).expect("inverse");
        for value in recovered.iter().copied() {
            assert_abs_diff_eq!(value, constant, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn complex_harmonic_roundtrip_recovers_single_mode() {
        let plan = ShtPlan::new(14, 31, 5).expect("plan");
        let degree = 2;
        let order = 1;
        let samples = Array2::from_shape_fn(
            [plan.grid().latitudes(), plan.grid().longitudes()],
            |[i, j]| spherical_harmonic(degree, order, plan.theta(i), plan.phi(j)),
        );

        let coefficients = plan.forward_complex(&samples).expect("forward");

        for l in 0..=plan.grid().max_degree() {
            for m in -(l as isize)..=(l as isize) {
                let expected = if l == degree && m == order {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert_abs_diff_eq!(coefficients.get(l, m).re, expected.re, epsilon = 1.0e-12);
                assert_abs_diff_eq!(coefficients.get(l, m).im, expected.im, epsilon = 1.0e-12);
            }
        }

        let recovered = plan.inverse_complex(&coefficients).expect("inverse");
        for (actual, expected) in recovered.iter().zip(samples.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn rejects_mismatched_sample_and_coefficient_shapes() {
        let plan = ShtPlan::new(8, 17, 3).expect("plan");
        let bad_samples = Array2::<f64>::zeros([7, 17]);
        assert_eq!(
            plan.forward_real(&bad_samples).unwrap_err(),
            ShtError::SampleShapeMismatch
        );

        let bad_coefficients = SphericalHarmonicCoefficients::zeros(2);
        assert_eq!(
            plan.inverse_real(&bad_coefficients).unwrap_err(),
            ShtError::CoefficientShapeMismatch
        );
    }

    #[test]
    fn associated_legendre_known_values() {
        use crate::infrastructure::kernel::spherical_harmonic::associated_legendre;
        // P_0^0(x) = 1 for all x
        assert!(
            (associated_legendre(0, 0, 0.5) - 1.0).abs() < 1e-14,
            "P_0^0 should be 1"
        );

        // P_1^0(x) = x
        assert!(
            (associated_legendre(1, 0, 0.7) - 0.7).abs() < 1e-14,
            "P_1^0 should be x"
        );

        // P_1^1(x) = -(1-x^2)^{1/2} Condon-Shortley convention
        let x = 0.5_f64;
        let expected_p11 = -(1.0 - x * x).sqrt();
        assert!(
            (associated_legendre(1, 1, x) - expected_p11).abs() < 1e-14,
            "P_1^1(0.5): expected {expected_p11}, got {}",
            associated_legendre(1, 1, x)
        );

        // P_2^0(x) = (3x^2 - 1)/2
        let x = 0.3_f64;
        let expected_p20 = (3.0 * x * x - 1.0) / 2.0;
        assert!(
            (associated_legendre(2, 0, x) - expected_p20).abs() < 1e-14,
            "P_2^0(0.3): expected {expected_p20}, got {}",
            associated_legendre(2, 0, x)
        );

        // P_2^0(0) = -1/2
        assert!(
            (associated_legendre(2, 0, 0.0) + 0.5).abs() < 1e-14,
            "P_2^0(0) should be -0.5"
        );
    }

    #[test]
    fn gauss_legendre_weights_sum_to_2() {
        use crate::infrastructure::kernel::spherical_harmonic::gauss_legendre_nodes_weights;
        // GL weights for n points sum to 2 = integral_{-1}^{1} 1 dx.
        for n in [2_usize, 4, 8, 16] {
            let (_nodes, weights) = gauss_legendre_nodes_weights(n);
            let sum: f64 = weights.iter().sum();
            assert!(
                (sum - 2.0).abs() < 1e-12,
                "GL weights for n={n} should sum to 2.0, got {sum}"
            );
        }
    }

    #[test]
    fn gauss_legendre_integrates_polynomial_exactly() {
        use crate::infrastructure::kernel::spherical_harmonic::gauss_legendre_nodes_weights;
        // GL with n=2 integrates polynomials of degree <= 3 exactly.
        // integral_{-1}^{1} x^2 dx = 2/3.
        let (nodes, weights) = gauss_legendre_nodes_weights(2);
        let integral: f64 = nodes
            .iter()
            .zip(weights.iter())
            .map(|(x, w)| w * x * x)
            .sum();
        assert!(
            (integral - 2.0 / 3.0).abs() < 1e-14,
            "GL quadrature of x^2 should be 2/3, got {integral}"
        );
    }

    #[test]
    fn gauss_legendre_exactness_for_degree_2n_minus_1_polynomials() {
        // Theorem 2: the n-point GL rule integrates x^k exactly for k = 0, 1, ..., 2n-1.
        // ∫_{-1}^{1} x^k dx = 2/(k+1) when k is even, 0 when k is odd.
        for &n in &[2_usize, 3, 4, 5, 8, 16] {
            let (nodes, weights) = gauss_legendre_nodes_weights(n);
            for k in 0..=(2 * n - 1) {
                let exact = if k % 2 == 0 {
                    2.0 / (k + 1) as f64
                } else {
                    0.0
                };
                let computed: f64 = nodes
                    .iter()
                    .zip(weights.iter())
                    .map(|(x, w)| w * x.powi(k as i32))
                    .sum();
                assert!(
                    (computed - exact).abs() < 1e-10,
                    "GL n={n}, k={k}: expected {exact}, got {computed}, err={}",
                    (computed - exact).abs()
                );
            }
        }
    }

    #[test]
    fn spherical_harmonic_y00_is_analytical_value() {
        // Y_0^0(theta, phi) = 1/sqrt(4π) for all (theta, phi).
        // Real part must equal 1/sqrt(4π); imaginary part must be 0.
        let expected = 1.0 / (4.0 * std::f64::consts::PI).sqrt();
        for &theta in &[
            0.0_f64,
            0.5,
            1.0,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
        ] {
            for &phi in &[0.0_f64, 1.0, std::f64::consts::PI, std::f64::consts::TAU] {
                let y00 = spherical_harmonic(0, 0, theta, phi);
                assert!(
                    (y00.re - expected).abs() < 1e-14,
                    "Y_0^0 real part at (theta={theta}, phi={phi}): expected {expected}, got {}",
                    y00.re
                );
                assert!(
                    y00.im.abs() < 1e-14,
                    "Y_0^0 imaginary part at (theta={theta}, phi={phi}): expected 0, got {}",
                    y00.im
                );
            }
        }
    }

    #[test]
    fn sht_parseval_energy_conservation() {
        // Parseval identity: Σ_{lm} |a_lm|^2 = ∫_{S²} |f(Ω)|^2 dΩ for orthonormal Y_lm.
        // With f = Y_0^0 = 1/sqrt(4π), orthonormality gives a_00 = 1 and all other a_lm = 0.
        // Therefore: Σ |a_lm|^2 = 1, consistent with ∫ |Y_0^0|^2 dΩ = 1.
        let plan = ShtPlan::new(6, 7, 2).expect("plan");
        let samples: Array2<f64> = Array2::from_shape_fn(
            [plan.grid().latitudes(), plan.grid().longitudes()],
            |[i, j]| spherical_harmonic(0, 0, plan.theta(i), plan.phi(j)).re,
        );
        let coefficients = plan.forward_real(&samples).expect("forward");

        // a_00 must equal 1.0 — all energy is in the l=0, m=0 mode.
        assert_abs_diff_eq!(coefficients.get(0, 0).re, 1.0, epsilon = 0.01);
        assert_abs_diff_eq!(coefficients.get(0, 0).im, 0.0, epsilon = 0.01);

        // All higher-degree coefficients must vanish by orthonormality.
        for degree in 1..=plan.grid().max_degree() {
            for order in -(degree as isize)..=(degree as isize) {
                assert!(
                    coefficients.get(degree, order).norm() < 0.05,
                    "a_{degree},{order} = {} should be ~0",
                    coefficients.get(degree, order).norm()
                );
            }
        }

        // Parseval sum Σ |a_lm|^2 must equal 1.0.
        let parseval_sum: f64 = (0..=plan.grid().max_degree())
            .flat_map(|l| (-(l as isize)..=(l as isize)).map(move |m| (l, m)))
            .map(|(l, m)| coefficients.get(l, m).norm_sqr())
            .sum();
        assert!(
            (parseval_sum - 1.0).abs() < 0.01,
            "Parseval sum = {parseval_sum}, expected ≈ 1.0"
        );
    }

    #[test]
    fn sht_orthonormal_basis_roundtrip_for_l1_modes() {
        // For each mode (l, m) in {(0,0), (1,-1), (1,0), (1,1)}: set a_{lm} = 1, all
        // others = 0, compute inverse SHT, then forward SHT. The recovered coefficient
        // for (l, m) must be ≈ 1 and all others must be ≈ 0 (roundtrip exactness,
        // Theorem 4). Grid: N_lat=4 > L=1, N_lon=5 >= 2*1+1=3.
        let plan = ShtPlan::new(4, 5, 1).expect("plan");
        let modes: &[(usize, isize)] = &[(0, 0), (1, -1), (1, 0), (1, 1)];
        for &(l, m) in modes {
            let mut coeffs = SphericalHarmonicCoefficients::zeros(1);
            coeffs.set(l, m, Complex64::new(1.0, 0.0));

            let field = plan.inverse_complex(&coeffs).expect("inverse");
            let recovered = plan.forward_complex(&field).expect("forward");

            assert_abs_diff_eq!(recovered.get(l, m).re, 1.0, epsilon = 1e-4);
            assert_abs_diff_eq!(recovered.get(l, m).im, 0.0, epsilon = 1e-4);

            for &(l2, m2) in modes {
                if l2 == l && m2 == m {
                    continue;
                }
                assert!(
                    recovered.get(l2, m2).norm() < 1e-4,
                    "mode ({l},{m}) set: a_{l2},{m2} = {} should be ~0",
                    recovered.get(l2, m2).norm()
                );
            }
        }
    }

    #[test]
    fn associated_legendre_recurrence_parity_against_known_p22() {
        // P_2^2(x) = 3(1 - x^2) (Condon-Shortley: (-1)^2 = 1, so same as standard).
        // At x = 0.5: P_2^2(0.5) = 3 * (1 - 0.25) = 2.25.
        let x = 0.5_f64;
        let expected_p22 = 3.0 * (1.0 - x * x);
        assert!(
            (associated_legendre(2, 2, x) - expected_p22).abs() < 1e-12,
            "P_2^2(0.5): expected {expected_p22}, got {}",
            associated_legendre(2, 2, x)
        );

        // P_3^1(x) = -3/2 * (5x^2 - 1) * sqrt(1 - x^2) (Condon-Shortley: (-1)^1 = -1).
        // At x = 0.5: -3/2 * (1.25 - 1) * sqrt(0.75) = -3/2 * 0.25 * sqrt(0.75).
        let expected_p31 = -1.5_f64 * (5.0 * x * x - 1.0) * (1.0 - x * x).sqrt();
        assert!(
            (associated_legendre(3, 1, x) - expected_p31).abs() < 1e-10,
            "P_3^1(0.5): expected {expected_p31}, got {}",
            associated_legendre(3, 1, x)
        );
    }

    proptest! {
        /// SHT forward-then-inverse recovers a band-limited field within floating-point precision.
        ///
        /// A field produced by `inverse_complex` on arbitrary coefficients lies in the
        /// band-limited subspace spanned by Y_l^m for l <= lmax. GL quadrature with
        /// n_lat = lmax+1 nodes is exact for polynomials of degree <= 2*lmax+1, which
        /// covers all products Y_l^m * conj(Y_{l2}^{m2}) for l, l2 <= lmax. The DFT on
        /// n_lon = 2*lmax+2 points aliases at |m| >= lmax+1, outside the valid range.
        /// Roundtrip error is therefore bounded by floating-point rounding only.
        #[test]
        fn sht_roundtrip_small_degree(
            lmax in 1_usize..4,
            seed in 0_u64..20,
        ) {
            let n_lat = lmax + 1;
            let n_lon = 2 * lmax + 2;
            let plan = ShtPlan::new(n_lat, n_lon, lmax).unwrap();

            // Build a band-limited test field via random coefficients then inverse SHT.
            let mut coeffs = SphericalHarmonicCoefficients::zeros(lmax);
            for l in 0..=lmax {
                for m in -(l as isize)..=(l as isize) {
                    let idx = l as u64 * 100 + (m + lmax as isize) as u64;
                    let re =
                        (seed.wrapping_mul(idx.wrapping_add(1)).wrapping_mul(6364136223846793005)
                            >> 33) as f64
                            / ((1_u64 << 31) as f64)
                            - 1.0;
                    let im =
                        (seed.wrapping_mul(idx.wrapping_add(2)).wrapping_mul(2862933555777941757)
                            >> 33) as f64
                            / ((1_u64 << 31) as f64)
                            - 1.0;
                    coeffs.set(l, m, Complex64::new(re, im));
                }
            }

            let field = plan.inverse_complex(&coeffs).unwrap();
            let recovered_coeffs = plan.forward_complex(&field).unwrap();
            let recovered = plan.inverse_complex(&recovered_coeffs).unwrap();

            let err: f64 = field
                .iter()
                .zip(recovered.iter())
                .map(|(a, b)| (a - b).norm())
                .fold(0.0_f64, f64::max);
            prop_assert!(
                err < 1e-6,
                "SHT roundtrip failed: lmax={lmax}, seed={seed}, err={err}"
            );
        }
    }
}
