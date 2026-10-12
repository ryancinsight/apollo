use super::*;
use crate::domain::metadata::grid::UniformDomain1D;
use crate::DEFAULT_NUFFT_KERNEL_WIDTH;
use proptest::prelude::*;

proptest! {
    /// DC mode invariant: output[0] = Σ_j c_j because exp(-2πi·0·x_j / L) = 1 for all j.
    ///
    /// Formal contract: ∀ n ∈ [4,8], ∀ (x_j, c_j) with x_j ∈ [0,L):
    ///   |f_0 - Σ c_j| < 1e-10
    #[test]
    fn dc_mode_equals_sum_for_arbitrary_positions_and_values(
        n in 4_usize..=8_usize,
        pos_parts in prop::collection::vec(0.0_f64..1.0_f64, 4),
        re_parts  in prop::collection::vec(-1.0_f64..1.0_f64, 4),
        im_parts  in prop::collection::vec(-1.0_f64..1.0_f64, 4),
    ) {
        let domain = UniformDomain1D::new(n, 1.0 / n as f64).unwrap();
        // positions in [0, 0.99) ⊂ [0, length=1.0)
        let positions: Vec<f64> = pos_parts.iter().map(|&p| p * 0.99).collect();
        let values: Vec<Complex64> = re_parts
            .iter()
            .zip(im_parts.iter())
            .map(|(&re, &im)| Complex64::new(re, im))
            .collect();
        let output = nufft_type1_1d(&positions, &values, domain);
        let expected_dc: Complex64 = values.iter().copied().sum();
        let err = (output[[0]] - expected_dc).norm();
        prop_assert!(
            err < 1e-10,
            "DC mode err={}: got {:?}, expected {:?}",
            err, output[[0]], expected_dc
        );
    }

    /// Fast path tracks exact direct summation within relative tolerance 1e-3.
    ///
    /// Formal contract: ∀ n ∈ [8,16], ∀ (x_j, c_j):
    ///   ‖fast_k - exact_k‖ / max_k(‖exact_k‖) < 1e-3
    #[test]
    fn fast_tracks_exact_for_arbitrary_inputs(
        n in 8_usize..=16_usize,
        pos_parts in prop::collection::vec(0.0_f64..1.0_f64, 4),
        re_parts  in prop::collection::vec(-1.0_f64..1.0_f64, 4),
        im_parts  in prop::collection::vec(-1.0_f64..1.0_f64, 4),
    ) {
        let domain = UniformDomain1D::new(n, 1.0 / n as f64).unwrap();
        let positions: Vec<f64> = pos_parts.iter().map(|&p| p * 0.99).collect();
        let values: Vec<Complex64> = re_parts
            .iter()
            .zip(im_parts.iter())
            .map(|(&re, &im)| Complex64::new(re, im))
            .collect();
        let exact = nufft_type1_1d(&positions, &values, domain);
        let fast  = nufft_type1_1d_fast(&positions, &values, domain, DEFAULT_NUFFT_KERNEL_WIDTH);
        let scale = exact
            .iter()
            .map(|v| v.norm())
            .fold(1.0_f64, f64::max)
            .max(1e-30);
        for (k, (e, f)) in exact.iter().zip(fast.iter()).enumerate() {
            let err = (e - f).norm() / scale;
            prop_assert!(
                err < 1e-3,
                "k={}: exact={:?}, fast={:?}, rel_err={}",
                k, e, f, err
            );
        }
    }

    /// Type-1 NUFFT is linear in the source values: T1(s·c) = s·T1(c).
    ///
    /// Formal contract: ∀ n ∈ [4,8], ∀ s ∈ ℤ ∩ [-3,3], ∀ (x_j, c_j):
    ///   ‖T1(s·c)_k - s·T1(c)_k‖ / max_k(‖s·T1(c)_k‖) < 1e-9
    #[test]
    fn type1_is_linear_in_values(
        n      in 4_usize..=8_usize,
        scalar in -3_i32..3_i32,
        pos_parts in prop::collection::vec(0.0_f64..1.0_f64, 4),
        re_parts  in prop::collection::vec(-1.0_f64..1.0_f64, 4),
        im_parts  in prop::collection::vec(-1.0_f64..1.0_f64, 4),
    ) {
        let domain = UniformDomain1D::new(n, 1.0 / n as f64).unwrap();
        let positions: Vec<f64> = pos_parts.iter().map(|&p| p * 0.99).collect();
        let values: Vec<Complex64> = re_parts
            .iter()
            .zip(im_parts.iter())
            .map(|(&re, &im)| Complex64::new(re, im))
            .collect();
        let s = Complex64::new(f64::from(scalar), 0.0);
        let scaled_values: Vec<Complex64> = values.iter().map(|&v| s * v).collect();
        // lhs = T1(s·c);  rhs = s·T1(c)
        let lhs      = nufft_type1_1d(&positions, &scaled_values, domain);
        let rhs_base = nufft_type1_1d(&positions, &values,        domain);
        let rhs: Vec<Complex64> = rhs_base.iter().map(|&v| s * v).collect();
        let scale = rhs
            .iter()
            .map(|v| v.norm())
            .fold(1.0_f64, f64::max)
            .max(1e-30);
        for (k, (l, r)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let err = (l - r).norm() / scale;
            prop_assert!(
                err < 1e-9,
                "k={}: lhs={:?}, rhs={:?}, rel_err={}",
                k, l, r, err
            );
        }
    }
}
