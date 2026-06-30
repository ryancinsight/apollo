//! Verification modules for quantum Fourier transforms.

#[cfg(test)]
mod tests {
    use crate::{iqft, is_valid_length, qft, QftError, QftPlan, QuantumStateDimension};
    use approx::assert_relative_eq;
    use leto::Array1;
    use eunomia::Complex64;
    use proptest::prelude::*;
    use proptest::proptest;

    fn norm_sqr(values: &Array1<Complex64>) -> f64 {
        values.iter().map(|value| value.norm_sqr()).sum()
    }

    #[test]
    fn two_point_qft_matches_reference() {
        let plan = QftPlan::new(QuantumStateDimension::new(2).expect("valid dim"));
        let input = Array1::from(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        let output = plan.forward(&input).expect("forward");
        let scale = 1.0 / 2.0_f64.sqrt();
        assert_relative_eq!(output[0].re, scale, epsilon = 1.0e-12);
        assert_relative_eq!(output[0].im, 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(output[1].re, scale, epsilon = 1.0e-12);
        assert_relative_eq!(output[1].im, 0.0, epsilon = 1.0e-12);
    }

    #[test]
    fn roundtrip_and_norm_preservation_hold() {
        let plan = QftPlan::new(QuantumStateDimension::new(4).expect("valid dim"));
        let input = Array1::from(vec![
            Complex64::new(1.0, -1.0),
            Complex64::new(0.5, 0.25),
            Complex64::new(-0.75, 0.125),
            Complex64::new(0.0, 0.5),
        ]);
        let spectrum = plan.forward(&input).expect("forward");
        assert_relative_eq!(norm_sqr(&spectrum), norm_sqr(&input), epsilon = 1.0e-12);
        let recovered = plan.inverse(&spectrum).expect("inverse");
        for (actual, expected) in recovered.iter().zip(input.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn inplace_and_convenience_paths_match_allocating_paths() {
        let plan = QftPlan::new(QuantumStateDimension::new(8).expect("valid dim"));
        let input = Array1::from(
            (0..8)
                .map(|i| Complex64::new(i as f64, -(i as f64) * 0.25))
                .collect::<Vec<_>>(),
        );
        let mut inplace = input.clone();
        plan.forward_inplace(&mut inplace).expect("forward inplace");
        let allocated = plan.forward(&input).expect("forward");
        let convenience = qft(&input).expect("qft");
        for ((actual, expected), wrapped) in
            inplace.iter().zip(allocated.iter()).zip(convenience.iter())
        {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
            assert_relative_eq!(wrapped.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(wrapped.im, expected.im, epsilon = 1.0e-12);
        }
        let recovered = iqft(&allocated).expect("iqft");
        for (actual, expected) in recovered.iter().zip(input.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn rejects_invalid_lengths() {
        assert!(!is_valid_length(0));
        assert!(is_valid_length(1));
        assert!(matches!(
            QuantumStateDimension::new(0),
            Err(QftError::EmptyLength)
        ));
        let plan = QftPlan::new(QuantumStateDimension::new(4).expect("valid dim"));
        let bad = Array1::from(vec![Complex64::new(0.0, 0.0); 3]);
        assert!(matches!(plan.forward(&bad), Err(QftError::LengthMismatch)));
        assert!(matches!(plan.inverse(&bad), Err(QftError::LengthMismatch)));
    }

    proptest! {
        #[test]
        fn forward_inverse_roundtrip_holds_for_random_vectors(
            len in 1usize..16,
            data in prop::collection::vec((-5.0f64..5.0f64, -5.0f64..5.0f64), 1usize..16)
        ) {
            let plan = QftPlan::new(QuantumStateDimension::new(len).expect("valid dim"));
            let mut values = Vec::with_capacity(len);
            for (re, im) in data.into_iter().cycle().take(len) {
                values.push(Complex64::new(re, im));
            }
            let input = Array1::from(values);
            let spectrum = plan.forward(&input).expect("forward");
            prop_assert!((norm_sqr(&spectrum) - norm_sqr(&input)).abs() < 1.0e-9);
            let recovered = plan.inverse(&spectrum).expect("inverse");
            for (actual, expected) in recovered.iter().zip(input.iter()) {
                prop_assert!((actual.re - expected.re).abs() < 1.0e-10);
                prop_assert!((actual.im - expected.im).abs() < 1.0e-10);
            }
        }
    }
    #[test]
    fn qft_matrix_is_unitary() {
        // For n=4, verify U * U_dagger = I:
        // columns of U are orthonormal under the inner product <u_i, u_j> = delta_ij.
        let n = 4usize;
        let plan = QftPlan::new(QuantumStateDimension::new(n).expect("valid dim"));
        let mut u_cols: Vec<Array1<eunomia::Complex64>> = Vec::new();
        for j in 0..n {
            let mut basis = Array1::zeros([n]);
            basis[j] = eunomia::Complex64::new(1.0, 0.0);
            u_cols.push(plan.forward(&basis).expect("forward"));
        }
        for i in 0..n {
            for j in 0..n {
                let dot: eunomia::Complex64 = u_cols[i]
                    .iter()
                    .zip(u_cols[j].iter())
                    .map(|(a, b)| a.conj() * b)
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot.re - expected).abs() < 1e-10 && dot.im.abs() < 1e-10,
                    "U not unitary at ({i},{j}): dot = {dot:?}"
                );
            }
        }
    }

    #[test]
    fn qft_n1_is_identity() {
        // n=1: scale = 1/sqrt(1)=1, twiddle[0]=exp(0)=1. Output = input.
        let plan = QftPlan::new(QuantumStateDimension::new(1).expect("valid dim"));
        let input = Array1::from(vec![eunomia::Complex64::new(3.7, -1.2)]);
        let result = plan.forward(&input).expect("forward");
        assert!((result[0] - input[0]).norm() < 1e-14);
    }

    #[test]
    fn qft_unitarity_holds_for_multiple_sizes() {
        // QFT satisfies U†U = I for all n ≥ 1 (Shor 1994, Nielsen & Chuang 2010).
        // Proof: M[k,j] = exp(2πijk/n)/√n. Then (M†M)[j,j'] = (1/n) Σ_k exp(2πi(j-j')k/n)
        //   = δ(j,j') by DFT orthogonality, so M†M = I and ||QFT(x)||² = ||x||² for all x.
        use approx::assert_abs_diff_eq;
        use std::f64::consts::TAU;
        for n in [2usize, 3, 4, 5, 6, 8] {
            let dim = QuantumStateDimension::new(n).expect("valid dim");
            let plan = QftPlan::new(dim);
            // x[j] = cos(2πj/n) + i·sin(4πj/n): analytically constructed, no round numbers.
            let input: Array1<Complex64> = Array1::from_shape_fn([n], |[j]| {
                Complex64::new(
                    (TAU * j as f64 / n as f64).cos(),
                    (TAU * 2.0 * j as f64 / n as f64).sin(),
                )
            });
            let energy_in: f64 = input.iter().map(|c| c.norm_sqr()).sum();
            let spectrum = plan.forward(&input).expect("forward");
            let energy_out: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum();
            assert_abs_diff_eq!(energy_out, energy_in, epsilon = 1e-10);
        }
    }

    proptest! {
        #[test]
        fn qft_unitarity_holds_for_random_size_and_input(
            n in 2usize..=8,
            // Always generate 8 pairs so we can cycle-take n regardless of generated n.
            data in proptest::collection::vec((-5.0f64..5.0f64, -5.0f64..5.0f64), 8usize..=8)
        ) {
            // QFT unitarity (Shor 1994, Nielsen & Chuang 2010):
            // ||QFT_n(x)||² = ||x||² for all n ≥ 1 and x ∈ ℂⁿ.
            let dim = QuantumStateDimension::new(n).expect("valid dim");
            let plan = QftPlan::new(dim);
            let input: Array1<Complex64> = Array1::from(
                data.into_iter().cycle().take(n).map(|(re, im)| Complex64::new(re, im)).collect::<Vec<_>>(),
            );
            let energy_in: f64 = input.iter().map(|c| c.norm_sqr()).sum();
            let spectrum = plan.forward(&input).expect("forward");
            let energy_out: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum();
            prop_assert!(
                (energy_in - energy_out).abs() < 1e-10,
                "n={n}, energy_in={energy_in:.15}, energy_out={energy_out:.15}, diff={}",
                (energy_in - energy_out).abs()
            );
        }
    }

    #[test]
    fn qft_n3_roundtrip() {
        // n=3 is not a power of two; the dense QFT handles arbitrary positive lengths.
        let plan = QftPlan::new(QuantumStateDimension::new(3).expect("valid dim"));
        let input = Array1::from(vec![
            eunomia::Complex64::new(1.0, 0.5),
            eunomia::Complex64::new(-0.5, 1.0),
            eunomia::Complex64::new(0.25, -0.75),
        ]);
        let spectrum = plan.forward(&input).expect("forward");
        let recovered = plan.inverse(&spectrum).expect("inverse");
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert!((a - b).norm() < 1e-12, "n=3 roundtrip failed");
        }
    }
}
