//! Sparse Fourier transform tests.

use crate::{SparseFftPlan, SparseSpectrum};
use apollo_fft::ApolloError;
use num_complex::Complex64;

fn approx_eq(a: Complex64, b: Complex64, eps: f64) -> bool {
    (a.re - b.re).abs() <= eps && (a.im - b.im).abs() <= eps
}

#[test]
fn constructor_sets_expected_parameters() {
    let plan = SparseFftPlan::new(64, 2).expect("plan");
    assert_eq!(plan.len(), 64);
    assert_eq!(plan.sparsity(), 2);
    assert_eq!(plan.bucket_count(), 8);
    assert!(plan.trials() >= 4);
    assert_eq!(plan.threshold(), 0.0);
}

#[test]
fn rejects_zero_length() {
    let err = SparseFftPlan::new(0, 1).expect_err("expected validation error");
    match err {
        ApolloError::Validation {
            field,
            value,
            constraint,
        } => {
            assert_eq!(field, "n");
            assert_eq!(value, "0");
            assert!(constraint.contains("non-zero"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn rejects_zero_sparsity() {
    let err = SparseFftPlan::new(8, 0).expect_err("expected validation error");
    match err {
        ApolloError::Validation {
            field,
            value,
            constraint,
        } => {
            assert_eq!(field, "k");
            assert_eq!(value, "0");
            assert!(constraint.contains("non-zero"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn sparse_spectrum_insert_and_validate() {
    let mut spectrum = SparseSpectrum::new(16);
    spectrum
        .insert(3, Complex64::new(1.0, -2.0))
        .expect("insert");
    spectrum
        .insert(7, Complex64::new(-0.5, 0.25))
        .expect("insert");
    spectrum.validate().expect("valid spectrum");
    assert_eq!(spectrum.frequencies, vec![3, 7]);
    assert_eq!(spectrum.values.len(), 2);
}

#[test]
fn forward_retains_largest_coefficients() {
    let plan = SparseFftPlan::new(8, 2).expect("plan");
    let signal = vec![
        Complex64::new(4.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];

    let spectrum = plan.forward(&signal).expect("forward");
    assert_eq!(spectrum.len(), 2);
    assert_eq!(spectrum.frequencies, vec![0, 2]);
    assert!(approx_eq(
        spectrum.values[0],
        Complex64::new(5.0, 0.0),
        1.0e-12
    ));
    assert!(approx_eq(
        spectrum.values[1],
        Complex64::new(5.0, 0.0),
        1.0e-12
    ));
}

#[test]
fn inverse_roundtrip_on_retained_support() {
    let plan = SparseFftPlan::new(8, 2).expect("plan");
    let mut spectrum = SparseSpectrum::new(8);
    spectrum
        .insert(1, Complex64::new(2.0, -1.0))
        .expect("insert");
    spectrum
        .insert(6, Complex64::new(-0.5, 0.75))
        .expect("insert");

    let signal = plan.inverse(&spectrum).expect("inverse");
    let recovered = plan.forward(&signal).expect("forward");
    assert_eq!(recovered.frequencies, spectrum.frequencies);
    for (actual, expected) in recovered.values.iter().zip(spectrum.values.iter()) {
        assert!(approx_eq(*actual, *expected, 1.0e-10));
    }
}

#[test]
fn support_export_is_consistent() {
    let plan = SparseFftPlan::new(8, 2).expect("plan");
    let mut spectrum = SparseSpectrum::new(8);
    spectrum
        .insert(2, Complex64::new(1.0, 0.0))
        .expect("insert");
    spectrum
        .insert(5, Complex64::new(0.0, 2.0))
        .expect("insert");

    let support = plan.support(&spectrum);
    assert_eq!(
        support,
        vec![(2, Complex64::new(1.0, 0.0)), (5, Complex64::new(0.0, 2.0))]
    );
}

#[test]
fn dense_roundtrip_matches_inverse_formula() {
    let plan = SparseFftPlan::new(4, 2).expect("plan");
    let signal = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, -1.0),
    ];
    let spectrum = plan.forward(&signal).expect("forward");
    let reconstructed = plan.inverse(&spectrum).expect("inverse");
    let ref_spectrum = crate::infrastructure::kernel::direct::dft(&signal, false);
    let ref_signal = crate::infrastructure::kernel::direct::dft(&ref_spectrum, true);
    for (actual, expected) in reconstructed.iter().zip(ref_signal.iter()) {
        assert!(approx_eq(*actual, *expected, 1.0e-10));
    }
}

#[test]
fn forward_all_zero_signal_returns_all_zero_spectrum() {
    let plan = SparseFftPlan::new(8, 3).unwrap();
    let signal = vec![Complex64::new(0.0, 0.0); 8];
    let result = plan.forward(&signal).unwrap();
    for (_, coeff) in result.coefficients() {
        assert!(
            coeff.norm() < 1e-14,
            "Expected zero spectrum for zero signal, got {:?}",
            coeff
        );
    }
}

#[test]
fn forward_single_tone_recovers_dominant_frequency() {
    // Pure tone at frequency freq=3: x[n] = cos(2 pi 3 n / N).
    // DFT: X[3] = X[13] = N/2 = 8; all other bins = 0.
    let n = 16usize;
    let freq = 3usize;
    let signal: Vec<Complex64> = (0..n)
        .map(|i| {
            Complex64::new(
                (std::f64::consts::TAU * freq as f64 * i as f64 / n as f64).cos(),
                0.0,
            )
        })
        .collect();
    let plan = SparseFftPlan::new(n, 4).unwrap();
    let result = plan.forward(&signal).unwrap();
    let dominant: Vec<usize> = result
        .coefficients()
        .filter(|(_, c)| c.norm() > 5.0)
        .map(|(idx, _)| idx)
        .collect();
    assert!(
        dominant.contains(&freq) || dominant.contains(&(n - freq)),
        "dominant frequency not in top coefficients: {:?}",
        dominant
    );
}

#[test]
fn forward_k_equals_n_is_full_spectrum() {
    // Dirac delta: DFT[k] = 1 for all k, guaranteeing all n bins nonzero.
    let n = 8usize;
    let plan = SparseFftPlan::new(n, n).unwrap();
    let mut signal = vec![Complex64::new(0.0, 0.0); n];
    signal[0] = Complex64::new(1.0, 0.0);
    let result = plan.forward(&signal).unwrap();
    assert_eq!(
        result.len(),
        n,
        "k=n with Dirac delta must retain all n DFT bins"
    );
}

#[test]
fn forward_k_equals_one_returns_dc_for_constant_signal() {
    // Constant signal: X[0] = N * 2.0 = 16, X[k!=0] = 0.
    // With k=1 and threshold=0, only X[0] is retained.
    let n = 8usize;
    let plan = SparseFftPlan::new(n, 1).unwrap();
    let signal = vec![Complex64::new(2.0, 0.0); n];
    let result = plan.forward(&signal).unwrap();
    let coeffs: Vec<(usize, Complex64)> = result.coefficients().collect();
    assert_eq!(coeffs.len(), 1, "Expected exactly 1 coefficient");
    assert_eq!(coeffs[0].0, 0, "DC bin must be at frequency index 0");
    let dc_mag = coeffs[0].1.norm();
    assert!(
        (dc_mag - 16.0).abs() < 1e-10,
        "DC magnitude: expected 16.0, got {dc_mag}"
    );
}

#[test]
fn top_k_minimizes_parseval_energy_residual() {
    // DC spike: x = [1, 0, 0, 0, 0, 0, 0, 0], N=8.
    // DFT(x)[k] = sum_{n=0}^{7} x[n] exp(-2πi·kn/8) = x[0] = 1 for all k.
    // All 8 DFT bins have equal magnitude 1.0; tie-breaking by lowest index
    // selects frequency 0 as the single retained coefficient for K=1.
    // Parseval residual = (1/N) * sum_{k=1}^{7} |X[k]|² = 7/8; this is the
    // minimum achievable for any single-bin selection, proved by the
    // Top-K Coefficient Selection Optimality theorem in the module doc.
    let n = 8usize;
    let mut signal = vec![Complex64::new(0.0, 0.0); n];
    signal[0] = Complex64::new(1.0, 0.0);

    let plan = SparseFftPlan::new(n, 1).expect("plan");
    let spectrum = plan.forward(&signal).expect("forward");

    let coeffs: Vec<(usize, Complex64)> = spectrum.coefficients().collect();
    assert_eq!(coeffs.len(), 1, "K=1 must retain exactly 1 coefficient");
    assert_eq!(
        coeffs[0].0, 0,
        "Tie-breaking must select lowest frequency index 0, got {}",
        coeffs[0].0
    );
    let val = coeffs[0].1;
    assert!(
        (val.re - 1.0).abs() < 1e-10 && val.im.abs() < 1e-10,
        "DFT[0] of DC spike must be 1.0+0.0i, got {:?}",
        val
    );
}

#[test]
fn exactly_sparse_signal_recovers_exact_coefficients() {
    // Construct X with exactly 2 nonzero DFT components at known positions:
    //   X[0] = 3 + 0i  (|X[0]| = 3.0)
    //   X[3] = 0 + 2i  (|X[3]| = 2.0)
    //   X[k] = 0 for k ∉ {0, 3}
    // Compute x = IFFT(X) via SparseFftPlan::inverse. The resulting signal is
    // 2-sparse in the frequency domain. SparseFftPlan::forward(x) with K=2
    // must recover exactly X[0] and X[3] by the Exact Recovery theorem.
    let n = 8usize;
    let mut x_freq = SparseSpectrum::new(n);
    x_freq
        .insert(0, Complex64::new(3.0, 0.0))
        .expect("insert X[0]");
    x_freq
        .insert(3, Complex64::new(0.0, 2.0))
        .expect("insert X[3]");

    let plan = SparseFftPlan::new(n, 2).expect("plan");
    let signal = plan.inverse(&x_freq).expect("ifft to time domain");
    let recovered = plan.forward(&signal).expect("forward");

    let support: Vec<(usize, Complex64)> = recovered.coefficients().collect();
    assert_eq!(
        support.len(),
        2,
        "Exactly 2 nonzero DFT components must be recovered; got {:?}",
        support
    );

    let coeff_at = |idx: usize| -> Option<Complex64> {
        support.iter().find(|(i, _)| *i == idx).map(|(_, v)| *v)
    };

    let c0 = coeff_at(0).expect("coefficient at frequency 0 not found");
    let c3 = coeff_at(3).expect("coefficient at frequency 3 not found");

    assert!(
        (c0.norm() - 3.0_f64).abs() < 1e-6,
        "|X[0]| must be 3.0, got {}",
        c0.norm()
    );
    assert!(
        (c3.norm() - 2.0_f64).abs() < 1e-6,
        "|X[3]| must be 2.0, got {}",
        c3.norm()
    );
}
