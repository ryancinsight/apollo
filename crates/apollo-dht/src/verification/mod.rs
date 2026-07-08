//! Verification modules for the DHT.

#[cfg(test)]
mod tests {
    use crate::{DhtError, DhtPlan, HartleySpectrum};
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;
    use proptest::proptest;

    #[test]
    fn impulse_maps_to_unit_spectrum() {
        let plan = DhtPlan::new(8).expect("plan");
        let spectrum = plan
            .forward(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .expect("forward");
        for value in spectrum.values() {
            assert_abs_diff_eq!(*value, 1.0, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn constant_signal_has_only_dc_component() {
        let plan = DhtPlan::new(6).expect("plan");
        let spectrum = plan.forward(&[2.0; 6]).expect("forward");
        assert_abs_diff_eq!(spectrum.values()[0], 12.0, epsilon = 1.0e-12);
        for value in &spectrum.values()[1..] {
            assert_abs_diff_eq!(*value, 0.0, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn inverse_reuses_hartley_involution() {
        let signal = [3.0, -1.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
        let plan = DhtPlan::new(signal.len()).expect("plan");
        let spectrum = plan.forward(&signal).expect("forward");
        let recovered = plan.inverse(&spectrum).expect("inverse");
        for (actual, expected) in recovered.iter().zip(signal.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-11);
        }
    }

    #[test]
    fn raw_double_transform_equals_length_scaled_input() {
        let signal = [1.0, -2.0, 4.0, 0.25];
        let plan = DhtPlan::new(signal.len()).expect("plan");
        let first = plan.transform_unscaled(&signal).expect("first");
        let second = plan.transform_unscaled(&first).expect("second");
        for (actual, expected) in second.iter().zip(signal.iter()) {
            assert_abs_diff_eq!(*actual, signal.len() as f64 * expected, epsilon = 1.0e-11);
        }
    }

    #[test]
    fn rejects_invalid_contracts() {
        assert_eq!(DhtPlan::new(0).unwrap_err(), DhtError::EmptySignal);
        let plan = DhtPlan::new(4).expect("plan");
        assert_eq!(
            plan.forward(&[1.0, 2.0]).unwrap_err(),
            DhtError::LengthMismatch
        );
    }

    #[test]
    fn parseval_identity_holds_for_unnormalized_dht() {
        let signal = [1.0, -3.0, 2.0, 5.0, -0.5];
        let plan = DhtPlan::new(signal.len()).expect("plan");
        let spectrum = plan.forward(&signal).expect("forward");
        let signal_energy: f64 = signal.iter().map(|value| value * value).sum();
        let spectrum_energy: f64 = spectrum.values().iter().map(|value| value * value).sum();
        assert_abs_diff_eq!(
            spectrum_energy,
            signal.len() as f64 * signal_energy,
            epsilon = 1.0e-10
        );
    }

    /// DHT involution: DHT(DHT(x)) = N * x.
    ///
    /// Proof sketch: the DHT matrix H satisfies H^2 = N*I because the Hartley
    /// basis functions are self-dual under the circular convolution.
    #[test]
    fn dht_involution_property() {
        for n in [4usize, 7, 16, 32, 257] {
            let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.23).cos()).collect();
            let plan = DhtPlan::new(n).expect("plan");
            let once = plan.forward(&signal).expect("first pass");
            let twice = plan.forward(once.values()).expect("second pass");
            let n_f = n as f64;
            let err: f64 = signal
                .iter()
                .zip(twice.values().iter())
                .map(|(a, b)| (a * n_f - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                err < 1e-8,
                "DHT involution n={n}: DHT(DHT(x)) != N*x, err={err}"
            );
        }
    }

    /// Parseval identity: sum_k H[k]^2 = N * sum_n x[n]^2.
    #[test]
    fn dht_parseval_identity() {
        let n = 16usize;
        let signal: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.37).sin() + 0.5).collect();
        let plan = DhtPlan::new(n).unwrap();
        let spectrum = plan.forward(&signal).unwrap();
        let energy_time: f64 = signal.iter().map(|x| x * x).sum();
        let energy_freq: f64 = spectrum.values().iter().map(|h| h * h).sum();
        assert!(
            (energy_freq - n as f64 * energy_time).abs() < 1e-8,
            "Parseval failed: freq={energy_freq}, n*time={}",
            n as f64 * energy_time
        );
    }

    /// Inverse rejects a spectrum whose length does not match the plan.
    #[test]
    fn inverse_rejects_length_mismatch() {
        let plan = DhtPlan::new(8).unwrap();
        let wrong_spectrum = HartleySpectrum::new(vec![0.0; 5]);
        let result = plan.inverse(&wrong_spectrum);
        assert!(matches!(result, Err(DhtError::LengthMismatch)));
    }

    /// Theorem (Hartley–Fourier Relationship): For any real sequence x ∈ ℝᴺ,
    /// the unnormalized DHT output H[k] and the DFT output F[k] = DFT(x)[k] satisfy:
    ///
    /// ```text
    /// H[k] = Re(F[k]) - Im(F[k])
    /// ```
    ///
    /// **Proof:** By definition,
    /// H[k] = Σ_n x[n] · cas(2πkn/N) = Σ_n x[n] · (cos(2πkn/N) + sin(2πkn/N))
    /// F[k] = Σ_n x[n] · exp(-2πikn/N) = Σ_n x[n] · (cos(2πkn/N) - i·sin(2πkn/N))
    ///
    /// Therefore Re(F[k]) = Σ_n x[n]·cos(2πkn/N) and -Im(F[k]) = Σ_n x[n]·sin(2πkn/N).
    /// Combining: H[k] = Re(F[k]) + (-Im(F[k])) = Re(F[k]) - Im(F[k]). □
    ///
    /// Cross-check independence: for N=8 (below FAST_KERNEL_THRESHOLD=512), `DhtPlan::forward`
    /// uses the O(N²) direct `transform_real` kernel. The DFT is computed via
    /// `fft_forward` which selects the radix-2 Cooley-Tukey path for N=8. These are
    /// entirely separate code paths, so any sign or index error in either would be detected.
    #[test]
    fn dht_equals_re_minus_im_of_independent_dft() {
        use apollo_fft::application::execution::kernel::fft_forward;
        use eunomia::Complex64;
        let signal = [3.0_f64, -1.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
        let plan = crate::DhtPlan::new(signal.len()).expect("plan");
        let h = plan.forward(&signal).expect("forward");
        // Independent DFT: embed real signal in complex, apply radix-2 FFT
        let mut scratch: Vec<Complex64> = signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        fft_forward(&mut scratch);
        // H[k] = Re(F[k]) - Im(F[k])
        for (k, (hk, fk)) in h.values().iter().zip(scratch.iter()).enumerate() {
            let expected = fk.re - fk.im;
            let err = (hk - expected).abs();
            assert!(
                err < 1e-10,
                "DHT-DFT relationship fails at k={k}: H[k]={hk:.6}, \
                 Re(F)-Im(F)={expected:.6}, err={err:.3e}"
            );
        }
    }

    /// Fast-path correctness: at and above FAST_KERNEL_THRESHOLD the FFT-mapped DHT
    /// must match the independent direct Hartley kernel value-for-value within f64 error.
    #[test]
    fn fast_kernel_matches_direct_hartley_at_threshold() {
        use crate::infrastructure::kernel::direct::transform_real;

        let n = 512usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.173).sin() + 0.25 * (i as f64 * 0.071).cos())
            .collect();
        let plan = crate::DhtPlan::new(n).expect("plan");

        let got = plan.forward(&signal).expect("fast forward");
        let mut expected = vec![0.0; n];
        transform_real(&signal, &mut expected).expect("direct hartley");

        let err = got
            .values()
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            err < 1.0e-10,
            "fast DHT mismatch vs direct kernel: err={err}"
        );
    }

    #[test]
    fn fast_wrapper_matches_direct_hartley_output() {
        use crate::infrastructure::kernel::direct::transform_real;
        use crate::infrastructure::kernel::fast::dht_fast;

        let n = 512usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.137).cos() - 0.5 * (i as f64 * 0.041).sin())
            .collect();
        let mut got = vec![0.0; n];
        let mut expected = vec![0.0; n];

        dht_fast(&signal, &mut got);
        transform_real(&signal, &mut expected).expect("direct hartley");

        let err = got
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            err < 1.0e-10,
            "fast wrapper mismatch vs direct kernel: err={err}"
        );
    }

    proptest! {
        /// Serial-range roundtrip: for n in [2,16], DHT inverse recovers the signal.
        #[test]
        fn dht_roundtrip_serial_range(
            data in prop::collection::vec(-10.0f64..10.0f64, 2..17)
        ) {
            let plan = DhtPlan::new(data.len()).expect("plan");
            let spectrum = plan.forward(&data).expect("forward");
            let recovered = plan.inverse(&spectrum).expect("inverse");
            for (actual, expected) in recovered.iter().zip(data.iter()) {
                prop_assert!((actual - expected).abs() < 1.0e-9);
            }
        }

        /// Parallel-range roundtrip: n in [257,512] exercises the Moirai parallel path
        /// and (at n=512) the FFT-based fast kernel (FAST_KERNEL_THRESHOLD=512).
        #[test]
        fn dht_roundtrip_parallel_range(
            n in 257usize..513,
        ) {
            let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.173).sin()).collect();
            let plan = DhtPlan::new(n).unwrap();
            let spectrum = plan.forward(&signal).unwrap();
            let recovered = plan.inverse(&spectrum).unwrap();
            let err: f64 = signal
                .iter()
                .zip(recovered.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            prop_assert!(err < 1e-10, "parallel DHT roundtrip failed: err={}", err);
        }
    }

    /// 2D DHT separability: applying forward_2d then again = N²·x.
    ///
    /// Proof: by the involutory property of the 1D DHT applied per axis,
    /// H_{2D}(H_{2D}(X))[m,n] = N·(N·X[m,n]) = N²·X[m,n].
    #[test]
    fn forward_2d_involution_equals_n_squared_times_input() {
        let input = leto::Array2::from_shape_vec(
            [3, 3],
            vec![1.0_f64, -2.0, 0.5, 0.25, 3.0, -1.5, -0.75, 0.5, 2.0],
        )
        .unwrap();
        let plan = DhtPlan::new(3).expect("plan");
        let first = plan.forward_2d(&input).expect("first 2D forward");
        let second = plan.forward_2d(&first).expect("second 2D forward");
        let n2 = 9.0_f64;
        for r in 0..3 {
            for c in 0..3 {
                let diff = (second[[r, c]] - n2 * input[[r, c]]).abs();
                assert!(
                    diff < 1.0e-10,
                    "2D involution at [{r},{c}]: got {}, expected {}",
                    second[[r, c]],
                    n2 * input[[r, c]]
                );
            }
        }
    }

    /// 2D DHT inverse roundtrip recovers the original signal.
    ///
    /// inverse_2d = (1/N²) · forward_2d; combined with the involution result,
    /// inverse_2d(forward_2d(X)) = (1/N²)·N²·X = X exactly.
    #[test]
    fn inverse_2d_recovers_input() {
        let input = leto::Array2::from_shape_vec(
            [3, 3],
            vec![1.0_f64, -2.0, 0.5, 0.25, 3.0, -1.5, -0.75, 0.5, 2.0],
        )
        .unwrap();
        let plan = DhtPlan::new(3).expect("plan");
        let spectrum = plan.forward_2d(&input).expect("2D forward");
        let recovered = plan.inverse_2d(&spectrum).expect("2D inverse");
        for r in 0..3 {
            for c in 0..3 {
                let diff = (recovered[[r, c]] - input[[r, c]]).abs();
                assert!(
                    diff < 1.0e-10,
                    "2D inverse roundtrip mismatch at [{r},{c}]: recovered={}, original={}",
                    recovered[[r, c]],
                    input[[r, c]]
                );
            }
        }
    }

    #[test]
    fn forward_and_inverse_2d_into_match_allocating_paths() {
        let input = leto::Array2::from_shape_vec(
            [3, 3],
            vec![1.25_f64, -0.5, 2.0, -1.0, 0.75, 3.5, 0.125, -2.25, 1.5],
        )
        .unwrap();
        let plan = DhtPlan::new(3).expect("plan");

        let expected_forward = plan.forward_2d(&input).expect("forward 2D");
        let mut forward_into = leto::Array2::<f64>::zeros([3, 3]);
        plan.forward_2d_into(&input, &mut forward_into)
            .expect("forward_2d_into");
        for r in 0..3 {
            for c in 0..3 {
                assert_abs_diff_eq!(
                    forward_into[[r, c]],
                    expected_forward[[r, c]],
                    epsilon = 1.0e-12
                );
            }
        }

        let expected_inverse = plan.inverse_2d(&expected_forward).expect("inverse 2D");
        let mut inverse_into = leto::Array2::<f64>::zeros([3, 3]);
        plan.inverse_2d_into(&expected_forward, &mut inverse_into)
            .expect("inverse_2d_into");
        for r in 0..3 {
            for c in 0..3 {
                assert_abs_diff_eq!(
                    inverse_into[[r, c]],
                    expected_inverse[[r, c]],
                    epsilon = 1.0e-12
                );
            }
        }
    }

    /// 2D DHT of a 4×4 separable signal: the row-DHT of each row then column-DHT of each
    /// column matches manual computation using the known 1D DHT formula.
    #[test]
    fn forward_2d_matches_separable_manual_application() {
        use leto::Array2;
        let n = 4_usize;
        // Separable signal: outer product of a 1D signal with itself.
        let row = [1.0_f64, 2.0, -1.0, 0.5];
        let mut input = Array2::<f64>::zeros([n, n]);
        for r in 0..n {
            for c in 0..n {
                input[[r, c]] = row[r] * row[c];
            }
        }
        let plan = DhtPlan::new(n).expect("plan");
        let result = plan.forward_2d(&input).expect("2D forward");

        // CPU reference: apply 1D DHT row-then-column.
        let mut tmp = [[0.0_f64; 4]; 4];
        for r in 0..n {
            let out = plan.forward(&row).expect("1D row forward");
            for c in 0..n {
                tmp[r][c] = out.values()[c] * row[r];
            }
        }
        // The separable product means each entry is row_dht[r] * row_dht[c].
        let dht_row = plan.forward(&row).expect("1D DHT of row");
        for r in 0..n {
            for c in 0..n {
                let expected = dht_row.values()[r] * dht_row.values()[c];
                let diff = (result[[r, c]] - expected).abs();
                assert!(
                    diff < 1.0e-10,
                    "separable 2D DHT mismatch at [{r},{c}]: got {}, expected {}",
                    result[[r, c]],
                    expected
                );
            }
        }
        let _ = tmp;
    }

    /// 3D DHT inverse roundtrip recovers the original signal.
    #[test]
    fn inverse_3d_recovers_input() {
        use leto::Array3;
        let n = 3_usize;
        let flat: [f64; 27] = [
            1.0, -2.0, 0.5, 0.25, 3.0, -1.5, -0.75, 0.5, 2.0, 0.1, -0.3, 1.2, -0.5, 2.1, -1.1, 0.7,
            -0.9, 0.3, 1.5, -0.2, 0.8, -1.4, 0.6, -0.1, 0.9, -2.5, 1.3,
        ];
        let mut input = Array3::<f64>::zeros([n, n, n]);
        let mut idx = 0;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    input[[i, j, k]] = flat[idx];
                    idx += 1;
                }
            }
        }
        let plan = DhtPlan::new(n).expect("plan");
        let spectrum = plan.forward_3d(&input).expect("3D forward");
        let recovered = plan.inverse_3d(&spectrum).expect("3D inverse");
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let diff = (recovered[[i, j, k]] - input[[i, j, k]]).abs();
                    assert!(
                        diff < 1.0e-10,
                        "3D inverse roundtrip mismatch at [{i},{j},{k}]: recovered={}, original={}",
                        recovered[[i, j, k]],
                        input[[i, j, k]]
                    );
                }
            }
        }
    }

    #[test]
    fn forward_and_inverse_3d_into_match_allocating_paths() {
        use leto::Array3;

        let n = 3_usize;
        let mut input = Array3::<f64>::zeros([n, n, n]);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    input[[i, j, k]] = (i as f64 * 0.5) - (j as f64 * 1.25) + (k as f64 * 0.75);
                }
            }
        }

        let plan = DhtPlan::new(n).expect("plan");
        let expected_forward = plan.forward_3d(&input).expect("forward 3D");
        let mut forward_into = Array3::<f64>::zeros([n, n, n]);
        plan.forward_3d_into(&input, &mut forward_into)
            .expect("forward_3d_into");
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    assert_abs_diff_eq!(
                        forward_into[[i, j, k]],
                        expected_forward[[i, j, k]],
                        epsilon = 1.0e-12
                    );
                }
            }
        }

        let expected_inverse = plan.inverse_3d(&expected_forward).expect("inverse 3D");
        let mut inverse_into = Array3::<f64>::zeros([n, n, n]);
        plan.inverse_3d_into(&expected_forward, &mut inverse_into)
            .expect("inverse_3d_into");
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    assert_abs_diff_eq!(
                        inverse_into[[i, j, k]],
                        expected_inverse[[i, j, k]],
                        epsilon = 1.0e-12
                    );
                }
            }
        }
    }

    /// Shape mismatch errors are returned for non-square 2D and non-cubic 3D inputs.
    #[test]
    fn rejects_non_square_2d_and_non_cubic_3d() {
        use leto::{Array2, Array3};
        let plan = DhtPlan::new(3).expect("plan");
        let non_square = Array2::<f64>::zeros([2, 3]);
        assert!(matches!(
            plan.forward_2d(&non_square),
            Err(DhtError::ShapeMismatch2d {
                expected: 3,
                rows: 2,
                cols: 3
            })
        ));
        let non_cubic = Array3::<f64>::zeros([2, 3, 3]);
        assert!(matches!(
            plan.forward_3d(&non_cubic),
            Err(DhtError::ShapeMismatch3d {
                expected: 3,
                d0: 2,
                d1: 3,
                d2: 3
            })
        ));
    }
}
