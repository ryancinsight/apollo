use crate::{DhtError, DhtPlan};
use eunomia::assert_abs_diff_eq;

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
