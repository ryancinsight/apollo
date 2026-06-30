//! Tests for forward/inverse complex FFT API (part 1).

use crate::*;
use leto::{Array1, Array2, Array3};
use eunomia::{Complex32, Complex64};

#[test]
fn complex_into_wrappers_match_allocating_paths() {
    let signal1 = Array1::from_shape_fn([16], |[i]| {
        let x = i as f64;
        Complex64::new((0.17 * x).sin(), (0.29 * x).cos())
    });
    let expected1 = fft_1d_complex(&signal1);
    let mut actual1 = Array1::<Complex64>::zeros([16]);
    fft_1d_complex_into(&signal1, &mut actual1);
    for (expected, actual) in expected1.iter().zip(actual1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered1 = ifft_1d_complex(&expected1);
    let mut actual_recovered1 = Array1::<Complex64>::zeros([16]);
    ifft_1d_complex_into(&expected1, &mut actual_recovered1);
    for (expected, actual) in recovered1.iter().zip(actual_recovered1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal2 = Array2::from_shape_fn([4, 8], |[i, j]| {
        let x = (i * 8 + j) as f64;
        Complex64::new((0.13 * x).sin(), (0.23 * x).cos())
    });
    let expected2 = fft_2d_complex(&signal2);
    let mut actual2 = Array2::<Complex64>::zeros([4, 8]);
    fft_2d_complex_into(&signal2, &mut actual2);
    for (expected, actual) in expected2.iter().zip(actual2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered2 = ifft_2d_complex(&expected2);
    let mut actual_recovered2 = Array2::<Complex64>::zeros([4, 8]);
    ifft_2d_complex_into(&expected2, &mut actual_recovered2);
    for (expected, actual) in recovered2.iter().zip(actual_recovered2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal3 = Array3::from_shape_fn([4, 4, 4], |[i, j, k]| {
        let x = (i * 16 + j * 4 + k) as f64;
        Complex64::new((0.11 * x).sin(), (0.19 * x).cos())
    });
    let expected3 = fft_3d_complex(&signal3);
    let mut actual3 = Array3::<Complex64>::zeros([4, 4, 4]);
    fft_3d_complex_into(&signal3, &mut actual3);
    for (expected, actual) in expected3.iter().zip(actual3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered3 = ifft_3d_complex(&expected3);
    let mut actual_recovered3 = Array3::<Complex64>::zeros([4, 4, 4]);
    ifft_3d_complex_into(&expected3, &mut actual_recovered3);
    for (expected, actual) in recovered3.iter().zip(actual_recovered3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
}

#[test]
fn owned_complex_wrappers_reuse_input_allocation() {
    let signal1 = Array1::from_shape_fn([16], |[i]| {
        let x = i as f64;
        Complex64::new((0.17 * x).sin(), (0.29 * x).cos())
    });
    let expected1 = fft_1d_complex(&signal1);
    let ptr1 = signal1.as_slice().unwrap().as_ptr();
    let actual1 = fft_1d_complex_owned(signal1);
    assert_eq!(ptr1, actual1.as_slice().unwrap().as_ptr());
    for (expected, actual) in expected1.iter().zip(actual1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered1 = ifft_1d_complex(&expected1);
    let ptr_recovered1 = expected1.as_slice().unwrap().as_ptr();
    let actual_recovered1 = ifft_1d_complex_owned(expected1);
    assert_eq!(ptr_recovered1, actual_recovered1.as_slice().unwrap().as_ptr());
    for (expected, actual) in recovered1.iter().zip(actual_recovered1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal2 = Array2::from_shape_fn([4, 8], |[i, j]| {
        let x = (i * 8 + j) as f64;
        Complex64::new((0.13 * x).sin(), (0.23 * x).cos())
    });
    let expected2 = fft_2d_complex(&signal2);
    let ptr2 = signal2.as_slice().unwrap().as_ptr();
    let actual2 = fft_2d_complex_owned(signal2);
    assert_eq!(ptr2, actual2.as_slice().unwrap().as_ptr());
    for (expected, actual) in expected2.iter().zip(actual2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered2 = ifft_2d_complex(&expected2);
    let ptr_recovered2 = expected2.as_slice().unwrap().as_ptr();
    let actual_recovered2 = ifft_2d_complex_owned(expected2);
    assert_eq!(ptr_recovered2, actual_recovered2.as_slice().unwrap().as_ptr());
    for (expected, actual) in recovered2.iter().zip(actual_recovered2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal3 = Array3::from_shape_fn([4, 4, 4], |[i, j, k]| {
        let x = (i * 16 + j * 4 + k) as f64;
        Complex64::new((0.11 * x).sin(), (0.19 * x).cos())
    });
    let expected3 = fft_3d_complex(&signal3);
    let ptr3 = signal3.as_slice().unwrap().as_ptr();
    let actual3 = fft_3d_complex_owned(signal3);
    assert_eq!(ptr3, actual3.as_slice().unwrap().as_ptr());
    for (expected, actual) in expected3.iter().zip(actual3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered3 = ifft_3d_complex(&expected3);
    let ptr_recovered3 = expected3.as_slice().unwrap().as_ptr();
    let actual_recovered3 = ifft_3d_complex_owned(expected3);
    assert_eq!(ptr_recovered3, actual_recovered3.as_slice().unwrap().as_ptr());
    for (expected, actual) in recovered3.iter().zip(actual_recovered3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
}

#[test]
fn static_complex_wrappers_match_dynamic_paths() {
    let signal1 = Array1::from_shape_fn([16], |[i]| {
        let x = i as f64;
        Complex64::new((0.17 * x).sin(), (0.29 * x).cos())
    });
    let expected1 = fft_1d_complex(&signal1);
    let mut actual1 = signal1.clone();
    fft_1d_complex_static_inplace::<16>(&mut actual1);
    for (expected, actual) in expected1.iter().zip(actual1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
    ifft_1d_complex_static_inplace::<16>(&mut actual1);
    for (expected, actual) in signal1.iter().zip(actual1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal2 = Array2::from_shape_fn([4, 5], |[i, j]| {
        let x = (i * 5 + j) as f64;
        Complex64::new((0.13 * x).sin(), (0.23 * x).cos())
    });
    let expected2 = fft_2d_complex(&signal2);
    let mut actual2 = signal2.clone();
    fft_2d_complex_static_inplace::<4, 5>(&mut actual2);
    for (expected, actual) in expected2.iter().zip(actual2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
    ifft_2d_complex_static_inplace::<4, 5>(&mut actual2);
    for (expected, actual) in signal2.iter().zip(actual2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal3 = Array3::from_shape_fn([3, 4, 5], |[i, j, k]| {
        let x = ((i * 4 + j) * 5 + k) as f64;
        Complex64::new((0.11 * x).sin(), (0.19 * x).cos())
    });
    let expected3 = fft_3d_complex(&signal3);
    let mut actual3 = signal3.clone();
    fft_3d_complex_static_inplace::<3, 4, 5>(&mut actual3);
    for (expected, actual) in expected3.iter().zip(actual3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
    ifft_3d_complex_static_inplace::<3, 4, 5>(&mut actual3);
    for (expected, actual) in signal3.iter().zip(actual3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
}

#[test]
fn static_complex_into_wrappers_match_allocating_paths() {
    let signal1 = Array1::from_shape_fn([16], |[i]| {
        let x = i as f64;
        Complex64::new((0.17 * x).sin(), (0.29 * x).cos())
    });
    let expected1 = fft_1d_complex(&signal1);
    let mut actual1 = Array1::<Complex64>::zeros([16]);
    fft_1d_complex_static_into::<16>(&signal1, &mut actual1);
    for (expected, actual) in expected1.iter().zip(actual1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered1 = ifft_1d_complex(&expected1);
    let mut actual_recovered1 = Array1::<Complex64>::zeros([16]);
    ifft_1d_complex_static_into::<16>(&expected1, &mut actual_recovered1);
    for (expected, actual) in recovered1.iter().zip(actual_recovered1.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal2 = Array2::from_shape_fn([4, 5], |[i, j]| {
        let x = (i * 5 + j) as f64;
        Complex64::new((0.13 * x).sin(), (0.23 * x).cos())
    });
    let expected2 = fft_2d_complex(&signal2);
    let mut actual2 = Array2::<Complex64>::zeros([4, 5]);
    fft_2d_complex_static_into::<4, 5>(&signal2, &mut actual2);
    for (expected, actual) in expected2.iter().zip(actual2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered2 = ifft_2d_complex(&expected2);
    let mut actual_recovered2 = Array2::<Complex64>::zeros([4, 5]);
    ifft_2d_complex_static_into::<4, 5>(&expected2, &mut actual_recovered2);
    for (expected, actual) in recovered2.iter().zip(actual_recovered2.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let signal3 = Array3::from_shape_fn([3, 4, 5], |[i, j, k]| {
        let x = ((i * 4 + j) * 5 + k) as f64;
        Complex64::new((0.11 * x).sin(), (0.19 * x).cos())
    });
    let expected3 = fft_3d_complex(&signal3);
    let mut actual3 = Array3::<Complex64>::zeros([3, 4, 5]);
    fft_3d_complex_static_into::<3, 4, 5>(&signal3, &mut actual3);
    for (expected, actual) in expected3.iter().zip(actual3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }

    let recovered3 = ifft_3d_complex(&expected3);
    let mut actual_recovered3 = Array3::<Complex64>::zeros([3, 4, 5]);
    ifft_3d_complex_static_into::<3, 4, 5>(&expected3, &mut actual_recovered3);
    for (expected, actual) in recovered3.iter().zip(actual_recovered3.iter()) {
        assert!((expected - actual).norm() < 1e-13);
    }
}

#[test]
fn test_bench_pot_sizes() {
    use crate::application::execution::kernel::FftPrecision;
    use std::time::Instant;

    let sizes = [2, 4, 8, 16, 32, 64];
    for &n in &sizes {
        let input: Vec<Complex32> = (0..n)
            .map(|k| Complex32::new((k as f32 * 0.17).sin(), (k as f32 * 0.29).cos()))
            .collect();
        let mut got = input.clone();
        let start = Instant::now();
        #[cfg(debug_assertions)]
        let iters = 1_000;
        #[cfg(not(debug_assertions))]
        let iters = 1_000_000;
        for _ in 0..iters {
            got.copy_from_slice(&input);
            Complex32::fft_forward(&mut got);
        }
        let elapsed = start.elapsed();
        println!(
            "Size {}: {:.2} ns per iteration (including copy)",
            n,
            (elapsed.as_secs_f64() * 1e9) / iters as f64
        );
    }
}

#[test]
fn test_f64_pot_dfts_correctness() {
    let sizes = [2, 4, 8, 16, 32, 64];
    for &n in &sizes {
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * 0.17).sin(), (k as f64 * 0.29).cos()))
            .collect();

        let mut got_fwd = leto::Array1::from_shape_vec([input.len()], input.clone()).unwrap();
        fft_1d_complex_inplace(&mut got_fwd);

        // Compare with naive DFT
        let mut expected_fwd = vec![Complex64::new(0.0, 0.0); n];
        for k in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for j in 0..n {
                let angle = -2.0 * std::f64::consts::PI * j as f64 * k as f64 / n as f64;
                let w = Complex64::new(angle.cos(), angle.sin());
                sum += input[j] * w;
            }
            expected_fwd[k] = sum;
        }
        for k in 0..n {
            let err = (got_fwd[k] - expected_fwd[k]).norm();
            assert!(
                err < 1e-12,
                "forward: n = {}, k = {}, got_fwd = {:?}, expected = {:?}, err = {}",
                n,
                k,
                got_fwd[k],
                expected_fwd[k],
                err
            );
        }

        let mut got_inv = leto::Array1::from_shape_vec([input.len()], input.clone()).unwrap();
        ifft_1d_complex_inplace(&mut got_inv);
        // Naive IDFT (with normalization)
        let mut expected_inv = vec![Complex64::new(0.0, 0.0); n];
        for k in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for j in 0..n {
                let angle = 2.0 * std::f64::consts::PI * j as f64 * k as f64 / n as f64;
                let w = Complex64::new(angle.cos(), angle.sin());
                sum += input[j] * w;
            }
            expected_inv[k] = sum / n as f64;
        }
        for k in 0..n {
            let err = (got_inv[k] - expected_inv[k]).norm();
            assert!(
                err < 1e-12,
                "inverse: n = {}, k = {}, got_inv = {:?}, expected = {:?}, err = {}",
                n,
                k,
                got_inv[k],
                expected_inv[k],
                err
            );
        }
    }
}

#[test]
fn test_debug_twiddles() {
    use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
    let tw_table = <f64 as MixedRadixScalar>::small_pot_twiddles::<false>(32);
    println!("tw_table len: {}", tw_table.len());

    let get_twiddle = |idx: usize| {
        let half = 16;
        if idx < half {
            tw_table[half - 1 + idx]
        } else {
            -tw_table[half - 1 + idx - half]
        }
    };

    for k in 0..32 {
        let tw = get_twiddle(k);
        let angle = -2.0 * std::f64::consts::PI * k as f64 / 32.0;
        let expected = Complex64::new(angle.cos(), angle.sin());
        let diff = (tw - expected).norm();
        println!(
            "k = {}: got = {:?}, expected = {:?}, diff = {}",
            k, tw, expected, diff
        );
        assert!(diff < 1e-12);
    }
}

#[test]
fn test_f32_pot_plans_correctness() {
    let sizes = [2, 4, 8, 16, 32, 64];
    for &n in &sizes {
        let input: Vec<Complex32> = (0..n)
            .map(|k| Complex32::new((k as f32 * 0.17).sin(), (k as f32 * 0.29).cos()))
            .collect();

        let plan = f32::get_1d_plan(crate::Shape1D::new(n).unwrap());
        let mut got_fwd = input.clone();
        plan.forward_complex_slice_inplace(&mut got_fwd);

        // Compare with naive DFT
        let mut expected_fwd = vec![Complex32::new(0.0, 0.0); n];
        for k in 0..n {
            let mut sum = eunomia::Complex64::new(0.0, 0.0);
            for j in 0..n {
                let angle = -2.0 * std::f64::consts::PI * j as f64 * k as f64 / n as f64;
                let w = eunomia::Complex64::new(angle.cos(), angle.sin());
                sum += eunomia::Complex64::new(input[j].re as f64, input[j].im as f64) * w;
            }
            expected_fwd[k] = Complex32::new(sum.re as f32, sum.im as f32);
        }
        for k in 0..n {
            let err = (got_fwd[k] - expected_fwd[k]).norm();
            assert!(
                err < 1e-5,
                "forward: n = {}, k = {}, got_fwd = {:?}, expected = {:?}, err = {}",
                n,
                k,
                got_fwd[k],
                expected_fwd[k],
                err
            );
        }

        let mut got_inv = input.clone();
        plan.inverse_complex_slice_inplace(&mut got_inv);
        // Naive IDFT (with normalization)
        let mut expected_inv = vec![Complex32::new(0.0, 0.0); n];
        for k in 0..n {
            let mut sum = eunomia::Complex64::new(0.0, 0.0);
            for j in 0..n {
                let angle = 2.0 * std::f64::consts::PI * j as f64 * k as f64 / n as f64;
                let w = eunomia::Complex64::new(angle.cos(), angle.sin());
                sum += eunomia::Complex64::new(input[j].re as f64, input[j].im as f64) * w;
            }
            expected_inv[k] =
                Complex32::new((sum.re / n as f64) as f32, (sum.im / n as f64) as f32);
        }
        for k in 0..n {
            let err = (got_inv[k] - expected_inv[k]).norm();
            assert!(
                err < 1e-5,
                "inverse: n = {}, k = {}, got_inv = {:?}, expected = {:?}, err = {}",
                n,
                k,
                got_inv[k],
                expected_inv[k],
                err
            );
        }
    }
}
