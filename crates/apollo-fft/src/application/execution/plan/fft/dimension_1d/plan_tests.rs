use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
use crate::application::execution::plan::fft::dimension_1d::helpers::PlanStrategy;
use crate::application::execution::plan::fft::dimension_1d::{FftPlan1D, StaticFftPlan1D};
use crate::domain::metadata::shape::Shape1D;
use eunomia::{Complex32, Complex64};

fn signal64(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|k| {
            let x = k as f64;
            Complex64::new(
                (0.17 * x).sin() + 0.11 * (0.07 * x).cos(),
                0.23 * (0.31 * x).cos(),
            )
        })
        .collect()
}

fn signal32(n: usize) -> Vec<Complex32> {
    (0..n)
        .map(|k| {
            let x = k as f32;
            Complex32::new(
                (0.17_f32 * x).sin() + 0.11_f32 * (0.07_f32 * x).cos(),
                0.23_f32 * (0.31_f32 * x).cos(),
            )
        })
        .collect()
}

fn assert_planned_f64_forward_matches_direct(n: usize, tolerance: f64) {
    let plan = FftPlan1D::<f64>::new(Shape1D::new(n).expect("shape"));
    let input = signal64(n);
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= tolerance,
        "planned f64 N={n} forward mismatch max_err={max_err:.2e}"
    );
}

fn assert_static_f64_forward_matches_direct<const N: usize>(tolerance: f64) {
    let plan = StaticFftPlan1D::<f64, N>::new();
    let input = signal64(N);
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= tolerance,
        "static f64 N={N} forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn static_fft_plan_is_zero_sized() {
    assert_eq!(std::mem::size_of::<StaticFftPlan1D<f64, 512>>(), 0);
    assert_eq!(std::mem::size_of::<StaticFftPlan1D<f32, 200>>(), 0);
    assert_eq!(StaticFftPlan1D::<f64, 512>::new().len(), 512);
}

#[test]
fn static_fft_plan_matches_direct_for_pot_composite_and_rader() {
    assert_static_f64_forward_matches_direct::<512>(1.0e-10);
    assert_static_f64_forward_matches_direct::<200>(1.0e-10);
    assert_static_f64_forward_matches_direct::<359>(1.0e-10);
}

#[test]
fn tiny_runtime_and_static_n3_match_direct() {
    let input64 = signal64(3);
    let expected64 = dft_forward(&input64);
    let inverse64 = dft_inverse(&expected64);

    let plan64 = FftPlan1D::<f64>::new(Shape1D::new(3).expect("shape"));
    let mut runtime_forward64 = input64.clone();
    plan64.forward_complex_slice_inplace(&mut runtime_forward64);
    let mut static_forward64 = input64.clone();
    StaticFftPlan1D::<f64, 3>::new().forward_complex_slice_inplace(&mut static_forward64);
    let mut runtime_inverse64 = expected64.clone();
    plan64.inverse_complex_slice_inplace(&mut runtime_inverse64);

    for ((runtime, static_), expected) in runtime_forward64
        .iter()
        .zip(static_forward64.iter())
        .zip(expected64.iter())
    {
        assert!((*runtime - *expected).norm() <= 1.0e-12);
        assert!((*static_ - *expected).norm() <= 1.0e-12);
    }
    for (actual, expected) in runtime_inverse64.iter().zip(inverse64.iter()) {
        assert!((*actual - *expected).norm() <= 1.0e-12);
    }

    let input32 = signal32(3);
    let expected32 = dft_forward(&input32);
    let mut runtime_forward32 = input32.clone();
    FftPlan1D::<f32>::new(Shape1D::new(3).expect("shape"))
        .forward_complex_slice_inplace(&mut runtime_forward32);
    let mut static_forward32 = input32;
    StaticFftPlan1D::<f32, 3>::new().forward_complex_slice_inplace(&mut static_forward32);

    for ((runtime, static_), expected) in runtime_forward32
        .iter()
        .zip(static_forward32.iter())
        .zip(expected32.iter())
    {
        assert!(f64::from((*runtime - *expected).norm()) <= 1.0e-5);
        assert!(f64::from((*static_ - *expected).norm()) <= 1.0e-5);
    }
}

fn assert_planned_f32_forward_matches_direct(n: usize, tolerance: f64) {
    let plan = FftPlan1D::<f32>::new(Shape1D::new(n).expect("shape"));
    let input = signal32(n);
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| f64::from((*a - *b).norm()))
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= tolerance,
        "planned f32 N={n} forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n200_201_202_f64_forward_matches_direct() {
    let plan_200 = FftPlan1D::<f64>::new(Shape1D::new(200).expect("shape"));
    match &plan_200.strategy {
        PlanStrategy::Composite { radices } => assert_eq!(&**radices, &[4, 5, 5, 2]),
        _ => panic!("f64 N=200 must use the measured composite route"),
    }
    assert_planned_f64_forward_matches_direct(200, 1.0e-10);
    assert_planned_f64_forward_matches_direct(201, 2.0e-10);
    assert_planned_f64_forward_matches_direct(202, 2.0e-10);
}

#[test]
fn planned_n200_201_202_f32_forward_matches_direct() {
    let plan_200 = FftPlan1D::<f32>::new(Shape1D::new(200).expect("shape"));
    match &plan_200.strategy {
        PlanStrategy::Composite { radices } => assert_eq!(&**radices, &[4, 2, 5, 5]),
        _ => panic!("f32 N=200 must use the measured composite route"),
    }
    assert_planned_f32_forward_matches_direct(200, 8.0e-4);
    assert_planned_f32_forward_matches_direct(201, 1.5e-3);
    assert_planned_f32_forward_matches_direct(202, 1.5e-3);
}

#[test]
fn planned_power_of_two_lengths_never_route_to_good_thomas() {
    for n in [2usize, 4, 8, 16, 32, 64, 128, 256, 512] {
        let expected_log2 = n.trailing_zeros();

        let plan64 = FftPlan1D::<f64>::new(Shape1D::new(n).expect("shape"));
        match &plan64.strategy {
            PlanStrategy::PowerOfTwo { log2, .. } => assert_eq!(*log2, expected_log2),
            PlanStrategy::GoodThomas { .. } => {
                panic!("f64 power-of-two N={n} must not use Good-Thomas")
            }
            _ => panic!("f64 power-of-two N={n} must use the power-of-two route"),
        }

        let plan32 = FftPlan1D::<f32>::new(Shape1D::new(n).expect("shape"));
        match &plan32.strategy {
            PlanStrategy::PowerOfTwo { log2, .. } => assert_eq!(*log2, expected_log2),
            PlanStrategy::GoodThomas { .. } => {
                panic!("f32 power-of-two N={n} must not use Good-Thomas")
            }
            _ => panic!("f32 power-of-two N={n} must use the power-of-two route"),
        }
    }
}

#[test]
fn planned_good_thomas_n90_forward_matches_direct() {
    let n = 90usize;
    let plan = FftPlan1D::<f64>::new(Shape1D::new(n).expect("shape"));
    let input: Vec<Complex64> = (0..n)
        .map(|k| {
            let x = k as f64;
            Complex64::new((0.19 * x).sin(), 0.25 * (0.37 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| f64::from((*a - *b).norm()))
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 1.0e-10,
        "planned Good-Thomas N=90 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n48_f64_composite_forward_matches_direct() {
    let n = 48usize;
    let plan = FftPlan1D::<f64>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::Composite { radices } => assert_eq!(&**radices, &[4, 4, 3]),
        _ => panic!("f64 N=48 must use the planned composite route"),
    }
    let input: Vec<Complex64> = (0..n)
        .map(|k| {
            let x = k as f64;
            Complex64::new((0.23 * x).sin(), 0.31 * (0.41 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 1.0e-10,
        "planned f64 composite N=48 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n48_f32_composite_forward_matches_direct() {
    let n = 48usize;
    let plan = FftPlan1D::<f32>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::Composite { radices } => assert_eq!(&**radices, &[4, 4, 3]),
        _ => panic!("f32 N=48 must use the planned composite route"),
    }
    let input: Vec<Complex32> = (0..n)
        .map(|k| {
            let x = k as f32;
            Complex32::new((0.23 * x).sin(), 0.31 * (0.41 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| f64::from((*a - *b).norm()))
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 2.0e-4,
        "planned f32 composite N=48 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n385_f64_composite_forward_matches_direct() {
    let n = 385usize;
    let plan = FftPlan1D::<f64>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::Composite { radices } => assert_eq!(&**radices, &[11, 5, 7]),
        _ => panic!("f64 N=385 must use the planned composite route"),
    }
    let input: Vec<Complex64> = (0..n)
        .map(|k| {
            let x = k as f64;
            Complex64::new((0.17 * x).sin(), 0.29 * (0.43 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 1.0e-9,
        "planned f64 composite N=385 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n385_f32_composite_forward_matches_direct() {
    let n = 385usize;
    let plan = FftPlan1D::<f32>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::Composite { radices } => assert_eq!(&**radices, &[11, 5, 7]),
        _ => panic!("f32 N=385 must use the planned composite route"),
    }
    let input: Vec<Complex32> = (0..n)
        .map(|k| {
            let x = k as f32;
            Complex32::new((0.17 * x).sin(), 0.29 * (0.43 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| f64::from((*a - *b).norm()))
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 1.0e-3,
        "planned f32 composite N=385 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n180_f64_composite_forward_matches_direct() {
    let n = 180usize;
    let plan = FftPlan1D::<f64>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::Composite { radices } => assert_eq!(&**radices, &[5, 3, 3, 4]),
        _ => panic!("f64 N=180 must use the planned composite probe route"),
    }
    let input: Vec<Complex64> = (0..n)
        .map(|k| {
            let x = k as f64;
            Complex64::new((0.21 * x).sin(), 0.27 * (0.39 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 1.0e-10,
        "planned f64 composite N=180 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n180_f32_composite_forward_matches_direct() {
    let n = 180usize;
    let plan = FftPlan1D::<f32>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::Composite { radices } => assert_eq!(&**radices, &[5, 3, 3, 4]),
        _ => panic!("f32 N=180 must use the planned composite probe route"),
    }
    let input: Vec<Complex32> = (0..n)
        .map(|k| {
            let x = k as f32;
            Complex32::new((0.21 * x).sin(), 0.27 * (0.39 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| f64::from((*a - *b).norm()))
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 4.0e-4,
        "planned f32 composite N=180 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n144_f64_composite_forward_matches_direct() {
    let n = 144usize;
    let plan = FftPlan1D::<f64>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::Composite { radices } => assert_eq!(&**radices, &[4, 4, 3, 3]),
        _ => panic!("f64 N=144 must use the planned composite probe route"),
    }
    let input: Vec<Complex64> = (0..n)
        .map(|k| {
            let x = k as f64;
            Complex64::new((0.19 * x).sin(), 0.33 * (0.37 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 1.0e-10,
        "planned f64 composite N=144 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n144_f32_composite_forward_matches_direct() {
    let n = 144usize;
    let plan = FftPlan1D::<f32>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::Composite { radices } => assert_eq!(&**radices, &[4, 4, 3, 3]),
        _ => panic!("f32 N=144 must use the planned composite probe route"),
    }
    let input: Vec<Complex32> = (0..n)
        .map(|k| {
            let x = k as f32;
            Complex32::new((0.19 * x).sin(), 0.33 * (0.37 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| f64::from((*a - *b).norm()))
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 3.0e-4,
        "planned f32 composite N=144 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n176_f64_composite_forward_matches_direct() {
    let n = 176usize;
    let plan = FftPlan1D::<f64>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::Composite { radices } => assert_eq!(&**radices, &[11, 4, 4]),
        _ => panic!("f64 N=176 must use the planned composite probe route"),
    }
    let input: Vec<Complex64> = (0..n)
        .map(|k| {
            let x = k as f64;
            Complex64::new((0.17 * x).sin(), 0.35 * (0.31 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 1.0e-10,
        "planned f64 composite N=176 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n176_f32_composite_forward_matches_direct() {
    let n = 176usize;
    let plan = FftPlan1D::<f32>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::Composite { radices } => assert_eq!(&**radices, &[11, 4, 4]),
        _ => panic!("f32 N=176 must use the planned composite probe route"),
    }
    let input: Vec<Complex32> = (0..n)
        .map(|k| {
            let x = k as f32;
            Complex32::new((0.17 * x).sin(), 0.35 * (0.31 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| f64::from((*a - *b).norm()))
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 4.0e-4,
        "planned f32 composite N=176 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n36_f64_composite_forward_matches_direct() {
    let n = 36usize;
    let plan = FftPlan1D::<f64>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::Composite { radices } => assert_eq!(&**radices, &[4, 3, 3]),
        _ => panic!("f64 N=36 must use the planned composite route"),
    }
    let input: Vec<Complex64> = (0..n)
        .map(|k| {
            let x = k as f64;
            Complex64::new((0.13 * x).sin(), 0.19 * (0.23 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 1.0e-10,
        "planned f64 composite N=36 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n36_f32_composite_forward_matches_direct() {
    let n = 36usize;
    let plan = FftPlan1D::<f32>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::Composite { radices } => assert_eq!(&**radices, &[4, 3, 3]),
        _ => panic!("f32 N=36 must use the planned composite route"),
    }
    let input: Vec<Complex32> = (0..n)
        .map(|k| {
            let x = k as f32;
            Complex32::new((0.13 * x).sin(), 0.19 * (0.23 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| f64::from((*a - *b).norm()))
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 2.0e-4,
        "planned f32 composite N=36 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n63_f64_winograd_forward_matches_direct() {
    let n = 63usize;
    let plan = FftPlan1D::<f64>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::ShortWinograd => {}
        _ => panic!("f64 N=63 must use Winograd (precision-specific routing)"),
    }
    let input: Vec<Complex64> = (0..n)
        .map(|k| {
            let x = k as f64;
            Complex64::new((0.13 * x).sin(), 0.19 * (0.23 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 1.0e-10,
        "planned f64 Winograd N=63 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n63_f32_composite_forward_matches_direct() {
    let n = 63usize;
    let plan = FftPlan1D::<f32>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::Composite { radices } => assert_eq!(&**radices, &[3, 3, 7]),
        _ => panic!("f32 N=63 must use Composite (precision-specific routing)"),
    }
    let input: Vec<Complex32> = (0..n)
        .map(|k| {
            let x = k as f32;
            Complex32::new((0.13 * x).sin(), 0.19 * (0.23 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| f64::from((*a - *b).norm()))
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 5.0e-4,
        "planned f32 composite N=63 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n72_f32_codelet_forward_matches_direct() {
    assert_f32_codelet_forward_matches_direct(72, 1.0e-4);
}

#[test]
fn planned_n108_f32_codelet_forward_matches_direct() {
    assert_f32_codelet_forward_matches_direct(108, 2.0e-4);
}

#[test]
fn planned_n112_f32_codelet_forward_matches_direct() {
    assert_f32_codelet_forward_matches_direct(112, 3.0e-4);
}

#[test]
fn planned_n120_f32_codelet_forward_matches_direct() {
    assert_f32_codelet_forward_matches_direct(120, 2.0e-4);
}

#[test]
fn planned_n121_f32_codelet_forward_matches_direct() {
    assert_f32_codelet_forward_matches_direct(121, 3.0e-4);
}

#[test]
fn planned_n126_f32_codelet_forward_matches_direct() {
    assert_f32_codelet_forward_matches_direct(126, 2.0e-4);
}

#[test]
fn planned_n154_f32_codelet_forward_matches_direct() {
    assert_f32_codelet_forward_matches_direct(154, 3.0e-4);
}

#[test]
fn planned_n168_f32_codelet_forward_matches_direct() {
    assert_f32_codelet_forward_matches_direct(168, 3.0e-4);
}

#[test]
fn planned_n189_f32_codelet_forward_matches_direct() {
    assert_f32_codelet_forward_matches_direct(189, 4.0e-4);
}

#[test]
fn planned_n242_f32_codelet_forward_matches_direct() {
    assert_f32_codelet_forward_matches_direct(242, 5.0e-4);
}

#[test]
fn planned_n275_f32_codelet_forward_matches_direct() {
    assert_f32_codelet_forward_matches_direct(275, 5.0e-4);
}

#[test]
fn planned_n280_f32_codelet_forward_matches_direct() {
    assert_f32_codelet_forward_matches_direct(280, 6.0e-4);
}

#[test]
fn planned_n363_f32_codelet_forward_matches_direct() {
    assert_f32_codelet_forward_matches_direct(363, 8.0e-4);
}

#[test]
fn planned_n400_f32_codelet_forward_matches_direct() {
    assert_f32_codelet_forward_matches_direct(400, 8.0e-4);
}

#[test]
fn planned_n484_f32_codelet_forward_matches_direct() {
    assert_f32_codelet_forward_matches_direct(484, 1.0e-3);
}

#[test]
fn planned_n511_f32_good_thomas_forward_matches_direct() {
    let n = 511usize;
    let plan = FftPlan1D::<f32>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::GoodThomas { .. } => {}
        _ => {}
    }
    let input: Vec<Complex32> = (0..n)
        .map(|k| {
            let x = k as f32;
            Complex32::new((0.23 * x).sin(), 0.31 * (0.41 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| f64::from((*a - *b).norm()))
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 1.2e-3,
        "planned f32 Good-Thomas N=511 forward mismatch max_err={max_err:.2e}"
    );
}

fn assert_f32_codelet_forward_matches_direct(n: usize, tolerance: f64) {
    let plan = FftPlan1D::<f32>::new(Shape1D::new(n).expect("shape"));
    let input: Vec<Complex32> = (0..n)
        .map(|k| {
            let x = k as f32;
            Complex32::new((0.23 * x).sin(), 0.31 * (0.41 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| f64::from((*a - *b).norm()))
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= tolerance,
        "planned f32 N={n} forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n72_f64_codelet_forward_matches_direct() {
    let n = 72usize;
    let plan = FftPlan1D::<f64>::new(Shape1D::new(n).expect("shape"));
    assert!(
        matches!(plan.strategy, PlanStrategy::ShortWinograd),
        "f64 N=72 did not map to ShortWinograd"
    );
    let input: Vec<Complex64> = (0..n)
        .map(|k| {
            let x = k as f64;
            Complex64::new((0.23 * x).sin(), 0.31 * (0.41 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 1.0e-10,
        "planned f64 Good-Thomas N=72 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n96_f32_codelet_forward_matches_direct() {
    assert_f32_codelet_forward_matches_direct(96, 2.0e-4);
}

#[test]
fn planned_n99_f32_codelet_forward_matches_direct() {
    assert_f32_codelet_forward_matches_direct(99, 2.0e-4);
}

#[test]
fn planned_rader_n359_f64_forward_matches_direct() {
    let n = 359usize;
    let plan = FftPlan1D::<f64>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::Rader => {}
        _ => panic!("f64 N=359 must use the planned Rader route"),
    }
    let input: Vec<Complex64> = (0..n)
        .map(|k| {
            let x = k as f64;
            Complex64::new((0.11 * x).sin(), 0.17 * (0.07 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 1.0e-9,
        "planned f64 Rader N=359 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_rader_n359_f32_forward_matches_direct() {
    let n = 359usize;
    let plan = FftPlan1D::<f32>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::Rader => {}
        _ => panic!("f32 N=359 must use the planned Rader route"),
    }
    let input: Vec<Complex32> = (0..n)
        .map(|k| {
            let x = k as f32;
            Complex32::new((0.11 * x).sin(), 0.17 * (0.07 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| f64::from((*a - *b).norm()))
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 5.0e-4,
        "planned f32 Rader N=359 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_rader_n113_f32_forward_matches_direct() {
    let n = 113usize;
    let plan = FftPlan1D::<f32>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::Rader => {}
        _ => panic!("f32 N=113 must use the planned Rader route"),
    }
    let input: Vec<Complex32> = (0..n)
        .map(|k| {
            let x = k as f32;
            Complex32::new((0.11 * x).sin(), 0.17 * (0.07 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| f64::from((*a - *b).norm()))
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 3.0e-4,
        "planned f32 Rader N=113 forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n512_f64_pot_zst_forward_matches_direct() {
    let n = 512usize;
    let plan = FftPlan1D::<f64>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::PowerOfTwo { log2, .. } => assert_eq!(*log2, 9),
        _ => panic!("N=512 must use PowerOfTwo (ZST-wired) route"),
    }
    let input: Vec<Complex64> = (0..n)
        .map(|k| {
            let x = k as f64;
            Complex64::new((0.11 * x).sin(), 0.17 * (0.07 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 1.0e-9,
        "planned f64 PoT N=512 (ZST) forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_n512_f32_pot_zst_forward_matches_direct() {
    let n = 512usize;
    let plan = FftPlan1D::<f32>::new(Shape1D::new(n).expect("shape"));
    match &plan.strategy {
        PlanStrategy::PowerOfTwo { log2, .. } => assert_eq!(*log2, 9),
        _ => panic!("f32 N=512 must use PowerOfTwo (ZST-wired) route"),
    }
    let input: Vec<Complex32> = (0..n)
        .map(|k| {
            let x = k as f32;
            Complex32::new((0.11 * x).sin(), 0.17 * (0.07 * x).cos())
        })
        .collect();
    let expected = dft_forward(&input);
    let mut actual = input;
    plan.forward_complex_slice_inplace(&mut actual);
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| f64::from((*a - *b).norm()))
        .fold(0.0f64, f64::max);
    assert!(
        max_err <= 1.0e-3,
        "planned f32 PoT N=512 (ZST) forward mismatch max_err={max_err:.2e}"
    );
}

#[test]
fn planned_new_winograd_composite_sizes_match_direct() {
    let sizes = [
        72, 81, 96, 99, 108, 112, 120, 121, 126, 128, 144, 154, 168, 180, 189, 222, 242, 246, 259, 275, 280, 296, 363, 400, 484
    ];
    for &n in &sizes {
        // test f64
        let plan64 = FftPlan1D::<f64>::new(Shape1D::new(n).expect("shape"));
        if n != 128 && n != 144 && n != 180 {
            assert!(
                matches!(plan64.strategy, PlanStrategy::ShortWinograd),
                "f64 size {n} did not map to ShortWinograd"
            );
        }
        let input64 = signal64(n);
        let expected64 = dft_forward(&input64);
        let mut actual64 = input64;
        plan64.forward_complex_slice_inplace(&mut actual64);
        let err64 = actual64.iter().zip(expected64.iter()).map(|(a, b)| (*a - *b).norm()).fold(0.0f64, f64::max);
        assert!(err64 <= 1.0e-9, "f64 size {n} mismatch err={err64:.2e}");

        // test f32
        let plan32 = FftPlan1D::<f32>::new(Shape1D::new(n).expect("shape"));
        if n != 128 && n != 144 && n != 180 {
            assert!(
                matches!(plan32.strategy, PlanStrategy::ShortWinograd),
                "f32 size {n} did not map to ShortWinograd"
            );
        }
        let input32 = signal32(n);
        let expected32 = dft_forward(&input32);
        let mut actual32 = input32;
        plan32.forward_complex_slice_inplace(&mut actual32);
        let err32 = actual32.iter().zip(expected32.iter()).map(|(a, b)| f64::from((*a - *b).norm())).fold(0.0f64, f64::max);
        assert!(err32 <= 1.0e-3, "f32 size {n} mismatch err={err32:.2e}");
    }
}

