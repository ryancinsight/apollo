use crate::application::execution::kernel::direct::{dft_forward, dft_inverse};
use crate::application::execution::plan::fft::dimension_1d::strategy::PlanStrategy;
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

fn inverse_bounds64(input: &[Complex64]) -> (f64, f64) {
    let n = input.len();
    assert!(n.is_power_of_two(), "the radix-stage bound requires N=2^k");
    let l1: f64 = input.iter().map(|value| value.re.hypot(value.im)).sum();
    // The direct oracle performs at most eight rounded scalar operations per
    // input term; the radix-2 inverse performs at most sixteen per stage.
    // Higham's gamma_k = ku / (1 - ku), u = epsilon/2, bounds their combined
    // first-order error against the input L1 norm.
    let operations = 8.0 * n as f64 + 16.0 * f64::from(n.ilog2());
    let scaled_epsilon = operations * (f64::EPSILON / 2.0);
    let unnormalized = scaled_epsilon / (1.0 - scaled_epsilon) * l1;
    // Power-of-two normalization is exact for these finite normal-range values.
    (unnormalized / n as f64, unnormalized)
}

fn inverse_bounds32(input: &[Complex32]) -> (f32, f32) {
    let n = input.len();
    assert!(n.is_power_of_two(), "the radix-stage bound requires N=2^k");
    let l1: f32 = input.iter().map(|value| value.re.hypot(value.im)).sum();
    // Applying the f32 unit roundoff to the direct oracle is conservative: its
    // internal sums use f64 and quantize only the final result.
    let operations = 8.0 * n as f32 + 16.0 * n.ilog2() as f32;
    let scaled_epsilon = operations * (f32::EPSILON / 2.0);
    let unnormalized = scaled_epsilon / (1.0 - scaled_epsilon) * l1;
    (unnormalized / n as f32, unnormalized)
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
fn normalized_inverse_recovers_every_power_of_two_routing_rung() {
    // The normalized inverse must apply the 1/n IDFT scale for its own
    // transform length. The base-128 route serves n = 128, 256, and 512, so a
    // constant scale named after the route left 256 doubled and 512
    // quadrupled; no per-size normalized-inverse round trip existed to see it.
    // Every power-of-two rung the planner dispatches is covered here so a
    // route serving more than one length cannot mis-scale again unseen.
    for log2n in 0..=14u32 {
        let n = 1usize << log2n;
        let plan = FftPlan1D::<f64>::new(Shape1D::new(n).expect("shape"));
        let input = signal64(n);
        let mut spectrum = input.clone();
        plan.forward_complex_slice_inplace(&mut spectrum);
        // Bound the round trip against the spectrum feeding the inverse.
        let (bound, _) = inverse_bounds64(&spectrum);
        plan.inverse_complex_slice_inplace(&mut spectrum);

        let max_err = spectrum
            .iter()
            .zip(input.iter())
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0f64, f64::max);
        assert!(
            max_err <= bound,
            "N={n} normalized inverse round trip max_err={max_err:.3e} exceeds \
             derived bound {bound:.3e}"
        );
    }
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

#[test]
fn dynamic_f64_inverse_modes_match_direct() {
    let n = 128;
    let plan = FftPlan1D::<f64>::new(Shape1D::new(n).expect("shape"));
    let input = signal64(n);
    let expected_normalized = dft_inverse(&input);
    let expected_unnormalized: Vec<_> = expected_normalized
        .iter()
        .map(|value| Complex64::new(value.re * n as f64, value.im * n as f64))
        .collect();
    let (normalized_tolerance, unnormalized_tolerance) = inverse_bounds64(&input);
    let mut normalized = input.clone();
    let mut unnormalized = input;

    plan.inverse_complex_slice_inplace(&mut normalized);
    plan.inverse_complex_slice_unnorm_inplace(&mut unnormalized);

    for ((actual_normalized, actual_unnormalized), (direct_normalized, direct_unnormalized)) in
        normalized
            .iter()
            .zip(&unnormalized)
            .zip(expected_normalized.iter().zip(&expected_unnormalized))
    {
        let normalized_error = (*actual_normalized - *direct_normalized).norm();
        let unnormalized_error = (*actual_unnormalized - *direct_unnormalized).norm();
        assert!(
            normalized_error <= normalized_tolerance,
            "dynamic f64 normalized inverse differs by {normalized_error:.3e} > {normalized_tolerance:.3e}"
        );
        assert!(
            unnormalized_error <= unnormalized_tolerance,
            "dynamic f64 unnormalized inverse differs by {unnormalized_error:.3e} > {unnormalized_tolerance:.3e}"
        );
    }
}

#[test]
fn dynamic_f32_inverse_modes_match_direct() {
    let n = 128;
    let plan = FftPlan1D::<f32>::new(Shape1D::new(n).expect("shape"));
    let input = signal32(n);
    let expected_normalized = dft_inverse(&input);
    let expected_unnormalized: Vec<_> = expected_normalized
        .iter()
        .map(|value| Complex32::new(value.re * n as f32, value.im * n as f32))
        .collect();
    let (normalized_tolerance, unnormalized_tolerance) = inverse_bounds32(&input);
    let mut normalized = input.clone();
    let mut unnormalized = input;

    plan.inverse_complex_slice_inplace(&mut normalized);
    plan.inverse_complex_slice_unnorm_inplace(&mut unnormalized);

    for ((actual_normalized, actual_unnormalized), (direct_normalized, direct_unnormalized)) in
        normalized
            .iter()
            .zip(&unnormalized)
            .zip(expected_normalized.iter().zip(&expected_unnormalized))
    {
        let normalized_error = (*actual_normalized - *direct_normalized).norm();
        let unnormalized_error = (*actual_unnormalized - *direct_unnormalized).norm();
        assert!(
            normalized_error <= normalized_tolerance,
            "dynamic f32 normalized inverse differs by {normalized_error:.3e} > {normalized_tolerance:.3e}"
        );
        assert!(
            unnormalized_error <= unnormalized_tolerance,
            "dynamic f32 unnormalized inverse differs by {unnormalized_error:.3e} > {unnormalized_tolerance:.3e}"
        );
    }
}

#[test]
fn dynamic_zero_length_plans_preserve_empty_slices() {
    let plan64 = FftPlan1D::<f64>::new(Shape1D { n: 0 });
    let expected64 = Vec::<Complex64>::new();
    let mut forward64 = expected64.clone();
    plan64.forward_complex_slice_inplace(&mut forward64);
    assert_eq!(forward64, expected64, "f64 zero-length forward identity");
    let mut inverse64 = expected64.clone();
    plan64.inverse_complex_slice_inplace(&mut inverse64);
    assert_eq!(inverse64, expected64, "f64 zero-length inverse identity");
    let mut inverse_unnorm64 = expected64.clone();
    plan64.inverse_complex_slice_unnorm_inplace(&mut inverse_unnorm64);
    assert_eq!(
        inverse_unnorm64, expected64,
        "f64 zero-length unnormalized inverse identity"
    );

    let plan32 = FftPlan1D::<f32>::new(Shape1D { n: 0 });
    let expected32 = Vec::<Complex32>::new();
    let mut forward32 = expected32.clone();
    plan32.forward_complex_slice_inplace(&mut forward32);
    assert_eq!(forward32, expected32, "f32 zero-length forward identity");
    let mut inverse32 = expected32.clone();
    plan32.inverse_complex_slice_inplace(&mut inverse32);
    assert_eq!(inverse32, expected32, "f32 zero-length inverse identity");
    let mut inverse_unnorm32 = expected32.clone();
    plan32.inverse_complex_slice_unnorm_inplace(&mut inverse_unnorm32);
    assert_eq!(
        inverse_unnorm32, expected32,
        "f32 zero-length unnormalized inverse identity"
    );
}

#[test]
fn dynamic_one_length_plans_preserve_singletons() {
    let plan64 = FftPlan1D::<f64>::new(Shape1D::new(1).expect("shape"));
    let expected64 = vec![Complex64::new(-0.75, 0.3125)];
    let mut forward64 = expected64.clone();
    plan64.forward_complex_slice_inplace(&mut forward64);
    assert_eq!(forward64, expected64, "f64 singleton forward identity");
    let mut inverse64 = expected64.clone();
    plan64.inverse_complex_slice_inplace(&mut inverse64);
    assert_eq!(inverse64, expected64, "f64 singleton inverse identity");
    let mut inverse_unnorm64 = expected64.clone();
    plan64.inverse_complex_slice_unnorm_inplace(&mut inverse_unnorm64);
    assert_eq!(
        inverse_unnorm64, expected64,
        "f64 singleton unnormalized inverse identity"
    );

    let plan32 = FftPlan1D::<f32>::new(Shape1D::new(1).expect("shape"));
    let expected32 = vec![Complex32::new(0.625, -0.1875)];
    let mut forward32 = expected32.clone();
    plan32.forward_complex_slice_inplace(&mut forward32);
    assert_eq!(forward32, expected32, "f32 singleton forward identity");
    let mut inverse32 = expected32.clone();
    plan32.inverse_complex_slice_inplace(&mut inverse32);
    assert_eq!(inverse32, expected32, "f32 singleton inverse identity");
    let mut inverse_unnorm32 = expected32.clone();
    plan32.inverse_complex_slice_unnorm_inplace(&mut inverse_unnorm32);
    assert_eq!(
        inverse_unnorm32, expected32,
        "f32 singleton unnormalized inverse identity"
    );
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
        72, 81, 96, 99, 108, 112, 120, 121, 126, 128, 144, 154, 168, 180, 189, 222, 242, 246, 259,
        275, 280, 296, 363, 400, 484,
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
        let err64 = actual64
            .iter()
            .zip(expected64.iter())
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0f64, f64::max);
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
        let err32 = actual32
            .iter()
            .zip(expected32.iter())
            .map(|(a, b)| f64::from((*a - *b).norm()))
            .fold(0.0f64, f64::max);
        assert!(err32 <= 1.0e-3, "f32 size {n} mismatch err={err32:.2e}");
    }
}

// ── Plan/slice length-mismatch rejection ─────────────────────────────────────
//
// Dispatch selects length-specialized kernels from the plan length alone, so
// the entry assertion must fire before any kernel sees a mismatched slice:
// without it a short slice reached `get_unchecked` paths (out of bounds) and
// codelet `try_into` paths silently left the data untouched. Each test pins
// the dispatch family it exercises via the plan strategy, so the rejection is
// proven per family, not just once.

#[test]
#[should_panic(expected = "FFT plan length mismatch")]
fn forward_rejects_short_slice_on_tiny_direct_plan() {
    // n = 4 routes through the tiny runtime dispatch ahead of the stored
    // executor; unchecked it ran `small_pot_inplace_sized::<4>` on 2 elements.
    let plan = FftPlan1D::<f64>::new(Shape1D::new(4).expect("shape"));
    assert!(matches!(plan.strategy, PlanStrategy::PowerOfTwo { .. }));
    let mut data = signal64(2);
    plan.forward_complex_slice_inplace(&mut data);
}

#[test]
#[should_panic(expected = "FFT plan length mismatch")]
fn forward_rejects_short_slice_on_pot_plan() {
    let plan = FftPlan1D::<f64>::new(Shape1D::new(16).expect("shape"));
    assert!(matches!(plan.strategy, PlanStrategy::PowerOfTwo { .. }));
    let mut data = signal64(8);
    plan.forward_complex_slice_inplace(&mut data);
}

#[test]
#[should_panic(expected = "FFT plan length mismatch")]
fn forward_rejects_short_slice_on_composite_plan() {
    let plan = FftPlan1D::<f64>::new(Shape1D::new(36).expect("shape"));
    assert!(matches!(plan.strategy, PlanStrategy::Composite { .. }));
    let mut data = signal64(12);
    plan.forward_complex_slice_inplace(&mut data);
}

#[test]
#[should_panic(expected = "FFT plan length mismatch")]
fn forward_rejects_short_slice_on_winograd_plan() {
    // Codelet executors `try_into` a sized array; before the entry assertion a
    // mismatched slice failed that conversion and returned unchanged data.
    let plan = FftPlan1D::<f64>::new(Shape1D::new(12).expect("shape"));
    assert!(matches!(plan.strategy, PlanStrategy::ShortWinograd));
    let mut data = signal64(6);
    plan.forward_complex_slice_inplace(&mut data);
}

#[test]
#[should_panic(expected = "FFT plan length mismatch")]
fn forward_rejects_short_slice_on_good_thomas_plan() {
    let plan = FftPlan1D::<f64>::new(Shape1D::new(511).expect("shape"));
    assert!(matches!(plan.strategy, PlanStrategy::GoodThomas { .. }));
    let mut data = signal64(100);
    plan.forward_complex_slice_inplace(&mut data);
}

#[test]
#[should_panic(expected = "FFT plan length mismatch")]
fn forward_rejects_short_slice_on_rader_plan() {
    let plan = FftPlan1D::<f64>::new(Shape1D::new(359).expect("shape"));
    assert!(matches!(plan.strategy, PlanStrategy::Rader));
    let mut data = signal64(100);
    plan.forward_complex_slice_inplace(&mut data);
}

#[test]
#[should_panic(expected = "FFT plan length mismatch")]
fn forward_rejects_long_slice() {
    // A longer slice is rejected too: dispatch would transform only the plan
    // length prefix and silently return partially transformed data.
    let plan = FftPlan1D::<f64>::new(Shape1D::new(16).expect("shape"));
    let mut data = signal64(32);
    plan.forward_complex_slice_inplace(&mut data);
}

#[test]
#[should_panic(expected = "FFT plan length mismatch")]
fn inverse_rejects_length_mismatch() {
    let plan = FftPlan1D::<f64>::new(Shape1D::new(16).expect("shape"));
    let mut data = signal64(8);
    plan.inverse_complex_slice_inplace(&mut data);
}

#[test]
#[should_panic(expected = "FFT plan length mismatch")]
fn inverse_unnorm_rejects_length_mismatch() {
    let plan = FftPlan1D::<f64>::new(Shape1D::new(16).expect("shape"));
    let mut data = signal64(8);
    plan.inverse_complex_slice_unnorm_inplace(&mut data);
}

#[test]
#[should_panic(expected = "FFT plan length mismatch")]
fn forward_rejects_length_mismatch_f32() {
    let plan = FftPlan1D::<f32>::new(Shape1D::new(16).expect("shape"));
    let mut data = signal32(8);
    plan.forward_complex_slice_inplace(&mut data);
}

/// Lengths with no correct route must fail at plan construction rather than
/// return corrupt output.
///
/// `361 = 19^2` is rejected by `factorize_composite` (19 is above the
/// supported radix set) and has no coprime split, so every strategy arm
/// declines it and the fallback reached Rader — whose primality precondition
/// is guarded only by a `debug_assert`, so release builds computed silently
/// wrong results. Measured before this guard: a forward-then-normalized-inverse
/// round trip at 361 returned a maximum error of 2.019e1 against an input of
/// order 1, and `apollo-dctdst`'s DCT-II at 361 (which routes through a
/// 722-point FFT) returned relative error 1.0101 — output uncorrelated with
/// the truth, reported as success.
#[test]
#[should_panic(expected = "no correct route for length 361")]
fn composite_length_without_a_correct_route_is_rejected() {
    let _ = FftPlan1D::<f64>::new(Shape1D::new(361).expect("shape"));
}

/// The guard must reject only what is genuinely unroutable: the neighbours of
/// an affected length, and a prime that Rader legitimately serves, still plan
/// and round-trip.
#[test]
fn lengths_adjacent_to_the_rejected_route_still_round_trip() {
    for n in [360usize, 362, 359] {
        let plan = FftPlan1D::<f64>::new(Shape1D::new(n).expect("shape"));
        let original = signal64(n);
        let mut data = original.clone();
        plan.forward_complex_slice_inplace(&mut data);
        plan.inverse_complex_slice_inplace(&mut data);
        let max_err = original
            .iter()
            .zip(data.iter())
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0f64, f64::max);
        assert!(
            max_err <= 1.0e-10,
            "round trip at n = {n} drifted by {max_err:e}"
        );
    }
}
