use crate::infrastructure::kernel::direct::{dct1, dct4, dst1, dst4};
use crate::{DctDstPlan, RealTransformKind};
use proptest::prelude::*;

proptest! {
    /// Property: (2/N) * DCT-III(DCT-II(x)) = x, L-inf err < 1e-9, for n in [2,32].
    #[test]
    fn dct2_dct3_inverse_pair(
        signal in proptest::collection::vec(-1.0f64..1.0f64, 2..33usize),
    ) {
        let n = signal.len();
        let plan_forward = DctDstPlan::new(n, RealTransformKind::DctII).unwrap();
        let plan_inverse = DctDstPlan::new(n, RealTransformKind::DctIII).unwrap();
        let forward = plan_forward.forward(&signal).unwrap();
        let recovered_raw = plan_inverse.forward(&forward).unwrap();
        let scale = 2.0 / n as f64;
        let recovered: Vec<f64> = recovered_raw.into_iter().map(|v| v * scale).collect();
        let err: f64 = signal
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        prop_assert!(err < 1e-9, "DCT-II/III inverse pair failed: err={}", err);
    }

    /// Property: (2/N) * DST-III(DST-II(x)) = x, L-inf err < 1e-9, for n in [2,32].
    #[test]
    fn dst2_dst3_inverse_pair(
        signal in proptest::collection::vec(-1.0f64..1.0f64, 2..33usize),
    ) {
        let n = signal.len();
        let plan_forward = DctDstPlan::new(n, RealTransformKind::DstII).unwrap();
        let plan_inverse = DctDstPlan::new(n, RealTransformKind::DstIII).unwrap();
        let forward = plan_forward.forward(&signal).unwrap();
        let recovered_raw = plan_inverse.forward(&forward).unwrap();
        let scale = 2.0 / n as f64;
        let recovered: Vec<f64> = recovered_raw.into_iter().map(|v| v * scale).collect();
        let err: f64 = signal
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        prop_assert!(err < 1e-9, "DST-II/III inverse pair failed: err={}", err);
    }

    /// Property: DCT-I(DCT-I(x)) = 2(N−1)·x, L-inf err < 1e-9, for n in [2,32].
    #[test]
    fn dct1_self_inverse_property(
        signal in proptest::collection::vec(-1.0f64..1.0f64, 2..33usize),
    ) {
        let n = signal.len();
        let mut first = vec![0.0_f64; n];
        let mut second = vec![0.0_f64; n];
        dct1(&signal, &mut first);
        dct1(&first, &mut second);
        let scale = 2.0 * (n - 1) as f64;
        let err: f64 = signal
            .iter()
            .zip(second.iter())
            .map(|(x, y)| (y - x * scale).abs())
            .fold(0.0_f64, f64::max);
        prop_assert!(err < 1e-9, "DCT-I self-inverse failed: err={}", err);
    }

    /// Property: plan.inverse(plan.forward(x)) = x for DctI, L-inf err < 1e-9, n in [2,32].
    #[test]
    fn plan_dct1_roundtrip(
        signal in proptest::collection::vec(-1.0f64..1.0f64, 2..33usize),
    ) {
        let n = signal.len();
        let plan = DctDstPlan::new(n, RealTransformKind::DctI).unwrap();
        let forward = plan.forward(&signal).unwrap();
        let recovered = plan.inverse(&forward).unwrap();
        let err: f64 = signal
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        prop_assert!(err < 1e-9, "DctI plan roundtrip failed: err={}", err);
    }

    /// Property: DCT-IV(DCT-IV(x)) = (N/2)·x, L-inf err < 1e-9, for n in [1,32].
    #[test]
    fn dct4_self_inverse_property(
        signal in proptest::collection::vec(-1.0f64..1.0f64, 1..33usize),
    ) {
        let n = signal.len();
        let mut first = vec![0.0_f64; n];
        let mut second = vec![0.0_f64; n];
        dct4(&signal, &mut first);
        dct4(&first, &mut second);
        let scale = n as f64 / 2.0;
        let err: f64 = signal
            .iter()
            .zip(second.iter())
            .map(|(x, y)| (y - x * scale).abs())
            .fold(0.0_f64, f64::max);
        prop_assert!(err < 1e-9, "DCT-IV self-inverse failed: err={}", err);
    }

    /// Property: plan.inverse(plan.forward(x)) = x for DctIV, L-inf err < 1e-9, n in [1,32].
    #[test]
    fn plan_dct4_roundtrip(
        signal in proptest::collection::vec(-1.0f64..1.0f64, 1..33usize),
    ) {
        let n = signal.len();
        let plan = DctDstPlan::new(n, RealTransformKind::DctIV).unwrap();
        let forward = plan.forward(&signal).unwrap();
        let recovered = plan.inverse(&forward).unwrap();
        let err: f64 = signal
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        prop_assert!(err < 1e-9, "DctIV plan roundtrip failed: err={}", err);
    }

    /// Property: DST-I(DST-I(x)) = 2(N+1)·x, L-inf err < 1e-9, for n in [1,32].
    #[test]
    fn dst1_self_inverse_property(
        signal in proptest::collection::vec(-1.0f64..1.0f64, 1..33usize),
    ) {
        let n = signal.len();
        let mut first = vec![0.0_f64; n];
        let mut second = vec![0.0_f64; n];
        dst1(&signal, &mut first);
        dst1(&first, &mut second);
        let scale = 2.0 * (n + 1) as f64;
        let err: f64 = signal
            .iter()
            .zip(second.iter())
            .map(|(x, y)| (y - x * scale).abs())
            .fold(0.0_f64, f64::max);
        prop_assert!(err < 1e-9, "DST-I self-inverse failed: err={}", err);
    }

    /// Property: plan.inverse(plan.forward(x)) = x for DstI, L-inf err < 1e-9, n in [1,32].
    #[test]
    fn plan_dst1_roundtrip(
        signal in proptest::collection::vec(-1.0f64..1.0f64, 1..33usize),
    ) {
        let n = signal.len();
        let plan = DctDstPlan::new(n, RealTransformKind::DstI).unwrap();
        let forward = plan.forward(&signal).unwrap();
        let recovered = plan.inverse(&forward).unwrap();
        let err: f64 = signal
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        prop_assert!(err < 1e-9, "DstI plan roundtrip failed: err={}", err);
    }

    /// Property: DST-IV(DST-IV(x)) = (N/2)·x, L-inf err < 1e-9, for n in [1,32].
    #[test]
    fn dst4_self_inverse_property(
        signal in proptest::collection::vec(-1.0f64..1.0f64, 1..33usize),
    ) {
        let n = signal.len();
        let mut first = vec![0.0_f64; n];
        let mut second = vec![0.0_f64; n];
        dst4(&signal, &mut first);
        dst4(&first, &mut second);
        let scale = n as f64 / 2.0;
        let err: f64 = signal
            .iter()
            .zip(second.iter())
            .map(|(x, y)| (y - x * scale).abs())
            .fold(0.0_f64, f64::max);
        prop_assert!(err < 1e-9, "DST-IV self-inverse failed: err={}", err);
    }

    /// Property: plan.inverse(plan.forward(x)) = x for DstIV, L-inf err < 1e-9, n in [1,32].
    #[test]
    fn plan_dst4_roundtrip(
        signal in proptest::collection::vec(-1.0f64..1.0f64, 1..33usize),
    ) {
        let n = signal.len();
        let plan = DctDstPlan::new(n, RealTransformKind::DstIV).unwrap();
        let forward = plan.forward(&signal).unwrap();
        let recovered = plan.inverse(&forward).unwrap();
        let err: f64 = signal
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        prop_assert!(err < 1e-9, "DstIV plan roundtrip failed: err={}", err);
    }
}
