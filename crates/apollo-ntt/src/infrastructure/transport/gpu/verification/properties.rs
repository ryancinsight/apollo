//! Exact finite-field properties supporting the GPU transport contracts.

#[cfg(feature = "proptest_gpu")]
mod gpu_roundtrip {
    use crate::{infrastructure::transport::gpu::NttWgpuBackend, DEFAULT_MODULUS};
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 64,
            ..Default::default()
        })]

        /// `INTT(NTT(x)) = x` for generated power-of-two residue vectors.
        #[test]
        fn gpu_roundtrip_preserves_residue_class(
            log2_n in 0_u32..=8_u32,
            seed in any::<u64>(),
        ) {
            let length = 1_usize << log2_n;
            let mut state = seed;
            let input = (0..length)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    state % DEFAULT_MODULUS
                })
                .collect::<Vec<_>>();
            let device = match hephaestus_wgpu::WgpuDevice::try_default("apollo-ntt-wgpu") {
                Ok(device) => device,
                Err(hephaestus_core::HephaestusError::AdapterUnavailable { .. }) => return Ok(()),
                Err(error) => panic!("NTT GPU verification requires a working provider: {error}"),
            };
            let backend = NttWgpuBackend::new(device);
            let plan = backend.plan(length);
            let spectrum = backend
                .execute_forward(&plan, &input)
                .expect("GPU forward must succeed");
            let recovered = backend
                .execute_inverse(&plan, &spectrum)
                .expect("GPU inverse must succeed");
            prop_assert_eq!(recovered, input);
        }
    }
}

mod cpu_reference {
    use crate::{NttPlan, DEFAULT_MODULUS};
    use leto::Array1;
    use proptest::prelude::*;

    proptest! {
        /// `INTT(NTT(x)) = x` for generated exact residue vectors.
        #[test]
        fn cpu_roundtrip_preserves_residue_class(
            values in prop::collection::vec(0_u64..DEFAULT_MODULUS, 1..=32),
        ) {
            let length = values.len().next_power_of_two();
            let mut padded = values;
            padded.resize(length, 0);
            let input = Array1::from(padded);
            let plan = NttPlan::new(length).expect("power-of-two CPU plan");
            let spectrum = plan.forward(&input).expect("CPU forward");
            let recovered = plan.inverse(&spectrum).expect("CPU inverse");
            prop_assert_eq!(recovered, input);
        }

        /// `INTT(NTT(a) ⊙ NTT(b)) = a ★ b` in the residue field.
        #[test]
        fn convolution_theorem_holds_for_arbitrary_pairs(
            first_values in prop::collection::vec(0_u64..1000_u64, 2..=8),
            second_values in prop::collection::vec(0_u64..1000_u64, 2..=8),
        ) {
            let length = first_values
                .len()
                .max(second_values.len())
                .next_power_of_two()
                * 2;
            let mut first = first_values;
            first.resize(length, 0);
            let mut second = second_values;
            second.resize(length, 0);
            let plan = NttPlan::new(length).expect("CPU plan");
            let transformed_first = plan.forward(&Array1::from(first.clone())).expect("first forward");
            let transformed_second = plan.forward(&Array1::from(second.clone())).expect("second forward");
            let pointwise = transformed_first
                .iter()
                .zip(transformed_second.iter())
                .map(|(&first, &second)| {
                    ((first as u128 * second as u128) % DEFAULT_MODULUS as u128) as u64
                })
                .collect::<Vec<_>>();
            let actual = plan.inverse(&Array1::from(pointwise)).expect("inverse");

            let mut expected = vec![0_u64; length];
            for (first_index, &first_value) in first.iter().enumerate() {
                for (second_index, &second_value) in second.iter().enumerate() {
                    let product =
                        (first_value as u128 * second_value as u128 % DEFAULT_MODULUS as u128)
                            as u64;
                    expected[(first_index + second_index) % length] =
                        (expected[(first_index + second_index) % length] + product) % DEFAULT_MODULUS;
                }
            }
            prop_assert_eq!(actual.into_vec(), expected);
        }
    }
}
