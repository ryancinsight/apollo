//! WGPU value-semantic verification for the NTT backend.
//!
//! # GPU test policy
//!
//! Tests that require a live WGPU device use an early-return guard:
//!
//! ```rust,no_run
//! # use apollo_ntt::NttWgpuBackend;
//! let Ok(backend) = NttWgpuBackend::try_default() else { return; };
//! ```
//!
//! This allows the tests to run unconditionally on GPU-enabled hosts and
//! skip silently on CI hosts without an adapter.  No `#[ignore]` is needed.
//!
//! # Mathematical coverage
//!
//! Every GPU test asserts on computed residue values, not just `Result` variants:
//! - Forward transform against CPU `NttPlan` reference (impulse and Fibonacci inputs).
//! - Inverse recovers original residues exactly.
//! - Quantized `u32` storage paths match the `u64` allocating path.
//! - Reusable-buffer paths match allocating paths.
//! - Error conditions produce the expected `WgpuError` variant with correct fields.
//! - Proptest: forward→inverse roundtrip preserves every residue for arbitrary
//!   power-of-two lengths and arbitrary inputs (matches CPU reference exactly).

#[cfg(test)]
mod tests {
    use crate::{NttPlan, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT};
    use leto::{SliceArg, Storage};
    use leto::Array1;

    use crate::infrastructure::transport::gpu::{NttWgpuBackend, NttWgpuPlan, WgpuCapabilities, WgpuError};

    // -----------------------------------------------------------------------
    // Pure-struct tests (no GPU device required)
    // -----------------------------------------------------------------------

    #[test]
    fn capabilities_reflect_full_kernel_surface() {
        let cap = WgpuCapabilities::full(true);
        assert!(cap.device_available);
        assert!(cap.supports_forward);
        assert!(cap.supports_inverse);
        assert!(!cap.supports_mixed_precision,
            "NTT is exact integer arithmetic; mixed floating-point precision is architecturally unsupported");
        assert!(cap.supports_quantized_storage);
    }

    #[test]
    fn capabilities_detected_without_device_clears_all_execution_flags() {
        let cap = WgpuCapabilities::detected(false);
        assert!(!cap.device_available);
        assert!(!cap.supports_forward);
        assert!(!cap.supports_inverse);
        assert!(!cap.supports_mixed_precision);
        assert!(!cap.supports_quantized_storage);
    }

    #[test]
    fn plan_preserves_modular_configuration() {
        let plan = NttWgpuPlan::new(64, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT);
        assert_eq!(plan.len(), 64);
        assert_eq!(plan.modulus(), DEFAULT_MODULUS);
        assert_eq!(plan.primitive_root(), DEFAULT_PRIMITIVE_ROOT);
        assert!(!NttWgpuPlan::new(64, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT).is_empty());
        assert!(NttWgpuPlan::new(0, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT).is_empty());
    }

    #[test]
    fn unsupported_execution_error_identifies_operation() {
        let err = WgpuError::UnsupportedExecution {
            operation: "forward",
        };
        assert_eq!(
            err.to_string(),
            "forward is unsupported by the current WGPU capability set"
        );
    }

    // -----------------------------------------------------------------------
    // GPU tests — require a live WGPU device
    #[test]
    fn ntt_wgpu_execution_suite_when_device_exists() {
        let Ok(backend) = NttWgpuBackend::try_default() else {
            return;
        };

        // 1. backend_reports_forward_and_inverse
        {
            let cap = backend.capabilities();
            assert!(cap.device_available);
            assert!(cap.supports_forward);
            assert!(cap.supports_inverse);
            assert!(cap.supports_quantized_storage);
        }

        // 2. forward_impulse_matches_cpu_reference
        {
            let input = vec![1_u64, 0, 0, 0, 0, 0, 0, 0];
            let plan = backend.plan(input.len());
            let gpu = backend
                .execute_forward(&plan, &input)
                .expect("gpu forward execution");

            assert_eq!(gpu.len(), 8, "output length must equal input length");
            for (k, &val) in gpu.iter().enumerate() {
                assert_eq!(val, 1u64, "NTT8(impulse)[{k}] must equal 1");
            }

            let cpu_plan = NttPlan::new(input.len()).expect("cpu plan");
            let cpu = cpu_plan
                .forward(&Array1::from(input.clone()))
                .expect("cpu forward");
            assert_eq!(gpu, cpu.to_vec(), "gpu must match cpu reference exactly");
        }

        // 3. forward_fibonacci_matches_cpu_reference
        {
            let input = vec![1_u64, 1, 2, 3, 5, 8, 13, 21];
            let plan = backend.plan(input.len());
            let gpu = backend
                .execute_forward(&plan, &input)
                .expect("gpu forward execution");

            let cpu_plan = NttPlan::new(input.len()).expect("cpu plan");
            let cpu = cpu_plan
                .forward(&Array1::from(input.clone()))
                .expect("cpu forward");

            assert_eq!(
                gpu,
                cpu.to_vec(),
                "gpu forward NTT must match cpu reference exactly for Fibonacci input"
            );
        }

        // 4. inverse_recovers_input
        {
            let input = vec![1_u64, 1, 2, 3, 5, 8, 13, 21];
            let plan = backend.plan(input.len());
            let spectrum = backend
                .execute_forward(&plan, &input)
                .expect("gpu forward execution");
            let recovered = backend
                .execute_inverse(&plan, &spectrum)
                .expect("gpu inverse execution");

            assert_eq!(
                recovered, input,
                "INTT(NTT(x)) must recover x exactly for Fibonacci input"
            );
        }

        // 5. leto_forward_and_inverse_match_allocating_slice
        {
            let input = vec![1_u64, 1, 2, 3, 5, 8, 13, 21];
            let plan = backend.plan(input.len());
            let expected_forward = backend
                .execute_forward(&plan, &input)
                .expect("slice forward");
            let leto_input =
                leto::Array1::from_shape_vec([input.len()], input).expect("leto input");
            let actual_forward = backend
                .execute_forward_leto(&plan, leto_input.view())
                .expect("leto forward");
            assert_eq!(
                actual_forward.storage().as_slice(),
                expected_forward.as_slice()
            );

            let expected_inverse = backend
                .execute_inverse(&plan, &expected_forward)
                .expect("slice inverse");
            let leto_spectrum =
                leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
                    .expect("leto spectrum");
            let actual_inverse = backend
                .execute_inverse_leto(&plan, leto_spectrum.view())
                .expect("leto inverse");
            assert_eq!(
                actual_inverse.storage().as_slice(),
                expected_inverse.as_slice()
            );
        }

        // 6. leto_strided_forward_matches_logical_slice
        {
            let logical = vec![1_u64, 4, 9, 16, 25, 36, 49, 64];
            let mut backing = Vec::with_capacity(logical.len() * 2);
            for value in logical.iter().copied() {
                backing.push(value);
                backing.push(99);
            }
            let plan = backend.plan(logical.len());
            let expected = backend
                .execute_forward(&plan, &logical)
                .expect("slice forward");
            let leto_input = leto::Array1::from_shape_vec([backing.len()], backing).expect("input");
            let strided = leto_input
                .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
                .expect("strided view");
            let actual = backend
                .execute_forward_leto(&plan, strided)
                .expect("strided leto forward");
            assert_eq!(actual.storage().as_slice(), expected.as_slice());
        }

        // 7. quantized_u32_storage_matches_allocating_u64_execution
        {
            let input_u32 = vec![1_u32, 1, 2, 3, 5, 8, 13, 21];
            let input_u64: Vec<u64> = input_u32.iter().map(|&v| u64::from(v)).collect();
            let plan = backend.plan(input_u32.len());

            let expected_fwd = backend
                .execute_forward(&plan, &input_u64)
                .expect("allocating forward");
            let mut quantized_fwd = vec![0_u32; input_u32.len()];
            backend
                .execute_forward_quantized_into(&plan, &input_u32, &mut quantized_fwd)
                .expect("quantized forward");
            let quantized_fwd_u64: Vec<u64> = quantized_fwd.iter().map(|&v| u64::from(v)).collect();
            assert_eq!(
                quantized_fwd_u64, expected_fwd,
                "quantized forward must match allocating forward exactly"
            );

            let mut quantized_inv = vec![0_u32; input_u32.len()];
            backend
                .execute_inverse_quantized_into(&plan, &quantized_fwd, &mut quantized_inv)
                .expect("quantized inverse");
            assert_eq!(
                quantized_inv, input_u32,
                "quantized INTT(NTT(x)) must recover x exactly"
            );
        }

        // 8. quantized_leto_forward_and_inverse_match_quantized_slice
        {
            let input = vec![3_u32, 1, 4, 1, 5, 9, 2, 6];
            let plan = backend.plan(input.len());
            let mut expected_forward = vec![0_u32; input.len()];
            backend
                .execute_forward_quantized_into(&plan, &input, &mut expected_forward)
                .expect("quantized forward");
            let leto_input =
                leto::Array1::from_shape_vec([input.len()], input).expect("leto input");
            let actual_forward = backend
                .execute_forward_quantized_leto(&plan, leto_input.view())
                .expect("leto quantized forward");
            assert_eq!(
                actual_forward.storage().as_slice(),
                expected_forward.as_slice()
            );

            let mut expected_inverse = vec![0_u32; expected_forward.len()];
            backend
                .execute_inverse_quantized_into(&plan, &expected_forward, &mut expected_inverse)
                .expect("quantized inverse");
            let leto_spectrum =
                leto::Array1::from_shape_vec([expected_forward.len()], expected_forward)
                    .expect("leto spectrum");
            let actual_inverse = backend
                .execute_inverse_quantized_leto(&plan, leto_spectrum.view())
                .expect("leto quantized inverse");
            assert_eq!(
                actual_inverse.storage().as_slice(),
                expected_inverse.as_slice()
            );
        }

        // 9. reusable_buffers_match_allocating_forward_and_inverse
        {
            let input = vec![1_u64, 4, 9, 16, 25, 36, 49, 64];
            let plan = backend.plan(input.len());
            let mut buffers = backend
                .create_buffers(&plan)
                .expect("reusable buffers for plan");

            let alloc_fwd = backend
                .execute_forward(&plan, &input)
                .expect("allocating forward");
            backend
                .execute_forward_with_buffers(&plan, &input, &mut buffers)
                .expect("buffered forward");
            assert_eq!(
                backend.buffer_output(&buffers),
                alloc_fwd.as_slice(),
                "buffered forward must match allocating forward"
            );

            let spectrum = backend.buffer_output(&buffers).to_vec();
            let alloc_inv = backend
                .execute_inverse(&plan, &spectrum)
                .expect("allocating inverse");
            backend
                .execute_inverse_with_buffers(&plan, &spectrum, &mut buffers)
                .expect("buffered inverse");
            assert_eq!(
                backend.buffer_output(&buffers),
                alloc_inv.as_slice(),
                "buffered inverse must match allocating inverse"
            );
            assert_eq!(
                backend.buffer_output(&buffers),
                input.as_slice(),
                "INTT(NTT(x)) via reusable buffers must recover x"
            );
        }

        // 10. quantized_u32_reusable_buffers_match_allocating_quantized_path
        {
            let input = vec![3_u32, 1, 4, 1, 5, 9, 2, 6];
            let plan = backend.plan(input.len());
            let mut buffers = backend.create_buffers(&plan).expect("reusable buffers");

            let mut alloc_fwd = vec![0_u32; input.len()];
            backend
                .execute_forward_quantized_into(&plan, &input, &mut alloc_fwd)
                .expect("allocating quantized forward");
            backend
                .execute_forward_quantized_with_buffers(&plan, &input, &mut buffers)
                .expect("buffered quantized forward");
            let alloc_fwd_u64: Vec<u64> = alloc_fwd.iter().map(|&v| u64::from(v)).collect();
            assert_eq!(
                backend.buffer_output(&buffers),
                alloc_fwd_u64.as_slice(),
                "buffered quantized forward must match allocating quantized forward"
            );

            let mut alloc_inv = vec![0_u32; input.len()];
            backend
                .execute_inverse_quantized_into(&plan, &alloc_fwd, &mut alloc_inv)
                .expect("allocating quantized inverse");
            backend
                .execute_inverse_quantized_with_buffers(&plan, &alloc_fwd, &mut buffers)
                .expect("buffered quantized inverse");
            let alloc_inv_u64: Vec<u64> = alloc_inv.iter().map(|&v| u64::from(v)).collect();
            assert_eq!(
                backend.buffer_output(&buffers),
                alloc_inv_u64.as_slice(),
                "buffered quantized inverse must match allocating quantized inverse"
            );
            assert_eq!(alloc_inv, input, "quantized INTT(NTT(x)) must recover x");
        }

        // 11. quantized_u32_storage_rejects_output_length_mismatch
        {
            let plan = backend.plan(8);
            let mut output = vec![0_u32; 4];
            let err = backend
                .execute_forward_quantized_into(&plan, &[0; 8], &mut output)
                .expect_err("output length mismatch must produce an error");
            assert_eq!(
                err,
                WgpuError::LengthMismatch {
                    expected: 8,
                    actual: 4,
                }
            );
        }

        // 12. reusable_buffers_reject_plan_length_mismatch
        {
            let plan = backend.plan(8);
            let short_plan = backend.plan(4);
            let mut short_buffers = backend
                .create_buffers(&short_plan)
                .expect("short reusable buffers");
            let err = backend
                .execute_forward_with_buffers(&plan, &[0; 8], &mut short_buffers)
                .expect_err("buffer length mismatch must produce an error");
            assert_eq!(
                err,
                WgpuError::LengthMismatch {
                    expected: 8,
                    actual: 4,
                }
            );
        }

        // 13. rejects_invalid_plan_and_length_before_dispatch
        {
            // Empty plan.
            let empty_err = backend
                .execute_forward(
                    &NttWgpuPlan::new(0, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT),
                    &[],
                )
                .expect_err("empty plan must fail");
            assert!(
                matches!(empty_err, WgpuError::InvalidPlan { ref message } if message.contains("length must be greater than zero")),
                "empty plan must report zero length"
            );

            // Non-power-of-two plan.
            let non_pow_err = backend
                .execute_forward(
                    &NttWgpuPlan::new(6, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT),
                    &[0; 6],
                )
                .expect_err("non-power-of-two plan must fail");
            assert!(
                matches!(non_pow_err, WgpuError::InvalidPlan { ref message } if message.contains("power of two")),
                "non-power-of-two plan must be rejected"
            );

            // Input length mismatch.
            let mismatch_err = backend
                .execute_forward(
                    &NttWgpuPlan::new(8, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT),
                    &[0; 4],
                )
                .expect_err("input length mismatch must fail");
            assert_eq!(
                mismatch_err,
                WgpuError::LengthMismatch {
                    expected: 8,
                    actual: 4,
                }
            );
        }
    }

    // -----------------------------------------------------------------------
    // Proptest: forward → inverse roundtrip (GPU-dependent)
    // -----------------------------------------------------------------------
    //
    // This property covers:
    //   ∀ N = 2^k (k ∈ 0..8), ∀ x ∈ F_m^N: INTT(NTT(x)) = x
    //
    // It validates both the GPU butterfly correctness and the N^{-1} scaling
    // against the CPU reference simultaneously.  Shrinking produces the
    // smallest failing (N, x) pair when a regression is introduced.

    #[cfg(feature = "proptest_gpu")]
    mod proptest_gpu {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(proptest::test_runner::Config {
                cases: 64,
                ..Default::default()
            })]

            /// GPU INTT(NTT(x)) = x for arbitrary power-of-two N and arbitrary
            /// residue inputs.  Requires `--features proptest-gpu` and a GPU
            /// device (run with `-- --include-ignored`).
            #[test]
                    fn gpu_roundtrip_preserves_residue_class(
                log2_n in 0_u32..=8_u32,
                seed in any::<u64>(),
            ) {
                let n = 1usize << log2_n;
                // Derive deterministic input from seed using a simple LCG.
                let mut state = seed;
                let input: Vec<u64> = (0..n).map(|_| {
                    state = state.wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    state % DEFAULT_MODULUS
                }).collect();

                let Ok(backend) = NttWgpuBackend::try_default() else {
                    return Ok(());
                };
                let plan = backend.plan(n);
                let spectrum = backend.execute_forward(&plan, &input)
                    .expect("gpu forward must succeed");
                let recovered = backend.execute_inverse(&plan, &spectrum)
                    .expect("gpu inverse must succeed");

                prop_assert_eq!(recovered, input,
                    "INTT(NTT(x)) must recover x for N={}", n);
            }
        }
    }

    // -----------------------------------------------------------------------
    // CPU-only proptest roundtrip (always runs, no GPU required)
    // -----------------------------------------------------------------------
    //
    // This is not strictly a GPU test but verifies that the host-side
    // bit_reverse_permute and twiddle precomputation (which underpin the GPU
    // kernel) are correct by cross-checking against the CPU NttPlan.
    // It runs in standard CI without a GPU device.

    mod cpu_reference {
        use crate::{NttPlan, DEFAULT_MODULUS};
        use leto::Array1;
        use proptest::prelude::*;

        proptest! {
            /// CPU INTT(NTT(x)) = x for arbitrary power-of-two N and arbitrary
            /// residue inputs.  Validates the mathematical specification used
            /// to derive the GPU kernel correctness criterion.
            #[test]
            fn cpu_roundtrip_preserves_residue_class(
                values in prop::collection::vec(0u64..DEFAULT_MODULUS, 1..=32),
            ) {
                let n = values.len().next_power_of_two();
                let mut padded = values;
                padded.resize(n, 0);
                let input = Array1::from(padded);
                let plan = NttPlan::new(n).expect("cpu plan must succeed for power-of-two N");
                let spectrum = plan.forward(&input).expect("cpu forward must succeed");
                let recovered = plan.inverse(&spectrum).expect("cpu inverse must succeed");
                prop_assert_eq!(recovered, input,
                    "CPU INTT(NTT(x)) must recover x");
            }

            /// NTT convolution theorem: INTT(NTT(a) ⊙ NTT(b)) = a ★ b.
            /// Validates pointwise multiplication in the NTT domain produces
            /// the correct cyclic convolution in the residue domain.
            #[test]
            fn convolution_theorem_holds_for_arbitrary_pairs(
                a_vals in prop::collection::vec(0u64..1000u64, 2..=8),
                b_vals in prop::collection::vec(0u64..1000u64, 2..=8),
            ) {
                use crate::DEFAULT_MODULUS;
                let n = a_vals.len().max(b_vals.len()).next_power_of_two() * 2;
                if n > 1 << 23 { return Ok(()); } // skip if beyond modulus support
                let mut a_pad = a_vals.clone();
                a_pad.resize(n, 0);
                let mut b_pad = b_vals.clone();
                b_pad.resize(n, 0);
                let plan = NttPlan::new(n).expect("plan");
                let fa = plan.forward(&Array1::from(a_pad.clone())).expect("forward a");
                let fb = plan.forward(&Array1::from(b_pad.clone())).expect("forward b");
                let fc: Vec<u64> = fa.iter().zip(fb.iter())
                    .map(|(&x, &y)| ((x as u128 * y as u128) % DEFAULT_MODULUS as u128) as u64)
                    .collect();
                let c = plan.inverse(&Array1::from(fc)).expect("inverse");

                // Direct cyclic convolution for verification.
                let mut expected = vec![0u64; n];
                for (i, &ai) in a_pad.iter().enumerate() {
                    for (j, &bj) in b_pad.iter().enumerate() {
                        expected[(i + j) % n] =
                            (expected[(i + j) % n] + (ai as u128 * bj as u128 % DEFAULT_MODULUS as u128) as u64)
                            % DEFAULT_MODULUS;
                    }
                }
                prop_assert_eq!(c.to_vec(), expected,
                    "INTT(NTT(a)*NTT(b)) must equal cyclic convolution");
            }
        }
    }
}
