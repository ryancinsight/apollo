//! WGPU value-semantic verification for the GFT GPU backend.

#[cfg(test)]
mod tests {
    use crate::{GftWgpuBackend, GftWgpuPlan, WgpuCapabilities, WgpuError};

    #[test]
    fn capabilities_reflect_implemented_kernel_surface() {
        let capabilities = WgpuCapabilities::implemented(true);
        assert!(capabilities.device_available);
        assert!(capabilities.supports_forward);
        assert!(capabilities.supports_inverse);
        assert!(capabilities.supports_mixed_precision);
        assert_eq!(
            capabilities.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
    }

    #[test]
    fn plan_preserves_logical_length() {
        let plan = GftWgpuPlan::new(4);
        assert_eq!(plan.len(), 4);
        assert!(!GftWgpuPlan::new(4).is_empty());
        assert!(GftWgpuPlan::new(0).is_empty());
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

    #[test]
    fn backend_reports_forward_and_inverse_when_device_exists() {
        let Ok(backend) = GftWgpuBackend::try_default() else {
            return;
        };
        let caps = backend.capabilities();
        assert!(caps.device_available);
        assert!(caps.supports_forward);
        assert!(caps.supports_inverse);
    }

    /// Build the 4-node path graph CPU plan and extract basis as f32.
    /// Adjacency: [[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]]
    fn path4_plan_and_basis() -> (apollo_gft::GftPlan, Vec<f32>, Vec<f32>) {
        let adj = leto::Array2::from_shape_vec(
            [4, 4],
            vec![
                0.0_f64, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            ],
        )
        .expect("path-4 adjacency shape");
        let cpu_plan = apollo_gft::GftPlan::from_adjacency(adj.view()).expect("path-4 gft plan");
        let basis_f32: Vec<f32> = cpu_plan.basis().iter().map(|&v| v as f32).collect();
        let signal_f32 = vec![1.0_f32, -0.5, 2.0, 0.5];
        (cpu_plan, basis_f32, signal_f32)
    }

    #[test]
    fn forward_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = GftWgpuBackend::try_default() else {
            return;
        };
        let (cpu_plan, basis_f32, signal_f32) = path4_plan_and_basis();
        let gpu_plan = GftWgpuPlan::new(4);
        let gpu_fwd = backend
            .execute_forward(&gpu_plan, &signal_f32, &basis_f32)
            .expect("gft forward");
        let signal_f64 = ndarray::Array1::from_vec(signal_f32.iter().map(|&v| v as f64).collect());
        let cpu_fwd = cpu_plan.forward(&signal_f64).expect("cpu gft forward");
        assert_eq!(gpu_fwd.len(), 4);
        for (k, (g, c)) in gpu_fwd.iter().zip(cpu_fwd.iter()).enumerate() {
            assert!(
                (*g as f64 - c).abs() < 1.0e-3_f64,
                "forward k={}: gpu={} cpu={}",
                k,
                g,
                c
            );
        }
    }

    #[test]
    fn inverse_matches_cpu_reference_when_device_exists() {
        let Ok(backend) = GftWgpuBackend::try_default() else {
            return;
        };
        let (cpu_plan, basis_f32, signal_f32) = path4_plan_and_basis();
        let signal_f64 = ndarray::Array1::from_vec(signal_f32.iter().map(|&v| v as f64).collect());
        // Use the CPU forward spectrum as input for the inverse.
        let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("cpu spectrum");
        let spectrum_f32: Vec<f32> = cpu_spectrum.iter().map(|&v| v as f32).collect();
        let gpu_plan = GftWgpuPlan::new(4);
        let gpu_inv = backend
            .execute_inverse(&gpu_plan, &spectrum_f32, &basis_f32)
            .expect("gft inverse");
        let cpu_inv = cpu_plan.inverse(&cpu_spectrum).expect("cpu inverse");
        assert_eq!(gpu_inv.len(), 4);
        for (k, (g, c)) in gpu_inv.iter().zip(cpu_inv.iter()).enumerate() {
            assert!(
                (*g as f64 - c).abs() < 1.0e-3_f64,
                "inverse k={}: gpu={} cpu={}",
                k,
                g,
                c
            );
        }
    }

    #[test]
    fn roundtrip_recovers_signal_when_device_exists() {
        let Ok(backend) = GftWgpuBackend::try_default() else {
            return;
        };
        let (_cpu_plan, basis_f32, signal_f32) = path4_plan_and_basis();
        let gpu_plan = GftWgpuPlan::new(4);
        let fwd = backend
            .execute_forward(&gpu_plan, &signal_f32, &basis_f32)
            .expect("roundtrip forward");
        let recovered = backend
            .execute_inverse(&gpu_plan, &fwd, &basis_f32)
            .expect("roundtrip inverse");
        assert_eq!(recovered.len(), 4);
        for (k, (actual, expected)) in recovered.iter().zip(signal_f32.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1.0e-3_f32,
                "roundtrip k={}: got={} want={}",
                k,
                actual,
                expected
            );
        }
    }

    #[test]
    fn typed_mixed_storage_matches_represented_f32_execution_when_device_exists() {
        let Ok(backend) = GftWgpuBackend::try_default() else {
            return;
        };
        use apollo_fft::{f16, PrecisionProfile};

        let (_cpu_plan, basis_f32, signal_f32) = path4_plan_and_basis();
        let input: Vec<f16> = signal_f32.iter().copied().map(f16::from_f32).collect();
        let represented_input: Vec<f32> = input.iter().map(|v| v.to_f64() as f32).collect();
        let gpu_plan = GftWgpuPlan::new(4);
        let expected_fwd = backend
            .execute_forward(&gpu_plan, &represented_input, &basis_f32)
            .expect("represented forward");
        let mut typed_fwd = vec![f16::from_f32(0.0); input.len()];
        backend
            .execute_forward_typed_into(
                &gpu_plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &input,
                &basis_f32,
                &mut typed_fwd,
            )
            .expect("typed mixed forward");
        assert_eq!(typed_fwd.len(), expected_fwd.len());
        for (actual, expected) in typed_fwd.iter().zip(expected_fwd.iter()) {
            let expected_f16 = f16::from_f32(*expected);
            assert_eq!(actual.to_bits(), expected_f16.to_bits());
        }

        let expected_inv = backend
            .execute_inverse(&gpu_plan, &expected_fwd, &basis_f32)
            .expect("represented inverse");
        let mut typed_inv = vec![f16::from_f32(0.0); input.len()];
        backend
            .execute_inverse_typed_into(
                &gpu_plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &typed_fwd,
                &basis_f32,
                &mut typed_inv,
            )
            .expect("typed mixed inverse");
        for (actual, expected) in typed_inv.iter().zip(expected_inv.iter()) {
            let q = expected.abs() * 2.0_f32.powi(-10) + f32::from(f16::MIN_POSITIVE);
            assert!(
                (actual.to_f32() - expected).abs() <= q,
                "f16 quantization mismatch: actual={}, expected={}",
                actual.to_f32(),
                expected
            );
        }
    }

    #[test]
    fn typed_path_rejects_profile_mismatch_when_device_exists() {
        let Ok(backend) = GftWgpuBackend::try_default() else {
            return;
        };
        use apollo_fft::{f16, PrecisionProfile};
        let (_cpu_plan, basis_f32, _) = path4_plan_and_basis();
        let plan = GftWgpuPlan::new(4);
        let input = vec![f16::from_f32(1.0); 4];
        let mut output = vec![f16::from_f32(0.0); 4];
        let err = backend
            .execute_forward_typed_into(
                &plan,
                PrecisionProfile::LOW_PRECISION_F32,
                &input,
                &basis_f32,
                &mut output,
            )
            .expect_err("profile mismatch must fail");
        assert_eq!(err, WgpuError::InvalidPrecisionProfile);
    }
}
