//! Value-semantic verification for the STFT WGPU backend.

#[cfg(test)]
mod tests {
    use crate::infrastructure::transport::gpu::{StftWgpuBackend, StftWgpuPlan, WgpuCapabilities, WgpuError};
    use apollo_fft::{f16, PrecisionProfile};
    use leto::{SliceArg, Storage};
    use leto::Array1;
    use eunomia::Complex32;

    // -----------------------------------------------------------------------
    // Structural / plan tests (no GPU required)
    // -----------------------------------------------------------------------

    #[test]
    fn capabilities_reflect_forward_only_surface() {
        let caps = WgpuCapabilities::forward_only(true);
        assert!(caps.device_available);
        assert!(caps.supports_forward);
        assert!(!caps.supports_inverse);
        assert!(caps.supports_mixed_precision);
        assert_eq!(
            caps.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
        let caps_off = WgpuCapabilities::forward_only(false);
        assert!(!caps_off.device_available);
        assert!(!caps_off.supports_forward);
        assert!(!caps_off.supports_inverse);
        assert!(caps_off.supports_mixed_precision);
        assert_eq!(
            caps_off.default_precision_profile,
            apollo_fft::PrecisionProfile::LOW_PRECISION_F32
        );
    }

    #[test]
    fn capabilities_reflect_forward_and_inverse_surface() {
        let caps = WgpuCapabilities::forward_and_inverse(true);
        assert!(caps.device_available);
        assert!(caps.supports_forward);
        assert!(caps.supports_inverse);
        assert!(caps.supports_mixed_precision);
        let caps_off = WgpuCapabilities::forward_and_inverse(false);
        assert!(!caps_off.device_available);
        assert!(!caps_off.supports_forward);
        assert!(!caps_off.supports_inverse);
    }

    #[test]
    fn plan_preserves_frame_and_hop_length() {
        let plan = StftWgpuPlan::new(8, 4);
        assert_eq!(plan.frame_len(), 8);
        assert_eq!(plan.hop_len(), 4);
        assert_eq!(plan.len(), 8);
        assert!(!plan.is_empty());
        assert!(StftWgpuPlan::new(0, 4).is_empty());
        assert!(StftWgpuPlan::new(8, 0).is_empty());
    }

    #[test]
    fn unsupported_execution_error_identifies_operation() {
        let err = WgpuError::UnsupportedExecution {
            operation: "inverse",
        };
        assert_eq!(
            err.to_string(),
            "inverse is unsupported by the current WGPU capability set"
        );
    }

    #[test]
    fn stft_wgpu_rejects_invalid_plan() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        // zero frame_len
        let r = backend.execute_forward(&StftWgpuPlan::new(0, 4), &[0.0f32; 8]);
        assert!(matches!(r, Err(WgpuError::InvalidPlan { .. })), "{r:?}");

        // zero hop_len
        let r = backend.execute_forward(&StftWgpuPlan::new(8, 0), &[0.0f32; 8]);
        assert!(matches!(r, Err(WgpuError::InvalidPlan { .. })), "{r:?}");

        // hop > frame
        let r = backend.execute_forward(&StftWgpuPlan::new(4, 8), &[0.0f32; 4]);
        assert!(matches!(r, Err(WgpuError::InvalidPlan { .. })), "{r:?}");

        // signal too short
        let r = backend.execute_forward(&StftWgpuPlan::new(8, 4), &[0.0f32; 4]);
        assert!(matches!(r, Err(WgpuError::InputTooShort { .. })), "{r:?}");
    }

    #[test]
    fn stft_wgpu_capabilities() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        let caps = backend.capabilities();
        assert!(caps.device_available);
        assert!(caps.supports_forward);
        assert!(caps.supports_inverse);
    }

    #[test]
    fn stft_wgpu_forward_matches_cpu() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        // 16-sample alternating signal: values 0, 1, 0, -1, ...
        let signal_f32: Vec<f32> = vec![
            0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0,
        ];
        let signal_f64: Array1<f64> =
            Array1::from(signal_f32.iter().map(|x| *x as f64).collect());

        let plan = StftWgpuPlan::new(8, 4);

        let gpu_out = backend
            .execute_forward(&plan, &signal_f32)
            .expect("GPU forward STFT");

        let cpu_plan = crate::StftPlan::new(8, 4).expect("CPU plan");
        let cpu_out = cpu_plan.forward(&signal_f64).expect("CPU forward STFT");

        assert_eq!(
            gpu_out.len(),
            cpu_out.len(),
            "output length mismatch: gpu={}, cpu={}",
            gpu_out.len(),
            cpu_out.len()
        );

        const TOL: f32 = 1e-3;
        for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            let re_err = (g.re - c.re as f32).abs();
            let im_err = (g.im - c.im as f32).abs();
            assert!(
                re_err < TOL,
                "re mismatch at index {i}: gpu={:.6}, cpu={:.6}, err={:.2e}",
                g.re,
                c.re,
                re_err
            );
            assert!(
                im_err < TOL,
                "im mismatch at index {i}: gpu={:.6}, cpu={:.6}, err={:.2e}",
                g.im,
                c.im,
                im_err
            );
        }
    }

    #[test]
    fn stft_wgpu_inverse_roundtrip_cola() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        // Alternating signal, smooth, non-trivial, well inside f32 dynamic range.
        let signal_f32: Vec<f32> = vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
        let signal_f64: Array1<f64> =
            Array1::from(signal_f32.iter().map(|&x| x as f64).collect());
        let signal_len = 8usize;

        let plan = StftWgpuPlan::new(8, 4);

        // Compute spectrum on CPU (f64) as the authoritative input.
        // frame_count = 1 + 8.div_ceil(4) = 3; spectrum_len = 3 * 8 = 24.
        let cpu_plan = crate::StftPlan::new(8, 4).expect("CPU plan");
        let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("CPU forward");

        // Downcast spectrum to f32 for GPU inverse.
        let gpu_spectrum: Vec<Complex32> = cpu_spectrum
            .iter()
            .map(|c| Complex32::new(c.re as f32, c.im as f32))
            .collect();

        let recovered = backend
            .execute_inverse(&plan, &gpu_spectrum, signal_len)
            .expect("GPU inverse STFT");

        // CPU inverse as reference for value-semantic comparison.
        let cpu_recovered = cpu_plan
            .inverse(&cpu_spectrum, signal_len)
            .expect("CPU inverse");

        assert_eq!(
            recovered.len(),
            signal_len,
            "recovered length mismatch: got {}, expected {signal_len}",
            recovered.len()
        );
        assert_eq!(
            cpu_recovered.len(),
            signal_len,
            "cpu_recovered length mismatch"
        );

        // Tolerance accounts for f64→f32 downcast of spectrum and f32 GPU arithmetic.
        const TOL: f32 = 5e-4;
        for (i, (gpu_val, cpu_val)) in recovered.iter().zip(cpu_recovered.iter()).enumerate() {
            let err = (gpu_val - *cpu_val as f32).abs();
            assert!(
                err < TOL,
                "mismatch at {i}: gpu={gpu_val:.6}, cpu={cpu_val:.6}, err={err:.2e}"
            );
        }
    }

    #[test]
    fn stft_wgpu_inverse_matches_cpu() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        let signal_f32: Vec<f32> = vec![
            0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0,
        ];
        let signal_f64: Array1<f64> =
            Array1::from(signal_f32.iter().map(|&x| x as f64).collect());
        let signal_len = 16usize;

        let plan = StftWgpuPlan::new(8, 4);

        // frame_count = 1 + 16.div_ceil(4) = 5; spectrum_len = 5 * 8 = 40.
        let cpu_plan = crate::StftPlan::new(8, 4).expect("CPU plan");
        let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("CPU forward");

        let gpu_spectrum: Vec<Complex32> = cpu_spectrum
            .iter()
            .map(|c| Complex32::new(c.re as f32, c.im as f32))
            .collect();

        let recovered = backend
            .execute_inverse(&plan, &gpu_spectrum, signal_len)
            .expect("GPU inverse STFT");

        let cpu_recovered = cpu_plan
            .inverse(&cpu_spectrum, signal_len)
            .expect("CPU inverse");

        assert_eq!(
            recovered.len(),
            signal_len,
            "recovered length mismatch: got {}, expected {signal_len}",
            recovered.len()
        );

        const TOL: f32 = 5e-4;
        for (i, (gpu_val, cpu_val)) in recovered.iter().zip(cpu_recovered.iter()).enumerate() {
            let err = (gpu_val - *cpu_val as f32).abs();
            assert!(
                err < TOL,
                "mismatch at {i}: gpu={gpu_val:.6}, cpu={cpu_val:.6}, err={err:.2e}"
            );
        }
    }

    #[test]
    fn stft_wgpu_typed_mixed_storage() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        // 8-sample signal: f16 quantization of the alternating pattern is exact for these values.
        let signal_f32: Vec<f32> = vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
        let signal_f16: Vec<f16> = signal_f32.iter().map(|&x| f16::from_f32(x)).collect();
        // Represented input: f16 promoted back to f32 (round-trip defines the reference).
        let represented: Vec<f32> = signal_f16.iter().map(|v| v.to_f32()).collect();
        let plan = StftWgpuPlan::new(4, 2);
        // frame_count = 1 + 8.div_ceil(2) = 5; output_len = 5 * 4 = 20.
        let f32_result = backend
            .execute_forward(&plan, &represented)
            .expect("f32 reference");
        let mut typed_out: Vec<[f16; 2]> =
            vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; f32_result.len()];
        backend
            .execute_forward_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &signal_f16,
                &mut typed_out,
            )
            .expect("typed mixed forward");
        for (actual, expected) in typed_out.iter().zip(f32_result.iter()) {
            let expected_f16 = [f16::from_f32(expected.re), f16::from_f32(expected.im)];
            assert_eq!(
                actual[0].to_bits(),
                expected_f16[0].to_bits(),
                "re bits mismatch: actual={:?} expected={:?}",
                actual[0],
                expected_f16[0]
            );
            assert_eq!(
                actual[1].to_bits(),
                expected_f16[1].to_bits(),
                "im bits mismatch: actual={:?} expected={:?}",
                actual[1],
                expected_f16[1]
            );
        }
    }

    #[test]
    fn stft_wgpu_leto_forward_and_inverse() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        let signal: Vec<f32> = vec![
            0.0, 1.0, 0.0, -1.0, 0.25, 0.75, -0.25, -0.75, 0.0, 1.0, 0.0, -1.0, 0.25, 0.75,
            -0.25, -0.75,
        ];
        let plan = StftWgpuPlan::new(8, 4);
        let leto_signal =
            leto::Array1::from_shape_vec([signal.len()], signal.clone()).expect("signal");

        let expected_forward = backend
            .execute_forward(&plan, &signal)
            .expect("slice forward");
        let actual_forward = backend
            .execute_forward_leto(&plan, leto_signal.view())
            .expect("leto forward");
        assert_eq!(
            actual_forward.storage().as_slice(),
            expected_forward.as_slice()
        );

        let leto_spectrum =
            leto::Array1::from_shape_vec([expected_forward.len()], expected_forward.clone())
                .expect("spectrum");
        let expected_inverse = backend
            .execute_inverse(&plan, &expected_forward, signal.len())
            .expect("slice inverse");
        let actual_inverse = backend
            .execute_inverse_leto(&plan, leto_spectrum.view(), signal.len())
            .expect("leto inverse");
        assert_eq!(
            actual_inverse.storage().as_slice(),
            expected_inverse.as_slice()
        );
    }

    #[test]
    fn stft_wgpu_leto_strided_forward() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        let logical: Vec<f32> = vec![
            0.0, 1.0, 0.0, -1.0, 0.25, 0.75, -0.25, -0.75, 0.0, 1.0, 0.0, -1.0, 0.25, 0.75,
            -0.25, -0.75,
        ];
        let mut backing = Vec::with_capacity(logical.len() * 2);
        for value in &logical {
            backing.push(*value);
            backing.push(99.0);
        }
        let leto_signal =
            leto::Array1::from_shape_vec([backing.len()], backing).expect("signal");
        let strided = leto_signal
            .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
            .expect("strided signal");

        let plan = StftWgpuPlan::new(8, 4);
        let expected = backend
            .execute_forward(&plan, &logical)
            .expect("slice forward");
        let actual = backend
            .execute_forward_leto(&plan, strided)
            .expect("leto strided forward");
        assert_eq!(actual.storage().as_slice(), expected.as_slice());
    }

    #[test]
    fn stft_wgpu_typed_leto() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        let signal_f16: Vec<f16> = [0.0_f32, 1.0, 0.0, -1.0, 0.25, 0.75, -0.25, -0.75]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let plan = StftWgpuPlan::new(4, 2);
        let frame_count = 1 + signal_f16.len().div_ceil(plan.hop_len());
        let output_len = frame_count * plan.frame_len();

        let mut expected_forward: Vec<[f16; 2]> =
            vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; output_len];
        backend
            .execute_forward_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &signal_f16,
                &mut expected_forward,
            )
            .expect("typed slice forward");

        let leto_signal = leto::Array1::from_shape_vec([signal_f16.len()], signal_f16.clone())
            .expect("signal");
        let actual_forward = backend
            .execute_forward_leto_typed::<f16, [f16; 2]>(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                leto_signal.view(),
            )
            .expect("typed leto forward");
        assert_eq!(
            actual_forward.storage().as_slice(),
            expected_forward.as_slice()
        );

        let mut expected_inverse = vec![f16::from_f32(0.0); signal_f16.len()];
        backend
            .execute_inverse_typed_into(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &expected_forward,
                signal_f16.len(),
                &mut expected_inverse,
            )
            .expect("typed slice inverse");

        let actual_inverse = backend
            .execute_inverse_leto_typed::<[f16; 2], f16>(
                &plan,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                actual_forward.view(),
                signal_f16.len(),
            )
            .expect("typed leto inverse");
        assert_eq!(
            actual_inverse.storage().as_slice(),
            expected_inverse.as_slice()
        );
    }

    #[test]
    fn stft_wgpu_typed_profile_mismatch() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        // f16 signal: I::PROFILE == MIXED_PRECISION_F16_F32.
        // Passing LOW_PRECISION_F32 as input_precision must trigger InvalidPrecisionProfile
        // before any GPU work is attempted.
        let signal_f16: Vec<f16> = vec![
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(-1.0),
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(-1.0),
        ];
        let plan = StftWgpuPlan::new(4, 2);
        let frame_count = 1 + signal_f16.len().div_ceil(plan.hop_len());
        let output_len = frame_count * plan.frame_len();
        let mut out: Vec<[f16; 2]> = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; output_len];
        let error = backend
            .execute_forward_typed_into(
                &plan,
                PrecisionProfile::LOW_PRECISION_F32,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
                &signal_f16,
                &mut out,
            )
            .expect_err("profile mismatch must fail");
        assert_eq!(error, WgpuError::InvalidPrecisionProfile);
    }

    #[test]
    fn stft_wgpu_multiple_cola_sets() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        let run_roundtrip = |frame_len: usize, hop_len: usize, signal: &[f32]| {
            let signal_len = signal.len();
            let plan = StftWgpuPlan::new(frame_len, hop_len);
            let cpu_plan = crate::StftPlan::new(frame_len, hop_len).expect("cpu plan");
            let signal_f64 = Array1::from(signal.iter().map(|&x| x as f64).collect());
            let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("cpu forward");
            let gpu_spectrum: Vec<Complex32> = cpu_spectrum
                .iter()
                .map(|c| Complex32::new(c.re as f32, c.im as f32))
                .collect();
            let recovered = backend
                .execute_inverse(&plan, &gpu_spectrum, signal_len)
                .unwrap_or_else(|e| {
                    panic!(
                        "GPU inverse failed for frame_len={frame_len}, hop_len={hop_len}: {e}"
                    )
                });
            let cpu_recovered = cpu_plan
                .inverse(&cpu_spectrum, signal_len)
                .expect("cpu inverse");
            assert_eq!(
                recovered.len(),
                signal_len,
                "length mismatch (frame={frame_len}, hop={hop_len})"
            );
            const TOL: f32 = 5e-3;
            for (i, (g, c)) in recovered.iter().zip(cpu_recovered.iter()).enumerate() {
                let err = (g - *c as f32).abs();
                assert!(
                    err < TOL,
                    "mismatch at sample {i} (frame={frame_len}, hop={hop_len}): \
                     gpu={g:.6}, cpu={c:.6}, err={err:.2e}"
                );
            }
        };

        // Case 1: frame_len=8, hop_len=4 — canonical 50% overlap, single frame.
        run_roundtrip(8, 4, &[0.5_f32, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5]);

        // Case 2: frame_len=16, hop_len=8 — 50% overlap, sine wave reference signal.
        let sig16: Vec<f32> = (0..16)
            .map(|i| ((i as f32) * std::f32::consts::FRAC_PI_4).sin())
            .collect();
        run_roundtrip(16, 8, &sig16);

        // Case 3: frame_len=32, hop_len=16 — 50% overlap, cosine reference signal.
        let sig32: Vec<f32> = (0..32)
            .map(|i| ((i as f32) * std::f32::consts::PI / 8.0).cos())
            .collect();
        run_roundtrip(32, 16, &sig32);
    }

    #[test]
    fn stft_wgpu_inverse_non_pot() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        let plan = StftWgpuPlan::new(6, 3);
        let dummy_spectrum = vec![Complex32::new(0.0, 0.0); 18];
        let result = backend.execute_inverse(&plan, &dummy_spectrum, 6);
        assert!(
            result.is_ok(),
            "expected Chirp-Z inverse acceptance; got {:?}",
            result
        );
    }

    #[test]
    fn stft_wgpu_inverse_large_frame() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        const FRAME_LEN: usize = 1024;
        const HOP_LEN: usize = 512;
        const SIGNAL_LEN: usize = 8192;
        const TOL: f32 = 5e-3;

        let signal_f32: Vec<f32> = (0..SIGNAL_LEN)
            .map(|n| (std::f32::consts::TAU * n as f32 / SIGNAL_LEN as f32).sin())
            .collect();

        let plan = StftWgpuPlan::new(FRAME_LEN, HOP_LEN);
        let cpu_plan = crate::StftPlan::new(FRAME_LEN, HOP_LEN).expect("cpu plan");
        let signal_f64 = Array1::from(signal_f32.iter().map(|&x| x as f64).collect());
        let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("cpu forward");
        let gpu_spectrum: Vec<Complex32> = cpu_spectrum
            .iter()
            .map(|c| Complex32::new(c.re as f32, c.im as f32))
            .collect();

        let gpu_result = backend
            .execute_inverse(&plan, &gpu_spectrum, SIGNAL_LEN)
            .expect("GPU inverse must succeed for power-of-two frame_len");

        let cpu_reference = cpu_plan
            .inverse(&cpu_spectrum, SIGNAL_LEN)
            .expect("cpu inverse");

        assert_eq!(
            gpu_result.len(),
            SIGNAL_LEN,
            "output length must equal signal_len"
        );
        let max_err = gpu_result
            .iter()
            .zip(cpu_reference.iter())
            .map(|(g, c)| (g - *c as f32).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_err <= TOL,
            "max |GPU - CPU_ref| = {max_err:.2e} exceeds TOL = {TOL:.2e} for frame_len={FRAME_LEN}"
        );
    }

    #[test]
    fn stft_wgpu_forward_non_pot() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        let signal = vec![0.0f32; 12];
        let plan = StftWgpuPlan::new(6, 3);
        let r = backend.execute_forward(&plan, &signal);
        assert!(r.is_ok(), "expected Chirp-Z forward acceptance; got {r:?}");
    }

    #[test]
    fn stft_wgpu_forward_large_frame() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        const FRAME_LEN: usize = 1024;
        const HOP_LEN: usize = 512;
        const SIGNAL_LEN: usize = 4096;
        const TOL: f32 = 1e-2;

        let signal_f32: Vec<f32> = (0..SIGNAL_LEN)
            .map(|n| {
                let t = n as f32;
                (2.0 * std::f32::consts::PI * 16.0 * t / FRAME_LEN as f32).sin()
                    + 0.5 * (2.0 * std::f32::consts::PI * 64.0 * t / FRAME_LEN as f32).sin()
            })
            .collect();

        let plan = StftWgpuPlan::new(FRAME_LEN, HOP_LEN);
        let frame_count = 1 + SIGNAL_LEN.div_ceil(HOP_LEN);

        let spectrum = backend
            .execute_forward(&plan, &signal_f32)
            .expect("GPU forward FFT");
        assert_eq!(
            spectrum.len(),
            frame_count * FRAME_LEN,
            "spectrum length mismatch"
        );

        let recovered = backend
            .execute_inverse(&plan, &spectrum, SIGNAL_LEN)
            .expect("GPU inverse FFT");
        assert_eq!(recovered.len(), SIGNAL_LEN, "recovered length mismatch");

        let margin = FRAME_LEN / 2;
        for i in margin..(SIGNAL_LEN - margin) {
            let err = (recovered[i] - signal_f32[i]).abs();
            assert!(
                err < TOL,
                "sample {i}: recovered={:.6}, expected={:.6}, err={:.2e}",
                recovered[i],
                signal_f32[i],
                err
            );
        }
    }

    #[test]
    fn stft_wgpu_reusable_buffers() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        const FRAME_LEN: usize = 512;
        const HOP_LEN: usize = 256;
        const SIGNAL_LEN: usize = 2048;
        const TOL: f32 = 1e-6;

        let signal: Vec<f32> = (0..SIGNAL_LEN)
            .map(|n| (2.0 * std::f32::consts::PI * 16.0 * n as f32 / FRAME_LEN as f32).sin())
            .collect();

        let plan = StftWgpuPlan::new(FRAME_LEN, HOP_LEN);
        let frame_count = 1 + SIGNAL_LEN.div_ceil(HOP_LEN);

        // ── Allocating path (reference) ───────────────────────────────────────
        let alloc_fwd = backend
            .execute_forward(&plan, &signal)
            .expect("allocating forward");
        assert_eq!(alloc_fwd.len(), frame_count * FRAME_LEN);

        let alloc_inv = backend
            .execute_inverse(&plan, &alloc_fwd, SIGNAL_LEN)
            .expect("allocating inverse");
        assert_eq!(alloc_inv.len(), SIGNAL_LEN);

        // ── Buffered path ─────────────────────────────────────────────────────
        let mut buffers = backend
            .make_buffers(&plan, SIGNAL_LEN)
            .expect("make_buffers");

        backend
            .execute_forward_with_buffers(&plan, &signal, &mut buffers)
            .expect("buffered forward");
        let buffered_fwd = buffers.fwd_output();
        assert_eq!(buffered_fwd.len(), frame_count * FRAME_LEN);

        // Forward output must match allocating path.
        let max_fwd_err = alloc_fwd
            .iter()
            .zip(buffered_fwd.iter())
            .map(|(a, b)| {
                let re_err = (a.re - b.re).abs();
                let im_err = (a.im - b.im).abs();
                re_err.max(im_err)
            })
            .fold(0.0f32, f32::max);
        assert!(
            max_fwd_err < TOL,
            "forward max error {max_fwd_err:.2e} exceeds tolerance {TOL:.2e}"
        );

        backend
            .execute_inverse_with_buffers(&plan, &alloc_fwd, SIGNAL_LEN, &mut buffers)
            .expect("buffered inverse");
        let buffered_inv = buffers.inv_output();
        assert_eq!(buffered_inv.len(), SIGNAL_LEN);

        // Inverse output must match allocating path.
        let max_inv_err = alloc_inv
            .iter()
            .zip(buffered_inv.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_inv_err < TOL,
            "inverse max error {max_inv_err:.2e} exceeds tolerance {TOL:.2e}"
        );

        // Second call with same buffers: verify buffer reuse (no panic / corruption).
        backend
            .execute_forward_with_buffers(&plan, &signal, &mut buffers)
            .expect("buffered forward second call");
        let buffered_fwd2 = buffers.fwd_output();
        let max_fwd2_err = alloc_fwd
            .iter()
            .zip(buffered_fwd2.iter())
            .map(|(a, b)| {
                let re_err = (a.re - b.re).abs();
                let im_err = (a.im - b.im).abs();
                re_err.max(im_err)
            })
            .fold(0.0f32, f32::max);
        assert!(
            max_fwd2_err < TOL,
            "second-call forward max error {max_fwd2_err:.2e} exceeds tolerance {TOL:.2e}"
        );
    }

    #[test]
    fn stft_wgpu_non_pot_structural() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        let signal = vec![0.0f32; 24];
        let plan = StftWgpuPlan::new(6, 3);
        let r = backend.execute_forward(&plan, &signal);
        assert!(
            r.is_ok(),
            "expected Chirp-Z structural forward acceptance; got {r:?}"
        );
    }

    #[test]
    fn stft_wgpu_forward_chirpz_400() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        use crate::StftPlan;

        const FRAME_LEN: usize = 400;
        const HOP_LEN: usize = 200;
        const SIGNAL_LEN: usize = 2000;
        const TOL: f32 = 2e-2;

        let signal_f32: Vec<f32> = (0..SIGNAL_LEN)
            .map(|n| (2.0 * std::f32::consts::PI * 10.0 * n as f32 / FRAME_LEN as f32).sin())
            .collect();
        let signal_f64: leto::Array1<f64> =
            leto::Array1::from(signal_f32.iter().map(|&x| x as f64).collect());

        let gpu_plan = StftWgpuPlan::new(FRAME_LEN, HOP_LEN);
        let gpu_out = backend
            .execute_forward(&gpu_plan, &signal_f32)
            .expect("GPU forward Chirp-Z");

        let cpu_plan = StftPlan::new(FRAME_LEN, HOP_LEN).expect("CPU plan");
        let cpu_out = cpu_plan.forward(&signal_f64).expect("CPU forward STFT");

        assert_eq!(
            gpu_out.len(),
            cpu_out.len(),
            "length mismatch: gpu={}, cpu={}",
            gpu_out.len(),
            cpu_out.len()
        );

        let max_err = gpu_out
            .iter()
            .zip(cpu_out.iter())
            .map(|(g, c)| {
                let re = (g.re - c.re as f32).abs();
                let im = (g.im - c.im as f32).abs();
                re.max(im)
            })
            .fold(0.0f32, f32::max);

        assert!(
            max_err < TOL,
            "max Chirp-Z forward error {max_err:.2e} exceeds tolerance {TOL:.2e}"
        );
    }

    #[test]
    fn stft_wgpu_inverse_chirpz_400() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        const FRAME_LEN: usize = 400;
        const HOP_LEN: usize = 200;
        const SIGNAL_LEN: usize = 2000;
        const TOL: f32 = 5e-2;
        const INTERIOR_START: usize = FRAME_LEN;
        const INTERIOR_END: usize = SIGNAL_LEN - FRAME_LEN;

        let signal_f32: Vec<f32> = (0..SIGNAL_LEN)
            .map(|n| (2.0 * std::f32::consts::PI * 10.0 * n as f32 / FRAME_LEN as f32).sin())
            .collect();
        let signal_f64: leto::Array1<f64> =
            leto::Array1::from(signal_f32.iter().map(|&x| x as f64).collect());

        let plan = StftWgpuPlan::new(FRAME_LEN, HOP_LEN);

        let cpu_plan = crate::StftPlan::new(FRAME_LEN, HOP_LEN).expect("CPU plan");
        let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("CPU forward");
        let spectrum_f32: Vec<Complex32> = cpu_spectrum
            .iter()
            .map(|c| Complex32::new(c.re as f32, c.im as f32))
            .collect();

        let recovered = backend
            .execute_inverse(&plan, &spectrum_f32, SIGNAL_LEN)
            .expect("GPU inverse Chirp-Z");

        assert_eq!(recovered.len(), SIGNAL_LEN);

        for i in INTERIOR_START..INTERIOR_END {
            let err = (recovered[i] - signal_f32[i]).abs();
            assert!(
                err < TOL,
                "sample {i}: recovered={:.6}, expected={:.6}, err={:.2e}",
                recovered[i],
                signal_f32[i],
                err
            );
        }
    }

    #[test]
    fn stft_wgpu_make_buffers_non_pot() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        let plan = StftWgpuPlan::new(6, 3);
        let signal_len = 24usize;
        let r = backend.make_buffers(&plan, signal_len);
        assert!(r.is_ok(), "expected non-PoT buffer acceptance; got {r:?}");
    }

    #[test]
    fn stft_wgpu_forward_buffers_400() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        use crate::StftPlan;

        const FRAME_LEN: usize = 400;
        const HOP_LEN: usize = 200;
        const SIGNAL_LEN: usize = 1000;
        const TOL: f32 = 2e-2;

        let signal_f32: Vec<f32> = (0..SIGNAL_LEN)
            .map(|n| (2.0 * std::f32::consts::PI * 10.0 * n as f32 / FRAME_LEN as f32).sin())
            .collect();
        let signal_f64: leto::Array1<f64> =
            leto::Array1::from(signal_f32.iter().map(|&x| x as f64).collect());

        let plan = StftWgpuPlan::new(FRAME_LEN, HOP_LEN);
        let mut buffers = backend
            .make_buffers(&plan, SIGNAL_LEN)
            .expect("make_buffers non-PoT");

        backend
            .execute_forward_with_buffers(&plan, &signal_f32, &mut buffers)
            .expect("forward with buffers");

        let cpu_plan = StftPlan::new(FRAME_LEN, HOP_LEN).expect("CPU plan");
        let cpu_out = cpu_plan.forward(&signal_f64).expect("CPU forward");

        let gpu_spectrum = buffers.fwd_output();
        assert_eq!(gpu_spectrum.len(), cpu_out.len());

        let max_err = gpu_spectrum
            .iter()
            .zip(cpu_out.iter())
            .map(|(g, c)| {
                let re = (g.re - c.re as f32).abs();
                let im = (g.im - c.im as f32).abs();
                re.max(im)
            })
            .fold(0.0f32, f32::max);

        assert!(
            max_err < TOL,
            "max forward buffered error {max_err:.2e} exceeds tolerance {TOL:.2e}"
        );
    }

    #[test]
    fn stft_wgpu_inverse_buffers_400() {
        let Ok(backend) = StftWgpuBackend::try_default() else {
            return;
        };
        use crate::StftPlan;

        const FRAME_LEN: usize = 400;
        const HOP_LEN: usize = 200;
        const SIGNAL_LEN: usize = 1000;
        const TOL: f32 = 5e-2;
        const INTERIOR_START: usize = FRAME_LEN;
        const INTERIOR_END: usize = SIGNAL_LEN - FRAME_LEN;

        let signal_f32: Vec<f32> = (0..SIGNAL_LEN)
            .map(|n| (2.0 * std::f32::consts::PI * 10.0 * n as f32 / FRAME_LEN as f32).sin())
            .collect();
        let signal_f64: leto::Array1<f64> =
            leto::Array1::from(signal_f32.iter().map(|&x| x as f64).collect());

        let plan = StftWgpuPlan::new(FRAME_LEN, HOP_LEN);
        let cpu_plan = StftPlan::new(FRAME_LEN, HOP_LEN).expect("CPU plan");
        let cpu_spectrum = cpu_plan.forward(&signal_f64).expect("CPU forward");
        let spectrum_f32: Vec<Complex32> = cpu_spectrum
            .iter()
            .map(|c| Complex32::new(c.re as f32, c.im as f32))
            .collect();

        let mut buffers = backend
            .make_buffers(&plan, SIGNAL_LEN)
            .expect("make_buffers non-PoT");

        backend
            .execute_inverse_with_buffers(&plan, &spectrum_f32, SIGNAL_LEN, &mut buffers)
            .expect("inverse with buffers");

        let recovered = buffers.inv_output().to_vec();
        assert_eq!(recovered.len(), SIGNAL_LEN);

        for i in INTERIOR_START..INTERIOR_END {
            let err = (recovered[i] - signal_f32[i]).abs();
            assert!(
                err < TOL,
                "sample {i}: recovered={:.6}, expected={:.6}, err={:.2e}",
                recovered[i],
                signal_f32[i],
                err
            );
        }
    }
}
