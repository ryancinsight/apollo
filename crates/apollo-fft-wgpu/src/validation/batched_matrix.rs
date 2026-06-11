//! Batched matrix-dispatch validation against analytical and CPU references.

use crate::infrastructure::gpu_fft::batched_matrix::prime::GpuPrimeBatch;
use crate::infrastructure::gpu_fft::batched_matrix::radix5::GpuRadix5Batch;
use apollo_fft::application::execution::kernel::direct::dft_forward;
use apollo_fft::domain::storage::FftPlanarMut;
use apollo_fft::f16;
use apollo_fft::infrastructure::cpu::simd::batched::{
    radix11_batched_f32, radix3_batched_f32, radix5_batched_f32,
};
use num_complex::Complex32;

fn to_column_major_complex(re: &[[f16; 8]; 5], im: &[[f16; 8]; 5], lane: usize) -> [Complex32; 5] {
    let mut out = [Complex32::new(0.0, 0.0); 5];
    for point in 0..5 {
        out[point] = Complex32::new(re[point][lane].to_f32(), im[point][lane].to_f32());
    }
    out
}

fn assert_gpu_prime_matches_analytical<const R: usize>(tolerance: f32) {
    let Ok((device, queue)) = GpuPrimeBatch::<R>::try_default_device() else {
        return;
    };
    let gpu = GpuPrimeBatch::<R>::new(device, queue, 8);

    let mut re = [[f16::ZERO; 8]; R];
    let mut im = [[f16::ZERO; 8]; R];
    for lane in 0..8 {
        for point in 0..R {
            let x = lane as f32 + 1.0;
            let t = point as f32;
            re[point][lane] = f16::from_f32((0.17 * x * t).sin());
            im[point][lane] = f16::from_f32((0.11 * x + 0.23 * t).cos());
        }
    }

    let gpu_out = gpu.forward_f16(&re, &im);
    for lane in 0..8 {
        let mut input = [Complex32::new(0.0, 0.0); R];
        for point in 0..R {
            input[point] = Complex32::new(re[point][lane].to_f32(), im[point][lane].to_f32());
        }
        let expected = dft_forward(&input);
        for point in 0..R {
            let idx = point * 8 + lane;
            let got = gpu_out[idx];
            let err = (got - expected[point]).norm();
            assert!(
                err <= tolerance,
                "R={R} lane={lane} point={point} err={err:.4e}"
            );
        }
    }
}

fn assert_gpu_matches_cpu_prime<const R: usize>(tolerance: f32) {
    let Ok((device, queue)) = GpuPrimeBatch::<R>::try_default_device() else {
        return;
    };
    let gpu = GpuPrimeBatch::<R>::new(device, queue, 8);

    let mut re = [[f16::ZERO; 8]; R];
    let mut im = [[f16::ZERO; 8]; R];
    for lane in 0..8 {
        for point in 0..R {
            let x = lane as f32 + 1.0;
            let t = point as f32;
            re[point][lane] = f16::from_f32((0.21 * x * t).sin() + 0.03 * t);
            im[point][lane] = f16::from_f32((0.09 * x + 0.27 * t).cos());
        }
    }

    let gpu_out = gpu.forward_f16(&re, &im);

    let mut re_cpu = [[0.0_f32; 8]; R];
    let mut im_cpu = [[0.0_f32; 8]; R];
    for lane in 0..8 {
        for point in 0..R {
            re_cpu[point][lane] = re[point][lane].to_f32();
            im_cpu[point][lane] = im[point][lane].to_f32();
        }
    }

    let mut storage = FftPlanarMut::new(re_cpu.as_flattened_mut(), im_cpu.as_flattened_mut(), R, 8);
    match R {
        3 => radix3_batched_f32(&mut storage, false),
        11 => radix11_batched_f32(&mut storage, false),
        _ => unreachable!("test helper currently supports radix-3 and radix-11"),
    }

    for lane in 0..8 {
        for point in 0..R {
            let idx = point * 8 + lane;
            let expected = Complex32::new(re_cpu[point][lane], im_cpu[point][lane]);
            let err = (gpu_out[idx] - expected).norm();
            assert!(
                err <= tolerance,
                "R={R} lane={lane} point={point} cpu-gpu err={err:.4e}"
            );
        }
    }
}

#[test]
fn gpu_batched_matrix_execution_suite_when_device_exists() {
    let Ok((device, queue)) = GpuRadix5Batch::try_default_device() else {
        return;
    };

    // 1. gpu_radix3_batch_matches_analytical
    assert_gpu_prime_matches_analytical::<3>(2.5e-3);

    // 2. gpu_radix3_batch_matches_cpu_backend
    assert_gpu_matches_cpu_prime::<3>(3.0e-3);

    // 3. gpu_radix7_batch_matches_analytical
    assert_gpu_prime_matches_analytical::<7>(4.0e-3);

    // 4. gpu_radix11_batch_matches_analytical
    assert_gpu_prime_matches_analytical::<11>(6.0e-3);

    // 5. gpu_radix11_batch_matches_cpu_backend
    assert_gpu_matches_cpu_prime::<11>(6.0e-3);

    // 6. gpu_radix5_batch_matches_analytical_direct_dft_per_lane
    {
        let gpu = GpuRadix5Batch::new(device.clone(), queue.clone(), 8);

        let mut re = [[f16::ZERO; 8]; 5];
        let mut im = [[f16::ZERO; 8]; 5];
        for lane in 0..8 {
            for point in 0..5 {
                let x = lane as f32 + 1.0;
                let t = point as f32;
                re[point][lane] = f16::from_f32((0.13 * x * t).sin() + 0.07 * t);
                im[point][lane] = f16::from_f32((0.17 * x + 0.19 * t).cos());
            }
        }

        let gpu_out = gpu.forward_f16(&re, &im);
        for lane in 0..8 {
            let input = to_column_major_complex(&re, &im, lane);
            let expected = dft_forward(&input);
            for point in 0..5 {
                let got = gpu_out[point * 8 + lane];
                let err = (got - expected[point]).norm();
                assert!(
                    err <= 2.5e-3,
                    "lane={lane} point={point} err={err:.4e} got={got:?} expected={:?}",
                    expected[point]
                );
            }
        }
    }

    // 7. gpu_radix5_batch_matches_cpu_backend
    {
        let gpu = GpuRadix5Batch::new(device.clone(), queue.clone(), 8);

        let mut re = [[f16::ZERO; 8]; 5];
        let mut im = [[f16::ZERO; 8]; 5];
        for lane in 0..8 {
            for point in 0..5 {
                let x = lane as f32 + 1.0;
                let t = point as f32;
                re[point][lane] = f16::from_f32((0.15 * x * t).sin() + 0.04 * t);
                im[point][lane] = f16::from_f32((0.12 * x + 0.19 * t).cos());
            }
        }

        let gpu_out = gpu.forward_f16(&re, &im);

        let mut re_cpu = [[0.0_f32; 8]; 5];
        let mut im_cpu = [[0.0_f32; 8]; 5];
        for lane in 0..8 {
            for point in 0..5 {
                re_cpu[point][lane] = re[point][lane].to_f32();
                im_cpu[point][lane] = im[point][lane].to_f32();
            }
        }
        let mut storage = FftPlanarMut::new(re_cpu.as_flattened_mut(), im_cpu.as_flattened_mut(), 5, 8);
        radix5_batched_f32(&mut storage, false);

        for lane in 0..8 {
            for point in 0..5 {
                let idx = point * 8 + lane;
                let expected = Complex32::new(re_cpu[point][lane], im_cpu[point][lane]);
                let err = (gpu_out[idx] - expected).norm();
                assert!(
                    err <= 3.0e-3,
                    "lane={lane} point={point} cpu-gpu err={err:.4e}"
                );
            }
        }
    }

    // 8. gpu_radix5_inverse_obeys_one_over_n_normalization
    {
        let gpu = GpuRadix5Batch::new(device, queue, 8);

        let mut re = [[f16::ZERO; 8]; 5];
        let mut im = [[f16::ZERO; 8]; 5];
        for lane in 0..8 {
            for point in 0..5 {
                let x = lane as f32 + 1.0;
                let t = point as f32;
                re[point][lane] = f16::from_f32((0.09 * x * t).sin());
                im[point][lane] = f16::from_f32((0.21 * x + 0.14 * t).cos());
            }
        }

        let mut spec_re = [[f16::ZERO; 8]; 5];
        let mut spec_im = [[f16::ZERO; 8]; 5];
        let gpu_fwd = gpu.forward_f16(&re, &im);
        for lane in 0..8 {
            for point in 0..5 {
                let v = gpu_fwd[point * 8 + lane];
                spec_re[point][lane] = f16::from_f32(v.re);
                spec_im[point][lane] = f16::from_f32(v.im);
            }
        }
        let gpu_inv = gpu.transform_f16(&spec_re, &spec_im, true);

        for lane in 0..8 {
            let expected_input = to_column_major_complex(&re, &im, lane);

            for point in 0..5 {
                let got = gpu_inv[point * 8 + lane] / 5.0;
                let err = (got - expected_input[point]).norm();
                assert!(
                    err <= 5.5e-3,
                    "lane={lane} point={point} err={err:.4e} got={got:?} expected={:?}",
                    expected_input[point]
                );
            }
        }
    }
}
