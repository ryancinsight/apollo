use super::*;

fn make_signal(nx: usize, ny: usize, nz: usize) -> Array3<f64> {
    Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        (i as f64 * 0.31 + j as f64 * 0.17 + k as f64 * 0.41).sin()
            + 0.5 * (i as f64 * 0.07 + j as f64 * 0.23 + k as f64 * 0.13).cos()
    })
}

/// Roundtrip identity: inverse(forward(x)) == x for asymmetric non-power-of-two sizes.
#[test]
fn roundtrip_recovers_asymmetric_inputs() {
    for (nx, ny, nz) in [(7usize, 13usize, 5usize), (16, 8, 9)] {
        let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
        let plan = FftPlan3D::new(shape);
        let input = make_signal(nx, ny, nz);
        let recovered = plan.inverse(&plan.forward(&input));
        for (a, b) in input.iter().zip(recovered.iter()) {
            let err = (a - b).abs();
            assert!(err < 1e-10, "roundtrip n=({nx},{ny},{nz}) err={err:.2e}");
        }
    }
}

/// Linearity: forward(a*s1 + b*s2) == a*forward(s1) + b*forward(s2), eps 1e-9.
#[test]
fn forward_is_linear() {
    let (nx, ny, nz) = (5usize, 7usize, 3usize);
    let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
    let plan = FftPlan3D::new(shape);
    let a = 2.3f64;
    let b = -1.7f64;
    let s1 = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        (i as f64 * 0.3 + j as f64 * 0.2 + k as f64 * 0.5).sin()
    });
    let s2 = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        (i as f64 * 0.7 + j as f64 * 0.4 + k as f64 * 0.1).cos()
    });
    let combined = &s1 * a + &s2 * b;
    let lhs = plan.forward(&combined);
    let rhs = plan.forward(&s1).mapv(|v| v * a) + plan.forward(&s2).mapv(|v| v * b);
    for (l, r) in lhs.iter().zip(rhs.iter()) {
        let err = (l - r).norm();
        assert!(err < 1e-9, "linearity err={err:.2e}");
    }
}

/// Parseval: sum|x|^2 == sum|X|^2 / (nx*ny*nz), eps 1e-6.
#[test]
fn parseval_identity_holds() {
    let (nx, ny, nz) = (8usize, 6usize, 5usize);
    let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
    let plan = FftPlan3D::new(shape);
    let input = make_signal(nx, ny, nz);
    let spectrum = plan.forward(&input);
    let time_energy: f64 = input.iter().map(|x| x * x).sum();
    let spectral_energy: f64 =
        spectrum.iter().map(|x| x.norm_sqr()).sum::<f64>() / (nx * ny * nz) as f64;
    let err = (time_energy - spectral_energy).abs();
    assert!(err < 1e-6, "Parseval err={err:.2e}");
}

/// Complex in-place forward then inverse recovers original, eps 1e-10.
#[test]
fn complex_inplace_roundtrip() {
    let (nx, ny, nz) = (8usize, 4usize, 6usize);
    let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
    let plan = FftPlan3D::new(shape);
    let input = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        Complex64::new(
            (i as f64 * 0.2).sin(),
            (j as f64 * 0.3 + k as f64 * 0.1).cos(),
        )
    });
    let mut data = input.clone();
    plan.forward_complex_inplace(&mut data);
    plan.inverse_complex_inplace(&mut data);
    for (a, b) in input.iter().zip(data.iter()) {
        let err = (a - b).norm();
        assert!(err < 1e-10, "complex roundtrip err={err:.2e}");
    }
}

/// inverse_complex_to_real_into matches the allocating inverse_complex_to_real.
#[test]
fn caller_owned_inverse_matches_allocating() {
    let (nx, ny, nz) = (6usize, 5usize, 4usize);
    let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
    let plan = FftPlan3D::new(shape);
    let input = Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        Complex64::new(
            (i as f64 * 0.5 + j as f64 * 0.3).sin(),
            (k as f64 * 0.7).cos(),
        )
    });
    let alloc_result = plan.inverse_complex_to_real(&input);
    let mut out = Array3::<f64>::zeros((nx, ny, nz));
    let mut scratch = Array3::<Complex64>::zeros((nx, ny, nz));
    plan.inverse_complex_to_real_into(&input, &mut out, &mut scratch);
    for (a, b) in alloc_result.iter().zip(out.iter()) {
        let err = (a - b).abs();
        assert!(err < 1e-14, "caller-owned vs alloc mismatch: {err:.2e}");
    }
}

/// forward_real_to_complex_into panics on wrong output shape.
#[test]
#[should_panic(expected = "forward output shape mismatch")]
fn forward_rejects_wrong_shape() {
    let shape = Shape3D::new(4, 4, 4).expect("valid shape");
    let plan = FftPlan3D::new(shape);
    let input = Array3::<f64>::zeros((4, 4, 4));
    let mut wrong_output = Array3::<Complex64>::zeros((4, 4, 3));
    plan.forward_real_to_complex_into(&input, &mut wrong_output);
}

/// LOW_PRECISION_F32 typed roundtrip stays within f32 tolerance.
#[test]
fn typed_low_precision_roundtrip() {
    let (nx, ny, nz) = (8usize, 8usize, 8usize);
    let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
    let plan = FftPlan3D::with_precision(shape, PrecisionProfile::LOW_PRECISION_F32);
    let input = Array3::<f32>::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        (i as f32 * 0.31 + j as f32 * 0.17 + k as f32 * 0.41).sin()
    });
    let spectrum = plan.forward_typed(&input);
    let recovered: Array3<f32> = plan.inverse_typed(&spectrum);
    for (a, b) in input.iter().zip(recovered.iter()) {
        let err = (a - b).abs();
        assert!(err < 1e-4, "low-precision roundtrip err={err:.2e}");
    }
}

/// R2C forward then C2R inverse recovers the original real signal, eps 1e-10.
///
/// Validates the Cooley-Tukey split formula and its inverse for power-of-two sizes.
#[test]
fn r2c_roundtrip_power_of_two() {
    for (nx, ny, nz) in [(4usize, 4usize, 4usize), (8, 8, 8), (16, 4, 8)] {
        let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
        let plan = FftPlan3D::new(shape);
        let input = make_signal(nx, ny, nz);
        let spectrum = plan.forward_r2c(&input);
        assert_eq!(spectrum.dim(), (nx, ny, nz / 2 + 1));
        let recovered = plan.inverse_c2r(&spectrum);
        assert_eq!(recovered.dim(), (nx, ny, nz));
        for (a, b) in input.iter().zip(recovered.iter()) {
            let err = (a - b).abs();
            assert!(err < 1e-10, "r2c roundtrip ({nx},{ny},{nz}) err={err:.2e}");
        }
    }
}

/// R2C for non-power-of-two nz uses the Bluestein fallback.
#[test]
fn r2c_roundtrip_non_power_of_two_nz() {
    // nz = 6 (not power of two), so sub-FFT length m=3 falls back to Bluestein.
    let (nx, ny, nz) = (4usize, 4usize, 6usize);
    let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
    let plan = FftPlan3D::new(shape);
    let input = make_signal(nx, ny, nz);
    let spectrum = plan.forward_r2c(&input);
    let recovered = plan.inverse_c2r(&spectrum);
    for (a, b) in input.iter().zip(recovered.iter()) {
        let err = (a - b).abs();
        assert!(err < 1e-9, "r2c non-pow2 roundtrip err={err:.2e}");
    }
}

/// R2C for odd z lengths uses the full-spectrum fallback.
#[test]
fn r2c_roundtrip_odd_nz() {
    for (nx, ny, nz) in [(4usize, 4usize, 3usize), (5, 7, 5)] {
        let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
        let plan = FftPlan3D::new(shape);
        let input = make_signal(nx, ny, nz);
        let spectrum = plan.forward_r2c(&input);
        assert_eq!(spectrum.dim(), (nx, ny, nz / 2 + 1));
        let recovered = plan.inverse_c2r(&spectrum);
        for (a, b) in input.iter().zip(recovered.iter()) {
            let err = (a - b).abs();
            assert!(err < 1e-10, "r2c odd-nz ({nx},{ny},{nz}) err={err:.2e}");
        }
    }
}

/// R2C half-spectrum matches the first nz_c rows of the full-complex forward FFT.
///
/// Correctness invariant: `forward_r2c(x)[i,j,k] == forward(x)[i,j,k]` for k = 0..nz_c-1.
#[test]
fn r2c_spectrum_matches_full_forward() {
    for (nx, ny, nz) in [(8usize, 6usize, 8usize), (4, 4, 3)] {
        let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
        let plan = FftPlan3D::new(shape);
        let input = make_signal(nx, ny, nz);

        let full_spectrum = plan.forward(&input);
        let half_spectrum = plan.forward_r2c(&input);

        let nz_c = nz / 2 + 1;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz_c {
                    let full = full_spectrum[[i, j, k]];
                    let half = half_spectrum[[i, j, k]];
                    let err = (full - half).norm();
                    assert!(
                        err < 1e-10,
                        "r2c vs full ({nx},{ny},{nz}): [{i},{j},{k}] \
                         full={full} half={half} err={err:.2e}"
                    );
                }
            }
        }
    }
}

/// Parseval identity for r2c: sum|x|² == sum|X_half|² * 2 / (nx*ny*nz), eps 1e-6.
///
/// For real x, the full spectrum satisfies sum|X|² = nx*ny*nz * sum|x|².
/// The half-spectrum has nz_c = nz/2+1 slabs; the interior slabs (k=1..nz/2-1)
/// are duplicated (Hermitian symmetry), so their energy counts double:
///   sum|x|² = [|X[*,*,0]|² + |X[*,*,nz/2]|² + 2*sum_{k=1}^{nz/2-1} |X[*,*,k]|²] / (nx*ny*nz)
#[test]
fn r2c_parseval_holds() {
    let (nx, ny, nz) = (8usize, 6usize, 8usize);
    let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
    let plan = FftPlan3D::new(shape);
    let input = make_signal(nx, ny, nz);

    let half_spectrum = plan.forward_r2c(&input);
    let nz_c = nz / 2 + 1;

    let time_energy: f64 = input.iter().map(|&x| x * x).sum();

    // Compute weighted spectral energy accounting for Hermitian symmetry.
    let mut spectral_energy = 0.0_f64;
    for i in 0..nx {
        for j in 0..ny {
            // k = 0 and k = nz/2 (if nz even): boundary terms, weight 1.
            spectral_energy += half_spectrum[[i, j, 0]].norm_sqr();
            if nz % 2 == 0 {
                spectral_energy += half_spectrum[[i, j, nz_c - 1]].norm_sqr();
            }
            // k = 1..nz_c-2: interior, each mode appears twice (Hermitian pair).
            let k_end = if nz % 2 == 0 { nz_c - 1 } else { nz_c };
            for k in 1..k_end {
                spectral_energy += 2.0 * half_spectrum[[i, j, k]].norm_sqr();
            }
        }
    }
    spectral_energy /= (nx * ny * nz) as f64;

    let err = (time_energy - spectral_energy).abs();
    assert!(
        err < 1e-6,
        "r2c Parseval err={err:.2e}: time_energy={time_energy:.6e} spectral={spectral_energy:.6e}"
    );
}

/// C2R inverse using caller-owned scratch matches the allocating inverse_c2r.
#[test]
fn c2r_caller_owned_matches_allocating() {
    let (nx, ny, nz) = (6usize, 4usize, 8usize);
    let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
    let plan = FftPlan3D::new(shape);
    let input = make_signal(nx, ny, nz);
    let spectrum = plan.forward_r2c(&input);

    let alloc = plan.inverse_c2r(&spectrum);
    let mut out = Array3::<f64>::zeros((nx, ny, nz));
    let mut scratch = Array3::<Complex64>::zeros((nx, ny, nz / 2 + 1));
    plan.inverse_c2r_into(&spectrum, &mut out, &mut scratch);

    for (a, b) in alloc.iter().zip(out.iter()) {
        let err = (a - b).abs();
        assert!(err < 1e-14, "c2r caller-owned mismatch: {err:.2e}");
    }
}

/// R2C/C2R roundtrip for nz=1 (1D grid): verifies that x/y FFT passes
/// still execute when nz=1 and that the z-trivial branch does not
/// short-circuit the full transform.
///
/// Physical motivation: kwavers 1D PSTD simulations use (nx, 1, 1) grids;
/// the bug this test guards against caused `forward_r2c_into` to return
/// after a real->complex cast without performing the x-axis FFT, making
/// every spectral derivative zero and the solver a no-op.
#[test]
fn r2c_roundtrip_nz1_1d_grid() {
    // 1D grid: (nx, 1, 1). The x-axis FFT must run; y/z are trivial.
    for nx in [8usize, 16, 32, 64] {
        let shape = Shape3D::new(nx, 1, 1).expect("valid shape");
        let plan = FftPlan3D::new(shape);
        let input = make_signal(nx, 1, 1);
        let spectrum = plan.forward_r2c(&input);
        // Half-spectrum shape: (nx, 1, nz/2+1) = (nx, 1, 1) for nz=1.
        assert_eq!(spectrum.dim(), (nx, 1, 1), "nz=1 half-spectrum shape");
        let recovered = plan.inverse_c2r(&spectrum);
        assert_eq!(recovered.dim(), (nx, 1, 1));
        for (a, b) in input.iter().zip(recovered.iter()) {
            let err = (a - b).abs();
            assert!(err < 1e-10, "r2c nz=1 1D roundtrip nx={nx} err={err:.2e}");
        }
    }
}

/// R2C/C2R roundtrip for nz=1, ny>1 (2D grid): verifies that both the
/// y-axis and x-axis FFT passes run when nz=1.
#[test]
fn r2c_roundtrip_nz1_2d_grid() {
    for (nx, ny) in [(8usize, 8usize), (16, 12), (6, 10)] {
        let shape = Shape3D::new(nx, ny, 1).expect("valid shape");
        let plan = FftPlan3D::new(shape);
        let input = make_signal(nx, ny, 1);
        let spectrum = plan.forward_r2c(&input);
        assert_eq!(spectrum.dim(), (nx, ny, 1), "nz=1 2D half-spectrum shape");
        let recovered = plan.inverse_c2r(&spectrum);
        for (a, b) in input.iter().zip(recovered.iter()) {
            let err = (a - b).abs();
            assert!(
                err < 1e-10,
                "r2c nz=1 2D roundtrip ({nx},{ny}) err={err:.2e}"
            );
        }
    }
}

/// Spectrum correctness for nz=1: forward_r2c must produce the same
/// DC-only spectrum as the full complex forward FFT for a 1D grid.
///
/// For nz=1, nz_c=1, and `forward_r2c(x)[i,0,0] == forward(x)[i,0,0]`
/// (the single frequency bin is the DC component, which equals the sum
/// of the x-axis DFT coefficients).
#[test]
fn r2c_spectrum_matches_full_forward_nz1() {
    let (nx, ny) = (16usize, 1usize);
    let shape = Shape3D::new(nx, ny, 1).expect("valid shape");
    let plan = FftPlan3D::new(shape);
    let input = make_signal(nx, ny, 1);

    let full_spectrum = plan.forward(&input);
    let half_spectrum = plan.forward_r2c(&input);

    for i in 0..nx {
        let full = full_spectrum[[i, 0, 0]];
        let half = half_spectrum[[i, 0, 0]];
        let err = (full - half).norm();
        assert!(
            err < 1e-10,
            "r2c vs full nz=1: [{i},0,0] full={full} half={half} err={err:.2e}"
        );
    }
}

/// MIXED_PRECISION_F16_F32 typed roundtrip stays within f16 tolerance.
#[test]
fn typed_mixed_precision_roundtrip() {
    let (nx, ny, nz) = (8usize, 8usize, 8usize);
    let shape = Shape3D::new(nx, ny, nz).expect("valid shape");
    let plan = FftPlan3D::with_precision(shape, PrecisionProfile::MIXED_PRECISION_F16_F32);
    let input = Array3::<f16>::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        f16::from_f32((i as f32 * 0.31 + j as f32 * 0.17 + k as f32 * 0.41).sin())
    });
    let spectrum = plan.forward_typed(&input);
    let recovered: Array3<f16> = plan.inverse_typed(&spectrum);
    for (a, b) in input.iter().zip(recovered.iter()) {
        let err = (a.to_f32() - b.to_f32()).abs();
        assert!(err < 5e-2, "mixed-precision roundtrip err={err:.2e}");
    }
}
