use crate::{nufft_type1_1d, nufft_type2_1d, nufft_type2_1d_fast, NufftPlan1D, UniformDomain1D};
use eunomia::Complex64;
use leto::Array1;

/// Theorem: KB approximation error decreases monotonically with kernel half-width
/// at fixed oversampling, and decreases further with higher oversampling.
///
/// For fixed `σ`, increasing `W` by one divides the Fessler–Sutton error bound
/// (2003, eq. 13) by a factor `> 1`, because the factor
/// `sinh(β·√(1-(π/β)²)) / β` grows super-linearly with `β ∝ W`.
///
/// **Cases verified (sigma, W, tolerance):**
/// - `(2, 4, 1e-4)`: conservative, `β = π·0.75·8 ≈ 18.85`
/// - `(2, 6, 1e-6)`: standard NUFFT setting, `β = π·0.75·12 ≈ 28.27`
/// - `(4, 6, 1e-8)`: higher oversampling, `β = π·0.875·12 ≈ 32.99`
#[test]
fn fast_1d_tracks_exact_at_varying_kernel_widths() {
    let domain = UniformDomain1D::new(16, 0.125).expect("domain");
    let positions = vec![0.1_f64, 0.5, 0.9, 1.4, 1.7];
    let values = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(-0.5, 0.5),
        Complex64::new(0.75, -0.25),
        Complex64::new(-0.25, 0.75),
        Complex64::new(0.5, -0.5),
    ];
    let exact = nufft_type1_1d(&positions, &values, domain);

    // (sigma, kernel_width, max_absolute_error_tolerance)
    let cases = [(2_usize, 4_usize, 1e-4_f64), (2, 6, 1e-6), (4, 6, 1e-8)];
    for &(sigma, w, tol) in cases.iter() {
        let fast = NufftPlan1D::new(domain, sigma, w).type1(&positions, &values);
        assert_eq!(
            fast.size(),
            exact.size(),
            "sigma={sigma}, W={w}: output length mismatch"
        );
        let max_err = exact
            .iter()
            .zip(fast.iter())
            .map(|(e, f)| (e - f).norm())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err <= tol,
            "sigma={sigma}, W={w}: max_err={max_err:.3e} > tol={tol:.0e}"
        );
    }
}

/// Theorem: Type-2 fast path preserves the inverse FFT normalization.
///
/// Apollo's `FftPlan1D::inverse_complex_slice_inplace` applies `1/M` for
/// an oversampled grid of length `M`. The type-2 NUFFT adjoint requires the
/// unnormalized inverse exponential sum before interpolation, so the fast
/// path must restore the factor `M` after the inverse FFT.
#[test]
fn fast_type2_1d_tracks_exact_after_inverse_fft_rescaling() {
    let domain = UniformDomain1D::new(32, 0.05).expect("domain");
    let positions: Vec<f64> = (0..20)
        .map(|i| (i as f64 * 0.137).rem_euclid(domain.length()))
        .collect();
    let coefficients = Array1::from_shape_fn([domain.n], |[k]| {
        Complex64::new((0.4 * k as f64).cos(), -(0.25 * k as f64).sin())
    });

    let exact = nufft_type2_1d(&coefficients, &positions, domain);
    let fast = nufft_type2_1d_fast(&coefficients, &positions, domain, 6);
    let max_relative_error = exact
        .iter()
        .zip(fast.iter())
        .map(|(lhs, rhs)| (lhs - rhs).norm() / lhs.norm().max(1.0))
        .fold(0.0_f64, f64::max);

    assert!(
        max_relative_error <= 1.0e-5,
        "type-2 fast relative error {max_relative_error:.3e} exceeded tolerance"
    );
}

/// Theorem: KB NUFFT approximation error decreases strictly with kernel
/// half-width `W` for fixed oversampling `σ = 2`.
///
/// The Fessler–Sutton error bound (2003, eq. 13) is an exponentially
/// decreasing function of `W` for the optimal `β = π·(1 - 1/(2σ))·2W`.
/// This test validates the strict ordering
///
/// ```text
/// err(W=2) > err(W=4) > err(W=6)
/// ```
///
/// for a bandlimited signal with 8 unit-magnitude sources at uniform positions
/// in `[0, L=8.0)`, and confirms `err(W=6) < 1e-5` (practical accuracy bound).
///
/// **Source amplitudes:** `exp(i·k·π/4)` for `k = 0, …, 7` (unit magnitude,
/// uniformly spaced phases — a standard benchmark for NUFFT error studies).
#[test]
fn nufft_approximation_error_decreases_with_kernel_width() {
    let domain = UniformDomain1D::new(32, 0.25).expect("domain");
    // 8 uniformly spaced positions in [0, domain.length() = 8.0)
    let positions: Vec<f64> = (0..8).map(|i| i as f64).collect();
    // Unit-magnitude values at angles k·π/4 for k = 0..8
    let values: Vec<Complex64> = (0..8)
        .map(|k| {
            let angle = k as f64 * std::f64::consts::PI / 4.0;
            Complex64::new(angle.cos(), angle.sin())
        })
        .collect();

    let exact = nufft_type1_1d(&positions, &values, domain);

    let max_abs_err = |w: usize| -> f64 {
        let fast = NufftPlan1D::new(domain, 2, w).type1(&positions, &values);
        exact
            .iter()
            .zip(fast.iter())
            .map(|(e, f)| (e - f).norm())
            .fold(0.0_f64, f64::max)
    };

    let err_2 = max_abs_err(2);
    let err_4 = max_abs_err(4);
    let err_6 = max_abs_err(6);

    assert!(
        err_2 > err_4,
        "error must decrease: err(W=2)={err_2:.3e} should exceed err(W=4)={err_4:.3e}"
    );
    assert!(
        err_4 > err_6,
        "error must decrease: err(W=4)={err_4:.3e} should exceed err(W=6)={err_6:.3e}"
    );
    assert!(
        err_6 < 1e-5,
        "err(W=6)={err_6:.3e} must be below practical accuracy threshold 1e-5"
    );
}
