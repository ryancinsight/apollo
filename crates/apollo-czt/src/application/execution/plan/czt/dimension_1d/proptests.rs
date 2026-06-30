//! Proptest suite for 1D Chirp Z-Transform.

use super::plan::CztPlan;
use leto::Array1;
use eunomia::Complex64;
use proptest::prelude::*;

proptest! {
    /// Bluestein (1970) convolution equivalence: for any n, m, a, w the
    /// fast path (FFT-based Bluestein convolution) and the direct O(nm)
    /// evaluation must agree to within a relative tolerance of 1e-9.
    #[test]
    fn bluestein_equals_direct_for_arbitrary_parameters(
        n in 1_usize..=8,
        m in 1_usize..=8,
        a_mag in 0.5_f64..1.5,
        a_arg in -1.0_f64..1.0,
        w_mag in 0.7_f64..1.3,
        w_arg in 0.1_f64..1.5,
        re_parts in prop::collection::vec(-2_i32..2, 8),
        im_parts in prop::collection::vec(-2_i32..2, 8),
    ) {
        let a = Complex64::from_polar(a_mag, a_arg);
        let w = Complex64::from_polar(w_mag, w_arg);
        let input = Array1::from_shape_fn([n], |[i]| {
            Complex64::new(re_parts[i % 8] as f64, im_parts[i % 8] as f64)
        });
        let plan = CztPlan::new(n, m, a, w).expect("plan");
        let fast = plan.forward(&input).expect("fast");
        let direct = plan.forward_direct(&input).expect("direct");
        prop_assert_eq!(fast.size(), direct.size());
        for k in 0..m {
            let diff = (fast[k] - direct[k]).norm();
            let scale = direct[k].norm().max(1.0_f64);
            prop_assert!(
                diff < 1e-9 * scale,
                "k={k}: |fast - direct| / |direct| = {} >= 1e-9 (a={a}, w={w}, diff={diff}, scale={scale})",
                diff / scale,
            );
        }
    }

    /// Spiral-collapse theorem: substituting A=1, W=exp(-2πi/N), M=N
    /// reduces the CZT sum exactly to the DFT sum.  The fast path must
    /// therefore agree with `apollo_fft::fft_1d_complex` to within 1e-9.
    #[test]
    fn czt_spiral_collapse_arbitrary_n(
        n in 2_usize..=8,
        re_parts in prop::collection::vec(-2_i32..2, 8),
        im_parts in prop::collection::vec(-2_i32..2, 8),
    ) {
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64);
        let input = Array1::from_shape_fn([n], |[i]| {
            Complex64::new(re_parts[i % 8] as f64, im_parts[i % 8] as f64)
        });
        let plan = CztPlan::new(n, n, a, w).expect("plan");
        let czt_out = plan.forward(&input).expect("czt");
        let fft_out = apollo_fft::fft_1d_complex(&input);
        prop_assert_eq!(czt_out.size(), fft_out.size());
        for k in 0..n {
            let diff = (czt_out[k] - fft_out[k]).norm();
            prop_assert!(
                diff < 1e-9,
                "k={k}: |czt - fft| = {diff} >= 1e-9 (n={n})"
            );
        }
    }

    /// Linearity of the CZT sum: for any scalar c,
    /// CZT(c·x)[k] = c · CZT(x)[k].
    #[test]
    fn czt_scalar_linearity(
        n in 1_usize..=7,
        m in 1_usize..=7,
        scalar_re in -2_i32..2,
        scalar_im in -2_i32..2,
        re_parts in prop::collection::vec(-2_i32..2, 8),
        im_parts in prop::collection::vec(-2_i32..2, 8),
    ) {
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::from_polar(
            1.0,
            -std::f64::consts::TAU / (n.max(m) as f64 + 1.0),
        );
        let scalar = Complex64::new(scalar_re as f64, scalar_im as f64);
        let input = Array1::from_shape_fn([n], |[i]| {
            Complex64::new(re_parts[i % 8] as f64, im_parts[i % 8] as f64)
        });
        let scaled_input = input.mapv(|v| scalar * v);
        let plan = CztPlan::new(n, m, a, w).expect("plan");
        let czt_scaled = plan.forward(&scaled_input).expect("czt scaled");
        let czt_base = plan.forward(&input).expect("czt base");
        prop_assert_eq!(czt_scaled.size(), czt_base.size());
        for k in 0..m {
            let expected = scalar * czt_base[k];
            let diff = (czt_scaled[k] - expected).norm();
            prop_assert!(
                diff < 1e-9,
                "k={k}: |CZT(c·x) - c·CZT(x)| = {diff} >= 1e-9 (scalar={scalar})"
            );
        }
    }
}
