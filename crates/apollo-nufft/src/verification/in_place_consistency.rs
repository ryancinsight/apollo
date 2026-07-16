use crate::{NufftPlan1D, UniformDomain1D};
use eunomia::Complex64;

/// Invariant: `NufftPlan1D::type1_into` produces bit-identical output to the
///
/// allocating `NufftPlan1D::type1`.
///
///
///
/// Both methods execute the same spreading → FFT → deconvolution pipeline.
///
/// `type1` allocates internal buffers and calls `type1_into`; the outputs
///
/// must therefore agree to within floating-point round-off (`< 1e-14`).
///
///
///
/// **Buffer sizes for domain.n=8, sigma=2:**
///
/// - `scratch_grid.len() = sigma * domain.n = 16`
///
/// - `output.len() = domain.n = 8`
///
#[test]

fn plan_1d_type1_into_matches_type1_allocating() {
    let domain = UniformDomain1D::new(8, 0.25).expect("domain");

    // sigma=2, kernel_width=6 ⟹ m = 2*8 = 16, n_out = 8

    let plan = NufftPlan1D::new(domain, 2, 6);

    let positions = vec![0.1_f64, 0.5, 1.3];

    let values = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(-0.5, 0.25),
        Complex64::new(0.25, -0.1),
    ];

    // Allocating path.

    let out_alloc = plan.type1(&positions, &values);

    assert_eq!(
        out_alloc.size(),
        domain.n,
        "allocating output length mismatch"
    );

    // In-place path: scratch_grid length = sigma * domain.n = 16.

    let sigma: usize = 2;

    let mut scratch_grid = vec![Complex64::new(0.0, 0.0); sigma * domain.n];

    let mut output = vec![Complex64::new(0.0, 0.0); domain.n];

    plan.type1_into(&positions, &values, &mut scratch_grid, &mut output);

    for (k, (a, b)) in out_alloc.iter().zip(output.iter()).enumerate() {
        let err = (a - b).norm();

        assert!(
            err < 1e-14,
            "k={k}: type1={a:?} vs type1_into={b:?}, err={err:.3e}"
        );
    }
}
