use crate::{nufft_type1_1d, nufft_type1_1d_fast, nufft_type1_3d, UniformDomain1D, UniformGrid3D};
use eunomia::Complex64;
use leto::Array1;

/// Theorem: Type-1 DC mode identity.
///
/// For any positions `x_j` and complex amplitudes `c_j`, the `k = 0`
/// Fourier coefficient of the type-1 NUFFT satisfies
///
/// ```text
/// F[0] = Σ_j c_j · exp(-2πi · 0 · x_j / L) = Σ_j c_j
/// ```
///
/// because `exp(0) = 1` for every `j` regardless of position. This identity
/// holds exactly for the direct transform and approximately for the fast path,
/// with the approximation error bounded by the KB spreading error.
///
/// **Verification data:**
/// - positions: `[0.1, 0.5, 1.3, 1.8]` (non-uniform, deliberately off-grid)
/// - values: `[(1.0,0), (0.5,-0.3), (-0.25,0.8), (0.1,0.1)]`
/// - `dc_exact = (1.0+0.5-0.25+0.1, 0-0.3+0.8+0.1) = (1.35, 0.6)`
#[test]
fn type1_dc_mode_is_sum_of_all_values_1d() {
    let domain = UniformDomain1D::new(8, 0.25).expect("domain");
    let positions = vec![0.1_f64, 0.5, 1.3, 1.8];
    let values = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.5, -0.3),
        Complex64::new(-0.25, 0.8),
        Complex64::new(0.1, 0.1),
    ];

    // Analytical DC = sum(c_j) = (1.35, 0.6)
    let dc_exact = Complex64::new(1.35, 0.6);

    let f_exact = nufft_type1_1d(&positions, &values, domain);
    let err_exact = (f_exact[[0]] - dc_exact).norm();
    assert!(
        err_exact < 1e-10,
        "exact DC mode error {err_exact}: got {:?}, expected {:?}",
        f_exact[[0]],
        dc_exact
    );
    // Verify the full output is finite.
    for (k, v) in f_exact.iter().enumerate() {
        assert!(
            v.norm().is_finite(),
            "exact output mode {k} is non-finite: {v:?}"
        );
    }

    // Fast path: kernel_width=6, oversampling=DEFAULT(2).
    let f_fast = nufft_type1_1d_fast(&positions, &values, domain, 6);
    let err_fast = (f_fast[[0]] - dc_exact).norm();
    assert!(
        err_fast < 1e-4,
        "fast DC mode error {err_fast}: got {:?}, expected {:?}",
        f_fast[[0]],
        dc_exact
    );
    for (k, v) in f_fast.iter().enumerate() {
        assert!(
            v.norm().is_finite(),
            "fast output mode {k} is non-finite: {v:?}"
        );
    }
}

/// Theorem: 3D type-1 DC mode identity.
///
/// The DC mode `(k_x=0, k_y=0, k_z=0)` of the 3D type-1 NUFFT satisfies
///
/// ```text
/// F[0,0,0] = Σ_j c_j
/// ```
///
/// because `exp(-2πi·(0·x_j/Lx + 0·y_j/Ly + 0·z_j/Lz)) = 1` for all `j`.
/// This is the separable extension of the 1D DC identity to the tensor-product
/// frequency lattice.
///
/// **Verification data:**
/// - values: `[(1.0,0), (0.5,0.5), (-0.75,0.25)]`
/// - `dc_exact = (1.0+0.5-0.75, 0+0.5+0.25) = (0.75, 0.75)`
#[test]
fn type1_3d_dc_mode_is_sum_of_values() {
    let grid = UniformGrid3D::new(2, 2, 2, 1.0, 1.0, 1.0).expect("grid");
    let positions = vec![(0.1_f64, 0.2, 0.3), (0.5, 0.6, 0.7), (0.8, 0.1, 0.5)];
    let values = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.5, 0.5),
        Complex64::new(-0.75, 0.25),
    ];

    // Analytical: dc = (1.0+0.5-0.75, 0.0+0.5+0.25) = (0.75, 0.75)
    let dc_exact = Complex64::new(0.75, 0.75);

    let f = nufft_type1_3d(&positions, &values, grid);
    assert_eq!(f.shape(), [2, 2, 2], "output shape mismatch");

    let err = (f[[0, 0, 0]] - dc_exact).norm();
    assert!(
        err < 1e-10,
        "3D DC mode error {err:.3e}: got {:?}, expected {:?}",
        f[[0, 0, 0]],
        dc_exact
    );

    for v in f.iter() {
        assert!(v.norm().is_finite(), "non-finite 3D output: {v:?}");
    }
}

/// Theorem: NUFFT type-1 on a uniform grid equals the standard DFT.
///
/// For positions `x_j = j · L/N = j · dx`, the type-1 NUFFT sum becomes:
///
/// ```text
/// f_k = Σ_j c_j · exp(-2πi · k_signed(k) · x_j / L)
///      = Σ_j c_j · exp(-2πi · k_signed(k) · j / N)
///      = Σ_j c_j · exp(-2πi · k · j / N)
/// ```
///
/// The last equality holds because `exp(2πi·N·j/N) = exp(2πij) = 1` for all
/// integer j, so `k_signed(k)·j/N` and `k·j/N` differ by exactly an integer.
///
/// This provides an independent published-reference cross-check: the NUFFT output
/// at uniform positions must match the `apollo-fft` Cooley-Tukey / Bluestein output
/// to within f64 round-off (< 1e-10).
#[test]
fn type1_on_uniform_grid_matches_standard_dft() {
    use apollo_fft::fft_1d_complex;
    let n = 8usize;
    let domain = UniformDomain1D::new(n, 0.125).expect("domain"); // L = n * dx = 1.0
                                                                  // Uniform grid positions: x_j = j * dx = j * L/N
    let positions: Vec<f64> = (0..n).map(|j| j as f64 * domain.dx).collect();
    let values: Vec<Complex64> = (0..n)
        .map(|j| Complex64::new((j as f64 * 0.3).sin(), (j as f64 * 0.17).cos()))
        .collect();
    // NUFFT type-1 at uniform positions (exact direct O(N²) path)
    let nufft_output = nufft_type1_1d(&positions, &values, domain);
    // Independent DFT via apollo_fft (separate Cooley-Tukey radix-2 kernel for N=8)
    let values_complex = Array1::from_shape_vec([values.len()], values)
        .expect("invariant: uniform-grid fixture length matches Array1 shape");
    let fft_complex = fft_1d_complex(&values_complex);
    for (k, (nv, fv)) in nufft_output.iter().zip(fft_complex.iter()).enumerate() {
        let err = (nv - fv).norm();
        assert!(
            err < 1e-10,
            "NUFFT type-1 on uniform grid differs from DFT at k={k}: \
             nufft={nv:?}, fft={fv:?}, err={err:.3e}"
        );
    }
}
