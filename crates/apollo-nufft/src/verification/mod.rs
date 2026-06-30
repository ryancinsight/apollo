//! Verification helpers for NUFFT.
//!
//! This module contains standalone verification tests that prove fundamental
//! mathematical invariants of the NUFFT implementation. Each test is derived
//! from a formal theorem and asserts on computed values, not merely on
//! `Result`/`Option` variants.
//!
//! ## Invariant hierarchy
//!
//! 1. **DC identity**: `F[0] = Σ_j c_j` (exact arithmetic consequence of `exp(0)=1`)
//! 2. **Adjoint identity**: `⟨A·c, f⟩ = ⟨c, A*·f⟩` (inner-product duality)
//! 3. **Kernel-width error ordering**: `err(W=2) > err(W=4) > err(W=6)` (Fessler–Sutton bound)
//! 4. **3D DC identity**: `F[0,0,0] = Σ_j c_j` (separable extension of invariant 1)
//! 5. **`type1_into` consistency**: allocating and in-place paths agree to machine precision
//! 6. **Monotone error decrease**: KB approximation error strictly decreases with `W`

#[cfg(test)]
mod tests {
    use crate::{
        nufft_type1_1d, nufft_type1_1d_fast, nufft_type1_3d, nufft_type2_1d, nufft_type2_1d_fast,
        NufftComplexStorage, NufftPlan1D, NufftPlan3D, UniformDomain1D, UniformGrid3D,
        DEFAULT_NUFFT_KERNEL_WIDTH, DEFAULT_NUFFT_OVERSAMPLING,
    };
    use apollo_fft::{f16, ApolloError, Complex32, PrecisionProfile};
    use leto::{Array1, Array3};
    use eunomia::Complex64;

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
        let err_exact = (f_exact[0] - dc_exact).norm();
        assert!(
            err_exact < 1e-10,
            "exact DC mode error {err_exact}: got {:?}, expected {:?}",
            f_exact[0],
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
        let err_fast = (f_fast[0] - dc_exact).norm();
        assert!(
            err_fast < 1e-4,
            "fast DC mode error {err_fast}: got {:?}, expected {:?}",
            f_fast[0],
            dc_exact
        );
        for (k, v) in f_fast.iter().enumerate() {
            assert!(
                v.norm().is_finite(),
                "fast output mode {k} is non-finite: {v:?}"
            );
        }
    }

    /// Theorem: Type-1 / Type-2 adjoint identity.
    ///
    /// Define the type-1 operator `A` and type-2 operator `A*` by
    ///
    /// ```text
    /// (A·c)_k  = Σ_j  c_j  exp(-2πi k x_j / L)   (type-1)
    /// (A*·f)_j = Σ_k  f_k  exp(+2πi k x_j / L)   (type-2)
    /// ```
    ///
    /// They satisfy the real inner-product adjoint identity:
    ///
    /// ```text
    /// Re(⟨A·c, f⟩) = Re(⟨c, A*·f⟩)
    /// ```
    ///
    /// **Proof:** Expand `(A·c)_k`, swap summation order, and factor:
    ///
    /// ```text
    /// Re(Σ_k conj((A·c)_k) · f_k)
    ///   = Re(Σ_k Σ_j conj(c_j) exp(+2πi k x_j/L) · f_k)
    ///   = Re(Σ_j conj(c_j) · Σ_k f_k exp(+2πi k x_j/L))
    ///   = Re(Σ_j conj(c_j) · (A*·f)_j)  □
    /// ```
    ///
    /// The residual `|LHS - RHS|` must be below round-off for the exact
    /// direct transform (asserted at `< 1e-10`).
    #[test]
    fn type1_and_type2_adjoint_relationship_1d() {
        let domain = UniformDomain1D::new(4, 0.5).expect("domain");
        // 3 non-uniform positions in [0, L=2.0)
        let positions = vec![0.1_f64, 0.5, 1.3];
        let values_c = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(-0.5, 0.25),
            Complex64::new(0.25, -0.1),
        ];

        // F = A·c, length = domain.n = 4
        let f = nufft_type1_1d(&positions, &values_c, domain);

        // Arbitrary frequency-domain vector of length 4.
        let freq_domain_data = [
            Complex64::new(2.0, 0.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(0.5, 0.3),
            Complex64::new(-0.25, 0.1),
        ];
        let freq_domain_array = Array1::from((freq_domain_data.iter().copied()).collect::<Vec<_>>());

        // G = A*·f, length = positions.len() = 3
        let g = nufft_type2_1d(&freq_domain_array, &positions, domain);

        // LHS = Re(⟨F, freq_domain⟩) = Re(Σ_k conj(F[k]) · freq_domain[k])
        let lhs: f64 = f
            .iter()
            .zip(freq_domain_data.iter())
            .map(|(f_k, fd_k)| (f_k.conj() * fd_k).re)
            .sum();

        // RHS = Re(⟨c, G⟩) = Re(Σ_j conj(c_j) · G[j])
        let rhs: f64 = values_c
            .iter()
            .zip(g.iter())
            .map(|(c_j, g_j)| (c_j.conj() * g_j).re)
            .sum();

        let residual = (lhs - rhs).abs();
        assert!(
            residual < 1e-10,
            "adjoint identity failed: LHS={lhs:.15e}, RHS={rhs:.15e}, residual={residual:.3e}"
        );
    }

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

    /// Invariant: `NufftPlan1D::type1_into` produces bit-identical output to the
    /// allocating `NufftPlan1D::type1`.
    ///
    /// Both methods execute the same spreading → FFT → deconvolution pipeline.
    /// `type1` allocates internal buffers and calls `type1_into`; the outputs
    /// must therefore agree to within floating-point round-off (`< 1e-14`).
    ///
    /// **Buffer sizes for domain.n=8, sigma=2:**
    /// - `scratch_grid.len() = sigma * domain.n = 16`
    /// - `output.len() = domain.n = 8`
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
        let values_complex = Array1::from(values);
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

    /// Invariant: `NufftPlan1D::type1_typed_into` produces value-identical output
    /// for `Complex64`, within-f32-precision for `Complex32`, and within-f16-
    /// quantization for `[f16; 2]`, and rejects profile mismatches.
    ///
    /// **Verification data** (DC-mode pattern):
    /// - positions: `[0.1, 0.5, 1.3, 1.8]` (non-uniform, off-grid)
    /// - values: `[(1,0), (0.5,-0.3), (-0.25,0.8), (0.1,0.1)]`
    /// - f64 path matches allocating `type1` to machine precision
    /// - f32 path matches f64 reference to within 1e-5 relative tolerance
    /// - f16 path matches f64 reference to within `|v|·2⁻¹⁰ + 2⁻¹⁴`
    /// - f32 storage with f64 profile returns `ApolloError::Validation { field: "precision_profile" }`
    #[test]
    fn typed_type1_1d_supports_complex64_complex32_and_f16_storage() {
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        let sigma = DEFAULT_NUFFT_OVERSAMPLING;
        let kernel_width = DEFAULT_NUFFT_KERNEL_WIDTH;
        let plan = NufftPlan1D::new(domain, sigma, kernel_width);
        // Buffer sizes derived from plan construction: m = sigma * n, n_out = n
        let m = sigma * domain.n;
        let n_out = domain.n;
        let positions = vec![0.1_f64, 0.5, 1.3, 1.8];
        let values64 = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, -0.3),
            Complex64::new(-0.25, 0.8),
            Complex64::new(0.1, 0.1),
        ];

        // Allocating reference (Complex64 owner path)
        let expected64 = plan.type1(&positions, &values64);

        // ── f64 path ──────────────────────────────────────────────────────
        let mut scratch = vec![Complex64::new(0.0, 0.0); m];
        let mut output64 = vec![Complex64::new(0.0, 0.0); n_out];
        plan.type1_typed_into(
            &positions,
            &values64,
            &mut scratch,
            &mut output64,
            PrecisionProfile::HIGH_ACCURACY_F64,
        )
        .expect("typed complex64 type1");
        for (actual, expected) in output64.iter().zip(expected64.iter()) {
            approx::assert_abs_diff_eq!(actual.re, expected.re);
            approx::assert_abs_diff_eq!(actual.im, expected.im);
        }

        // ── f32 path ──────────────────────────────────────────────────────
        let values32: Vec<Complex32> = values64
            .iter()
            .map(|v| Complex32::new(v.re as f32, v.im as f32))
            .collect();
        let represented32: Vec<Complex64> = values32
            .iter()
            .copied()
            .map(Complex32::to_complex64)
            .collect();
        let expected32 = plan.type1(&positions, &represented32);
        let mut output32 = vec![Complex32::new(0.0, 0.0); n_out];
        plan.type1_typed_into(
            &positions,
            &values32,
            &mut scratch,
            &mut output32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed complex32 type1");
        let mut max_rel_err_f32 = 0.0_f64;
        for (actual, expected) in output32.iter().zip(expected32.iter()) {
            let denom_re = expected.re.abs().max(1e-30);
            let denom_im = expected.im.abs().max(1e-30);
            max_rel_err_f32 = max_rel_err_f32
                .max((f64::from(actual.re) - expected.re).abs() / denom_re)
                .max((f64::from(actual.im) - expected.im).abs() / denom_im);
        }
        assert!(
            max_rel_err_f32 < 1e-5,
            "f32 type1 max relative error {max_rel_err_f32:.3e} exceeds 1e-5"
        );

        // ── f16 path ──────────────────────────────────────────────────────
        let values16: Vec<[f16; 2]> = values64
            .iter()
            .map(|v| [f16::from_f32(v.re as f32), f16::from_f32(v.im as f32)])
            .collect();
        let represented16: Vec<Complex64> = values16
            .iter()
            .copied()
            .map(<[f16; 2]>::to_complex64)
            .collect();
        let expected16 = plan.type1(&positions, &represented16);
        let mut output16 = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; n_out];
        plan.type1_typed_into(
            &positions,
            &values16,
            &mut scratch,
            &mut output16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed f16 type1");
        for (actual, expected) in output16.iter().zip(expected16.iter()) {
            let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!(
                (f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound,
                "f16 re: actual={}, expected={}, bound={re_bound:.3e}",
                f64::from(actual[0].to_f32()),
                expected.re
            );
            assert!(
                (f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound,
                "f16 im: actual={}, expected={}, bound={im_bound:.3e}",
                f64::from(actual[1].to_f32()),
                expected.im
            );
        }

        // ── profile mismatch ──────────────────────────────────────────────
        let mut mismatch_output = vec![Complex32::new(0.0, 0.0); n_out];
        let err = plan
            .type1_typed_into(
                &positions,
                &values32,
                &mut scratch,
                &mut mismatch_output,
                PrecisionProfile::HIGH_ACCURACY_F64,
            )
            .expect_err("profile mismatch must fail");
        assert!(
            matches!(err, ApolloError::Validation { ref field, .. } if field == "precision_profile"),
            "expected Validation {{ field: \"precision_profile\" }}, got {err:?}"
        );
    }

    /// Invariant: `NufftPlan1D::type2_typed_into` produces value-identical output
    /// for `Complex64`, within-f32-precision for `Complex32`, and within-f16-
    /// quantization for `[f16; 2]`, and rejects profile mismatches.
    ///
    /// **Verification strategy:**
    /// - Use the f64 type1 output as type2 input (round-trip guarantee)
    /// - f64 path matches allocating `type2` to machine precision
    /// - f32 path matches f64 reference to within 1e-5 relative tolerance
    /// - f16 path matches f64 reference to within `|v|·2⁻¹⁰ + 2⁻¹⁴`
    /// - f32 storage with f64 profile returns `ApolloError::Validation { field: "precision_profile" }`
    #[test]
    fn typed_type2_1d_supports_complex64_complex32_and_f16_storage() {
        let domain = UniformDomain1D::new(8, 0.25).expect("domain");
        let sigma = DEFAULT_NUFFT_OVERSAMPLING;
        let kernel_width = DEFAULT_NUFFT_KERNEL_WIDTH;
        let plan = NufftPlan1D::new(domain, sigma, kernel_width);
        let m = sigma * domain.n;
        let positions = vec![0.1_f64, 0.5, 1.3, 1.8];

        // Build type2 input from a type1 forward pass
        let values64 = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, -0.3),
            Complex64::new(-0.25, 0.8),
            Complex64::new(0.1, 0.1),
        ];
        let coeffs64_array = plan.type1(&positions, &values64);

        // Allocating reference (Complex64 owner path)
        let expected64 = plan.type2(&coeffs64_array, &positions);

        // ── f64 path ──────────────────────────────────────────────────────
        let mut scratch = vec![Complex64::new(0.0, 0.0); m];
        let mut output64 = vec![Complex64::new(0.0, 0.0); positions.len()];
        plan.type2_typed_into(
            coeffs64_array.as_slice().unwrap(),
            &positions,
            &mut scratch,
            &mut output64,
            PrecisionProfile::HIGH_ACCURACY_F64,
        )
        .expect("typed complex64 type2");
        for (actual, expected) in output64.iter().zip(expected64.iter()) {
            approx::assert_abs_diff_eq!(actual.re, expected.re);
            approx::assert_abs_diff_eq!(actual.im, expected.im);
        }

        // ── f32 path ──────────────────────────────────────────────────────
        let coeffs32: Vec<Complex32> = coeffs64_array
            .iter()
            .map(|v| Complex32::new(v.re as f32, v.im as f32))
            .collect();
        let represented32: Vec<Complex64> = coeffs32
            .iter()
            .copied()
            .map(Complex32::to_complex64)
            .collect();
        let expected32 = plan.type2(&Array1::from(represented32), &positions);
        let mut output32 = vec![Complex32::new(0.0, 0.0); positions.len()];
        plan.type2_typed_into(
            &coeffs32,
            &positions,
            &mut scratch,
            &mut output32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed complex32 type2");
        let mut max_rel_err_f32 = 0.0_f64;
        for (actual, expected) in output32.iter().zip(expected32.iter()) {
            let denom_re = expected.re.abs().max(1e-30);
            let denom_im = expected.im.abs().max(1e-30);
            max_rel_err_f32 = max_rel_err_f32
                .max((f64::from(actual.re) - expected.re).abs() / denom_re)
                .max((f64::from(actual.im) - expected.im).abs() / denom_im);
        }
        assert!(
            max_rel_err_f32 < 1e-5,
            "f32 type2 max relative error {max_rel_err_f32:.3e} exceeds 1e-5"
        );

        // ── f16 path ──────────────────────────────────────────────────────
        let coeffs16: Vec<[f16; 2]> = coeffs64_array
            .iter()
            .map(|v| [f16::from_f32(v.re as f32), f16::from_f32(v.im as f32)])
            .collect();
        let represented16: Vec<Complex64> = coeffs16
            .iter()
            .copied()
            .map(<[f16; 2]>::to_complex64)
            .collect();
        let expected16 = plan.type2(&Array1::from(represented16), &positions);
        let mut output16 = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; positions.len()];
        plan.type2_typed_into(
            &coeffs16,
            &positions,
            &mut scratch,
            &mut output16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed f16 type2");
        for (actual, expected) in output16.iter().zip(expected16.iter()) {
            let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!(
                (f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound,
                "f16 re: actual={}, expected={}, bound={re_bound:.3e}",
                f64::from(actual[0].to_f32()),
                expected.re
            );
            assert!(
                (f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound,
                "f16 im: actual={}, expected={}, bound={im_bound:.3e}",
                f64::from(actual[1].to_f32()),
                expected.im
            );
        }

        // ── profile mismatch ──────────────────────────────────────────────
        let mut mismatch_output = vec![Complex32::new(0.0, 0.0); positions.len()];
        let err = plan
            .type2_typed_into(
                &coeffs32,
                &positions,
                &mut scratch,
                &mut mismatch_output,
                PrecisionProfile::HIGH_ACCURACY_F64,
            )
            .expect_err("profile mismatch must fail");
        assert!(
            matches!(err, ApolloError::Validation { ref field, .. } if field == "precision_profile"),
            "expected Validation {{ field: \"precision_profile\" }}, got {err:?}"
        );
    }

    /// Invariant: `NufftPlan3D::type1_typed_into` produces value-identical output
    /// for `Complex64`, within-f32-precision for `Complex32`, and within-f16-
    /// quantization for `[f16; 2]`, and rejects profile mismatches.
    ///
    /// **Verification data** (3D DC-mode pattern):
    /// - positions: `[(0.1,0.2,0.3), (0.5,0.6,0.7), (0.8,0.1,0.5)]`
    /// - values: `[(1,0), (0.5,0.5), (-0.75,0.25)]`
    /// - f64 path matches allocating `type1` to machine precision
    /// - f32 path matches f64 reference to within 1e-5 relative tolerance
    /// - f16 path matches f64 reference to within `|v|·2⁻¹⁰ + 2⁻¹⁴`
    /// - f32 storage with f64 profile returns `ApolloError::Validation { field: "precision_profile" }`
    #[test]
    fn typed_type1_3d_supports_complex64_complex32_and_f16_storage() {
        let grid = UniformGrid3D::new(2, 2, 2, 1.0, 1.0, 1.0).expect("grid");
        let sigma = DEFAULT_NUFFT_OVERSAMPLING;
        let kernel_width = DEFAULT_NUFFT_KERNEL_WIDTH;
        let plan = NufftPlan3D::new(grid, sigma, kernel_width);
        // Buffer sizes: for n=2, sigma=2, kernel_width=6:
        //   oversampled = max(n*sigma, 2*kernel_width+1).next_power_of_two()
        //   = max(4, 13).next_power_of_two() = 16
        let mx = (grid.nx * sigma)
            .max(2 * kernel_width + 1)
            .next_power_of_two();
        let my = (grid.ny * sigma)
            .max(2 * kernel_width + 1)
            .next_power_of_two();
        let mz = (grid.nz * sigma)
            .max(2 * kernel_width + 1)
            .next_power_of_two();
        let w = kernel_width;
        let positions = vec![(0.1_f64, 0.2, 0.3), (0.5, 0.6, 0.7), (0.8, 0.1, 0.5)];
        let values64 = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.5),
            Complex64::new(-0.75, 0.25),
        ];

        // Allocating reference (Complex64 owner path)
        let expected64 = plan.type1(&positions, &values64);

        // ── f64 path ──────────────────────────────────────────────────────
        let mut scratch_grid = Array3::<Complex64>::zeros([mx, my, mz]);
        let mut wx = vec![0.0_f64; 2 * w + 1];
        let mut wy = vec![0.0_f64; 2 * w + 1];
        let mut wz = vec![0.0_f64; 2 * w + 1];
        let mut output64 = Array3::<Complex64>::zeros([grid.nx, grid.ny, grid.nz]);
        plan.type1_typed_into(
            &positions,
            &values64,
            &mut scratch_grid,
            &mut wx,
            &mut wy,
            &mut wz,
            &mut output64,
            PrecisionProfile::HIGH_ACCURACY_F64,
        )
        .expect("typed complex64 3d type1");
        for (actual, expected) in output64.iter().zip(expected64.iter()) {
            approx::assert_abs_diff_eq!(actual.re, expected.re);
            approx::assert_abs_diff_eq!(actual.im, expected.im);
        }

        // ── f32 path ──────────────────────────────────────────────────────
        let values32: Vec<Complex32> = values64
            .iter()
            .map(|v| Complex32::new(v.re as f32, v.im as f32))
            .collect();
        let represented32: Vec<Complex64> = values32
            .iter()
            .copied()
            .map(Complex32::to_complex64)
            .collect();
        let expected32 = plan.type1(&positions, &represented32);
        let mut output32 = Array3::<Complex32>::zeros([grid.nx, grid.ny, grid.nz]);
        plan.type1_typed_into(
            &positions,
            &values32,
            &mut scratch_grid,
            &mut wx,
            &mut wy,
            &mut wz,
            &mut output32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed complex32 3d type1");
        let mut max_rel_err_f32 = 0.0_f64;
        for (actual, expected) in output32.iter().zip(expected32.iter()) {
            let denom_re = expected.re.abs().max(1e-30);
            let denom_im = expected.im.abs().max(1e-30);
            max_rel_err_f32 = max_rel_err_f32
                .max((f64::from(actual.re) - expected.re).abs() / denom_re)
                .max((f64::from(actual.im) - expected.im).abs() / denom_im);
        }
        assert!(
            max_rel_err_f32 < 1e-5,
            "f32 3d type1 max relative error {max_rel_err_f32:.3e} exceeds 1e-5"
        );

        // ── f16 path ──────────────────────────────────────────────────────
        let values16: Vec<[f16; 2]> = values64
            .iter()
            .map(|v| [f16::from_f32(v.re as f32), f16::from_f32(v.im as f32)])
            .collect();
        let represented16: Vec<Complex64> = values16
            .iter()
            .copied()
            .map(<[f16; 2]>::to_complex64)
            .collect();
        let expected16 = plan.type1(&positions, &represented16);
        let mut output16 = Array3::from_shape_fn([grid.nx, grid.ny, grid.nz], |_| {
            [f16::from_f32(0.0), f16::from_f32(0.0)]
        });
        plan.type1_typed_into(
            &positions,
            &values16,
            &mut scratch_grid,
            &mut wx,
            &mut wy,
            &mut wz,
            &mut output16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed f16 3d type1");
        for (actual, expected) in output16.iter().zip(expected16.iter()) {
            let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!(
                (f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound,
                "f16 re: actual={}, expected={}, bound={re_bound:.3e}",
                f64::from(actual[0].to_f32()),
                expected.re
            );
            assert!(
                (f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound,
                "f16 im: actual={}, expected={}, bound={im_bound:.3e}",
                f64::from(actual[1].to_f32()),
                expected.im
            );
        }

        // ── profile mismatch ──────────────────────────────────────────────
        let mut mismatch_output = Array3::<Complex32>::zeros([grid.nx, grid.ny, grid.nz]);
        let err = plan
            .type1_typed_into(
                &positions,
                &values32,
                &mut scratch_grid,
                &mut wx,
                &mut wy,
                &mut wz,
                &mut mismatch_output,
                PrecisionProfile::HIGH_ACCURACY_F64,
            )
            .expect_err("profile mismatch must fail");
        assert!(
            matches!(err, ApolloError::Validation { ref field, .. } if field == "precision_profile"),
            "expected Validation {{ field: \"precision_profile\" }}, got {err:?}"
        );
    }

    /// Invariant: `NufftPlan3D::type2_typed_into` produces value-identical output
    /// for `Complex64`, within-f32-precision for `Complex32`, and within-f16-
    /// quantization for `[f16; 2]`, and rejects profile mismatches.
    ///
    /// **Verification strategy:**
    /// - Use the f64 type1 output as type2 input (round-trip guarantee)
    /// - f64 path matches allocating `type2` to machine precision
    /// - f32 path matches f64 reference to within 1e-5 relative tolerance
    /// - f16 path matches f64 reference to within `|v|·2⁻¹⁰ + 2⁻¹⁴`
    /// - f32 storage with f64 profile returns `ApolloError::Validation { field: "precision_profile" }`
    #[test]
    fn typed_type2_3d_supports_complex64_complex32_and_f16_storage() {
        let grid = UniformGrid3D::new(2, 2, 2, 1.0, 1.0, 1.0).expect("grid");
        let sigma = DEFAULT_NUFFT_OVERSAMPLING;
        let kernel_width = DEFAULT_NUFFT_KERNEL_WIDTH;
        let plan = NufftPlan3D::new(grid, sigma, kernel_width);
        let positions = vec![(0.1_f64, 0.2, 0.3), (0.5, 0.6, 0.7), (0.8, 0.1, 0.5)];

        // Build type2 input from a type1 forward pass
        let values64 = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.5),
            Complex64::new(-0.75, 0.25),
        ];
        let coeffs64 = plan.type1(&positions, &values64);

        // Allocating reference (Complex64 owner path)
        let expected64 = plan.type2(&positions, &coeffs64);

        // ── f64 path ──────────────────────────────────────────────────────
        let mut output64 = vec![Complex64::new(0.0, 0.0); positions.len()];
        plan.type2_typed_into(
            &positions,
            &coeffs64,
            &mut output64,
            PrecisionProfile::HIGH_ACCURACY_F64,
        )
        .expect("typed complex64 3d type2");
        for (actual, expected) in output64.iter().zip(expected64.iter()) {
            approx::assert_abs_diff_eq!(actual.re, expected.re);
            approx::assert_abs_diff_eq!(actual.im, expected.im);
        }

        // ── f32 path ──────────────────────────────────────────────────────
        let coeffs32 = coeffs64.mapv(|v| Complex32::new(v.re as f32, v.im as f32));
        let represented32 = coeffs32.mapv(Complex32::to_complex64);
        let expected32 = plan.type2(&positions, &represented32);
        let mut output32 = vec![Complex32::new(0.0, 0.0); positions.len()];
        plan.type2_typed_into(
            &positions,
            &coeffs32,
            &mut output32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed complex32 3d type2");
        let mut max_rel_err_f32 = 0.0_f64;
        for (actual, expected) in output32.iter().zip(expected32.iter()) {
            let denom_re = expected.re.abs().max(1e-30);
            let denom_im = expected.im.abs().max(1e-30);
            max_rel_err_f32 = max_rel_err_f32
                .max((f64::from(actual.re) - expected.re).abs() / denom_re)
                .max((f64::from(actual.im) - expected.im).abs() / denom_im);
        }
        assert!(
            max_rel_err_f32 < 1e-5,
            "f32 3d type2 max relative error {max_rel_err_f32:.3e} exceeds 1e-5"
        );

        // ── f16 path ──────────────────────────────────────────────────────
        let coeffs16 = coeffs64.mapv(|v| [f16::from_f32(v.re as f32), f16::from_f32(v.im as f32)]);
        let represented16 = coeffs16.mapv(<[f16; 2]>::to_complex64);
        let expected16 = plan.type2(&positions, &represented16);
        let mut output16 = vec![[f16::from_f32(0.0), f16::from_f32(0.0)]; positions.len()];
        plan.type2_typed_into(
            &positions,
            &coeffs16,
            &mut output16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed f16 3d type2");
        for (actual, expected) in output16.iter().zip(expected16.iter()) {
            let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!(
                (f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound,
                "f16 re: actual={}, expected={}, bound={re_bound:.3e}",
                f64::from(actual[0].to_f32()),
                expected.re
            );
            assert!(
                (f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound,
                "f16 im: actual={}, expected={}, bound={im_bound:.3e}",
                f64::from(actual[1].to_f32()),
                expected.im
            );
        }

        // ── profile mismatch ──────────────────────────────────────────────
        let mut mismatch_output = vec![Complex32::new(0.0, 0.0); positions.len()];
        let err = plan
            .type2_typed_into(
                &positions,
                &coeffs32,
                &mut mismatch_output,
                PrecisionProfile::HIGH_ACCURACY_F64,
            )
            .expect_err("profile mismatch must fail");
        assert!(
            matches!(err, ApolloError::Validation { ref field, .. } if field == "precision_profile"),
            "expected Validation {{ field: \"precision_profile\" }}, got {err:?}"
        );
    }
}
