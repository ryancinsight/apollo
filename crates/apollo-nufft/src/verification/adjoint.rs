use crate::{nufft_type1_1d, nufft_type2_1d, UniformDomain1D};
use eunomia::Complex64;
use leto::Array1;

/// Theorem: Type-1 / Type-2 adjoint identity.
///
///
///
/// Define the type-1 operator `A` and type-2 operator `A*` by
///
///
///
/// ```text
///
/// (A·c)_k  = Σ_j  c_j  exp(-2πi k x_j / L)   (type-1)
///
/// (A*·f)_j = Σ_k  f_k  exp(+2πi k x_j / L)   (type-2)
///
/// ```
///
///
///
/// They satisfy the real inner-product adjoint identity:
///
///
///
/// ```text
///
/// Re(⟨A·c, f⟩) = Re(⟨c, A*·f⟩)
///
/// ```
///
///
///
/// **Proof:** Expand `(A·c)_k`, swap summation order, and factor:
///
///
///
/// ```text
///
/// Re(Σ_k conj((A·c)_k) · f_k)
///
///   = Re(Σ_k Σ_j conj(c_j) exp(+2πi k x_j/L) · f_k)
///
///   = Re(Σ_j conj(c_j) · Σ_k f_k exp(+2πi k x_j/L))
///
///   = Re(Σ_j conj(c_j) · (A*·f)_j)  □
///
/// ```
///
///
///
/// The residual `|LHS - RHS|` must be below round-off for the exact
///
/// direct transform (asserted at `< 1e-10`).
///
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

    let freq_domain_values = freq_domain_data.to_vec();

    let freq_domain_array = Array1::from_shape_vec([freq_domain_values.len()], freq_domain_values)
        .expect("invariant: frequency-domain fixture length matches Array1 shape");

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
