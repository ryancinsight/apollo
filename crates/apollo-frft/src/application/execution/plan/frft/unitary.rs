//! Eigendecomposition-based unitary discrete fractional Fourier transform.
//!
//! # Theorem: Candan–Grünbaum Unitary DFrFT
//!
//! Let c = (N−1)/2 and let S ∈ ℝ^{N×N} be the palindrome-structured matrix:
//! ```text
//!   S[j,j]              = 2·cos(2π(j−c)/N) − 2       for j = 0..N−1
//!   S[j,(j+1) mod N]    = 1                           super-diagonal with wrap
//!   S[(j+1) mod N, j]   = 1                           sub-diagonal with wrap
//! ```
//! S is real symmetric with a palindrome diagonal (`S[j,j] = S[N−1−j, N−1−j]`),
//! which causes its eigenvectors to be either symmetric or antisymmetric under
//! index reversal. The eigendecomposition S = V Λ V^T gives an orthonormal
//! eigenbasis sorted by decreasing eigenvalue (column 0 = DC-like, column N−1
//! = most oscillatory).
//!
//! # Unitary discrete FrFT of order a
//!
//! ```text
//! DFrFT_a(x) = V · diag(exp(−iakπ/2), k=0..N−1) · V^T · x
//! ```
//!
//! # Unitarity proof
//!
//! V is orthogonal (V^T V = I) and |exp(−iakπ/2)| = 1 for real a,
//! so ‖DFrFT_a(x)‖₂ = ‖V diag(·) V^T x‖₂ = ‖x‖₂.
//!
//! # Complexity
//!
//! Construction: O(N³) (dense symmetric eigendecomposition via Leto).
//! Transform: O(N²) per call.
//!
//! # References
//!
//! - Candan, Ç., Kutay, M. A., & Ozaktas, H. M. (2000). The discrete fractional
//!   Fourier transform. *IEEE Trans. Signal Process.*, 48(5), 1329–1337.
//! - Grünbaum, F. A. (1982). The eigenvectors of the discrete Fourier transform.
//!   *J. Math. Anal. Appl.*, 88(1), 355–363.

use crate::domain::contracts::error::FrftError;
use eunomia::Complex64;
use leto::Array1;
use leto::Array2;
use leto_ops::symmetric_eigen_jacobi;
use moirai::ParallelSliceMut;
use std::f64::consts::PI;
use std::sync::Arc;

/// Below this O(N²) operation count, serial loops avoid parallel scheduling overhead.
const UNITARY_FRFT_PAR_OP_THRESHOLD: usize = 16_384;

thread_local! {
    static UNITARY_COEFF_SCRATCH: mnemosyne::scratch::ScratchPool<Complex64> = const { mnemosyne::scratch::ScratchPool::new() };
}

/// Sorted orthonormal eigenvector basis of the Grünbaum commuting matrix.
///
/// Column k corresponds to Hermite-Gauss order k; decreasing-eigenvalue sort
/// maps H_0 (DC-like, largest eigenvalue) to column 0 and H_{N−1} (most
/// oscillatory, smallest eigenvalue) to column N−1.
#[derive(Debug, Clone)]
pub struct GrunbaumBasis {
    eigenvectors: Arc<Array2<f64>>,
    n: usize,
}

impl GrunbaumBasis {
    /// Compute the Grünbaum basis for the centered DFT of length `n`.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    #[must_use]
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "GrunbaumBasis requires n > 0");
        let s = build_grunbaum_matrix(n);
        let v = sorted_eigenvectors(s);
        Self {
            eigenvectors: Arc::new(v),
            n,
        }
    }

    /// Return the transform length.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Return the sorted eigenvector matrix (N×N, row-major Leto storage).
    #[must_use]
    pub fn eigenvectors(&self) -> &Array2<f64> {
        &self.eigenvectors
    }

    /// Return the sorted eigenvectors as column-major `f32` values for WGPU storage buffers.
    #[must_use]
    pub fn eigenvectors_column_major_f32(&self) -> Vec<f32> {
        let mut values = Vec::with_capacity(self.n * self.n);
        for col in 0..self.n {
            for row in 0..self.n {
                values.push(
                    *self
                        .eigenvectors
                        .get([row, col])
                        .expect("eigenvector matrix shape matches basis order")
                        as f32,
                );
            }
        }
        values
    }
}

/// Build the palindrome-structured Grünbaum matrix S of length n.
///
/// The diagonal entry S[j,j] = 2·cos(2π(j−c)/N) − 2 with c = (N−1)/2 produces
/// a palindrome diagonal (S[j,j] = S[N−1−j, N−1−j]), ensuring eigenvectors are
/// symmetric or antisymmetric under index reversal. This property guarantees
/// DFrFT_2(x)[k] = x[N−1−k] (reversal) and ‖DFrFT_a(x)‖₂ = ‖x‖₂ for all
/// real a (unitarity follows from V^T V = I and |exp(−iakπ/2)| = 1).
fn build_grunbaum_matrix(n: usize) -> Array2<f64> {
    let mut values = vec![0.0; n * n];
    let center = (n as f64 - 1.0) / 2.0;
    // Diagonal: 2*cos(2*pi*(j - center)/n) - 2
    for j in 0..n {
        values[j * n + j] = 2.0 * (2.0 * PI * (j as f64 - center) / n as f64).cos() - 2.0;
    }
    // Off-diagonal with periodic wrap
    for j in 0..n.saturating_sub(1) {
        values[j * n + (j + 1)] = 1.0;
        values[(j + 1) * n + j] = 1.0;
    }
    if n >= 2 {
        values[n - 1] = 1.0;
        values[(n - 1) * n] = 1.0;
    }
    Array2::from_shape_vec([n, n], values).expect("Grunbaum matrix shape matches storage")
}

/// Eigendecompose S and return eigenvectors sorted by decreasing eigenvalue.
fn sorted_eigenvectors(s: Array2<f64>) -> Array2<f64> {
    let n = s.shape()[0];
    let decomp = symmetric_eigen_jacobi(&s.view())
        .expect("Grunbaum matrix construction must produce a finite symmetric matrix");
    let mut values = vec![0.0; n * n];
    // Leto returns increasing eigenvalues; reverse to place H_0 (DC-like) at column 0.
    for (new_col, old_col) in (0..n).rev().enumerate() {
        for row in 0..n {
            values[row * n + new_col] = *decomp
                .eigenvectors
                .get([row, old_col])
                .expect("eigenvector matrix shape matches basis order");
        }
    }
    Array2::from_shape_vec([n, n], values).expect("sorted eigenvector shape matches storage")
}

/// Unitary discrete fractional Fourier transform plan.
///
/// Uses the Candan (2000) eigendecomposition algorithm to guarantee
/// `‖DFrFT_a(x)‖₂ = ‖x‖₂` for all real orders `a` and all inputs `x`.
///
/// Construction is O(N³). Each forward or inverse call is O(N²).
#[derive(Debug, Clone)]
pub struct UnitaryFrftPlan {
    n: usize,
    order: f64,
    basis: GrunbaumBasis,
}

impl UnitaryFrftPlan {
    /// Create a validated unitary FrFT plan.
    ///
    /// # Errors
    ///
    /// Returns [`FrftError::EmptySignal`] if `n == 0` or
    /// [`FrftError::NonFiniteOrder`] if `order` is NaN or infinite.
    pub fn new(n: usize, order: f64) -> Result<Self, FrftError> {
        if n == 0 {
            return Err(FrftError::EmptySignal);
        }
        if !order.is_finite() {
            return Err(FrftError::NonFiniteOrder);
        }
        Ok(Self {
            n,
            order,
            basis: GrunbaumBasis::new(n),
        })
    }

    /// Return the transform length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.n
    }

    /// Return `true` if the plan length is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Return the fractional order.
    #[must_use]
    pub const fn order(&self) -> f64 {
        self.order
    }

    /// Execute the forward unitary DFrFT into a pre-allocated buffer.
    pub fn forward_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<Complex64>,
    ) -> Result<(), FrftError> {
        validate_io(input.size(), output.size(), self.n)?;
        apply_unitary_frft(
            self.basis.eigenvectors(),
            self.n,
            self.order,
            input.as_slice().expect("contiguous array"),
            output.as_slice_mut().expect("contiguous array"),
        );
        Ok(())
    }

    /// Execute the forward unitary DFrFT, returning an allocated output.
    pub fn forward(&self, input: &Array1<Complex64>) -> Result<Array1<Complex64>, FrftError> {
        let mut output = Array1::<Complex64>::zeros([self.n]);
        self.forward_into(input, &mut output)?;
        Ok(output)
    }

    /// Execute the inverse unitary DFrFT into a pre-allocated buffer.
    ///
    /// The inverse is DFrFT_{−a}: negate the order.
    pub fn inverse_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<Complex64>,
    ) -> Result<(), FrftError> {
        validate_io(input.size(), output.size(), self.n)?;
        apply_unitary_frft(
            self.basis.eigenvectors(),
            self.n,
            -self.order,
            input.as_slice().expect("contiguous array"),
            output.as_slice_mut().expect("contiguous array"),
        );
        Ok(())
    }

    /// Execute the inverse unitary DFrFT, returning an allocated output.
    pub fn inverse(&self, input: &Array1<Complex64>) -> Result<Array1<Complex64>, FrftError> {
        let mut output = Array1::<Complex64>::zeros([self.n]);
        self.inverse_into(input, &mut output)?;
        Ok(output)
    }
}

fn validate_io(input_len: usize, output_len: usize, n: usize) -> Result<(), FrftError> {
    if input_len != n {
        return Err(FrftError::LengthMismatch {
            input: input_len,
            plan: n,
        });
    }
    if output_len != n {
        return Err(FrftError::LengthMismatch {
            input: output_len,
            plan: n,
        });
    }
    Ok(())
}

/// Core unitary DFrFT computation: V · diag(exp(−iakπ/2)) · V^T · x.
///
/// Steps:
/// 1. Project input onto eigenbasis: c[k] = (V^T x)[k] = sum_j V[j,k] * x[j]
/// 2. Apply fractional phase: c[k] *= exp(−i·order·k·π/2)
/// 3. Reconstruct from eigenbasis: output[j] = (V c)[j] = sum_k V[j,k] * c[k]
fn apply_unitary_frft(
    v: &Array2<f64>,
    n: usize,
    order: f64,
    input: &[Complex64],
    output: &mut [Complex64],
) {
    with_coeff_scratch(n, |coeffs| {
        let work_items = n.saturating_mul(n);
        // Step 1: c = V^T x
        if work_items >= UNITARY_FRFT_PAR_OP_THRESHOLD {
            coeffs.par_mut().enumerate(|k, coeff| {
                *coeff = projection_row(v, input, n, k);
            });
        } else {
            coeffs.iter_mut().enumerate().for_each(|(k, coeff)| {
                *coeff = projection_row(v, input, n, k);
            });
        }
        // Step 2: phase c[k] *= exp(-i * order * k * pi / 2)
        if work_items >= UNITARY_FRFT_PAR_OP_THRESHOLD {
            coeffs.par_mut().enumerate(|k, coeff| {
                apply_phase(coeff, order, k);
            });
        } else {
            coeffs.iter_mut().enumerate().for_each(|(k, coeff)| {
                apply_phase(coeff, order, k);
            });
        }
        // Step 3: output = V c
        if work_items >= UNITARY_FRFT_PAR_OP_THRESHOLD {
            output.par_mut().enumerate(|j, slot| {
                *slot = reconstruction_row(v, coeffs, n, j);
            });
        } else {
            output.iter_mut().enumerate().for_each(|(j, slot)| {
                *slot = reconstruction_row(v, coeffs, n, j);
            });
        }
    });
}

#[inline]
fn projection_row(v: &Array2<f64>, input: &[Complex64], n: usize, k: usize) -> Complex64 {
    (0..n)
        .map(|j| input[j] * *v.get([j, k]).expect("basis index in bounds"))
        .sum()
}

#[inline]
fn apply_phase(coeff: &mut Complex64, order: f64, k: usize) {
    let phase = -order * k as f64 * PI / 2.0;
    *coeff *= Complex64::new(phase.cos(), phase.sin());
}

#[inline]
fn reconstruction_row(v: &Array2<f64>, coeffs: &[Complex64], n: usize, j: usize) -> Complex64 {
    (0..n)
        .map(|k| coeffs[k] * *v.get([j, k]).expect("basis index in bounds"))
        .sum()
}

#[inline]
fn with_coeff_scratch<R>(n: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
    UNITARY_COEFF_SCRATCH.with(|pool| pool.with_scratch(n, f))
}

#[cfg(test)]
fn coeff_scratch_capacity() -> usize {
    UNITARY_COEFF_SCRATCH.with(|pool| pool.capacity())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn unitary_order_zero_is_identity() {
        let n = 8;
        let plan = UnitaryFrftPlan::new(n, 0.0).expect("valid plan");
        let input = Array1::from_shape_fn([n], |[i]| {
            Complex64::new((i as f64 * 0.3).sin(), (i as f64 * 0.17).cos())
        });
        let result = plan.forward(&input).expect("forward");
        for (actual, expected) in result.iter().zip(input.iter()) {
            assert!(
                (actual - expected).norm() < 1.0e-12,
                "order 0 is not identity: diff = {}",
                (actual - expected).norm()
            );
        }
    }

    #[test]
    fn unitary_order_4_is_identity() {
        // 4 full cycles = identity
        let n = 8;
        let plan = UnitaryFrftPlan::new(n, 4.0).expect("valid plan");
        let input = Array1::from_shape_fn([n], |[i]| {
            Complex64::new(i as f64 * 0.25 - 1.0, 0.5 - i as f64 * 0.1)
        });
        let result = plan.forward(&input).expect("forward");
        for (actual, expected) in result.iter().zip(input.iter()) {
            assert!(
                (actual - expected).norm() < 1.0e-10,
                "order 4 is not identity: diff = {}",
                (actual - expected).norm()
            );
        }
    }

    #[test]
    fn unitary_order_1_squared_equals_reversal() {
        // Additivity: DFrFT_1 ∘ DFrFT_1 = DFrFT_2 = reversal (output[k] = input[N-1-k]).
        // This verifies the semigroup law at the integer boundary without depending
        // on any specific DFT centering convention.
        let n = 8;
        let plan = UnitaryFrftPlan::new(n, 1.0).expect("valid plan");
        let input = Array1::from_shape_fn([n], |[i]| Complex64::new((i as f64 * 0.31).sin(), 0.0));
        let after_twice = plan
            .forward(&plan.forward(&input).expect("first forward"))
            .expect("second forward");
        for k in 0..n {
            let expected = input[n - 1 - k];
            assert!(
                (after_twice[k] - expected).norm() < 1.0e-10,
                "DFrFT_1^2 != reversal at k={}: diff={}",
                k,
                (after_twice[k] - expected).norm()
            );
        }
    }

    #[test]
    fn unitary_order_2_is_reversal() {
        let n = 8;
        let plan = UnitaryFrftPlan::new(n, 2.0).expect("valid plan");
        let input = Array1::from_shape_fn([n], |[i]| Complex64::new(i as f64 + 1.0, 0.0));
        let result = plan.forward(&input).expect("forward");
        for k in 0..n {
            let expected = input[n - 1 - k];
            assert!(
                (result[k] - expected).norm() < 1.0e-10,
                "order 2 reversal failed at k={}: got {:?}, expected {:?}",
                k,
                result[k],
                expected
            );
        }
    }

    #[test]
    fn unitary_forward_inverse_roundtrip() {
        let n = 16;
        for order in [0.3_f64, 0.5, 0.7, 1.0, 1.3, 1.7, 2.5] {
            let plan = UnitaryFrftPlan::new(n, order).expect("valid plan");
            let input = Array1::from_shape_fn([n], |[i]| {
                Complex64::new((i as f64 * 0.23).sin(), (i as f64 * 0.31).cos())
            });
            let spectrum = plan.forward(&input).expect("forward");
            let recovered = plan.inverse(&spectrum).expect("inverse");
            for (actual, expected) in recovered.iter().zip(input.iter()) {
                assert!(
                    (actual - expected).norm() < 1.0e-10,
                    "roundtrip failed at order={}: diff = {}",
                    order,
                    (actual - expected).norm()
                );
            }
        }
    }

    #[test]
    fn unitary_transform_reuses_thread_local_coeff_workspace() {
        let n = 16;
        let plan = UnitaryFrftPlan::new(n, 0.5).expect("valid plan");
        let input = Array1::from_shape_fn([n], |[i]| {
            Complex64::new((i as f64 * 0.19).sin(), (i as f64 * 0.23).cos())
        });
        let mut first = Array1::<Complex64>::zeros([n]);
        let mut second = Array1::<Complex64>::zeros([n]);

        plan.forward_into(&input, &mut first)
            .expect("first forward");
        let first_capacity = coeff_scratch_capacity();
        plan.forward_into(&input, &mut second)
            .expect("second forward");
        let second_capacity = coeff_scratch_capacity();

        assert_eq!(first_capacity, second_capacity);
        assert!(first_capacity >= n);
        for (actual, expected) in second.iter().zip(first.iter()) {
            assert!((actual - expected).norm() < 1.0e-14);
        }
    }

    #[test]
    fn moirai_parallel_unitary_rows_match_serial_formula_at_threshold() {
        let n = 128;
        let order = 0.5;
        let basis = GrunbaumBasis::new(n);
        let input = Array1::from_shape_fn([n], |[i]| {
            Complex64::new((i as f64 * 0.11).sin(), (i as f64 * 0.19).cos())
        });
        let plan = UnitaryFrftPlan {
            n,
            order,
            basis: basis.clone(),
        };
        let actual = plan.forward(&input).expect("forward");
        let v = basis.eigenvectors();
        let mut coeffs = (0..n)
            .map(|k| projection_row(v, input.as_slice().expect("contiguous input"), n, k))
            .collect::<Vec<_>>();
        for (k, coeff) in coeffs.iter_mut().enumerate() {
            apply_phase(coeff, order, k);
        }

        for (row, value) in actual.iter().enumerate() {
            let expected = reconstruction_row(v, &coeffs, n, row);
            assert_eq!(value.re.to_bits(), expected.re.to_bits());
            assert_eq!(value.im.to_bits(), expected.im.to_bits());
        }
    }

    #[test]
    fn unitary_frft_preserves_l2_norm_for_non_integer_orders() {
        // Core unitarity test: ||DFrFT_a(x)||_2 = ||x||_2 for non-integer a.
        let n = 16;
        let input = Array1::from_shape_fn([n], |[i]| {
            Complex64::new((i as f64 * 0.37).cos(), (i as f64 * 0.41).sin())
        });
        let input_norm_sq: f64 = input.iter().map(|x| x.norm_sqr()).sum();

        for order in [0.1_f64, 0.3, 0.5, 0.7, 1.2, 1.5, 1.8, 2.3, 2.7, 3.1] {
            let plan = UnitaryFrftPlan::new(n, order).expect("valid plan");
            let result = plan.forward(&input).expect("forward");
            let output_norm_sq: f64 = result.iter().map(|x| x.norm_sqr()).sum();
            let rel_err = (output_norm_sq - input_norm_sq).abs() / (input_norm_sq + 1.0e-30);
            assert!(
                rel_err < 1.0e-10,
                "unitarity violated at order={}: ||output||²={}, ||input||²={}, rel_err={}",
                order,
                output_norm_sq,
                input_norm_sq,
                rel_err
            );
        }
    }

    #[test]
    fn unitary_frft_additive_order_property() {
        // DFrFT_{a+b}(x) = DFrFT_a(DFrFT_b(x)) for a=0.4, b=0.6
        let n = 8;
        let a = 0.4_f64;
        let b = 0.6_f64;
        let plan_a = UnitaryFrftPlan::new(n, a).expect("plan a");
        let plan_b = UnitaryFrftPlan::new(n, b).expect("plan b");
        let plan_ab = UnitaryFrftPlan::new(n, a + b).expect("plan a+b");

        let input = Array1::from_shape_fn([n], |[i]| {
            Complex64::new((i as f64 * 0.25).sin(), (i as f64 * 0.17).cos())
        });
        let composed = plan_a
            .forward(&plan_b.forward(&input).expect("forward b"))
            .expect("forward a");
        let direct = plan_ab.forward(&input).expect("forward ab");

        for (actual, expected) in composed.iter().zip(direct.iter()) {
            assert!(
                (actual - expected).norm() < 1.0e-9,
                "additivity failed: diff = {}",
                (actual - expected).norm()
            );
        }
    }

    #[test]
    fn rejects_invalid_plan_parameters() {
        assert!(matches!(
            UnitaryFrftPlan::new(0, 1.0),
            Err(FrftError::EmptySignal)
        ));
        assert!(matches!(
            UnitaryFrftPlan::new(4, f64::NAN),
            Err(FrftError::NonFiniteOrder)
        ));
        assert!(matches!(
            UnitaryFrftPlan::new(4, f64::INFINITY),
            Err(FrftError::NonFiniteOrder)
        ));
    }

    #[test]
    fn length_mismatch_is_rejected() {
        let plan = UnitaryFrftPlan::new(4, 0.5).expect("valid plan");
        let input = Array1::from_elem([3], Complex64::new(1.0, 0.0));
        let mut output = Array1::<Complex64>::zeros([4]);
        assert!(matches!(
            plan.forward_into(&input, &mut output),
            Err(FrftError::LengthMismatch { .. })
        ));
    }

    proptest! {
        /// UnitaryFrftPlan forward followed by inverse recovers the input signal.
        ///
        /// DFrFT_{-a}(DFrFT_a(x)) = x for all real a and all x ∈ ℂ^N.
        /// Proof: V·diag(e^{iakπ/2})·V^T·(V·diag(e^{-iakπ/2})·V^T·x)
        ///      = V·diag(e^{iakπ/2})·diag(e^{-iakπ/2})·V^T·x
        ///      = V·I·V^T·x = x (since V^T V = I).
        #[test]
        fn unitary_frft_roundtrip_for_arbitrary_order_and_signal(
            n_and_re in (2_usize..=12).prop_flat_map(|n| {
                prop::collection::vec(-2.0_f64..2.0_f64, n)
                    .prop_map(move |v| (n, v))
            }),
            // Fractional order: 10 steps of 0.1 offset by 0.05 to avoid integers
            order_step in 0_u32..40_u32,
        ) {
            let (n, re_parts) = n_and_re;
            let order = order_step as f64 * 0.1 + 0.05;
            let input = leto::Array1::from(
                re_parts.iter().map(|&r| eunomia::Complex64::new(r, 0.0)).collect::<Vec<_>>()
            );
            let plan = UnitaryFrftPlan::new(n, order).expect("plan");
            let forward = plan.forward(&input).expect("forward");
            let recovered = plan.inverse(&forward).expect("inverse");
            for (expected, actual) in input.iter().zip(recovered.iter()) {
                prop_assert!(
                    (expected - actual).norm() < 1.0e-9,
                    "roundtrip failed at order={order}: |expected-actual|={} for n={n}",
                    (expected - actual).norm()
                );
            }
        }

        /// UnitaryFrftPlan satisfies the semigroup (additivity) law: FrFT_{a+b} = FrFT_a ∘ FrFT_b.
        ///
        /// Proof: V·diag(e^{-i(a+b)kπ/2})·V^T·x
        ///      = V·diag(e^{-iakπ/2})·diag(e^{-ibkπ/2})·V^T·x
        ///      = DFrFT_a(DFrFT_b(x)).
        #[test]
        fn unitary_frft_additivity_of_order(
            n_and_re in (2_usize..=10).prop_flat_map(|n| {
                prop::collection::vec(-2.0_f64..2.0_f64, n)
                    .prop_map(move |v| (n, v))
            }),
            // Two non-integer orders whose sum is also non-integer
            a_step in 1_u32..18_u32,
            b_step in 1_u32..18_u32,
        ) {
            let (n, re_parts) = n_and_re;
            let a = a_step as f64 * 0.1 + 0.05;
            let b = b_step as f64 * 0.1 + 0.05;
            let input = leto::Array1::from(
                re_parts.iter().map(|&r| eunomia::Complex64::new(r, 0.0)).collect::<Vec<_>>()
            );
            let plan_ab = UnitaryFrftPlan::new(n, a + b).expect("plan_ab");
            let plan_b = UnitaryFrftPlan::new(n, b).expect("plan_b");
            let plan_a = UnitaryFrftPlan::new(n, a).expect("plan_a");
            let direct = plan_ab.forward(&input).expect("direct");
            let composed = plan_a.forward(&plan_b.forward(&input).expect("step_b")).expect("step_a");
            for (d, c) in direct.iter().zip(composed.iter()) {
                prop_assert!(
                    (d - c).norm() < 1.0e-9,
                    "additivity failed: a={a}, b={b}, |d-c|={} for n={n}",
                    (d - c).norm()
                );
            }
        }

        /// UnitaryFrftPlan is linear: FrFT_a(c·x) = c·FrFT_a(x) for real scalar c.
        ///
        /// Proof: The DFrFT sum is linear in the input vector; the Grünbaum
        /// eigenvector matrix V and diagonal phase factor are both independent of x.
        #[test]
        fn unitary_frft_scalar_linearity(
            n_and_re in (2_usize..=10).prop_flat_map(|n| {
                prop::collection::vec(-2.0_f64..2.0_f64, n)
                    .prop_map(move |v| (n, v))
            }),
            order_step in 1_u32..38_u32,
            scalar in -3.0_f64..3.0_f64,
        ) {
            let (n, re_parts) = n_and_re;
            let order = order_step as f64 * 0.1 + 0.05;
            let input = leto::Array1::from(
                re_parts.iter().map(|&r| eunomia::Complex64::new(r, 0.0)).collect::<Vec<_>>()
            );
            let scaled = leto::Array1::from(
                re_parts.iter().map(|&r| eunomia::Complex64::new(r * scalar, 0.0)).collect::<Vec<_>>()
            );
            let plan = UnitaryFrftPlan::new(n, order).expect("plan");
            let frft_scaled = plan.forward(&scaled).expect("frft_scaled");
            let frft_then_scale: Vec<eunomia::Complex64> = plan
                .forward(&input)
                .expect("frft_x")
                .iter()
                .map(|&v| v * scalar)
                .collect();
            for (a, b) in frft_scaled.iter().zip(frft_then_scale.iter()) {
                prop_assert!(
                    (a - b).norm() < 1.0e-9,
                    "linearity failed: scalar={scalar}, |a-b|={} for n={n}",
                    (a - b).norm()
                );
            }
        }
    }
}
