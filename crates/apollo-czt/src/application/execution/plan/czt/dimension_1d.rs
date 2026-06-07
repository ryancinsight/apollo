//! 1D Chirp Z-Transform Plan

use crate::application::execution::kernel::bluestein::{
    czt_bjork_pereyra_inverse_into, czt_bluestein_forward_into_with_workspace, czt_inverse_nodes,
};
use crate::application::execution::kernel::direct::czt_direct_forward;
use crate::domain::contracts::error::CztError;
use apollo_fft::{f16, FftPlan1D, PrecisionProfile, Shape1D};
use ndarray::Array1;
use num_complex::{Complex32, Complex64};
use std::cell::RefCell;

thread_local! {
    static TYPED_INPUT64_SCRATCH: RefCell<Vec<Complex64>> = const { RefCell::new(Vec::new()) };
    static TYPED_OUTPUT64_SCRATCH: RefCell<Vec<Complex64>> = const { RefCell::new(Vec::new()) };
    static FORWARD_WORKSPACE_SCRATCH: RefCell<Vec<Complex64>> = const { RefCell::new(Vec::new()) };
}

/// Run `f` with a thread-local Bluestein forward-convolution workspace sized to
/// `len`. Per-thread scratch replaces a per-plan `Mutex<Vec<_>>`: it removes the
/// lock from every `forward` call and lets concurrent threads transform through
/// a shared plan without serializing on one workspace. The buffer is grown on
/// demand and reused across calls on the same thread.
fn with_forward_workspace<R>(len: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
    FORWARD_WORKSPACE_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        if scratch.len() < len {
            scratch.resize(len, Complex64::new(0.0, 0.0));
        }
        f(&mut scratch[..len])
    })
}

/// Return whether a CZT input or output length satisfies the non-zero contract.
#[must_use]
pub fn is_valid_length(n: usize) -> bool {
    n > 0
}

/// Reusable chirp z-transform plan.
///
/// The plan is the single source of truth for CZT dimensions, spiral
/// parameters, chirp factors, convolution kernel, and FFT plan. Construction
/// validates non-zero lengths, finite non-zero `a` and `w`, and precomputes
/// all input-independent terms.
///
/// # Mathematical contract
///
/// For input `x[0..n)` this plan computes
/// `X[k] = sum_n x[n] a^-n w^(n k)` for `k in 0..m`.
/// `forward_direct` evaluates that definition directly. `forward` and
/// `forward_into` evaluate the same map through Bluestein convolution.
///
/// # Complexity
///
/// `forward_direct` costs `O(nm)` time. `forward` and `forward_into` cost
/// `O(p log p)` time with one `O(p)` convolution workspace, where
/// `p = convolution_len() >= n + m - 1`.
#[derive(Debug)]
pub struct CztPlan {
    n: usize,
    m: usize,
    a: Complex64,
    w: Complex64,
    convolution_len: usize,
    chirp_n: Vec<Complex64>,
    chirp_k: Vec<Complex64>,
    fft_kernel: Array1<Complex64>,
    fft_plan: FftPlan1D<f64>,
    inverse_nodes: Option<Vec<Complex64>>,
}

impl CztPlan {
    /// Create a validated CZT plan precomputing and caching the $O(P \log P)$ convolution kernel.
    pub fn new(n: usize, m: usize, a: Complex64, w: Complex64) -> Result<Self, CztError> {
        if n == 0 || m == 0 {
            return Err(CztError::EmptyLength);
        }
        if !a.re.is_finite() || !a.im.is_finite() || !w.re.is_finite() || !w.im.is_finite() {
            return Err(CztError::InvalidParameters);
        }
        if a == Complex64::new(0.0, 0.0) || w == Complex64::new(0.0, 0.0) {
            return Err(CztError::InvalidParameters);
        }

        let convolution_len = (n + m - 1).next_power_of_two();
        let fft_plan = FftPlan1D::<f64>::new(
            Shape1D::new(convolution_len).expect("CZT convolution length must be valid"),
        );

        let mut chirp_n = Vec::with_capacity(n);
        let mut chirp_k = Vec::with_capacity(m);
        let mut kernel = vec![Complex64::new(0.0, 0.0); convolution_len];

        for n_idx in 0..n {
            let nn = n_idx as f64;
            let phase = 0.5 * nn * nn;
            chirp_n.push((a.powf(-nn)) * w.powf(phase));
        }

        for k_idx in 0..m {
            let kk = k_idx as f64;
            let phase = 0.5 * kk * kk;
            chirp_k.push(w.powf(phase));
        }

        for k_idx in 0..m {
            kernel[k_idx] = w.powf(-0.5 * (k_idx as f64) * (k_idx as f64));
        }
        for k_idx in 1..n {
            kernel[convolution_len - k_idx] = w.powf(-0.5 * (k_idx as f64) * (k_idx as f64));
        }

        let mut fft_kernel = Array1::from_vec(kernel);
        fft_plan.forward_complex_inplace(&mut fft_kernel);
        let inverse_nodes = if n == m {
            Some(czt_inverse_nodes(n, w))
        } else {
            None
        };

        Ok(Self {
            n,
            m,
            a,
            w,
            convolution_len,
            chirp_n,
            chirp_k,
            fft_kernel,
            fft_plan,
            inverse_nodes,
        })
    }

    /// Return the input length.
    #[must_use]
    pub const fn input_len(&self) -> usize {
        self.n
    }

    /// Return the output length.
    #[must_use]
    pub const fn output_len(&self) -> usize {
        self.m
    }

    /// Return the convolution length used by the fast path.
    #[must_use]
    pub const fn convolution_len(&self) -> usize {
        self.convolution_len
    }

    #[cfg(test)]
    pub(crate) fn forward_workspace_capacity() -> usize {
        FORWARD_WORKSPACE_SCRATCH.with(|scratch| scratch.borrow().capacity())
    }

    /// Forward direct CZT evaluation.
    pub fn forward_direct(&self, input: &Array1<Complex64>) -> Result<Array1<Complex64>, CztError> {
        if input.len() != self.n {
            return Err(CztError::LengthMismatch);
        }
        czt_direct_forward(input, self.m, self.a, self.w)
    }

    /// Forward CZT using Bluestein's convolution identity with precomputed caching.
    pub fn forward(&self, input: &Array1<Complex64>) -> Result<Array1<Complex64>, CztError> {
        if input.len() != self.n {
            return Err(CztError::LengthMismatch);
        }

        let mut output = Array1::<Complex64>::zeros(self.m);
        self.forward_into(input, &mut output)?;
        Ok(output)
    }

    /// Forward CZT into caller-owned output storage.
    pub fn forward_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<Complex64>,
    ) -> Result<(), CztError> {
        if input.len() != self.n || output.len() != self.m {
            return Err(CztError::LengthMismatch);
        }
        self.forward_complex64_slice_into(
            input.as_slice().expect("CZT input must be contiguous"),
            output
                .as_slice_mut()
                .expect("CZT output must be contiguous"),
        )
    }

    /// Forward CZT over contiguous Complex64 slices.
    pub(crate) fn forward_complex64_slice_into(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), CztError> {
        if input.len() != self.n || output.len() != self.m {
            return Err(CztError::LengthMismatch);
        }
        with_forward_workspace(self.convolution_len, |workspace| {
            czt_bluestein_forward_into_with_workspace(
                input,
                output,
                workspace,
                self.convolution_len,
                &self.chirp_n,
                &self.chirp_k,
                self.fft_kernel
                    .as_slice()
                    .expect("CZT FFT kernel must be contiguous"),
                &self.fft_plan,
            );
        });
        Ok(())
    }

    /// Forward CZT for `Complex64`, `Complex32`, or mixed two-lane `f16` storage.
    ///
    /// `Complex64` uses the native high-accuracy path. `Complex32` and mixed
    /// `[f16; 2]` storage convert through the owner kernel and quantize once
    /// into the caller-owned output.
    pub fn forward_typed_into<T: CztStorage>(
        &self,
        input: &Array1<T>,
        output: &mut Array1<T>,
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        T::forward_into(self, input, output, profile)
    }

    /// In-place forward CZT.
    pub fn forward_inplace(&self, data: &mut Array1<Complex64>) -> Result<(), CztError> {
        let transformed = self.forward(data)?;
        *data = transformed;
        Ok(())
    }

    /// Inverse chirp z-transform via the Björck–Pereyra Vandermonde solve.
    ///
    /// # Mathematical contract
    ///
    /// Given a spectrum `X[k]` of length `M`, recovers `x[n]` of length `N`
    /// such that `forward(x) == X` (in exact arithmetic).  Inversion requires
    /// `M == N`; a rectangular (M ≠ N) CZT is not uniquely invertible.
    ///
    /// The evaluation points are `z_k = W^k` for `k = 0..N-1`.  If two points
    /// coincide the Vandermonde matrix is singular and `CztError::NotInvertible`
    /// is returned.
    ///
    /// # Complexity
    ///
    /// O(N²) time, O(N) additional space.
    pub fn inverse(&self, spectrum: &Array1<Complex64>) -> Result<Array1<Complex64>, CztError> {
        if spectrum.len() != self.m {
            return Err(CztError::LengthMismatch);
        }
        if self.n != self.m {
            return Err(CztError::NotInvertible {
                reason: "inverse is only defined for square (M == N) CZT plans",
            });
        }
        let mut output = Array1::<Complex64>::zeros(self.n);
        self.inverse_complex64_slice_into(
            spectrum
                .as_slice()
                .expect("CZT spectrum must be contiguous"),
            output
                .as_slice_mut()
                .expect("CZT inverse output must be contiguous"),
        )?;
        Ok(output)
    }

    /// Inverse CZT over contiguous Complex64 slices.
    pub(crate) fn inverse_complex64_slice_into(
        &self,
        spectrum: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), CztError> {
        if spectrum.len() != self.m || output.len() != self.n {
            return Err(CztError::LengthMismatch);
        }
        let nodes = self
            .inverse_nodes
            .as_deref()
            .ok_or(CztError::NotInvertible {
                reason: "inverse is only defined for square (M == N) CZT plans",
            })?;
        czt_bjork_pereyra_inverse_into(spectrum, output, nodes, self.a)
    }

    /// Inverse CZT for typed `Complex64`, `Complex32`, or mixed `[f16; 2]` storage.
    ///
    /// Converts input once to `Complex64`, applies the exact Vandermonde solve,
    /// and quantizes back to the requested storage type.
    pub fn inverse_typed_into<T: CztStorage>(
        &self,
        spectrum: &Array1<T>,
        output: &mut Array1<T>,
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        T::inverse_into(self, spectrum, output, profile)
    }
}

/// Complex storage accepted by typed CZT paths.
pub trait CztStorage: Copy + Send + Sync + 'static {
    /// Required precision profile for this storage type.
    const PROFILE: PrecisionProfile;

    /// Convert storage into the owner `Complex64` arithmetic path.
    fn to_complex64(self) -> Complex64;

    /// Convert owner arithmetic result back to storage.
    fn from_complex64(value: Complex64) -> Self;

    /// Execute forward transform into caller-owned storage.
    fn forward_into(
        plan: &CztPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        validate_profile(profile, Self::PROFILE)?;
        if input.len() != plan.input_len() || output.len() != plan.output_len() {
            return Err(CztError::LengthMismatch);
        }
        with_complex64_workspaces(plan.input_len(), plan.output_len(), |input64, output64| {
            for (slot, value) in input64.iter_mut().zip(input.iter().copied()) {
                *slot = Self::to_complex64(value);
            }
            plan.forward_complex64_slice_into(input64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                *slot = Self::from_complex64(value);
            }
            Ok(())
        })
    }

    /// Execute inverse transform into caller-owned storage.
    fn inverse_into(
        plan: &CztPlan,
        spectrum: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        validate_profile(profile, Self::PROFILE)?;
        if spectrum.len() != plan.output_len() || output.len() != plan.input_len() {
            return Err(CztError::LengthMismatch);
        }
        with_complex64_workspaces(
            plan.output_len(),
            plan.input_len(),
            |spectrum64, output64| {
                for (slot, value) in spectrum64.iter_mut().zip(spectrum.iter().copied()) {
                    *slot = Self::to_complex64(value);
                }
                plan.inverse_complex64_slice_into(spectrum64, output64)?;
                for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                    *slot = Self::from_complex64(value);
                }
                Ok(())
            },
        )
    }
}

impl CztStorage for Complex64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_complex64(self) -> Complex64 {
        self
    }

    fn from_complex64(value: Complex64) -> Self {
        value
    }

    fn forward_into(
        plan: &CztPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_into(input, output)
    }

    fn inverse_into(
        plan: &CztPlan,
        spectrum: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        validate_profile(profile, Self::PROFILE)?;
        if spectrum.len() != plan.output_len() || output.len() != plan.input_len() {
            return Err(CztError::LengthMismatch);
        }
        plan.inverse_complex64_slice_into(
            spectrum
                .as_slice()
                .expect("CZT spectrum must be contiguous"),
            output
                .as_slice_mut()
                .expect("CZT inverse output must be contiguous"),
        )
    }
}

impl CztStorage for Complex32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_complex64(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }

    fn from_complex64(value: Complex64) -> Self {
        Complex32::new(value.re as f32, value.im as f32)
    }
}

impl CztStorage for [f16; 2] {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_complex64(self) -> Complex64 {
        Complex64::new(f64::from(self[0].to_f32()), f64::from(self[1].to_f32()))
    }

    fn from_complex64(value: Complex64) -> Self {
        [
            f16::from_f32(value.re as f32),
            f16::from_f32(value.im as f32),
        ]
    }
}

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> Result<(), CztError> {
    if actual.storage == expected.storage && actual.compute == expected.compute {
        Ok(())
    } else {
        Err(CztError::PrecisionMismatch)
    }
}

fn with_complex64_workspaces<R>(
    input_len: usize,
    output_len: usize,
    f: impl FnOnce(&mut [Complex64], &mut [Complex64]) -> R,
) -> R {
    TYPED_INPUT64_SCRATCH.with(|input_scratch| {
        TYPED_OUTPUT64_SCRATCH.with(|output_scratch| {
            let mut input_scratch = input_scratch.borrow_mut();
            if input_scratch.len() < input_len {
                input_scratch.resize(input_len, Complex64::new(0.0, 0.0));
            }

            let mut output_scratch = output_scratch.borrow_mut();
            if output_scratch.len() < output_len {
                output_scratch.resize(output_len, Complex64::new(0.0, 0.0));
            }

            f(
                &mut input_scratch[..input_len],
                &mut output_scratch[..output_len],
            )
        })
    })
}

#[cfg(test)]
pub(crate) fn typed_scratch_capacities() -> (usize, usize) {
    TYPED_INPUT64_SCRATCH.with(|input_scratch| {
        TYPED_OUTPUT64_SCRATCH.with(|output_scratch| {
            (
                input_scratch.borrow().capacity(),
                output_scratch.borrow().capacity(),
            )
        })
    })
}

#[cfg(test)]
mod spiral_collapse_tests {
    use super::*;

    /// CZT with A=1, W=exp(-2pi*i/N), M=N equals the N-point DFT (spiral-collapse theorem).
    #[test]
    fn czt_with_dft_parameters_equals_dft() {
        let n = 8usize;
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64);
        let input = Array1::from_shape_fn(n, |i| {
            Complex64::new((i as f64 * 0.7).sin(), (i as f64 * 0.3).cos())
        });
        let plan = CztPlan::new(n, n, a, w).expect("valid DFT plan");
        let fast = plan.forward(&input).expect("fast");
        let direct = plan.forward_direct(&input).expect("direct");
        for (k, (cv, dv)) in fast.iter().zip(direct.iter()).enumerate() {
            let err = (cv - dv).norm();
            assert!(err < 1e-10, "CZT != DFT at k={k}: err={err}");
        }
    }

    /// Bluestein fast path matches direct for multiple non-trivial (N, M) pairs.
    #[test]
    fn czt_bluestein_equals_direct_for_fixed_inputs() {
        for (n, m) in [(3usize, 3usize), (5, 7), (11, 11), (13, 8)] {
            let a = Complex64::from_polar(0.9, 0.3);
            let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64 * 0.8);
            let inp = Array1::from_shape_fn(n, |i| {
                Complex64::new((i as f64 * 0.31).sin(), -(i as f64 * 0.19).cos())
            });
            let plan = CztPlan::new(n, m, a, w).expect("valid");
            let fast = plan.forward(&inp).expect("fast");
            let direct = plan.forward_direct(&inp).expect("direct");
            for (k, (f, d)) in fast.iter().zip(direct.iter()).enumerate() {
                let err = (f - d).norm();
                assert!(err < 1e-9, "n={n} m={m} k={k} err={err}");
            }
        }
    }

    /// Theorem (Spiral-Collapse, independent cross-check): CZT(x, N, exp(-2πi/N), 1)
    /// equals the N-point DFT, verified here against `apollo_fft::fft_1d_complex`
    /// which is entirely independent of the CZT implementation path.
    ///
    /// Proof: By Bluestein (1970), CZT with A=1, W=exp(-2πi/N), M=N reduces
    /// identically to the DFT summation Σ_n x[n] exp(-2πikn/N). The apollo_fft
    /// implementation uses a separate Cooley-Tukey / Bluestein kernel path; any
    /// shared sign or index bug in the CZT path would produce a measurable mismatch.
    #[test]
    fn czt_dft_parameters_match_independent_fft_implementation() {
        let n = 8usize;
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64);
        let input = Array1::from_shape_fn(n, |i| {
            Complex64::new((i as f64 * 0.7).sin(), (i as f64 * 0.3).cos())
        });
        let plan = CztPlan::new(n, n, a, w).expect("valid DFT plan");
        let czt_output = plan.forward(&input).expect("CZT forward");
        // Independent DFT via apollo_fft (separate Cooley-Tukey / Bluestein path)
        let fft_output = apollo_fft::fft_1d_complex(&input);
        for (k, (cv, fv)) in czt_output.iter().zip(fft_output.iter()).enumerate() {
            let err = (cv - fv).norm();
            assert!(
                err < 1e-9,
                "CZT does not match independent FFT at k={k}: czt={cv:?}, fft={fv:?}, err={err:.3e}"
            );
        }
    }

    #[test]
    fn rejects_zero_length() {
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::new(0.9, 0.1);
        assert!(matches!(
            CztPlan::new(0, 5, a, w),
            Err(CztError::EmptyLength)
        ));
        assert!(matches!(
            CztPlan::new(5, 0, a, w),
            Err(CztError::EmptyLength)
        ));
    }

    #[test]
    fn rejects_zero_a() {
        let a = Complex64::new(0.0, 0.0);
        let w = Complex64::new(0.9, 0.1);
        assert!(matches!(
            CztPlan::new(4, 4, a, w),
            Err(CztError::InvalidParameters)
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn direct_matches_reference_for_small_sequence() {
        let input = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.5, 0.25),
            Complex64::new(-0.75, 0.5),
        ]);
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / 8.0);
        let plan = CztPlan::new(input.len(), 4, a, w).expect("valid plan");
        let direct = plan.forward_direct(&input).expect("direct");
        let fast = plan.forward(&input).expect("fast");
        for (lhs, rhs) in direct.iter().zip(fast.iter()) {
            assert_relative_eq!(lhs.re, rhs.re, epsilon = 1.0e-9);
            assert_relative_eq!(lhs.im, rhs.im, epsilon = 1.0e-9);
        }
    }

    #[test]
    fn rejects_invalid_lengths() {
        assert!(matches!(
            CztPlan::new(0, 4, Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)),
            Err(CztError::EmptyLength)
        ));
        assert!(matches!(
            CztPlan::new(4, 0, Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)),
            Err(CztError::EmptyLength)
        ));
    }

    #[test]
    fn length_mismatch_is_rejected() {
        let plan = CztPlan::new(
            4,
            4,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, -std::f64::consts::TAU / 8.0),
        )
        .expect("valid plan");
        let bad = Array1::from_vec(vec![Complex64::new(0.0, 0.0); 3]);
        assert!(matches!(plan.forward(&bad), Err(CztError::LengthMismatch)));
        let good = Array1::from_vec(vec![Complex64::new(0.0, 0.0); 4]);
        let mut bad_output = Array1::from_vec(vec![Complex64::new(0.0, 0.0); 3]);
        assert!(matches!(
            plan.forward_into(&good, &mut bad_output),
            Err(CztError::LengthMismatch)
        ));
    }

    #[test]
    fn forward_into_matches_allocating_fast_path() {
        let input = Array1::from_vec(vec![
            Complex64::new(0.25, 0.5),
            Complex64::new(-0.75, 1.0),
            Complex64::new(1.25, -0.25),
            Complex64::new(0.5, 0.125),
            Complex64::new(-0.375, -0.75),
        ]);
        let plan = CztPlan::new(
            input.len(),
            7,
            Complex64::from_polar(1.0, 0.125),
            Complex64::from_polar(1.0, -std::f64::consts::TAU / 11.0),
        )
        .expect("valid plan");

        let expected = plan.forward(&input).expect("allocating fast path");
        let mut actual = Array1::<Complex64>::zeros(plan.output_len());
        plan.forward_into(&input, &mut actual)
            .expect("caller-owned fast path");

        for (lhs, rhs) in expected.iter().zip(actual.iter()) {
            assert_relative_eq!(lhs.re, rhs.re, epsilon = 1.0e-12);
            assert_relative_eq!(lhs.im, rhs.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn forward_into_reuses_plan_convolution_workspace() {
        let input = Array1::from_vec(vec![
            Complex64::new(0.25, -0.5),
            Complex64::new(-0.75, 1.0),
            Complex64::new(1.25, 0.25),
            Complex64::new(0.5, -0.125),
            Complex64::new(-0.375, 0.75),
        ]);
        let plan = CztPlan::new(
            input.len(),
            7,
            Complex64::from_polar(1.0, 0.125),
            Complex64::from_polar(1.0, -std::f64::consts::TAU / 11.0),
        )
        .expect("valid plan");
        let mut first = Array1::<Complex64>::zeros(plan.output_len());
        let mut second = Array1::<Complex64>::zeros(plan.output_len());

        plan.forward_into(&input, &mut first)
            .expect("first caller-owned fast path");
        // After the first transform the per-thread workspace is grown to at
        // least the plan convolution length; a larger capacity is admissible
        // when a wider plan ran earlier on this thread (shared thread-local).
        let capacity_after_first = CztPlan::forward_workspace_capacity();
        assert!(capacity_after_first >= plan.convolution_len());

        plan.forward_into(&input, &mut second)
            .expect("second caller-owned fast path");
        // The workspace is reused, not reallocated, across calls.
        assert_eq!(CztPlan::forward_workspace_capacity(), capacity_after_first);
        for (actual, expected) in second.iter().zip(first.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn typed_paths_support_complex64_complex32_and_mixed_f16_storage() {
        let input64 = Array1::from_vec(vec![
            Complex64::new(0.25, 0.5),
            Complex64::new(-0.75, 1.0),
            Complex64::new(1.25, -0.25),
            Complex64::new(0.5, 0.125),
            Complex64::new(-0.375, -0.75),
        ]);
        let plan = CztPlan::new(
            input64.len(),
            7,
            Complex64::from_polar(1.0, 0.125),
            Complex64::from_polar(1.0, -std::f64::consts::TAU / 11.0),
        )
        .expect("valid plan");
        let expected = plan.forward(&input64).expect("reference");

        let mut out64 = Array1::<Complex64>::zeros(plan.output_len());
        plan.forward_typed_into(&input64, &mut out64, PrecisionProfile::HIGH_ACCURACY_F64)
            .expect("complex64 typed");
        for (actual, expected) in out64.iter().zip(expected.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }

        let input32 = input64.mapv(|value| Complex32::new(value.re as f32, value.im as f32));
        let mut out32 = Array1::<Complex32>::zeros(plan.output_len());
        plan.forward_typed_into(&input32, &mut out32, PrecisionProfile::LOW_PRECISION_F32)
            .expect("complex32 typed");
        for (actual, expected) in out32.iter().zip(expected.iter()) {
            assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
            assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
        }

        let input16 = input64.mapv(|value| {
            [
                f16::from_f32(value.re as f32),
                f16::from_f32(value.im as f32),
            ]
        });
        let mut out16 = Array1::from_elem(plan.output_len(), [f16::from_f32(0.0); 2]);
        plan.forward_typed_into(
            &input16,
            &mut out16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("mixed f16 typed");
        for (actual, expected) in out16.iter().zip(expected.iter()) {
            let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!((f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound);
            assert!((f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound);
        }
    }

    #[test]
    fn typed_path_rejects_profile_storage_mismatch() {
        let plan = CztPlan::new(
            4,
            4,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, -std::f64::consts::TAU / 8.0),
        )
        .expect("valid plan");
        let input = Array1::from_vec(vec![Complex32::new(1.0, 0.0); 4]);
        let mut output = Array1::<Complex32>::zeros(4);
        assert!(matches!(
            plan.forward_typed_into(&input, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
            Err(CztError::PrecisionMismatch)
        ));
    }

    #[test]
    fn typed_complex32_forward_and_inverse_reuse_complex64_workspaces() {
        let n = 5usize;
        let input64 = Array1::from_shape_fn(n, |i| {
            Complex64::new((i as f64 * 0.23).sin(), (i as f64 * 0.41).cos())
        });
        let input32 = input64.mapv(|value| Complex32::new(value.re as f32, value.im as f32));
        let plan = CztPlan::new(
            n,
            n,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64),
        )
        .expect("valid DFT-equivalent CZT plan");
        let mut first_spectrum = Array1::<Complex32>::zeros(n);
        let mut second_spectrum = Array1::<Complex32>::zeros(n);
        let mut first_recovered = Array1::<Complex32>::zeros(n);
        let mut second_recovered = Array1::<Complex32>::zeros(n);

        plan.forward_typed_into(
            &input32,
            &mut first_spectrum,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("first typed forward");
        let forward_caps = typed_scratch_capacities();
        plan.forward_typed_into(
            &input32,
            &mut second_spectrum,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("second typed forward");
        assert_eq!(typed_scratch_capacities(), forward_caps);

        plan.inverse_typed_into(
            &first_spectrum,
            &mut first_recovered,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("first typed inverse");
        let inverse_caps = typed_scratch_capacities();
        plan.inverse_typed_into(
            &first_spectrum,
            &mut second_recovered,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("second typed inverse");
        assert_eq!(typed_scratch_capacities(), inverse_caps);
        assert!(inverse_caps.0 >= n);
        assert!(inverse_caps.1 >= n);

        for ((first, second), original) in first_recovered
            .iter()
            .zip(second_recovered.iter())
            .zip(input32.iter())
        {
            assert!((first.re - second.re).abs() < 1.0e-7);
            assert!((first.im - second.im).abs() < 1.0e-7);
            assert!((first.re - original.re).abs() < 1.0e-4);
            assert!((first.im - original.im).abs() < 1.0e-4);
        }
    }
}

#[cfg(test)]
mod proptest_suite {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Bluestein (1970) convolution equivalence: for any n, m, a, w the
        /// fast path (FFT-based Bluestein convolution) and the direct O(nm)
        /// evaluation must agree to within a relative tolerance of 1e-9.
        ///
        /// # Mathematical justification
        ///
        /// The Bluestein algorithm decomposes the CZT into three FFTs of
        /// length p = 2^⌈log₂(n+m-1)⌉.  Standard FFT rounding-error analysis
        /// (Higham, "Accuracy and Stability", §3.10) gives relative error
        ///
        ///   |fast[k] - exact[k]| / |exact[k]| ≤ C · log₂(p) · ε_machine
        ///
        /// where C is a small implementation constant and ε_machine = 2.2e-16.
        /// For p ≤ 16 (n, m ≤ 8): C · log₂(16) · ε_machine ≈ 12 · 2.2e-16 ≈
        /// 2.6e-15.  The 1e-9 relative threshold provides a ×3.8e5 safety
        /// margin over that bound.
        ///
        /// An absolute threshold is inappropriate when |w| > 1: the chirp
        /// factors w^(n²/2) amplify output magnitudes by up to |w|^((N-1)²/2),
        /// so a fixed 1e-9 absolute tolerance is violated at 3×10⁻⁹ relative
        /// error when |w| ≈ 1.28 and |a| = 0.5.  Scaling by max(|direct[k]|, 1)
        /// restores the invariant for all spiral parameters.
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
            let input = Array1::from_shape_fn(n, |i| {
                Complex64::new(re_parts[i % 8] as f64, im_parts[i % 8] as f64)
            });
            let plan = CztPlan::new(n, m, a, w).expect("plan");
            let fast = plan.forward(&input).expect("fast");
            let direct = plan.forward_direct(&input).expect("direct");
            prop_assert_eq!(fast.len(), direct.len());
            for k in 0..m {
                let diff = (fast[k] - direct[k]).norm();
                // Scale tolerance by max(|direct[k]|, 1.0) to convert the
                // absolute threshold to a relative one.  When the output is
                // amplified by chirp factors (|w|>1), both fast and direct
                // accumulate proportional floating-point error, so the
                // meaningful invariant is a relative error bound.
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
            let input = Array1::from_shape_fn(n, |i| {
                Complex64::new(re_parts[i % 8] as f64, im_parts[i % 8] as f64)
            });
            let plan = CztPlan::new(n, n, a, w).expect("plan");
            let czt_out = plan.forward(&input).expect("czt");
            let fft_out = apollo_fft::fft_1d_complex(&input);
            prop_assert_eq!(czt_out.len(), fft_out.len());
            for k in 0..n {
                let diff = (czt_out[k] - fft_out[k]).norm();
                prop_assert!(
                    diff < 1e-9,
                    "k={k}: |czt - fft| = {diff} >= 1e-9 (n={n})"
                );
            }
        }

        /// Linearity of the CZT sum: for any scalar c,
        /// CZT(c·x)[k] = c · CZT(x)[k].  The scalar factors out of the
        /// finite sum by distributivity of multiplication over addition.
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
            let input = Array1::from_shape_fn(n, |i| {
                Complex64::new(re_parts[i % 8] as f64, im_parts[i % 8] as f64)
            });
            let scaled_input = input.mapv(|v| scalar * v);
            let plan = CztPlan::new(n, m, a, w).expect("plan");
            let czt_scaled = plan.forward(&scaled_input).expect("czt scaled");
            let czt_base = plan.forward(&input).expect("czt base");
            prop_assert_eq!(czt_scaled.len(), czt_base.len());
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
}

#[cfg(test)]
mod inverse_tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Theorem (exact ICZT roundtrip, DFT case):
    /// For A=1, W=exp(-2πi/N), the CZT is the N-point DFT.
    /// `inverse(forward(x)) == x` must hold to machine precision.
    #[test]
    fn inverse_roundtrip_dft_parameters() {
        for n in [2usize, 3, 4, 5, 7, 8, 11, 13, 16] {
            let a = Complex64::new(1.0, 0.0);
            let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64);
            let input: Array1<Complex64> = Array1::from_shape_fn(n, |i| {
                Complex64::new((i as f64 * 0.31 + 0.7).sin(), (i as f64 * 0.17).cos())
            });
            let plan = CztPlan::new(n, n, a, w).expect("DFT plan");
            let spectrum = plan.forward(&input).expect("forward");
            let recovered = plan.inverse(&spectrum).expect("inverse");
            for i in 0..n {
                let err = (recovered[i] - input[i]).norm();
                assert!(err < 1e-10, "n={n} i={i} err={err:.3e}");
            }
        }
    }

    /// Theorem (exact ICZT roundtrip, general A≠1):
    /// For |A|=1 and W=exp(-2πi*θ) with θ irrational, the Vandermonde nodes
    /// z_k = W^k are distinct and Björck-Pereyra gives the exact inverse.
    #[test]
    fn inverse_roundtrip_general_a_parameter() {
        let n = 6usize;
        let a = Complex64::from_polar(1.0, 0.5);
        let w = Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64);
        let input: Array1<Complex64> = Array1::from_shape_fn(n, |i| {
            Complex64::new((i as f64 * 0.53 + 1.1).sin(), (i as f64 * 0.37 - 0.5).cos())
        });
        let plan = CztPlan::new(n, n, a, w).expect("plan");
        let spectrum = plan.forward(&input).expect("forward");
        let recovered = plan.inverse(&spectrum).expect("inverse");
        for i in 0..n {
            assert_abs_diff_eq!(recovered[i].re, input[i].re, epsilon = 1e-9);
            assert_abs_diff_eq!(recovered[i].im, input[i].im, epsilon = 1e-9);
        }
    }

    /// Theorem (exact ICZT roundtrip, |W|≠1):
    /// When |W| > 1 the spiral points z_k = W^k diverge; they are always
    /// distinct since |z_j - z_k| = |W|^k * |W^{j-k} - 1| > 0 for j≠k.
    #[test]
    fn inverse_roundtrip_non_unit_w() {
        let n = 5usize;
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::from_polar(1.1, -0.9);
        let input: Array1<Complex64> = Array1::from_shape_fn(n, |i| {
            Complex64::new((i as f64 * 0.7).cos(), (i as f64 * 0.4 + 0.3).sin())
        });
        let plan = CztPlan::new(n, n, a, w).expect("plan");
        let spectrum = plan.forward(&input).expect("forward");
        let recovered = plan.inverse(&spectrum).expect("inverse");
        for i in 0..n {
            assert_abs_diff_eq!(recovered[i].re, input[i].re, epsilon = 1e-8);
            assert_abs_diff_eq!(recovered[i].im, input[i].im, epsilon = 1e-8);
        }
    }

    /// Rectangular CZT (M ≠ N) must return NotInvertible.
    #[test]
    fn rectangular_czt_is_not_invertible() {
        let plan = CztPlan::new(
            4,
            6,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, -0.7),
        )
        .expect("plan");
        let spectrum = Array1::zeros(6);
        assert!(matches!(
            plan.inverse(&spectrum),
            Err(CztError::NotInvertible { .. })
        ));
    }

    /// Spectrum length mismatch must return LengthMismatch.
    #[test]
    fn inverse_rejects_wrong_spectrum_length() {
        let n = 4usize;
        let plan = CztPlan::new(
            n,
            n,
            Complex64::new(1.0, 0.0),
            Complex64::from_polar(1.0, -std::f64::consts::TAU / n as f64),
        )
        .expect("plan");
        let bad_spectrum = Array1::zeros(n + 1);
        assert!(matches!(
            plan.inverse(&bad_spectrum),
            Err(CztError::LengthMismatch)
        ));
    }
}
