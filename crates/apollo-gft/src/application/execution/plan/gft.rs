//! Reusable graph Fourier transform plan.
//!
//! For an undirected weighted graph with combinatorial Laplacian `L = D - A`,
//! the graph Fourier basis is the orthonormal eigenvector matrix `U` of `L`.
//! The forward transform is `U^T x`; the inverse transform is `U X`.

use crate::domain::contracts::error::{GftError, GftResult};
use crate::domain::graph::adjacency::GraphAdjacency;
use crate::infrastructure::kernel::laplacian::spectral_basis;
use apollo_fft::{f16, PrecisionProfile};
use leto::ArrayView2;
use mnemosyne::scratch::ScratchPool;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

thread_local! {
    static TYPED_INPUT64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    static TYPED_OUTPUT64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}

/// Reusable graph Fourier plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GftPlan {
    n: usize,
    eigenvalues: Vec<f64>,
    basis: Vec<f64>,
}

impl GftPlan {
    /// Create a graph Fourier plan from an undirected weighted adjacency matrix.
    pub fn from_adjacency(adjacency: ArrayView2<'_, f64>) -> GftResult<Self> {
        let graph = GraphAdjacency::from_view(adjacency)?;
        Self::from_graph(&graph)
    }

    /// Create a graph Fourier plan from a validated graph adjacency descriptor.
    pub fn from_graph(graph: &GraphAdjacency) -> GftResult<Self> {
        let basis = spectral_basis(graph);
        Ok(Self {
            n: graph.len(),
            eigenvalues: basis.eigenvalues,
            basis: basis.eigenvectors,
        })
    }

    /// Return graph order.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.n
    }

    /// Return true when graph order is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Return Laplacian eigenvalues.
    #[must_use]
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// Return the column-major graph Fourier basis.
    #[must_use]
    pub fn basis(&self) -> &[f64] {
        &self.basis
    }

    /// Forward graph Fourier transform `U^T x`.
    pub fn forward(&self, signal: &Array1<f64>) -> GftResult<Array1<f64>> {
        let mut output = Array1::zeros(self.n);
        self.forward_into(signal, &mut output)?;
        Ok(output)
    }

    /// Forward graph Fourier transform `U^T x` into caller-owned storage.
    pub fn forward_into(&self, signal: &Array1<f64>, output: &mut Array1<f64>) -> GftResult<()> {
        self.forward_f64_slice_into(
            signal.as_slice().expect("GFT signal must be contiguous"),
            output
                .as_slice_mut()
                .expect("GFT output must be contiguous"),
        )
    }

    /// Forward graph Fourier transform `U^T x` over contiguous f64 slices.
    pub(crate) fn forward_f64_slice_into(
        &self,
        signal: &[f64],
        output: &mut [f64],
    ) -> GftResult<()> {
        if signal.len() != self.n || output.len() != self.n {
            return Err(GftError::LengthMismatch);
        }
        for (k, slot) in output.iter_mut().enumerate() {
            *slot = (0..self.n)
                .map(|i| self.basis[i + k * self.n] * signal[i])
                .sum();
        }
        Ok(())
    }

    /// Forward GFT for `f64`, `f32`, or mixed `f16` storage.
    ///
    /// The mathematical owner path is the real `f64` graph-basis multiply.
    /// Lower storage profiles convert input into the owner representation and
    /// quantize once when writing the caller-owned output.
    pub fn forward_typed_into<T: GftStorage>(
        &self,
        signal: &Array1<T>,
        output: &mut Array1<T>,
        profile: PrecisionProfile,
    ) -> GftResult<()> {
        T::forward_into(self, signal, output, profile)
    }

    /// Inverse GFT for `f64`, `f32`, or mixed `f16` storage.
    pub fn inverse_typed_into<T: GftStorage>(
        &self,
        spectrum: &Array1<T>,
        output: &mut Array1<T>,
        profile: PrecisionProfile,
    ) -> GftResult<()> {
        T::inverse_into(self, spectrum, output, profile)
    }

    /// Inverse graph Fourier transform `U X`.
    pub fn inverse(&self, spectrum: &Array1<f64>) -> GftResult<Array1<f64>> {
        let mut output = Array1::zeros(self.n);
        self.inverse_into(spectrum, &mut output)?;
        Ok(output)
    }

    /// Inverse graph Fourier transform `U X` into caller-owned storage.
    pub fn inverse_into(&self, spectrum: &Array1<f64>, output: &mut Array1<f64>) -> GftResult<()> {
        self.inverse_f64_slice_into(
            spectrum
                .as_slice()
                .expect("GFT spectrum must be contiguous"),
            output
                .as_slice_mut()
                .expect("GFT output must be contiguous"),
        )
    }

    /// Inverse graph Fourier transform `U X` over contiguous f64 slices.
    pub(crate) fn inverse_f64_slice_into(
        &self,
        spectrum: &[f64],
        output: &mut [f64],
    ) -> GftResult<()> {
        if spectrum.len() != self.n || output.len() != self.n {
            return Err(GftError::LengthMismatch);
        }
        for (i, slot) in output.iter_mut().enumerate() {
            *slot = (0..self.n)
                .map(|k| self.basis[i + k * self.n] * spectrum[k])
                .sum();
        }
        Ok(())
    }
}

/// Real storage accepted by typed GFT paths.
pub trait GftStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage into the owner `f64` arithmetic path.
    fn to_f64(self) -> f64;

    /// Convert owner arithmetic result back to storage.
    fn from_f64(value: f64) -> Self;

    /// Execute forward transform into caller-owned storage.
    fn forward_into(
        plan: &GftPlan,
        signal: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> GftResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if signal.len() != plan.n || output.len() != plan.n {
            return Err(GftError::LengthMismatch);
        }
        with_f64_workspaces(plan.n, |input64, output64| {
            for (slot, value) in input64.iter_mut().zip(signal.iter().copied()) {
                *slot = Self::to_f64(value);
            }
            plan.forward_f64_slice_into(input64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                *slot = Self::from_f64(value);
            }
            Ok(())
        })
    }

    /// Execute inverse transform into caller-owned storage.
    fn inverse_into(
        plan: &GftPlan,
        spectrum: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> GftResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if spectrum.len() != plan.n || output.len() != plan.n {
            return Err(GftError::LengthMismatch);
        }
        with_f64_workspaces(plan.n, |input64, output64| {
            for (slot, value) in input64.iter_mut().zip(spectrum.iter().copied()) {
                *slot = Self::to_f64(value);
            }
            plan.inverse_f64_slice_into(input64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                *slot = Self::from_f64(value);
            }
            Ok(())
        })
    }
}

impl GftStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn forward_into(
        plan: &GftPlan,
        signal: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> GftResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_into(signal, output)
    }

    fn inverse_into(
        plan: &GftPlan,
        spectrum: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> GftResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.inverse_into(spectrum, output)
    }
}

impl GftStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl GftStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    fn from_f64(value: f64) -> Self {
        f16::from_f32(value as f32)
    }
}

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> GftResult<()> {
    if actual.storage == expected.storage && actual.compute == expected.compute {
        Ok(())
    } else {
        Err(GftError::PrecisionMismatch)
    }
}

fn with_f64_workspaces<R>(n: usize, f: impl FnOnce(&mut [f64], &mut [f64]) -> R) -> R {
    TYPED_INPUT64_SCRATCH.with(|in_pool| {
        in_pool.with_scratch(n, |input64| {
            TYPED_OUTPUT64_SCRATCH
                .with(|out_pool| out_pool.with_scratch(n, |output64| f(input64, output64)))
        })
    })
}

#[cfg(test)]
pub(crate) fn typed_scratch_capacities() -> (usize, usize) {
    TYPED_INPUT64_SCRATCH.with(|in_pool| {
        TYPED_OUTPUT64_SCRATCH.with(|out_pool| (in_pool.capacity(), out_pool.capacity()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn path_three_plan() -> GftPlan {
        let adjacency =
            leto::Array2::from_shape_vec([3, 3], vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
                .expect("valid adjacency shape");
        GftPlan::from_adjacency(adjacency.view()).expect("valid path graph")
    }

    #[test]
    fn caller_owned_forward_and_inverse_match_allocating_paths() {
        let plan = path_three_plan();
        let signal = Array1::from_vec(vec![1.25, -0.5, 2.0]);
        let spectrum = plan.forward(&signal).expect("forward");
        let mut forward_output = Array1::<f64>::zeros(plan.len());
        plan.forward_into(&signal, &mut forward_output)
            .expect("caller-owned forward");
        for (actual, expected) in forward_output.iter().zip(spectrum.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }

        let recovered = plan.inverse(&spectrum).expect("inverse");
        let mut inverse_output = Array1::<f64>::zeros(plan.len());
        plan.inverse_into(&spectrum, &mut inverse_output)
            .expect("caller-owned inverse");
        for ((actual, expected), original) in inverse_output
            .iter()
            .zip(recovered.iter())
            .zip(signal.iter())
        {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual, original, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn typed_paths_support_f64_f32_and_mixed_f16_storage() {
        let plan = path_three_plan();
        let signal64 = Array1::from_vec(vec![1.25, -0.5, 2.0]);
        let expected = plan.forward(&signal64).expect("forward");

        let mut out64 = Array1::<f64>::zeros(plan.len());
        plan.forward_typed_into(&signal64, &mut out64, PrecisionProfile::HIGH_ACCURACY_F64)
            .expect("typed f64 forward");
        for (actual, expected) in out64.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }

        let signal32 = signal64.mapv(|value| value as f32);
        let mut out32 = Array1::<f32>::zeros(plan.len());
        plan.forward_typed_into(&signal32, &mut out32, PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed f32 forward");
        for (actual, expected) in out32.iter().zip(expected.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-5);
        }

        let signal16 = signal64.mapv(|value| f16::from_f32(value as f32));
        let expected16_input =
            Array1::from_iter(signal16.iter().map(|value| f64::from(value.to_f32())));
        let expected16 = plan
            .forward(&expected16_input)
            .expect("f16 represented forward");
        let mut out16 = Array1::from_elem(plan.len(), f16::from_f32(0.0));
        plan.forward_typed_into(
            &signal16,
            &mut out16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed mixed f16 forward");
        for (actual, expected) in out16.iter().zip(expected16.iter()) {
            let quantization_bound = expected.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!((f64::from(actual.to_f32()) - *expected).abs() <= quantization_bound);
        }

        let mut recovered32 = Array1::<f32>::zeros(plan.len());
        plan.inverse_typed_into(
            &out32,
            &mut recovered32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed f32 inverse");
        for (actual, expected) in recovered32.iter().zip(signal32.iter()) {
            assert!((*actual - *expected).abs() < 1.0e-5);
        }
    }

    #[test]
    fn typed_f32_paths_reuse_f64_workspaces() {
        let plan = path_three_plan();
        let signal = Array1::from_vec(vec![1.25_f32, -0.5, 2.0]);
        let mut first_spectrum = Array1::<f32>::zeros(plan.len());
        let mut second_spectrum = Array1::<f32>::zeros(plan.len());
        let mut first_recovered = Array1::<f32>::zeros(plan.len());
        let mut second_recovered = Array1::<f32>::zeros(plan.len());

        plan.forward_typed_into(
            &signal,
            &mut first_spectrum,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("first typed f32 forward");
        let forward_caps = typed_scratch_capacities();
        plan.forward_typed_into(
            &signal,
            &mut second_spectrum,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("second typed f32 forward");

        assert_eq!(typed_scratch_capacities(), forward_caps);
        assert!(forward_caps.0 >= plan.len());
        assert!(forward_caps.1 >= plan.len());
        for (actual, expected) in second_spectrum.iter().zip(first_spectrum.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 0.0);
        }

        plan.inverse_typed_into(
            &first_spectrum,
            &mut first_recovered,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("first typed f32 inverse");
        let inverse_caps = typed_scratch_capacities();
        plan.inverse_typed_into(
            &first_spectrum,
            &mut second_recovered,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("second typed f32 inverse");

        assert_eq!(typed_scratch_capacities(), inverse_caps);
        assert!(inverse_caps.0 >= plan.len());
        assert!(inverse_caps.1 >= plan.len());
        for ((actual, expected), original) in second_recovered
            .iter()
            .zip(first_recovered.iter())
            .zip(signal.iter())
        {
            assert_abs_diff_eq!(actual, expected, epsilon = 0.0);
            assert!((*actual - *original).abs() < 1.0e-5);
        }
    }

    #[test]
    fn typed_path_rejects_profile_storage_mismatch() {
        let plan = path_three_plan();
        let signal = Array1::from_vec(vec![1.0_f32, -2.0, 0.5]);
        let mut output = Array1::<f32>::zeros(plan.len());
        assert!(matches!(
            plan.forward_typed_into(&signal, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
            Err(GftError::PrecisionMismatch)
        ));
    }
}
