//! Reusable dense quantum Fourier transform plan.
//!
//! For a state vector x in C^n, the forward QFT is
//! X_k = (1/sqrt(n)) sum_j x_j exp(2*pi*i*j*k/n). The inverse is the
//! conjugate transpose with the negative phase. Both maps are unitary.
//!
//! Twiddle factors exp(2*pi*i*k/n) for k=0..n are precomputed at plan
//! construction time and reused across all forward and inverse calls.

use crate::domain::contracts::error::{QftError, QftResult};
use crate::domain::state::dimension::QuantumStateDimension;
use crate::infrastructure::kernel::dense::{qft_forward_dense_into, qft_inverse_dense_into};
use apollo_fft::{f16, PrecisionProfile};
use ndarray::Array1;
use num_complex::{Complex32, Complex64};
use serde::{Deserialize, Serialize};
use mnemosyne::scratch::ScratchPool;

thread_local! {
    static TYPED_INPUT64_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
    static TYPED_OUTPUT64_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
}

/// Reusable QFT plan with precomputed twiddle factors.
///
/// `twiddles[k] = exp(2*pi*i*k/n)` for `k = 0..n`. The kernel indexes
/// `twiddles[(row*col) % n]` to obtain `exp(2*pi*i*row*col/n)` without
/// trigonometric evaluation per transform element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QftPlan {
    dimension: QuantumStateDimension,
    /// Precomputed twiddle factors: twiddles[k] = exp(2*pi*i*k/n).
    twiddles: Vec<Complex64>,
}

impl QftPlan {
    /// Create a QFT plan for a validated quantum state dimension.
    pub fn new(dimension: QuantumStateDimension) -> Self {
        let n = dimension.len();
        let twiddles: Vec<Complex64> = (0..n)
            .map(|k| {
                let angle = std::f64::consts::TAU * k as f64 / n as f64;
                Complex64::new(angle.cos(), angle.sin())
            })
            .collect();
        Self {
            dimension,
            twiddles,
        }
    }

    /// Return the plan length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.dimension.len()
    }

    /// Return true when the plan length is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dimension.is_empty()
    }

    /// Forward QFT of a complex amplitude vector.
    pub fn forward(&self, input: &Array1<Complex64>) -> QftResult<Array1<Complex64>> {
        let mut output = Array1::zeros(self.len());
        self.forward_into(input, &mut output)?;
        Ok(output)
    }

    /// Forward QFT into caller-owned storage.
    pub fn forward_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<Complex64>,
    ) -> QftResult<()> {
        self.forward_complex64_slice_into(
            input.as_slice().expect("QFT input must be contiguous"),
            output
                .as_slice_mut()
                .expect("QFT output must be contiguous"),
        )
    }

    /// Forward QFT over contiguous Complex64 slices.
    pub(crate) fn forward_complex64_slice_into(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> QftResult<()> {
        if input.len() != self.len() || output.len() != self.len() {
            return Err(QftError::LengthMismatch);
        }
        qft_forward_dense_into(input, output, &self.twiddles);
        Ok(())
    }

    /// Forward QFT for `Complex64`, `Complex32`, or mixed two-lane `f16` storage.
    pub fn forward_typed_into<T: QftStorage>(
        &self,
        input: &Array1<T>,
        output: &mut Array1<T>,
        profile: PrecisionProfile,
    ) -> QftResult<()> {
        T::forward_into(self, input, output, profile)
    }

    /// Inverse QFT of a complex amplitude vector.
    pub fn inverse(&self, input: &Array1<Complex64>) -> QftResult<Array1<Complex64>> {
        let mut output = Array1::zeros(self.len());
        self.inverse_into(input, &mut output)?;
        Ok(output)
    }

    /// Inverse QFT into caller-owned storage.
    pub fn inverse_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<Complex64>,
    ) -> QftResult<()> {
        self.inverse_complex64_slice_into(
            input.as_slice().expect("QFT input must be contiguous"),
            output
                .as_slice_mut()
                .expect("QFT output must be contiguous"),
        )
    }

    /// Inverse QFT over contiguous Complex64 slices.
    pub(crate) fn inverse_complex64_slice_into(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> QftResult<()> {
        if input.len() != self.len() || output.len() != self.len() {
            return Err(QftError::LengthMismatch);
        }
        qft_inverse_dense_into(input, output, &self.twiddles);
        Ok(())
    }

    /// Inverse QFT for `Complex64`, `Complex32`, or mixed two-lane `f16` storage.
    pub fn inverse_typed_into<T: QftStorage>(
        &self,
        input: &Array1<T>,
        output: &mut Array1<T>,
        profile: PrecisionProfile,
    ) -> QftResult<()> {
        T::inverse_into(self, input, output, profile)
    }

    /// Forward QFT executed in place.
    pub fn forward_inplace(&self, data: &mut Array1<Complex64>) -> QftResult<()> {
        let transformed = self.forward(data)?;
        *data = transformed;
        Ok(())
    }

    /// Inverse QFT executed in place.
    pub fn inverse_inplace(&self, data: &mut Array1<Complex64>) -> QftResult<()> {
        let transformed = self.inverse(data)?;
        *data = transformed;
        Ok(())
    }
}

/// Complex storage accepted by typed QFT paths.
pub trait QftStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage into the owner `Complex64` arithmetic path.
    fn to_complex64(self) -> Complex64;

    /// Convert owner arithmetic result back to storage.
    fn from_complex64(value: Complex64) -> Self;

    /// Execute forward transform into caller-owned storage.
    fn forward_into(
        plan: &QftPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> QftResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if input.len() != plan.len() || output.len() != plan.len() {
            return Err(QftError::LengthMismatch);
        }
        with_complex64_workspaces(plan.len(), |input64, output64| {
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
        plan: &QftPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> QftResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if input.len() != plan.len() || output.len() != plan.len() {
            return Err(QftError::LengthMismatch);
        }
        with_complex64_workspaces(plan.len(), |input64, output64| {
            for (slot, value) in input64.iter_mut().zip(input.iter().copied()) {
                *slot = Self::to_complex64(value);
            }
            plan.inverse_complex64_slice_into(input64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                *slot = Self::from_complex64(value);
            }
            Ok(())
        })
    }
}

impl QftStorage for Complex64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_complex64(self) -> Complex64 {
        self
    }

    fn from_complex64(value: Complex64) -> Self {
        value
    }

    fn forward_into(
        plan: &QftPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> QftResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_into(input, output)
    }

    fn inverse_into(
        plan: &QftPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> QftResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.inverse_into(input, output)
    }
}

impl QftStorage for Complex32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_complex64(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }

    fn from_complex64(value: Complex64) -> Self {
        Complex32::new(value.re as f32, value.im as f32)
    }
}

impl QftStorage for [f16; 2] {
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

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> QftResult<()> {
    if actual.storage == expected.storage && actual.compute == expected.compute {
        Ok(())
    } else {
        Err(QftError::PrecisionMismatch)
    }
}

fn with_complex64_workspaces<R>(
    n: usize,
    f: impl FnOnce(&mut [Complex64], &mut [Complex64]) -> R,
) -> R {
    TYPED_INPUT64_SCRATCH.with(|input_scratch| {
        input_scratch.with_scratch(n, |input64| {
            TYPED_OUTPUT64_SCRATCH.with(|output_scratch| {
                output_scratch.with_scratch(n, |output64| {
                    f(input64, output64)
                })
            })
        })
    })
}

#[cfg(test)]
pub(crate) fn typed_scratch_capacities() -> (usize, usize) {
    TYPED_INPUT64_SCRATCH.with(|input_scratch| {
        TYPED_OUTPUT64_SCRATCH.with(|output_scratch| {
            (
                input_scratch.capacity(),
                output_scratch.capacity(),
            )
        })
    })
}

/// Convenience wrapper for forward QFT.
pub fn qft(input: &Array1<Complex64>) -> QftResult<Array1<Complex64>> {
    QftPlan::new(QuantumStateDimension::new(input.len())?).forward(input)
}

/// Convenience wrapper for inverse QFT.
pub fn iqft(input: &Array1<Complex64>) -> QftResult<Array1<Complex64>> {
    QftPlan::new(QuantumStateDimension::new(input.len())?).inverse(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn plan4() -> QftPlan {
        QftPlan::new(QuantumStateDimension::new(4).expect("valid dimension"))
    }

    fn input64() -> Array1<Complex64> {
        Array1::from_vec(vec![
            Complex64::new(1.0, -0.5),
            Complex64::new(0.25, 0.75),
            Complex64::new(-0.5, 1.25),
            Complex64::new(1.5, 0.0),
        ])
    }

    #[test]
    fn caller_owned_forward_and_inverse_match_allocating_paths() {
        let plan = plan4();
        let input = input64();
        let expected = plan.forward(&input).expect("forward");
        let mut forward_output = Array1::<Complex64>::zeros(plan.len());
        plan.forward_into(&input, &mut forward_output)
            .expect("caller-owned forward");
        for (actual, expected) in forward_output.iter().zip(expected.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }

        let recovered = plan.inverse(&expected).expect("inverse");
        let mut inverse_output = Array1::<Complex64>::zeros(plan.len());
        plan.inverse_into(&expected, &mut inverse_output)
            .expect("caller-owned inverse");
        for ((actual, expected), original) in inverse_output
            .iter()
            .zip(recovered.iter())
            .zip(input.iter())
        {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
            assert_relative_eq!(actual.re, original.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, original.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn typed_paths_support_complex64_complex32_and_mixed_f16_storage() {
        let plan = plan4();
        let input = input64();
        let expected = plan.forward(&input).expect("forward");

        let mut out64 = Array1::<Complex64>::zeros(plan.len());
        plan.forward_typed_into(&input, &mut out64, PrecisionProfile::HIGH_ACCURACY_F64)
            .expect("typed complex64 forward");
        for (actual, expected) in out64.iter().zip(expected.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }

        let input32 = input.mapv(|value| Complex32::new(value.re as f32, value.im as f32));
        let represented32 =
            Array1::from_iter(input32.iter().copied().map(QftStorage::to_complex64));
        let expected32 = plan
            .forward(&represented32)
            .expect("represented f32 forward");
        let mut out32 = Array1::<Complex32>::zeros(plan.len());
        plan.forward_typed_into(&input32, &mut out32, PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed complex32 forward");
        for (actual, expected) in out32.iter().zip(expected32.iter()) {
            assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
            assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
        }

        let input16 = input.mapv(|value| {
            [
                f16::from_f32(value.re as f32),
                f16::from_f32(value.im as f32),
            ]
        });
        let represented16 =
            Array1::from_iter(input16.iter().copied().map(QftStorage::to_complex64));
        let expected16 = plan
            .forward(&represented16)
            .expect("represented f16 forward");
        let mut out16 = Array1::from_elem(plan.len(), [f16::from_f32(0.0); 2]);
        plan.forward_typed_into(
            &input16,
            &mut out16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed mixed f16 forward");
        for (actual, expected) in out16.iter().zip(expected16.iter()) {
            let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!((f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound);
            assert!((f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound);
        }

        let mut recovered32 = Array1::<Complex32>::zeros(plan.len());
        plan.inverse_typed_into(
            &out32,
            &mut recovered32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed complex32 inverse");
        for (actual, expected) in recovered32.iter().zip(input32.iter()) {
            assert!((actual.re - expected.re).abs() < 1.0e-5);
            assert!((actual.im - expected.im).abs() < 1.0e-5);
        }
    }

    #[test]
    fn typed_complex32_paths_reuse_complex64_workspaces() {
        let plan = plan4();
        let input = input64().mapv(|value| Complex32::new(value.re as f32, value.im as f32));
        let mut first_spectrum = Array1::<Complex32>::zeros(plan.len());
        let mut second_spectrum = Array1::<Complex32>::zeros(plan.len());
        let mut first_recovered = Array1::<Complex32>::zeros(plan.len());
        let mut second_recovered = Array1::<Complex32>::zeros(plan.len());

        plan.forward_typed_into(
            &input,
            &mut first_spectrum,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("first typed complex32 forward");
        let forward_caps = typed_scratch_capacities();
        plan.forward_typed_into(
            &input,
            &mut second_spectrum,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("second typed complex32 forward");

        assert_eq!(typed_scratch_capacities(), forward_caps);
        assert!(forward_caps.0 >= plan.len());
        assert!(forward_caps.1 >= plan.len());
        for (actual, expected) in second_spectrum.iter().zip(first_spectrum.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 0.0);
            assert_relative_eq!(actual.im, expected.im, epsilon = 0.0);
        }

        plan.inverse_typed_into(
            &first_spectrum,
            &mut first_recovered,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("first typed complex32 inverse");
        let inverse_caps = typed_scratch_capacities();
        plan.inverse_typed_into(
            &first_spectrum,
            &mut second_recovered,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("second typed complex32 inverse");

        assert_eq!(typed_scratch_capacities(), inverse_caps);
        assert!(inverse_caps.0 >= plan.len());
        assert!(inverse_caps.1 >= plan.len());
        for ((actual, expected), original) in second_recovered
            .iter()
            .zip(first_recovered.iter())
            .zip(input.iter())
        {
            assert_relative_eq!(actual.re, expected.re, epsilon = 0.0);
            assert_relative_eq!(actual.im, expected.im, epsilon = 0.0);
            assert!((actual.re - original.re).abs() < 1.0e-5);
            assert!((actual.im - original.im).abs() < 1.0e-5);
        }
    }

    #[test]
    fn typed_path_rejects_profile_storage_mismatch() {
        let plan = plan4();
        let input = Array1::from_vec(vec![Complex32::new(1.0, 0.0); 4]);
        let mut output = Array1::<Complex32>::zeros(plan.len());
        assert!(matches!(
            plan.forward_typed_into(&input, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
            Err(QftError::PrecisionMismatch)
        ));
    }
}
