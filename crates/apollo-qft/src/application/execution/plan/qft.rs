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
use eunomia::{Complex32, Complex64};
use leto::Array1;
use mnemosyne::scratch::ScratchPool;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

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
        let mut output = Array1::zeros([self.len()]);
        self.forward_into(input, &mut output)?;
        Ok(output)
    }

    /// Forward QFT over a Leto complex amplitude view.
    ///
    /// Contiguous views are borrowed. Strided views copy once into logical order
    /// before entering the canonical slice execution path.
    pub fn forward_leto(
        &self,
        input: leto::ArrayView1<'_, Complex64>,
    ) -> QftResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let signal = leto_view1_cow(&input);
        let mut output = vec![Complex64::new(0.0, 0.0); self.len()];
        self.forward_complex64_slice_into(&signal, &mut output)?;
        Ok(
            leto::Array::<Complex64, leto::MnemosyneStorage<Complex64>, 1>::from_mnemosyne_vec(
                [output.len()],
                output,
            )
            .expect("QFT output length must match Leto output shape"),
        )
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

    /// Forward QFT over a typed Leto complex amplitude view.
    pub fn forward_leto_typed<T: QftStorage>(
        &self,
        input: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> QftResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let signal = leto_view1_cow(&input);
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); self.len()];
        T::forward_slice_into(self, &signal, &mut output, profile)?;
        Ok(
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::from_mnemosyne_vec(
                [output.len()],
                output,
            )
            .expect("typed QFT output length must match Leto output shape"),
        )
    }

    /// Inverse QFT of a complex amplitude vector.
    pub fn inverse(&self, input: &Array1<Complex64>) -> QftResult<Array1<Complex64>> {
        let mut output = Array1::zeros([self.len()]);
        self.inverse_into(input, &mut output)?;
        Ok(output)
    }

    /// Inverse QFT over a Leto complex amplitude view.
    ///
    /// Contiguous views are borrowed. Strided views copy once into logical order
    /// before entering the canonical slice execution path.
    pub fn inverse_leto(
        &self,
        input: leto::ArrayView1<'_, Complex64>,
    ) -> QftResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let signal = leto_view1_cow(&input);
        let mut output = vec![Complex64::new(0.0, 0.0); self.len()];
        self.inverse_complex64_slice_into(&signal, &mut output)?;
        Ok(
            leto::Array::<Complex64, leto::MnemosyneStorage<Complex64>, 1>::from_mnemosyne_vec(
                [output.len()],
                output,
            )
            .expect("inverse QFT output length must match Leto output shape"),
        )
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

    /// Inverse QFT over a typed Leto complex amplitude view.
    pub fn inverse_leto_typed<T: QftStorage>(
        &self,
        input: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> QftResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let signal = leto_view1_cow(&input);
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); self.len()];
        T::inverse_slice_into(self, &signal, &mut output, profile)?;
        Ok(
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::from_mnemosyne_vec(
                [output.len()],
                output,
            )
            .expect("typed inverse QFT output length must match Leto output shape"),
        )
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

    /// View slice as `Complex32` if layout is identical.
    #[inline]
    fn as_c32_slice(slice: &[Self]) -> Option<&[Complex32]> {
        let _ = slice;
        None
    }

    /// View mutable slice as `Complex32` if layout is identical.
    #[inline]
    fn as_c32_slice_mut(slice: &mut [Self]) -> Option<&mut [Complex32]> {
        let _ = slice;
        None
    }

    /// Execute forward transform into caller-owned contiguous storage.
    fn forward_slice_into(
        plan: &QftPlan,
        input: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> QftResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        validate_lengths(plan, input.len(), output.len())?;
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

    /// Execute forward transform into caller-owned Leto storage.
    fn forward_into(
        plan: &QftPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> QftResult<()> {
        Self::forward_slice_into(
            plan,
            input.as_slice().expect("QFT input must be contiguous"),
            output
                .as_slice_mut()
                .expect("QFT output must be contiguous"),
            profile,
        )
    }

    /// Execute inverse transform into caller-owned contiguous storage.
    fn inverse_slice_into(
        plan: &QftPlan,
        input: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> QftResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        validate_lengths(plan, input.len(), output.len())?;
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

    /// Execute inverse transform into caller-owned Leto storage.
    fn inverse_into(
        plan: &QftPlan,
        input: &Array1<Self>,
        output: &mut Array1<Self>,
        profile: PrecisionProfile,
    ) -> QftResult<()> {
        Self::inverse_slice_into(
            plan,
            input.as_slice().expect("QFT input must be contiguous"),
            output
                .as_slice_mut()
                .expect("QFT output must be contiguous"),
            profile,
        )
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

    fn forward_slice_into(
        plan: &QftPlan,
        input: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> QftResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_complex64_slice_into(input, output)
    }

    fn inverse_slice_into(
        plan: &QftPlan,
        input: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> QftResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.inverse_complex64_slice_into(input, output)
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

    #[inline]
    fn as_c32_slice(slice: &[Self]) -> Option<&[Complex32]> {
        Some(slice)
    }

    #[inline]
    fn as_c32_slice_mut(slice: &mut [Self]) -> Option<&mut [Complex32]> {
        Some(slice)
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
    if apollo_fft::application::utilities::leto_interop::profile_matches(actual, expected) {
        Ok(())
    } else {
        Err(QftError::PrecisionMismatch)
    }
}

fn validate_lengths(plan: &QftPlan, input: usize, output: usize) -> QftResult<()> {
    if input == plan.len() && output == plan.len() {
        Ok(())
    } else {
        Err(QftError::LengthMismatch)
    }
}

fn with_complex64_workspaces<R>(
    n: usize,
    f: impl FnOnce(&mut [Complex64], &mut [Complex64]) -> R,
) -> R {
    TYPED_INPUT64_SCRATCH.with(|input_scratch| {
        input_scratch.with_scratch(n, |input64| {
            TYPED_OUTPUT64_SCRATCH.with(|output_scratch| {
                output_scratch.with_scratch(n, |output64| f(input64, output64))
            })
        })
    })
}

#[cfg(test)]
pub(crate) fn typed_scratch_capacities() -> (usize, usize) {
    TYPED_INPUT64_SCRATCH.with(|input_scratch| {
        TYPED_OUTPUT64_SCRATCH
            .with(|output_scratch| (input_scratch.capacity(), output_scratch.capacity()))
    })
}

fn leto_view1_cow<'a, T: Copy>(view: &leto::ArrayView1<'a, T>) -> Cow<'a, [T]> {
    apollo_fft::application::utilities::leto_interop::view1_cow(view)
}

/// Convenience wrapper for forward QFT.
pub fn qft(input: &Array1<Complex64>) -> QftResult<Array1<Complex64>> {
    QftPlan::new(QuantumStateDimension::new(input.size())?).forward(input)
}

/// Convenience wrapper for forward QFT over a Leto view.
pub fn qft_leto(
    input: leto::ArrayView1<'_, Complex64>,
) -> QftResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
    QftPlan::new(QuantumStateDimension::new(input.shape()[0])?).forward_leto(input)
}

/// Convenience wrapper for inverse QFT.
pub fn iqft(input: &Array1<Complex64>) -> QftResult<Array1<Complex64>> {
    QftPlan::new(QuantumStateDimension::new(input.size())?).inverse(input)
}

/// Convenience wrapper for inverse QFT over a Leto view.
pub fn iqft_leto(
    input: leto::ArrayView1<'_, Complex64>,
) -> QftResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
    QftPlan::new(QuantumStateDimension::new(input.shape()[0])?).inverse_leto(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn plan4() -> QftPlan {
        QftPlan::new(QuantumStateDimension::new(4).expect("valid dimension"))
    }

    fn input64() -> Array1<Complex64> {
        Array1::from(vec![
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
        let mut forward_output = Array1::<Complex64>::zeros([plan.len()]);
        plan.forward_into(&input, &mut forward_output)
            .expect("caller-owned forward");
        for (actual, expected) in forward_output.iter().zip(expected.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }

        let recovered = plan.inverse(&expected).expect("inverse");
        let mut inverse_output = Array1::<Complex64>::zeros([plan.len()]);
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
    fn leto_forward_and_inverse_match_leto_path() {
        use leto::Storage;

        let plan = plan4();
        let signal = input64().into_vec();
        let owned_input = Array1::from(signal.clone());
        let leto_input = leto::Array1::from_shape_vec([plan.len()], signal).expect("leto input");

        let leto_forward = plan.forward_leto(leto_input.view()).expect("leto forward");
        let owned_forward = plan.forward(&owned_input).expect("owned forward");
        for (actual, expected) in leto_forward
            .storage()
            .as_slice()
            .iter()
            .zip(owned_forward.iter())
        {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }

        let leto_inverse = plan
            .inverse_leto(leto_forward.view())
            .expect("leto inverse");
        let owned_inverse = plan.inverse(&owned_forward).expect("owned inverse");
        for (actual, expected) in leto_inverse
            .storage()
            .as_slice()
            .iter()
            .zip(owned_inverse.iter())
        {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_forward_accepts_strided_logical_view() {
        use leto::{SliceArg, Storage};

        let logical = input64().into_vec();
        let interleaved = logical
            .iter()
            .copied()
            .flat_map(|value| [value, Complex64::new(99.0, -99.0)])
            .collect::<Vec<_>>();
        let leto_input =
            leto::Array1::from_shape_vec([interleaved.len()], interleaved).expect("leto input");
        let strided = leto_input
            .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
            .expect("strided view");

        let actual = qft_leto(strided).expect("leto qft");
        let expected = qft(&Array1::from(logical)).expect("leto qft");
        for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_typed_complex32_matches_leto_typed_path() {
        use leto::Storage;

        let plan = plan4();
        let input = input64().mapv(|value| Complex32::new(value.re as f32, value.im as f32));
        let leto_input =
            leto::Array1::from_shape_vec([plan.len()], input.iter().copied().collect())
                .expect("leto input");
        let mut expected = Array1::<Complex32>::zeros([plan.len()]);
        plan.forward_typed_into(&input, &mut expected, PrecisionProfile::LOW_PRECISION_F32)
            .expect("leto typed forward");

        let actual = plan
            .forward_leto_typed(leto_input.view(), PrecisionProfile::LOW_PRECISION_F32)
            .expect("leto typed forward");
        for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 0.0);
            assert_relative_eq!(actual.im, expected.im, epsilon = 0.0);
        }
    }

    #[test]
    fn leto_typed_strided_f16_matches_leto_typed_path() {
        use leto::{SliceArg, Storage};

        let plan = plan4();
        let input = input64().mapv(|value| {
            [
                f16::from_f32(value.re as f32),
                f16::from_f32(value.im as f32),
            ]
        });
        let interleaved = input
            .iter()
            .copied()
            .flat_map(|value| [value, [f16::from_f32(99.0), f16::from_f32(-99.0)]])
            .collect::<Vec<_>>();
        let leto_input =
            leto::Array1::from_shape_vec([interleaved.len()], interleaved).expect("leto input");
        let strided = leto_input
            .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
            .expect("strided view");
        let mut expected = Array1::from_elem([plan.len()], [f16::from_f32(0.0); 2]);
        plan.forward_typed_into(
            &input,
            &mut expected,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("leto typed forward");

        let actual = plan
            .forward_leto_typed(strided, PrecisionProfile::MIXED_PRECISION_F16_F32)
            .expect("leto typed forward");
        for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
            assert_relative_eq!(actual[0].to_f32(), expected[0].to_f32(), epsilon = 0.0);
            assert_relative_eq!(actual[1].to_f32(), expected[1].to_f32(), epsilon = 0.0);
        }
    }

    #[test]
    fn typed_paths_support_complex64_complex32_and_mixed_f16_storage() {
        let plan = plan4();
        let input = input64();
        let expected = plan.forward(&input).expect("forward");

        let mut out64 = Array1::<Complex64>::zeros([plan.len()]);
        plan.forward_typed_into(&input, &mut out64, PrecisionProfile::HIGH_ACCURACY_F64)
            .expect("typed complex64 forward");
        for (actual, expected) in out64.iter().zip(expected.iter()) {
            assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }

        let input32 = input.mapv(|value| Complex32::new(value.re as f32, value.im as f32));
        let represented32 = Array1::from(
            (input32.iter().copied().map(QftStorage::to_complex64)).collect::<Vec<_>>(),
        );
        let expected32 = plan
            .forward(&represented32)
            .expect("represented f32 forward");
        let mut out32 = Array1::<Complex32>::zeros([plan.len()]);
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
        let represented16 = Array1::from(
            (input16.iter().copied().map(QftStorage::to_complex64)).collect::<Vec<_>>(),
        );
        let expected16 = plan
            .forward(&represented16)
            .expect("represented f16 forward");
        let mut out16 = Array1::from_elem([plan.len()], [f16::from_f32(0.0); 2]);
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

        let mut recovered32 = Array1::<Complex32>::zeros([plan.len()]);
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
        let mut first_spectrum = Array1::<Complex32>::zeros([plan.len()]);
        let mut second_spectrum = Array1::<Complex32>::zeros([plan.len()]);
        let mut first_recovered = Array1::<Complex32>::zeros([plan.len()]);
        let mut second_recovered = Array1::<Complex32>::zeros([plan.len()]);

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
        let input = Array1::from(vec![Complex32::new(1.0, 0.0); 4]);
        let mut output = Array1::<Complex32>::zeros([plan.len()]);
        assert!(matches!(
            plan.forward_typed_into(&input, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
            Err(QftError::PrecisionMismatch)
        ));
    }
}
