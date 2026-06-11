//! 1D Fractional Fourier Transform Plan

use super::helpers::leto_view1_cow;
use crate::application::execution::kernel::direct::direct_frft_forward_into;
use crate::application::execution::plan::frft::storage::FrftStorage;
use crate::domain::contracts::error::FrftError;
use apollo_fft::PrecisionProfile;
use ndarray::Array1;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::f64::consts::FRAC_PI_2;

/// Direct fractional Fourier transform plan.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrftPlan {
    n: usize,
    order: f64,
    cot: f64,
    csc: f64,
    scale: Complex64,
}

impl FrftPlan {
    /// Create a validated FrFT plan.
    pub fn new(n: usize, order: f64) -> Result<Self, FrftError> {
        if n == 0 {
            return Err(FrftError::EmptySignal);
        }
        if !order.is_finite() {
            return Err(FrftError::NonFiniteOrder);
        }
        let reduced = order.rem_euclid(4.0);
        let alpha = reduced * FRAC_PI_2;
        let integer_rotation = (reduced - reduced.round()).abs() < 1.0e-12;
        let (cot, csc, scale) = if integer_rotation {
            (0.0, 0.0, Complex64::new(1.0, 0.0))
        } else {
            let sin_alpha = alpha.sin();
            let cot = alpha.cos() / sin_alpha;
            let csc = 1.0 / sin_alpha;
            (
                cot,
                csc,
                (1.0 - Complex64::i() * cot).sqrt() / (n as f64).sqrt(),
            )
        };

        Ok(Self {
            n,
            order,
            cot,
            csc,
            scale,
        })
    }

    /// Return the transform length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.n
    }

    /// Return whether the plan length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.n == 0
    }

    /// Return the fractional order.
    #[must_use]
    pub const fn order(self) -> f64 {
        self.order
    }

    /// Execute the forward FrFT.
    pub fn forward(&self, input: &Array1<Complex64>) -> Result<Array1<Complex64>, FrftError> {
        let mut output = Array1::<Complex64>::zeros(self.n);
        self.forward_into(input, &mut output)?;
        Ok(output)
    }

    /// Execute the forward FrFT over a Leto complex view.
    pub fn forward_leto(
        &self,
        input: leto::ArrayView1<'_, Complex64>,
    ) -> Result<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>, FrftError> {
        let signal = leto_view1_cow(&input);
        let mut output = vec![Complex64::new(0.0, 0.0); self.n];
        self.forward_complex64_slice_into(&signal, &mut output)?;
        Ok(
            leto::Array::<Complex64, leto::MnemosyneStorage<Complex64>, 1>::from_mnemosyne_slice(
                [output.len()],
                &output,
            )
            .expect("FrFT output length must match Leto output shape"),
        )
    }

    /// Execute the forward FrFT over a typed Leto complex view.
    pub fn forward_leto_typed<T: FrftStorage>(
        &self,
        input: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> Result<leto::Array<T, leto::MnemosyneStorage<T>, 1>, FrftError> {
        let signal = leto_view1_cow(&input);
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); self.n];
        T::forward_slice_into(self, &signal, &mut output, profile)?;
        Ok(
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::from_mnemosyne_slice(
                [output.len()],
                &output,
            )
            .expect("typed FrFT output length must match Leto output shape"),
        )
    }

    /// Execute the forward FrFT into a pre-allocated output buffer.
    pub fn forward_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<Complex64>,
    ) -> Result<(), FrftError> {
        self.forward_complex64_slice_into(
            input.as_slice().expect("Array must be contiguous"),
            output.as_slice_mut().expect("Array must be contiguous"),
        )
    }

    /// Execute the forward FrFT over contiguous Complex64 slices.
    pub(crate) fn forward_complex64_slice_into(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), FrftError> {
        if input.len() != self.n {
            return Err(FrftError::LengthMismatch {
                input: input.len(),
                plan: self.n,
            });
        }
        if output.len() != self.n {
            return Err(FrftError::LengthMismatch {
                input: output.len(),
                plan: self.n,
            });
        }
        direct_frft_forward_into(input, output, self.order, self.cot, self.csc, self.scale);
        Ok(())
    }

    /// Execute the inverse FrFT, equivalent to a forward FrFT of order `-a`.
    pub fn inverse(&self, input: &Array1<Complex64>) -> Result<Array1<Complex64>, FrftError> {
        let mut output = Array1::<Complex64>::zeros(self.n);
        self.inverse_into(input, &mut output)?;
        Ok(output)
    }

    /// Execute the inverse FrFT over a Leto complex view.
    pub fn inverse_leto(
        &self,
        input: leto::ArrayView1<'_, Complex64>,
    ) -> Result<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>, FrftError> {
        let signal = leto_view1_cow(&input);
        let mut output = vec![Complex64::new(0.0, 0.0); self.n];
        self.inverse_complex64_slice_into(&signal, &mut output)?;
        Ok(
            leto::Array::<Complex64, leto::MnemosyneStorage<Complex64>, 1>::from_mnemosyne_slice(
                [output.len()],
                &output,
            )
            .expect("inverse FrFT output length must match Leto output shape"),
        )
    }

    /// Execute the inverse FrFT over a typed Leto complex view.
    pub fn inverse_leto_typed<T: FrftStorage>(
        &self,
        input: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> Result<leto::Array<T, leto::MnemosyneStorage<T>, 1>, FrftError> {
        let signal = leto_view1_cow(&input);
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); self.n];
        T::inverse_slice_into(self, &signal, &mut output, profile)?;
        Ok(
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::from_mnemosyne_slice(
                [output.len()],
                &output,
            )
            .expect("typed inverse FrFT output length must match Leto output shape"),
        )
    }

    /// Execute the inverse FrFT into a pre-allocated output buffer.
    pub fn inverse_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<Complex64>,
    ) -> Result<(), FrftError> {
        self.inverse_complex64_slice_into(
            input.as_slice().expect("Array must be contiguous"),
            output.as_slice_mut().expect("Array must be contiguous"),
        )
    }

    /// Execute the inverse FrFT over contiguous Complex64 slices.
    pub(crate) fn inverse_complex64_slice_into(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), FrftError> {
        let inverse_plan = Self::new(self.n, -self.order)?;
        inverse_plan.forward_complex64_slice_into(input, output)
    }

    /// Execute the forward FrFT for `Complex64`, `Complex32`, or mixed `[f16; 2]` storage.
    pub fn forward_typed_into<T: FrftStorage>(
        &self,
        input: &Array1<T>,
        output: &mut Array1<T>,
        profile: PrecisionProfile,
    ) -> Result<(), FrftError> {
        T::forward_into(self, input, output, profile)
    }

    /// Execute the inverse FrFT for `Complex64`, `Complex32`, or mixed `[f16; 2]` storage.
    pub fn inverse_typed_into<T: FrftStorage>(
        &self,
        input: &Array1<T>,
        output: &mut Array1<T>,
        profile: PrecisionProfile,
    ) -> Result<(), FrftError> {
        T::inverse_into(self, input, output, profile)
    }
}

/// Execute a single forward fractional Fourier transform on a 1D array.
pub fn frft(input: &Array1<Complex64>, order: f64) -> Result<Array1<Complex64>, FrftError> {
    FrftPlan::new(input.len(), order)?.forward(input)
}

/// Execute a single forward fractional Fourier transform on a Leto 1D view.
pub fn frft_leto(
    input: leto::ArrayView1<'_, Complex64>,
    order: f64,
) -> Result<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>, FrftError> {
    FrftPlan::new(input.shape()[0], order)?.forward_leto(input)
}

/// Execute a single forward fractional Fourier transform on a typed Leto 1D view.
pub fn frft_leto_typed<T: FrftStorage>(
    input: leto::ArrayView1<'_, T>,
    order: f64,
    profile: PrecisionProfile,
) -> Result<leto::Array<T, leto::MnemosyneStorage<T>, 1>, FrftError> {
    FrftPlan::new(input.shape()[0], order)?.forward_leto_typed(input, profile)
}
