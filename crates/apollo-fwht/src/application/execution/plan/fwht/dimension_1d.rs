//! 1D Fast Walsh-Hadamard Transform plan.

use crate::application::execution::kernel::direct::wht_inplace;
use crate::domain::contracts::error::FwhtError;
use apollo_fft::PrecisionProfile;
use ndarray::Array1;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

use super::storage::FwhtStorage;

/// Reusable FWHT plan.
///
/// Stores the validated transform length. All methods validate input length
/// and return Err(FwhtError::LengthMismatch) instead of panicking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FwhtPlan {
    n: usize,
}

impl FwhtPlan {
    /// Create a validated FWHT plan.
    ///
    /// # Errors
    /// Returns Err(FwhtError::EmptyInput) if n == 0.
    /// Returns Err(FwhtError::NonPowerOfTwo) if n is not a power of two.
    pub fn new(n: usize) -> Result<Self, FwhtError> {
        if n == 0 {
            return Err(FwhtError::EmptyInput);
        }
        if !n.is_power_of_two() {
            return Err(FwhtError::NonPowerOfTwo);
        }
        Ok(Self { n })
    }

    /// Return the transform length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.n
    }

    /// Return true when the plan length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.n == 0
    }

    /// Forward WHT of a real-valued vector. O(N log N).
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when input.len() != self.n.
    pub fn forward(&self, input: &Array1<f64>) -> Result<Array1<f64>, FwhtError> {
        let mut data = Array1::zeros(self.n);
        self.forward_into(input, &mut data)?;
        Ok(data)
    }

    /// Forward WHT over a Leto real-valued view.
    ///
    /// Contiguous views are borrowed. Strided views copy once into logical order
    /// before entering the canonical slice execution path.
    pub fn forward_leto(
        &self,
        input: leto::ArrayView1<'_, f64>,
    ) -> Result<leto::Array<f64, leto::MnemosyneStorage<f64>, 1>, FwhtError> {
        let signal = leto_view1_cow(&input);
        let mut output = vec![0.0; self.n];
        self.forward_f64_slice_into(&signal, &mut output)?;
        Ok(
            leto::Array::<f64, leto::MnemosyneStorage<f64>, 1>::from_mnemosyne_slice(
                [output.len()],
                &output,
            )
            .expect("FWHT output length must match Leto output shape"),
        )
    }

    /// Forward WHT over a typed Leto real-valued view.
    pub fn forward_leto_typed<T: FwhtStorage>(
        &self,
        input: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> Result<leto::Array<T, leto::MnemosyneStorage<T>, 1>, FwhtError> {
        let signal = leto_view1_cow(&input);
        let mut output = vec![T::from_f64(0.0); self.n];
        T::forward_slice_into(self, &signal, &mut output, profile)?;
        Ok(
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::from_mnemosyne_slice(
                [output.len()],
                &output,
            )
            .expect("typed FWHT output length must match Leto output shape"),
        )
    }

    /// Forward WHT into caller-owned output. O(N log N).
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when either array length differs
    /// from the plan.
    pub fn forward_into(
        &self,
        input: &Array1<f64>,
        output: &mut Array1<f64>,
    ) -> Result<(), FwhtError> {
        self.forward_f64_slice_into(
            input.as_slice().expect("Array must be contiguous"),
            output.as_slice_mut().expect("Array must be contiguous"),
        )
    }

    /// Forward WHT over contiguous f64 slices.
    pub(crate) fn forward_f64_slice_into(
        &self,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FwhtError> {
        if input.len() != self.n || output.len() != self.n {
            return Err(FwhtError::LengthMismatch);
        }
        output.copy_from_slice(input);
        wht_inplace(output);
        Ok(())
    }

    /// Forward WHT in-place. O(N log N) butterfly operations.
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when data.len() != self.n.
    pub fn forward_inplace(&self, data: &mut Array1<f64>) -> Result<(), FwhtError> {
        if data.len() != self.n {
            return Err(FwhtError::LengthMismatch);
        }
        wht_inplace(data.as_slice_mut().expect("Array must be contiguous"));
        Ok(())
    }

    /// Inverse WHT of a real-valued spectrum. Applies WHT then divides by N.
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when input.len() != self.n.
    pub fn inverse(&self, input: &Array1<f64>) -> Result<Array1<f64>, FwhtError> {
        let mut data = Array1::zeros(self.n);
        self.inverse_into(input, &mut data)?;
        Ok(data)
    }

    /// Inverse WHT over a Leto real-valued view.
    ///
    /// Contiguous views are borrowed. Strided views copy once into logical order
    /// before entering the canonical slice execution path.
    pub fn inverse_leto(
        &self,
        input: leto::ArrayView1<'_, f64>,
    ) -> Result<leto::Array<f64, leto::MnemosyneStorage<f64>, 1>, FwhtError> {
        let signal = leto_view1_cow(&input);
        let mut output = vec![0.0; self.n];
        self.inverse_f64_slice_into(&signal, &mut output)?;
        Ok(
            leto::Array::<f64, leto::MnemosyneStorage<f64>, 1>::from_mnemosyne_slice(
                [output.len()],
                &output,
            )
            .expect("inverse FWHT output length must match Leto output shape"),
        )
    }

    /// Inverse WHT over a typed Leto real-valued view.
    pub fn inverse_leto_typed<T: FwhtStorage>(
        &self,
        input: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> Result<leto::Array<T, leto::MnemosyneStorage<T>, 1>, FwhtError> {
        let signal = leto_view1_cow(&input);
        let mut output = vec![T::from_f64(0.0); self.n];
        T::inverse_slice_into(self, &signal, &mut output, profile)?;
        Ok(
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::from_mnemosyne_slice(
                [output.len()],
                &output,
            )
            .expect("typed inverse FWHT output length must match Leto output shape"),
        )
    }

    /// Inverse WHT into caller-owned output. Applies WHT then divides by N.
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when either array length differs
    /// from the plan.
    pub fn inverse_into(
        &self,
        input: &Array1<f64>,
        output: &mut Array1<f64>,
    ) -> Result<(), FwhtError> {
        self.inverse_f64_slice_into(
            input.as_slice().expect("Array must be contiguous"),
            output.as_slice_mut().expect("Array must be contiguous"),
        )
    }

    /// Inverse WHT over contiguous f64 slices.
    pub(crate) fn inverse_f64_slice_into(
        &self,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), FwhtError> {
        if input.len() != self.n || output.len() != self.n {
            return Err(FwhtError::LengthMismatch);
        }
        output.copy_from_slice(input);
        wht_inplace(output);
        let scale = 1.0 / self.n as f64;
        for value in output.iter_mut() {
            *value *= scale;
        }
        Ok(())
    }

    /// Inverse WHT in-place. Applies WHT then divides by N.
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when data.len() != self.n.
    pub fn inverse_inplace(&self, data: &mut Array1<f64>) -> Result<(), FwhtError> {
        if data.len() != self.n {
            return Err(FwhtError::LengthMismatch);
        }
        wht_inplace(data.as_slice_mut().expect("Array must be contiguous"));
        let scale = 1.0 / self.n as f64;
        data.mapv_inplace(|value| value * scale);
        Ok(())
    }

    /// Forward WHT of a complex-valued vector.
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when input.len() != self.n.
    pub fn forward_complex(
        &self,
        input: &Array1<Complex64>,
    ) -> Result<Array1<Complex64>, FwhtError> {
        let mut data = input.clone();
        self.forward_complex_into(input, &mut data)?;
        Ok(data)
    }

    /// Forward complex WHT into caller-owned output.
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when either array length differs
    /// from the plan.
    pub fn forward_complex_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<Complex64>,
    ) -> Result<(), FwhtError> {
        if input.len() != self.n || output.len() != self.n {
            return Err(FwhtError::LengthMismatch);
        }
        output.assign(input);
        self.forward_complex_inplace(output)
    }

    /// Forward complex WHT in-place.
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when data.len() != self.n.
    pub fn forward_complex_inplace(&self, data: &mut Array1<Complex64>) -> Result<(), FwhtError> {
        if data.len() != self.n {
            return Err(FwhtError::LengthMismatch);
        }
        wht_inplace(data.as_slice_mut().expect("Array must be contiguous"));
        Ok(())
    }

    /// Inverse WHT of a complex-valued spectrum. Applies WHT then divides by N.
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when input.len() != self.n.
    pub fn inverse_complex(
        &self,
        input: &Array1<Complex64>,
    ) -> Result<Array1<Complex64>, FwhtError> {
        let mut data = input.clone();
        self.inverse_complex_into(input, &mut data)?;
        Ok(data)
    }

    /// Inverse complex WHT into caller-owned output.
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when either array length differs
    /// from the plan.
    pub fn inverse_complex_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<Complex64>,
    ) -> Result<(), FwhtError> {
        if input.len() != self.n || output.len() != self.n {
            return Err(FwhtError::LengthMismatch);
        }
        output.assign(input);
        self.inverse_complex_inplace(output)
    }

    /// Inverse complex WHT in-place.
    ///
    /// # Errors
    /// Returns Err(FwhtError::LengthMismatch) when data.len() != self.n.
    pub fn inverse_complex_inplace(&self, data: &mut Array1<Complex64>) -> Result<(), FwhtError> {
        if data.len() != self.n {
            return Err(FwhtError::LengthMismatch);
        }
        wht_inplace(data.as_slice_mut().expect("Array must be contiguous"));
        let scale = 1.0 / self.n as f64;
        data.mapv_inplace(|value| value * scale);
        Ok(())
    }

    /// Execute the unnormalized FWHT for `f64`, `f32`, or mixed `f16` storage.
    ///
    /// `f64` and `f32` use native Hadamard butterflies. Mixed `f16` storage
    /// converts through `f32` compute and quantizes once into the caller-owned
    /// output.
    pub fn forward_typed_into<T: FwhtStorage>(
        &self,
        input: &Array1<T>,
        output: &mut Array1<T>,
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        T::forward_into(self, input, output, profile)
    }

    /// Execute the normalized inverse FWHT for `f64`, `f32`, or mixed `f16` storage.
    pub fn inverse_typed_into<T: FwhtStorage>(
        &self,
        input: &Array1<T>,
        output: &mut Array1<T>,
        profile: PrecisionProfile,
    ) -> Result<(), FwhtError> {
        T::inverse_into(self, input, output, profile)
    }
}

fn leto_view1_cow<'a, T: Copy>(view: &leto::ArrayView1<'a, T>) -> Cow<'a, [T]> {
    apollo_fft::application::utilities::leto_interop::view1_cow(view)
}

#[cfg(test)]
mod tests;
