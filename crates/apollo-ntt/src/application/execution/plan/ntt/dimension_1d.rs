//! 1D Number Theoretic Transform Plan

use crate::application::execution::kernel::direct::ntt_kernel;
use crate::domain::contracts::config::{DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT};
use crate::domain::contracts::error::NttError;
use crate::domain::contracts::math::{mod_inv, mod_mul, mod_pow};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Reusable radix-2 NTT plan.
///
/// # Theorem: Exact inverse over a supported prime field
///
/// Let `q` be prime, let `n | q - 1`, and let `omega` be a primitive
/// `n`-th root of unity in `F_q`. The forward NTT computes
///
/// `X[k] = sum_j x[j] omega^(j k)`.
///
/// The inverse computes
///
/// `x[j] = n^{-1} sum_k X[k] omega^(-j k)`.
///
/// Orthogonality gives `sum_k omega^((j-m)k) = n` when `j = m` and `0`
/// otherwise, so the inverse recovers every input residue exactly modulo `q`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NttPlan {
    n: usize,
    modulus: u64,
    primitive_root: u64,
    root: u64,
    root_inv: u64,
    n_inv: u64,
    forward_twiddles: Vec<u64>,
    inverse_twiddles: Vec<u64>,
}

impl NttPlan {
    /// Build a plan with the crate default NTT-friendly prime.
    pub fn new(n: usize) -> Result<Self, NttError> {
        Self::with_modulus(n, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT)
    }

    /// Build a plan with an explicit modulus and primitive root.
    pub fn with_modulus(n: usize, modulus: u64, primitive_root: u64) -> Result<Self, NttError> {
        if n == 0 {
            return Err(NttError::EmptyLength);
        }
        if !n.is_power_of_two() {
            return Err(NttError::NonPowerOfTwo);
        }
        if modulus < 2 {
            return Err(NttError::InvalidModulus);
        }
        if (modulus - 1) % n as u64 != 0 {
            return Err(NttError::UnsupportedLength);
        }

        let root = mod_pow(primitive_root, (modulus - 1) / n as u64, modulus);
        let root_inv = mod_inv(root, modulus);
        let n_inv = mod_inv(n as u64, modulus);
        let forward_twiddles = Self::calculate_twiddles(n, root, modulus);
        let inverse_twiddles = Self::calculate_twiddles(n, root_inv, modulus);

        Ok(Self {
            n,
            modulus,
            primitive_root,
            root,
            root_inv,
            n_inv,
            forward_twiddles,
            inverse_twiddles,
        })
    }

    /// Return the transform length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Return whether the plan length is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Return the modulus.
    #[must_use]
    pub fn modulus(&self) -> u64 {
        self.modulus
    }

    /// Return the primitive root used to derive the stage roots.
    #[must_use]
    pub fn primitive_root(&self) -> u64 {
        self.primitive_root
    }

    /// Allocate and execute the forward transform.
    pub fn forward(&self, input: &Array1<u64>) -> Result<Array1<u64>, NttError> {
        let mut output = Array1::zeros(self.n);
        self.forward_into(input, &mut output)?;
        Ok(output)
    }

    /// Allocate and execute the forward transform over a Leto view.
    ///
    /// Contiguous views are borrowed. Strided views copy once into logical order
    /// before entering the canonical slice execution path.
    pub fn forward_leto(
        &self,
        input: leto::ArrayView1<'_, u64>,
    ) -> Result<leto::Array<u64, leto::MnemosyneStorage<u64>, 1>, NttError> {
        let mut output =
            leto::Array::<u64, leto::MnemosyneStorage<u64>, 1>::zeros_mnemosyne([self.n]);
        self.forward_leto_into(input, output.view_mut())?;
        Ok(output)
    }

    /// Execute the forward transform into caller-owned Leto output storage.
    pub fn forward_leto_into(
        &self,
        input: leto::ArrayView1<'_, u64>,
        output: leto::ArrayViewMut1<'_, u64>,
    ) -> Result<(), NttError> {
        self.leto_into(input, output, |plan, input, output| {
            plan.forward_slice_into(input, output)
        })
    }

    /// Allocate and execute the inverse transform.
    pub fn inverse(&self, input: &Array1<u64>) -> Result<Array1<u64>, NttError> {
        let mut output = Array1::zeros(self.n);
        self.inverse_into(input, &mut output)?;
        Ok(output)
    }

    /// Allocate and execute the inverse transform over a Leto view.
    ///
    /// Contiguous views are borrowed. Strided views copy once into logical order
    /// before entering the canonical slice execution path.
    pub fn inverse_leto(
        &self,
        input: leto::ArrayView1<'_, u64>,
    ) -> Result<leto::Array<u64, leto::MnemosyneStorage<u64>, 1>, NttError> {
        let mut output =
            leto::Array::<u64, leto::MnemosyneStorage<u64>, 1>::zeros_mnemosyne([self.n]);
        self.inverse_leto_into(input, output.view_mut())?;
        Ok(output)
    }

    /// Execute the inverse transform into caller-owned Leto output storage.
    pub fn inverse_leto_into(
        &self,
        input: leto::ArrayView1<'_, u64>,
        output: leto::ArrayViewMut1<'_, u64>,
    ) -> Result<(), NttError> {
        self.leto_into(input, output, |plan, input, output| {
            plan.inverse_slice_into(input, output)
        })
    }

    /// Execute the forward transform into caller-owned output storage.
    pub fn forward_into(
        &self,
        input: &Array1<u64>,
        output: &mut Array1<u64>,
    ) -> Result<(), NttError> {
        self.forward_slice_into(
            input
                .as_slice()
                .expect("owned Array1 storage is contiguous"),
            output
                .as_slice_mut()
                .expect("owned Array1 storage is contiguous"),
        )
    }

    /// Execute the inverse transform into caller-owned output storage.
    pub fn inverse_into(
        &self,
        input: &Array1<u64>,
        output: &mut Array1<u64>,
    ) -> Result<(), NttError> {
        self.inverse_slice_into(
            input
                .as_slice()
                .expect("owned Array1 storage is contiguous"),
            output
                .as_slice_mut()
                .expect("owned Array1 storage is contiguous"),
        )
    }

    /// Execute the forward transform in place.
    pub fn forward_inplace(&self, data: &mut Array1<u64>) -> Result<(), NttError> {
        self.forward_slice_inplace(
            data.as_slice_mut()
                .expect("owned Array1 storage is contiguous"),
        )
    }

    /// Execute the inverse transform in place.
    pub fn inverse_inplace(&self, data: &mut Array1<u64>) -> Result<(), NttError> {
        self.inverse_slice_inplace(
            data.as_slice_mut()
                .expect("owned Array1 storage is contiguous"),
        )
    }

    /// Execute the forward transform into caller-owned contiguous output storage.
    pub fn forward_slice_into(&self, input: &[u64], output: &mut [u64]) -> Result<(), NttError> {
        self.check_len(input.len())?;
        self.check_len(output.len())?;
        output.copy_from_slice(input);
        self.forward_slice_inplace(output)
    }

    /// Execute the inverse transform into caller-owned contiguous output storage.
    pub fn inverse_slice_into(&self, input: &[u64], output: &mut [u64]) -> Result<(), NttError> {
        self.check_len(input.len())?;
        self.check_len(output.len())?;
        output.copy_from_slice(input);
        self.inverse_slice_inplace(output)
    }

    /// Execute the forward transform in caller-owned contiguous storage.
    pub fn forward_slice_inplace(&self, data: &mut [u64]) -> Result<(), NttError> {
        self.check_len(data.len())?;
        for value in data.iter_mut() {
            *value %= self.modulus;
        }
        ntt_kernel(data, &self.forward_twiddles, self.modulus);
        Ok(())
    }

    /// Execute the inverse transform in caller-owned contiguous storage.
    pub fn inverse_slice_inplace(&self, data: &mut [u64]) -> Result<(), NttError> {
        self.check_len(data.len())?;
        for value in data.iter_mut() {
            *value %= self.modulus;
        }
        ntt_kernel(data, &self.inverse_twiddles, self.modulus);
        for value in data.iter_mut() {
            *value = mod_mul(*value, self.n_inv, self.modulus);
        }
        Ok(())
    }

    fn check_len(&self, len: usize) -> Result<(), NttError> {
        if len == self.n {
            Ok(())
        } else {
            Err(NttError::LengthMismatch)
        }
    }

    fn leto_into<F>(
        &self,
        input: leto::ArrayView1<'_, u64>,
        mut output: leto::ArrayViewMut1<'_, u64>,
        execute: F,
    ) -> Result<(), NttError>
    where
        F: Fn(&Self, &[u64], &mut [u64]) -> Result<(), NttError>,
    {
        self.check_len(output.shape()[0])?;
        let signal = leto_view1_cow(&input);
        if let Some(output_slice) = output.as_mut_slice() {
            return execute(self, &signal, output_slice);
        }

        let mut values = vec![0; self.n];
        execute(self, &signal, &mut values)?;
        for (index, value) in values.into_iter().enumerate() {
            *output
                .get_mut([index])
                .expect("Leto output view shape and storage bounds must be valid") = value;
        }
        Ok(())
    }

    /// Precompute twiddle factors omega^j = g^j (mod p) for j = 0..n for the NTT butterfly stages.
    fn calculate_twiddles(n: usize, root: u64, modulus: u64) -> Vec<u64> {
        let mut twiddles = Vec::with_capacity(n.saturating_sub(1));
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let stage_root = mod_pow(root, (n / len) as u64, modulus);
            let mut value = 1;
            for _ in 0..half {
                twiddles.push(value);
                value = mod_mul(value, stage_root, modulus);
            }
            len *= 2;
        }
        twiddles
    }
}

fn leto_view1_cow<'a>(view: &leto::ArrayView1<'a, u64>) -> Cow<'a, [u64]> {
    if let Some(slice) = view.as_slice() {
        return Cow::Borrowed(slice);
    }

    let len = view.shape()[0];
    let mut values = Vec::with_capacity(len);
    for index in 0..len {
        values.push(
            *view
                .get([index])
                .expect("Leto view shape and storage bounds must be valid"),
        );
    }
    Cow::Owned(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use proptest::prelude::*;

    #[test]
    fn zero_and_one_point_cases() {
        assert_eq!(NttPlan::new(0), Err(NttError::EmptyLength));

        let plan = NttPlan::new(1).unwrap();
        let input = array![DEFAULT_MODULUS + 7];
        let spectrum = plan.forward(&input).unwrap();
        assert_eq!(spectrum, array![7]);
        let recovered = plan.inverse(&spectrum).unwrap();
        assert_eq!(recovered, array![7]);
    }

    #[test]
    fn forward_inverse_roundtrip_for_small_vector() {
        let plan = NttPlan::new(8).unwrap();
        let input = array![1, 1, 2, 3, 5, 8, 13, 21];
        let spectrum = plan.forward(&input).unwrap();
        assert_ne!(spectrum, input);
        let recovered = plan.inverse(&spectrum).unwrap();
        assert_eq!(recovered, input);
    }

    #[test]
    fn caller_owned_paths_match_allocating_paths() {
        let plan = NttPlan::new(4).unwrap();
        let input = array![2, 4, 8, 16];
        let expected = plan.forward(&input).unwrap();
        let mut actual = Array1::zeros(4);
        plan.forward_into(&input, &mut actual).unwrap();
        assert_eq!(actual, expected);

        let expected_inverse = plan.inverse(&expected).unwrap();
        plan.inverse_into(&expected, &mut actual).unwrap();
        assert_eq!(actual, expected_inverse);
    }

    #[test]
    fn leto_forward_and_inverse_match_ndarray_path() {
        use leto::Storage;

        let plan = NttPlan::new(8).unwrap();
        let signal = vec![1, 1, 2, 3, 5, 8, 13, 21];
        let ndarray_input = Array1::from_vec(signal.clone());
        let leto_input = leto::Array1::from_shape_vec([8], signal).unwrap();

        let leto_spectrum = plan.forward_leto(leto_input.view()).unwrap();
        let ndarray_spectrum = plan.forward(&ndarray_input).unwrap();
        assert_eq!(
            leto_spectrum.storage().as_slice(),
            ndarray_spectrum.as_slice().unwrap()
        );

        let leto_recovered = plan.inverse_leto(leto_spectrum.view()).unwrap();
        let ndarray_recovered = plan.inverse(&ndarray_spectrum).unwrap();
        assert_eq!(
            leto_recovered.storage().as_slice(),
            ndarray_recovered.as_slice().unwrap()
        );
    }

    #[test]
    fn leto_forward_accepts_strided_logical_view() {
        use leto::{SliceArg, Storage};

        let plan = NttPlan::new(8).unwrap();
        let logical = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let interleaved = logical
            .iter()
            .copied()
            .flat_map(|value| [value, DEFAULT_MODULUS - 1])
            .collect::<Vec<_>>();
        let leto_input = leto::Array1::from_shape_vec([interleaved.len()], interleaved).unwrap();
        let strided = leto_input
            .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
            .unwrap();

        let actual = plan.forward_leto(strided).unwrap();
        let expected = plan.forward(&Array1::from_vec(logical)).unwrap();
        assert_eq!(actual.storage().as_slice(), expected.as_slice().unwrap());
    }

    #[test]
    fn leto_into_paths_reuse_caller_owned_output_storage() {
        use leto::Storage;

        let plan = NttPlan::new(8).unwrap();
        let mut input = leto::Array1::from_shape_vec([8], vec![3, 1, 4, 1, 5, 9, 2, 6]).unwrap();
        let expected_forward = plan
            .forward(&Array1::from_vec(input.storage().as_slice().to_vec()))
            .unwrap();
        let mut output = leto::Array1::zeros([8]);

        plan.forward_leto_into(input.view(), output.view_mut())
            .unwrap();
        assert_eq!(
            output.storage().as_slice(),
            expected_forward.as_slice().unwrap()
        );

        let expected_inverse = plan.inverse(&expected_forward).unwrap();
        plan.inverse_leto_into(output.view(), input.view_mut())
            .unwrap();
        assert_eq!(
            input.storage().as_slice(),
            expected_inverse.as_slice().unwrap()
        );
    }

    #[test]
    fn leto_into_rejects_output_length_mismatch() {
        let plan = NttPlan::new(8).unwrap();
        let input = leto::Array1::from_shape_vec([8], vec![1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let mut output = leto::Array1::zeros([4]);

        assert_eq!(
            plan.forward_leto_into(input.view(), output.view_mut()),
            Err(NttError::LengthMismatch)
        );
    }

    #[test]
    fn leto_into_scatter_writes_strided_output_in_logical_order() {
        use leto::{SliceArg, Storage};

        let plan = NttPlan::new(8).unwrap();
        let logical = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let input = leto::Array1::from_shape_vec([8], logical.clone()).unwrap();
        let expected = plan.forward(&Array1::from_vec(logical)).unwrap();
        let mut backing =
            leto::Array1::from_shape_vec([16], vec![DEFAULT_MODULUS - 1; 16]).unwrap();
        {
            let output = backing
                .view_mut()
                .slice_with_mut::<1>(&[SliceArg::range(Some(0), None, 2)])
                .unwrap();
            plan.forward_leto_into(input.view(), output).unwrap();
        }

        let values = backing.storage().as_slice();
        for index in 0..8 {
            assert_eq!(values[index * 2], expected[index]);
            assert_eq!(values[index * 2 + 1], DEFAULT_MODULUS - 1);
        }
    }

    #[test]
    fn leto_transport_helpers_match_plan_paths() {
        use crate::{intt_leto, ntt_leto};
        use leto::Storage;

        let input = leto::Array1::from_shape_vec([4], vec![1, 2, 3, 4]).unwrap();
        let spectrum = ntt_leto(input.view()).unwrap();
        let recovered = intt_leto(spectrum.view()).unwrap();
        assert_eq!(recovered.storage().as_slice(), input.storage().as_slice());
    }

    #[test]
    fn inputs_are_normalized_to_residue_class() {
        let plan = NttPlan::new(4).unwrap();
        let input = array![DEFAULT_MODULUS + 1, DEFAULT_MODULUS + 2, 3, 4];
        let normalized = array![1, 2, 3, 4];
        assert_eq!(
            plan.forward(&input).unwrap(),
            plan.forward(&normalized).unwrap()
        );
    }

    #[test]
    fn rejects_invalid_lengths() {
        assert_eq!(NttPlan::new(3), Err(NttError::NonPowerOfTwo));
        let plan = NttPlan::new(4).unwrap();
        assert_eq!(plan.forward(&array![1, 2]), Err(NttError::LengthMismatch));
    }

    proptest! {
        #[test]
        fn roundtrip_preserves_residue_class(values in prop::collection::vec(0u64..DEFAULT_MODULUS, 1..=32)) {
            let n = values.len().next_power_of_two();
            let mut padded = values;
            padded.resize(n, 0);
            let input = Array1::from(padded);
            let plan = NttPlan::new(n).unwrap();
            let spectrum = plan.forward(&input).unwrap();
            let recovered = plan.inverse(&spectrum).unwrap();
            prop_assert_eq!(recovered, input);
        }
    }
    #[test]
    fn mod_pow_known_values() {
        use crate::domain::contracts::math::mod_pow;
        assert_eq!(mod_pow(2, 10, 7), 2); // 1024 mod 7 = 2
        assert_eq!(mod_pow(3, 0, 13), 1); // any base^0 = 1
        assert_eq!(mod_pow(3, 1, 13), 3); // identity exponent
        assert_eq!(mod_pow(3, 12, 13), 1); // Fermat: a^(p-1) = 1 mod p
    }

    #[test]
    fn mod_mul_overflow_safety() {
        use crate::domain::contracts::math::mod_mul;
        // (p-1)^2 mod p = 1  because p-1 = -1 mod p and (-1)^2 = 1.
        // Without 128-bit widening this overflows u64.
        let p: u64 = 998_244_353;
        assert_eq!(mod_mul(p - 1, p - 1, p), 1);
    }

    #[test]
    fn mod_inv_known_values() {
        use crate::domain::contracts::math::{mod_inv, mod_mul};
        let p: u64 = 998_244_353;
        let inv3 = mod_inv(3, p);
        assert_eq!(mod_mul(3, inv3, p), 1);
    }

    #[test]
    fn bit_reverse_permute_known_n4() {
        use crate::domain::contracts::math::bit_reverse_permute;
        // n=4, 2-bit indices: 0b00->0b00, 0b01->0b10, 0b10->0b01, 0b11->0b11 => [0,2,1,3]
        let mut data: Vec<u64> = vec![10, 20, 30, 40];
        bit_reverse_permute(&mut data);
        assert_eq!(data, vec![10, 30, 20, 40]);
    }

    #[test]
    fn polynomial_convolution_via_ntt() {
        use crate::domain::contracts::math::mod_mul;
        // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2; zero cubic term.
        // NTT convolution: forward both, multiply pointwise, inverse.
        let p = DEFAULT_MODULUS;
        let plan = NttPlan::new(4).unwrap();
        let a = Array1::from_vec(vec![1u64, 2, 0, 0]);
        let b = Array1::from_vec(vec![3u64, 4, 0, 0]);
        let fa = plan.forward(&a).unwrap();
        let fb = plan.forward(&b).unwrap();
        let fc = Array1::from_shape_fn(4, |i| mod_mul(fa[i], fb[i], p));
        let c = plan.inverse(&fc).unwrap();
        assert_eq!(c[0], 3, "constant term");
        assert_eq!(c[1], 10, "linear term");
        assert_eq!(c[2], 8, "quadratic term");
        assert_eq!(c[3], 0, "cubic term");
    }
}
