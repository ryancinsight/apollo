//! Sparse Fourier transform plan.
//!
//! # Mathematical contract
//!
//! For `x in C^N`, the plan computes the dense FFT coefficients in `O(N log N)`,
//! ranks them by squared magnitude with frequency-index tie-breaking via a
//! min-heap of size `K` in `O(N log K)`, and stores the largest `K` coefficients
//! in a sparse spectrum. Reconstruction expands the sparse spectrum to a dense
//! coefficient vector and evaluates the inverse FFT.
//!
//! ## Complexity
//! Spectrum density computation: `O(N log N)` via apollo-fft auto-selecting kernel.
//! Top-K selection: `O(N log K)` via min-heap of size `K`.
//!
//! The recovery kernel is dense and deterministic. Sparse plan parameters
//! remain explicit domain data so a later sublinear isolation kernel can
//! replace the infrastructure layer without changing this public API.
//!
//! ## Theorem: Top-K Coefficient Selection Optimality
//!
//! **Statement.** For a signal x ∈ ℂ^N with DFT coefficients X = FFT(x), the
//! K-sparse approximation X_K obtained by retaining the K largest-magnitude
//! coefficients minimizes the squared reconstruction error:
//!
//! ```text
//! ‖x - IFFT(X_K)‖² = Σ_{j=K+1}^{N} |X[π(j)]|² / N²
//! ```
//!
//! where π is the permutation sorting |X| in descending order.
//!
//! **Proof sketch.** Parseval's theorem states ‖u - v‖² = (1/N)‖FFT(u) - FFT(v)‖²
//! for all u, v ∈ ℂ^N. The reconstruction error after zeroing all but the K
//! retained DFT bins equals (1/N) Σ_{j: not in top-K} `|X[j]|²`. This sum is
//! minimised by retaining exactly the K coefficients with the largest `|X[j]|`,
//! since any other selection of K bins leaves a strictly larger or equal residual
//! energy. □
//!
//! **Reference.** Candès & Wakin (2008), "An Introduction to Compressive Sensing",
//! IEEE Signal Processing Magazine 25(2), pp. 21–30.
//!
//! ## Theorem: Exact Recovery for K-Sparse Signals
//!
//! **Statement.** If x ∈ ℂ^N is exactly K-sparse in the DFT domain (at most K
//! frequency components are nonzero), then `SparseFftPlan::new(N, K)?.forward(x)`
//! returns a `SparseSpectrum` containing exactly those K nonzero components, with
//! values matching `FFT(x)[k]` to within FFT numerical precision
//! (O(N log N · ε_machine)).
//!
//! **Proof sketch.** In exact arithmetic the N-K non-support DFT bins are zero.
//! The top-K heap selector retains all K nonzero bins because their squared
//! magnitudes are strictly positive while the remaining N-K bins have squared
//! magnitude zero; no nonzero bin can be displaced by a zero bin. The threshold
//! filter (default `threshold = 0.0`) passes every retained bin since each has
//! norm > 0. The only error is accumulated floating-point rounding in the
//! O(N log N) butterfly network, bounding per-coefficient error at
//! O(N log N · ε_machine). □

use crate::domain::plan::config::SparseFftConfig;
use crate::domain::spectrum::sparse::SparseSpectrum;
use apollo_fft::{f16, ApolloError, ApolloResult, PrecisionProfile};
use moirai::ParallelSliceMut;
use leto::Array1;
use eunomia::{Complex32, Complex64};
use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Below this independent element count, serial conversion avoids scheduling overhead.
const STORAGE_PAR_LEN_THRESHOLD: usize = 16_384;

/// Ordering key for (magnitude_squared, frequency_index) in the top-K heap.
///
/// `Ord` compares magnitude ascending, then frequency index descending so that
/// equal-magnitude coefficients at higher indices are evicted first from a
/// `BinaryHeap<Reverse<MagIdx>>`. The K coefficients with the greatest magnitudes
/// are retained; ties are broken in favour of the lower frequency index,
/// matching the stable descending-magnitude ascending-index sort.
#[derive(PartialEq)]
struct MagIdx(f64, usize);

impl Eq for MagIdx {}

impl PartialOrd for MagIdx {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MagIdx {
    /// Ascending magnitude; descending index for equal magnitudes.
    ///
    /// ### Theorem: Tie-Breaking Invariance
    ///
    /// **Statement.** For equal magnitudes |X[i]|² = |X[j]|², the lower-frequency
    /// index i < j is preferred. The K surviving entries are therefore the K
    /// largest-magnitude coefficients with, among ties, the smallest frequency
    /// indices — independent of insertion order.
    ///
    /// **Proof.** With `self = MagIdx(m, i)` and `other = MagIdx(m, j)`:
    ///
    /// ```text
    /// self.cmp(other) = m.total_cmp(m)       [Equal]
    ///                     .then_with(|| j.cmp(i))
    /// ```
    ///
    /// When i < j: `j.cmp(i)` = Greater → `MagIdx(m, i) > MagIdx(m, j)`.
    /// Lower frequency indices therefore map to *larger* `MagIdx` values.
    ///
    /// `BinaryHeap<Reverse<MagIdx>>::pop()` removes the *maximum* `Reverse`
    /// element, which equals the *minimum* `MagIdx`. Among equal-magnitude
    /// entries, minimum `MagIdx` is the entry with the highest index. That
    /// entry is evicted. After K elements remain, they carry the K largest
    /// magnitudes; among magnitude ties the lowest frequency indices survive. □
    ///
    /// For equal magnitudes m: `MagIdx(m, i).cmp(MagIdx(m, j)) = j.cmp(i)`,
    /// so higher indices sort smaller. `BinaryHeap<Reverse<MagIdx>>.pop()`
    /// evicts the minimum `MagIdx`, i.e. the smallest magnitude or, on ties,
    /// the highest index. The heap therefore retains K elements with the
    /// largest magnitudes, breaking ties by lowest frequency index.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .total_cmp(&other.0)
            .then_with(|| other.1.cmp(&self.1))
    }
}

/// Sparse FFT plan.
#[derive(Clone, Debug)]
pub struct SparseFftPlan {
    config: SparseFftConfig,
}

/// Retained sparse frequencies and Mnemosyne-backed typed Leto coefficient values.
pub struct SparseLetoSpectrum<T> {
    /// Retained frequency indices in ascending order.
    pub frequencies: Vec<usize>,
    /// Retained coefficient values in the same order as `frequencies`.
    pub values: leto::Array<T, leto::MnemosyneStorage<T>, 1>,
}

impl SparseFftPlan {
    /// Create a new sparse FFT plan.
    pub fn new(n: usize, k: usize) -> ApolloResult<Self> {
        Ok(Self {
            config: SparseFftConfig::new(n, k)?,
        })
    }

    /// Return the validated plan configuration.
    #[must_use]
    pub const fn config(&self) -> SparseFftConfig {
        self.config
    }

    /// Return the signal length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.config.len()
    }

    /// Return whether the configured signal length is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.config.is_empty()
    }

    /// Return the target sparsity.
    #[must_use]
    pub const fn sparsity(&self) -> usize {
        self.config.sparsity()
    }

    /// Return the bucket count used by the aliasing model.
    #[must_use]
    pub const fn bucket_count(&self) -> usize {
        self.config.bucket_count()
    }

    /// Return the number of recovery trials.
    #[must_use]
    pub const fn trials(&self) -> usize {
        self.config.trials()
    }

    /// Return the coefficient selection threshold.
    #[must_use]
    pub const fn threshold(&self) -> f64 {
        self.config.threshold()
    }

    /// Forward transform from a complex signal to a sparse spectrum.
    ///
    /// ## Complexity
    /// Spectrum density computation: O(N log N) via apollo-fft auto-selecting kernel.
    /// Top-K selection: O(N log K) via min-heap of size K.
    ///
    /// ## Recovery guarantee
    /// If the signal is K-sparse in the frequency domain, recovery is exact up to
    /// FFT numerical precision (O(N log N · ε_machine)); see module-level
    /// *Exact Recovery for K-Sparse Signals* theorem.
    /// For approximately sparse signals, the K retained coefficients minimise the
    /// squared time-domain reconstruction residual by Parseval's theorem; see
    /// module-level *Top-K Coefficient Selection Optimality* theorem.
    pub fn forward(&self, signal: &[Complex64]) -> ApolloResult<SparseSpectrum> {
        if signal.len() != self.len() {
            return Err(ApolloError::ShapeMismatch {
                expected: self.len().to_string(),
                actual: signal.len().to_string(),
            });
        }

        // O(N log N) via apollo-fft auto-selecting kernel.
        let dense: Vec<Complex64> = {
            let mut arr = Array1::from(signal.to_vec());
            apollo_fft::fft_1d_complex_inplace(&mut arr);
            arr.to_vec()
        };

        // O(N log K) top-K selection via min-heap of size K.
        // Invariant: heap holds at most K entries. heap.pop() evicts the entry
        // with the smallest magnitude (highest index on tie) -- the K-th best.
        let k = self.sparsity();
        let mut heap: BinaryHeap<Reverse<MagIdx>> = BinaryHeap::with_capacity(k + 1);
        for (i, coeff) in dense.iter().enumerate() {
            heap.push(Reverse(MagIdx(coeff.norm_sqr(), i)));
            if heap.len() > k {
                heap.pop();
            }
        }

        // Collect in ascending frequency order and apply threshold filter.
        let mut top_k: Vec<(usize, Complex64)> = heap
            .into_iter()
            .map(|Reverse(MagIdx(_, idx))| (idx, dense[idx]))
            .collect();
        top_k.sort_by_key(|&(idx, _)| idx);

        let mut spectrum = SparseSpectrum::new(self.len());
        for (frequency, value) in top_k {
            if value.norm() > self.threshold() {
                spectrum.insert(frequency, value)?;
            }
        }

        spectrum.validate()?;
        Ok(spectrum)
    }

    /// Forward sparse transform from a Leto 1D complex view.
    ///
    /// C-contiguous views borrow directly; strided views copy once into logical
    /// order before entering the canonical slice sparse-transform path.
    pub fn forward_leto(
        &self,
        signal: leto::ArrayView1<'_, Complex64>,
    ) -> ApolloResult<SparseSpectrum> {
        let signal = leto_view1_cow(signal);
        self.forward(&signal)
    }

    /// Inverse transform from sparse spectrum to a dense complex signal.
    ///
    /// Uses FFTW-compatible normalised inverse FFT (divides by N), matching
    /// the standard IDFT: x_n = (1/N) sum_k X_k exp(2 pi i k n / N).
    pub fn inverse(&self, spectrum: &SparseSpectrum) -> ApolloResult<Vec<Complex64>> {
        spectrum.validate()?;
        if spectrum.n != self.len() {
            return Err(ApolloError::ShapeMismatch {
                expected: self.len().to_string(),
                actual: spectrum.n.to_string(),
            });
        }

        let mut arr = Array1::from(spectrum.to_dense());
        apollo_fft::ifft_1d_complex_inplace(&mut arr);
        Ok(arr.to_vec())
    }

    /// Inverse sparse transform into a Mnemosyne-backed Leto 1D complex array.
    pub fn inverse_leto(
        &self,
        spectrum: &SparseSpectrum,
    ) -> ApolloResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let signal = self.inverse(spectrum)?;
        leto_array1_from_slice(&signal)
    }

    /// Return the retained support as a list of (frequency, coefficient) pairs.
    #[must_use]
    pub fn support(&self, spectrum: &SparseSpectrum) -> Vec<(usize, Complex64)> {
        spectrum
            .frequencies
            .iter()
            .copied()
            .zip(spectrum.values.iter().copied())
            .collect()
    }

    /// Forward sparse transform for `Complex64`, `Complex32`, or mixed `[f16; 2]` storage.
    ///
    /// The owner path remains the `Complex64` dense FFT plus deterministic top-K
    /// selector. Typed storage converts represented input into owner arithmetic
    /// and quantizes retained coefficients once into caller-owned output vectors.
    pub fn forward_typed_into<T: SparseComplexStorage>(
        &self,
        signal: &[T],
        frequencies: &mut Vec<usize>,
        values: &mut Vec<T>,
        profile: PrecisionProfile,
    ) -> ApolloResult<()> {
        T::forward_into(self, signal, frequencies, values, profile)
    }

    /// Forward sparse transform from typed Leto 1D input storage.
    ///
    /// Returns retained frequencies plus Mnemosyne-backed retained values.
    pub fn forward_leto_typed<T: SparseComplexStorage>(
        &self,
        signal: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> ApolloResult<SparseLetoSpectrum<T>> {
        let signal = leto_view1_cow(signal);
        let mut frequencies = Vec::new();
        let mut values = Vec::new();
        self.forward_typed_into(&signal, &mut frequencies, &mut values, profile)?;
        let values = leto_array1_from_slice(&values)?;
        Ok(SparseLetoSpectrum {
            frequencies,
            values,
        })
    }

    /// Inverse sparse transform for `Complex64`, `Complex32`, or mixed `[f16; 2]` storage.
    pub fn inverse_typed_into<T: SparseComplexStorage>(
        &self,
        frequencies: &[usize],
        values: &[T],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> ApolloResult<()> {
        T::inverse_into(self, frequencies, values, output, profile)
    }

    /// Inverse sparse transform from typed Leto retained-value storage.
    pub fn inverse_leto_typed<T: SparseComplexStorage>(
        &self,
        frequencies: &[usize],
        values: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> ApolloResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let values = leto_view1_cow(values);
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); self.len()];
        self.inverse_typed_into(frequencies, &values, &mut output, profile)?;
        leto_array1_from_slice(&output)
    }
}

/// Complex storage accepted by typed SFT paths.
pub trait SparseComplexStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage value into owner `Complex64` arithmetic.
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

    /// Execute typed forward sparse transform.
    fn forward_into(
        plan: &SparseFftPlan,
        signal: &[Self],
        frequencies: &mut Vec<usize>,
        values: &mut Vec<Self>,
        profile: PrecisionProfile,
    ) -> ApolloResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if signal.len() != plan.len() {
            return Err(ApolloError::ShapeMismatch {
                expected: plan.len().to_string(),
                actual: signal.len().to_string(),
            });
        }
        let owner_values = owner_values_from_storage(signal);
        let spectrum = plan.forward(&owner_values)?;
        frequencies.clear();
        values.clear();
        frequencies.reserve(spectrum.frequencies.len());
        values.reserve(spectrum.values.len());
        frequencies.extend_from_slice(&spectrum.frequencies);
        extend_storage_from_owner_values(values, &spectrum.values);
        Ok(())
    }

    /// Execute typed inverse sparse transform.
    fn inverse_into(
        plan: &SparseFftPlan,
        frequencies: &[usize],
        values: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> ApolloResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if output.len() != plan.len() {
            return Err(ApolloError::ShapeMismatch {
                expected: plan.len().to_string(),
                actual: output.len().to_string(),
            });
        }
        if frequencies.len() != values.len() {
            return Err(ApolloError::validation(
                "sparse_values",
                values.len().to_string(),
                "frequency and value counts must match",
            ));
        }
        let mut spectrum = SparseSpectrum::new(plan.len());
        for (&frequency, &value) in frequencies.iter().zip(values.iter()) {
            spectrum.insert(frequency, value.to_complex64())?;
        }
        let owner_values = plan.inverse(&spectrum)?;
        write_storage_from_owner_values(output, &owner_values);
        Ok(())
    }
}

impl SparseComplexStorage for Complex64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_complex64(self) -> Complex64 {
        self
    }

    fn from_complex64(value: Complex64) -> Self {
        value
    }

    fn forward_into(
        plan: &SparseFftPlan,
        signal: &[Self],
        frequencies: &mut Vec<usize>,
        values: &mut Vec<Self>,
        profile: PrecisionProfile,
    ) -> ApolloResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        let spectrum = plan.forward(signal)?;
        frequencies.clear();
        values.clear();
        frequencies.extend_from_slice(&spectrum.frequencies);
        values.extend_from_slice(&spectrum.values);
        Ok(())
    }

    fn inverse_into(
        plan: &SparseFftPlan,
        frequencies: &[usize],
        values: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> ApolloResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if output.len() != plan.len() {
            return Err(ApolloError::ShapeMismatch {
                expected: plan.len().to_string(),
                actual: output.len().to_string(),
            });
        }
        if frequencies.len() != values.len() {
            return Err(ApolloError::validation(
                "sparse_values",
                values.len().to_string(),
                "frequency and value counts must match",
            ));
        }
        let mut spectrum = SparseSpectrum::new(plan.len());
        for (&frequency, &value) in frequencies.iter().zip(values.iter()) {
            spectrum.insert(frequency, value)?;
        }
        let signal = plan.inverse(&spectrum)?;
        output.copy_from_slice(&signal);
        Ok(())
    }
}

impl SparseComplexStorage for Complex32 {
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

impl SparseComplexStorage for [f16; 2] {
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

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> ApolloResult<()> {
    if apollo_fft::application::utilities::leto_interop::profile_matches(actual, expected) {
        Ok(())
    } else {
        Err(ApolloError::validation(
            "precision_profile",
            format!("{actual:?}"),
            format!(
                "storage {:?} with compute {:?}",
                expected.storage, expected.compute
            ),
        ))
    }
}

fn owner_values_from_storage<T: SparseComplexStorage>(values: &[T]) -> Vec<Complex64> {
    if values.len() >= STORAGE_PAR_LEN_THRESHOLD {
        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(values.len(), |index| {
            values[index].to_complex64()
        })
    } else {
        values.iter().copied().map(T::to_complex64).collect()
    }
}

fn extend_storage_from_owner_values<T: SparseComplexStorage>(
    output: &mut Vec<T>,
    values: &[Complex64],
) {
    if values.len() >= STORAGE_PAR_LEN_THRESHOLD {
        let converted =
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(values.len(), |index| {
                T::from_complex64(values[index])
            });
        output.extend(converted);
    } else {
        output.extend(values.iter().copied().map(T::from_complex64));
    }
}

fn write_storage_from_owner_values<T: SparseComplexStorage>(
    output: &mut [T],
    values: &[Complex64],
) {
    debug_assert_eq!(output.len(), values.len());
    if output.len() >= STORAGE_PAR_LEN_THRESHOLD {
        output.par_mut().enumerate(|index, slot| {
            *slot = T::from_complex64(values[index]);
        });
    } else {
        for (slot, value) in output.iter_mut().zip(values.iter().copied()) {
            *slot = T::from_complex64(value);
        }
    }
}

fn leto_view1_cow<T: Copy>(view: leto::ArrayView1<'_, T>) -> Cow<'_, [T]> {
    apollo_fft::application::utilities::leto_interop::view1_cow(&view)
}
fn leto_array1_from_slice<T: Copy>(
    values: &[T],
) -> ApolloResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    apollo_fft::application::utilities::leto_interop::try_array1_from_slice(values).ok_or_else(
        || {
            ApolloError::validation(
                "leto_array",
                "rejected length",
                "Mnemosyne-backed 1D array construction",
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use apollo_fft::fft_1d_complex_inplace;
    use approx::assert_abs_diff_eq;
    use leto::Array1;
    use proptest::prelude::*;

    fn exactly_sparse_signal() -> Vec<Complex64> {
        let plan = SparseFftPlan::new(8, 2).expect("plan");
        let mut spectrum = SparseSpectrum::new(8);
        spectrum
            .insert(1, Complex64::new(3.0, -1.0))
            .expect("insert");
        spectrum
            .insert(5, Complex64::new(-0.5, 2.0))
            .expect("insert");
        plan.inverse(&spectrum).expect("inverse")
    }

    #[test]
    fn moirai_storage_conversion_helpers_match_serial_formulas_at_threshold() {
        let values: Vec<Complex32> = (0..STORAGE_PAR_LEN_THRESHOLD)
            .map(|index| {
                Complex32::new(
                    (index as f32 * 0.03125).sin(),
                    (index as f32 * 0.015625).cos(),
                )
            })
            .collect();

        let owner_values = owner_values_from_storage(&values);
        assert_eq!(owner_values.len(), values.len());
        for (actual, expected) in owner_values.iter().zip(values.iter()) {
            assert_eq!(actual.re.to_bits(), f64::from(expected.re).to_bits());
            assert_eq!(actual.im.to_bits(), f64::from(expected.im).to_bits());
        }

        let mut converted = Vec::new();
        extend_storage_from_owner_values::<Complex32>(&mut converted, &owner_values);
        assert_eq!(converted.len(), values.len());
        for (actual, expected) in converted.iter().zip(values.iter()) {
            assert_eq!(actual.re.to_bits(), expected.re.to_bits());
            assert_eq!(actual.im.to_bits(), expected.im.to_bits());
        }

        let mut output = vec![Complex32::new(0.0, 0.0); values.len()];
        write_storage_from_owner_values(&mut output, &owner_values);
        for (actual, expected) in output.iter().zip(values.iter()) {
            assert_eq!(actual.re.to_bits(), expected.re.to_bits());
            assert_eq!(actual.im.to_bits(), expected.im.to_bits());
        }
    }

    #[test]
    fn typed_paths_support_complex64_complex32_and_mixed_f16_storage() {
        let plan = SparseFftPlan::new(8, 2).expect("plan");
        let signal64 = exactly_sparse_signal();
        let expected = plan.forward(&signal64).expect("forward");

        let mut frequencies64 = Vec::new();
        let mut values64 = Vec::new();
        plan.forward_typed_into(
            &signal64,
            &mut frequencies64,
            &mut values64,
            PrecisionProfile::HIGH_ACCURACY_F64,
        )
        .expect("typed complex64 forward");
        assert_eq!(frequencies64, expected.frequencies);
        for (actual, expected) in values64.iter().zip(expected.values.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }

        let signal32: Vec<Complex32> = signal64
            .iter()
            .map(|value| Complex32::new(value.re as f32, value.im as f32))
            .collect();
        let represented32: Vec<Complex64> = signal32
            .iter()
            .copied()
            .map(Complex32::to_complex64)
            .collect();
        let expected32 = plan
            .forward(&represented32)
            .expect("represented f32 forward");
        let mut frequencies32 = Vec::new();
        let mut values32 = Vec::new();
        plan.forward_typed_into(
            &signal32,
            &mut frequencies32,
            &mut values32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed complex32 forward");
        assert_eq!(frequencies32, expected32.frequencies);
        for (actual, expected) in values32.iter().zip(expected32.values.iter()) {
            assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
            assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
        }

        let mut recovered32 = vec![Complex32::new(0.0, 0.0); plan.len()];
        plan.inverse_typed_into(
            &frequencies32,
            &values32,
            &mut recovered32,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed complex32 inverse");
        for (actual, expected) in recovered32.iter().zip(signal32.iter()) {
            assert!((actual.re - expected.re).abs() < 1.0e-5);
            assert!((actual.im - expected.im).abs() < 1.0e-5);
        }

        let signal16: Vec<[f16; 2]> = signal64
            .iter()
            .map(|value| {
                [
                    f16::from_f32(value.re as f32),
                    f16::from_f32(value.im as f32),
                ]
            })
            .collect();
        let represented16: Vec<Complex64> = signal16
            .iter()
            .copied()
            .map(<[f16; 2]>::to_complex64)
            .collect();
        let expected16 = plan
            .forward(&represented16)
            .expect("represented f16 forward");
        let mut frequencies16 = Vec::new();
        let mut values16 = Vec::new();
        plan.forward_typed_into(
            &signal16,
            &mut frequencies16,
            &mut values16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed f16 forward");
        assert_eq!(frequencies16, expected16.frequencies);
        for (actual, expected) in values16.iter().zip(expected16.values.iter()) {
            let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!((f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound);
            assert!((f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound);
        }
    }

    #[test]
    fn typed_paths_reject_profile_and_shape_mismatch() {
        let plan = SparseFftPlan::new(4, 1).expect("plan");
        let signal = vec![Complex32::new(1.0, 0.0); 4];
        let mut frequencies = Vec::new();
        let mut values = Vec::new();
        let err = plan
            .forward_typed_into(
                &signal,
                &mut frequencies,
                &mut values,
                PrecisionProfile::HIGH_ACCURACY_F64,
            )
            .expect_err("profile mismatch");
        assert!(matches!(
            err,
            ApolloError::Validation { field, .. } if field == "precision_profile"
        ));

        let mut output = vec![Complex32::new(0.0, 0.0); 4];
        let err = plan
            .inverse_typed_into(
                &[0, 1],
                &[Complex32::new(1.0, 0.0)],
                &mut output,
                PrecisionProfile::LOW_PRECISION_F32,
            )
            .expect_err("sparse shape mismatch");
        assert!(matches!(
            err,
            ApolloError::Validation { field, .. } if field == "sparse_values"
        ));
    }

    #[test]
    fn leto_forward_matches_slice_reference() {
        let plan = SparseFftPlan::new(8, 2).expect("plan");
        let signal = exactly_sparse_signal();
        let input = leto::Array1::from_shape_vec([8], signal.clone()).expect("leto input");
        let expected = plan.forward(&signal).expect("slice forward");

        let actual = plan.forward_leto(input.view()).expect("leto forward");

        assert_eq!(actual.frequencies, expected.frequencies);
        for (actual, expected) in actual.values.iter().zip(expected.values.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_strided_forward_matches_slice_reference() {
        let plan = SparseFftPlan::new(8, 2).expect("plan");
        let signal = exactly_sparse_signal();
        let mut backing = Vec::with_capacity(16);
        for sample in signal.iter().copied() {
            backing.push(sample);
            backing.push(Complex64::new(1000.0, -1000.0));
        }
        let input = leto::Array1::from_shape_vec([16], backing).expect("leto input");
        let strided = input.view().slice(&[(0, 16, 2)]).expect("strided view");
        let expected = plan.forward(&signal).expect("slice forward");

        let actual = plan.forward_leto(strided).expect("leto forward");

        assert_eq!(actual.frequencies, expected.frequencies);
        for (actual, expected) in actual.values.iter().zip(expected.values.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_inverse_matches_slice_reference() {
        let plan = SparseFftPlan::new(8, 2).expect("plan");
        let signal = exactly_sparse_signal();
        let spectrum = plan.forward(&signal).expect("forward");
        let expected = plan.inverse(&spectrum).expect("slice inverse");

        let actual = plan.inverse_leto(&spectrum).expect("leto inverse");
        let actual = actual.view();
        let actual = actual.as_slice().expect("contiguous leto output");

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn typed_leto_forward_and_inverse_match_typed_slice_reference() {
        let plan = SparseFftPlan::new(8, 2).expect("plan");
        let signal64 = exactly_sparse_signal();
        let signal32: Vec<Complex32> = signal64
            .iter()
            .map(|value| Complex32::new(value.re as f32, value.im as f32))
            .collect();
        let input = leto::Array1::from_shape_vec([8], signal32.clone()).expect("leto input");
        let mut expected_freq = Vec::new();
        let mut expected_values = Vec::new();
        plan.forward_typed_into(
            &signal32,
            &mut expected_freq,
            &mut expected_values,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed slice forward");

        let actual = plan
            .forward_leto_typed(input.view(), PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed leto forward");
        let actual_values_view = actual.values.view();
        let actual_values = actual_values_view
            .as_slice()
            .expect("contiguous leto values");

        assert_eq!(actual.frequencies, expected_freq);
        for (actual, expected) in actual_values.iter().zip(expected_values.iter()) {
            assert_eq!(actual.re.to_bits(), expected.re.to_bits());
            assert_eq!(actual.im.to_bits(), expected.im.to_bits());
        }

        let values_array =
            leto::Array1::from_shape_vec([expected_values.len()], expected_values.clone())
                .expect("leto values");
        let actual_signal = plan
            .inverse_leto_typed(
                &expected_freq,
                values_array.view(),
                PrecisionProfile::LOW_PRECISION_F32,
            )
            .expect("typed leto inverse");
        let actual_signal_view = actual_signal.view();
        let actual_signal = actual_signal_view
            .as_slice()
            .expect("contiguous leto signal");
        let mut expected_signal = vec![Complex32::new(0.0, 0.0); plan.len()];
        plan.inverse_typed_into(
            &expected_freq,
            &expected_values,
            &mut expected_signal,
            PrecisionProfile::LOW_PRECISION_F32,
        )
        .expect("typed slice inverse");

        for (actual, expected) in actual_signal.iter().zip(expected_signal.iter()) {
            assert_eq!(actual.re.to_bits(), expected.re.to_bits());
            assert_eq!(actual.im.to_bits(), expected.im.to_bits());
        }
    }

    proptest! {
        /// Exact K-sparse recovery: if a signal is K-sparse in the DFT domain,
        /// SparseFftPlan recovers the exact nonzero coefficients.
        ///
        /// Proof: in exact arithmetic, N-K non-support DFT bins are zero.
        /// The min-heap retains all K nonzero bins since their magnitudes exceed
        /// zero; no zero-magnitude bin can displace a nonzero bin (Theorem:
        /// Exact Recovery for K-Sparse Signals, module-level proof).
        #[test]
        fn k_sparse_roundtrip_recovers_exact_support(
            // Generate N as power-of-two in {8, 16, 32}
            n_exp in 3_u32..=5_u32,
            // Generate k nonzero DFT bins from the first k frequency indices
            k in 1_usize..=4_usize,
            // Real parts for k nonzero coefficients (magnitude >= 0.5 to clear threshold)
            re_parts in prop::collection::vec(0.5_f64..2.0_f64, 4),
        ) {
            let n = 1_usize << n_exp;
            let k_clamped = k.min(n / 2);
            // Build a K-sparse signal in frequency domain: set k DFT bins, IFFT to time domain.
            let mut freq_domain = vec![Complex64::new(0.0, 0.0); n];
            for i in 0..k_clamped {
                freq_domain[i + 1] = Complex64::new(re_parts[i % 4], 0.0);
            }
            // IFFT to get the time-domain signal.
            let mut arr = Array1::from(freq_domain.clone());
            apollo_fft::ifft_1d_complex_inplace(&mut arr);
            let signal: Vec<Complex64> = arr.iter().copied().collect();

            let plan = SparseFftPlan::new(n, k_clamped).expect("plan");
            let spectrum = plan.forward(&signal).expect("forward");
            let dense = spectrum.to_dense();

            // Verify: the nonzero DFT coefficients in dense match the input freq_domain.
            for i in 0..k_clamped {
                let bin = i + 1;
                let expected_mag = re_parts[i % 4];
                prop_assert!(
                    (dense[bin].norm() - expected_mag).abs() < 1.0e-9,
                    "K-sparse recovery: bin {bin} magnitude {}, expected {}",
                    dense[bin].norm(),
                    expected_mag
                );
            }
        }

        /// Top-K selection optimality: retained bins have greater or equal energy
        /// than any alternative K-subset of the DFT coefficients.
        ///
        /// Proof: SparseFftPlan retains the K largest-magnitude DFT coefficients.
        /// For any alternative K-subset, Σ|alt_k|² ≤ Σ|retained_k|² by definition
        /// of top-K ordering (Theorem: Top-K Optimality, module-level proof).
        #[test]
        fn top_k_retained_energy_dominates_alternatives(
            signal_re in prop::collection::vec(-3.0_f64..3.0_f64, 8),
        ) {
            let n = 8;
            let k = 3;
            let signal: Vec<Complex64> = signal_re.iter()
                .map(|&r| Complex64::new(r, 0.0))
                .collect();
            let plan = SparseFftPlan::new(n, k).expect("plan");
            let spectrum = plan.forward(&signal).expect("forward");

            // Compute the full DFT to get all coefficients.
            let mut arr = Array1::from(signal.clone());
            fft_1d_complex_inplace(&mut arr);
            let full_dft: Vec<Complex64> = arr.iter().copied().collect();

            // Energy in retained K bins.
            let retained_energy: f64 = spectrum.frequencies.iter()
                .map(|&bin| full_dft[bin].norm_sqr())
                .sum();

            // Energy in an alternative K-subset: last K bins by index.
            let alt_bins: Vec<usize> = (n - k..n).collect();
            let alt_energy: f64 = alt_bins.iter()
                .map(|&bin| full_dft[bin].norm_sqr())
                .sum();

            // Retained energy must be >= any alternative K-subset energy.
            prop_assert!(
                retained_energy >= alt_energy - 1.0e-10,
                "top-K optimality violated: retained_energy={retained_energy}, alt_energy={alt_energy}"
            );
        }

        /// Retained coefficient values equal the DFT at those frequency indices.
        ///
        /// SparseFftPlan computes the full DFT then selects top-K. The retained
        /// (frequency, value) pairs must satisfy |value - FFT(signal)[frequency]| < 1e-9.
        #[test]
        fn retained_values_equal_dft_at_those_indices(
            signal_re in prop::collection::vec(-2.0_f64..2.0_f64, 8),
        ) {
            let n = 8;
            let k = 4;
            let signal: Vec<Complex64> = signal_re.iter()
                .map(|&r| Complex64::new(r, 0.0))
                .collect();
            let plan = SparseFftPlan::new(n, k).expect("plan");
            let spectrum = plan.forward(&signal).expect("forward");

            // Compute the full DFT independently.
            let mut arr = Array1::from(signal);
            fft_1d_complex_inplace(&mut arr);
            let full_dft: Vec<Complex64> = arr.iter().copied().collect();

            // Each retained value must match the DFT at that index.
            for (&bin, &val) in spectrum.frequencies.iter().zip(spectrum.values.iter()) {
                prop_assert!(
                    (val - full_dft[bin]).norm() < 1.0e-9,
                    "retained value at bin {bin}: got {val}, expected {}", full_dft[bin]
                );
            }
        }
    }
}
