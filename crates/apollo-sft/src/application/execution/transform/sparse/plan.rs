//! Sparse FFT plan construction and dense top-K evaluation.

use super::storage::try_mnemosyne_array1_from_slice;
use super::{validate_transform_len, SparseComplexStorage};
use crate::domain::plan::config::{RecoveryRoute, SparseFftConfig};
use crate::domain::spectrum::sparse::SparseSpectrum;
use crate::infrastructure::kernel::downsampled;
use apollo_fft::{ApolloError, ApolloResult, PrecisionProfile};
use eunomia::Complex64;
use leto::Array1;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

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
    /// Create a new sparse FFT plan on the dense top-`K` route.
    pub fn new(n: usize, k: usize) -> ApolloResult<Self> {
        Ok(Self {
            config: SparseFftConfig::new(n, k)?,
        })
    }

    /// Create a sparse FFT plan on the downsampled route (ADR 0064): the
    /// support is found by aliasing the signal onto `bucket_count` buckets
    /// and decoding each, at `O(K log K)` while no bucket holds more than
    /// four tones and exactly for any exactly `K`-sparse input.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::new`], and a validation error on a length the
    /// route cannot alias (see [`SparseFftConfig::downsampled`]).
    pub fn downsampled(n: usize, k: usize) -> ApolloResult<Self> {
        Ok(Self {
            config: SparseFftConfig::downsampled(n, k)?,
        })
    }

    /// Return the recovery route.
    #[must_use]
    pub const fn route(&self) -> RecoveryRoute {
        self.config.route()
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
    /// On the dense route, spectrum density computation is O(N log N) via the
    /// apollo-fft auto-selecting kernel and top-K selection O(N log K) via a
    /// min-heap of size K. On the downsampled route the candidates come from
    /// `O(K log K)` aliased transforms while no bucket holds more than four
    /// tones (ADR 0064), the same heap ranking them.
    ///
    /// ## Recovery guarantee
    /// If the signal is K-sparse in the frequency domain, recovery is exact up to
    /// FFT numerical precision (O(N log N · ε_machine)); see module-level
    /// *Exact Recovery for K-Sparse Signals* theorem.
    /// For approximately sparse signals, the K retained coefficients minimise the
    /// squared time-domain reconstruction residual by Parseval's theorem; see
    /// module-level *Top-K Coefficient Selection Optimality* theorem.
    pub fn forward(&self, signal: &[Complex64]) -> ApolloResult<SparseSpectrum> {
        validate_transform_len(self.len(), signal.len())?;

        // Candidates in ascending frequency order: every bin on the dense
        // route, the resolved tones on the downsampled one, whose count is
        // what keeps that route's ranking sublinear.
        let candidates: Vec<(usize, Complex64)> = match self.route() {
            RecoveryRoute::DenseTopK => {
                // O(N log N) via apollo-fft auto-selecting kernel.
                let mut arr = Array1::from(signal.to_vec());
                apollo_fft::fft_1d_complex_inplace(&mut arr);
                arr.into_vec().into_iter().enumerate().collect()
            }
            RecoveryRoute::Downsampled => {
                let mut resolved = downsampled::recover(signal, self.bucket_count(), self.trials()).map_err(
                    |unresolved| {
                        ApolloError::validation(
                            "signal",
                            format!(
                                "{} of {} buckets unresolved",
                                unresolved.buckets, unresolved.bucket_count
                            ),
                            "the downsampled route resolves an exactly K-sparse signal; this one collided at every bucket count tried",
                        )
                    },
                )?;
                resolved.sort_by_key(|&(frequency, _)| frequency);
                resolved
            }
        };

        // O(N log K) top-K selection via min-heap of size K.
        // Invariant: heap holds at most K entries. heap.pop() evicts the entry
        // with the smallest magnitude (highest index on tie) -- the K-th best.
        let k = self.sparsity();
        let mut heap: BinaryHeap<Reverse<MagIdx>> = BinaryHeap::with_capacity(k + 1);
        for &(i, coeff) in &candidates {
            heap.push(Reverse(MagIdx(coeff.norm_sqr(), i)));
            if heap.len() > k {
                heap.pop();
            }
        }

        // Collect in ascending frequency order and apply threshold filter.
        let mut top_k: Vec<(usize, Complex64)> = heap
            .into_iter()
            .map(|Reverse(MagIdx(_, idx))| {
                let position = candidates
                    .binary_search_by_key(&idx, |&(frequency, _)| frequency)
                    .expect("invariant: the heap holds candidate frequencies");
                candidates[position]
            })
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
        let signal = apollo_leto_interop::view_cow(&signal);
        self.forward(&signal)
    }

    /// Inverse transform from sparse spectrum to a dense complex signal.
    ///
    /// Uses FFTW-compatible normalised inverse FFT (divides by N), matching
    /// the standard IDFT: x_n = (1/N) sum_k X_k exp(2 pi i k n / N).
    pub fn inverse(&self, spectrum: &SparseSpectrum) -> ApolloResult<Vec<Complex64>> {
        spectrum.validate()?;
        validate_transform_len(self.len(), spectrum.n)?;

        let mut arr = Array1::from(spectrum.to_dense());
        apollo_fft::ifft_1d_complex_inplace(&mut arr);
        Ok(arr.into_vec())
    }

    /// Inverse sparse transform into a Mnemosyne-backed Leto 1D complex array.
    pub fn inverse_leto(
        &self,
        spectrum: &SparseSpectrum,
    ) -> ApolloResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let signal = self.inverse(spectrum)?;
        try_mnemosyne_array1_from_slice(&signal)
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

    /// Forward sparse transform for `Complex64`, `Complex32`, or mixed `[F16; 2]` storage.
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
        let signal = apollo_leto_interop::view_cow(&signal);
        let mut frequencies = Vec::new();
        let mut values = Vec::new();
        self.forward_typed_into(&signal, &mut frequencies, &mut values, profile)?;
        let values = try_mnemosyne_array1_from_slice(&values)?;
        Ok(SparseLetoSpectrum {
            frequencies,
            values,
        })
    }

    /// Inverse sparse transform for `Complex64`, `Complex32`, or mixed `[F16; 2]` storage.
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
        let values = apollo_leto_interop::view_cow(&values);
        let mut output = vec![T::from_cpu(Complex64::new(0.0, 0.0)); self.len()];
        self.inverse_typed_into(frequencies, &values, &mut output, profile)?;
        try_mnemosyne_array1_from_slice(&output)
    }
}
